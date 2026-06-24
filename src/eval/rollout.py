#!/usr/bin/env python3
"""Closed-loop rollout runner: a model converses with the harness through the shared protocol
(src/sft/protocol.py) until answer_from_context; the answer is scored against the gold
SQL executed on the real DB. One runner serves three purposes: (a) few-shot baseline eval of
the base model, (b) post-SFT eval, (c) later, the RL environment interaction loop.

Modes
  live   : .venv/bin/python src/eval/rollout.py --base-url http://127.0.0.1:8000/v1 \
               --model Qwen/Qwen2.5-3B-Instruct [--n 50] [--few-shot 2] \
               [--result-dir data/results/run_name]
           (vLLM on the GPU server; reach it from the Mac via `ssh -L 8000:127.0.0.1:8000 NewGNN`)
  replay : .venv/bin/python src/eval/rollout.py --replay 50
           No model: re-executes recorded dev trajectories through the same loop machinery.
           Verified data must score ~100% — this validates executor wiring + scoring.

Scoring: prefer the rows of the evidence table the model cites (answers are grounded in
cited tables; also robust to >50-row answers, which the answer field truncates); fall back
to the literal answer rows for scalars / missing citation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from copy import deepcopy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "harness"))
sys.path.insert(0, os.path.join(ROOT, "src", "sft"))

from executor import Harness                                    # noqa: E402
from plan import resolve_cond, _value_ref_ids                  # noqa: E402
from scalar_grounding import extract_scalar                    # noqa: E402
from provenance import build_references                        # noqa: E402
from emitter import _catalog                                   # noqa: E402
from artifacts import ArtifactWriter                           # noqa: E402
from protocol import (SYSTEM_PROMPT, ProtocolError, TOOLS,      # noqa: E402
                      assistant_message, first_user_message, parse_assistant,
                      rows_equal, tool_output_message)

SPIDER = os.path.join(ROOT, "data", "spider_data")
MAX_CONSECUTIVE_ERRORS = 3   # error feedback turns allowed before aborting the trajectory
DEFAULT_FEWSHOT_IDS = ["spider_train_0", "spider_train_1"]


def db_path(db_id: str) -> str:
    return os.path.join(SPIDER, "database", db_id, f"{db_id}.sqlite")


def overview(h: Harness) -> dict:
    # V2-ctx: the opening overview is the lazy catalog (names + row_counts + FK relations, no columns)
    # — the SAME renderer the emitter uses, so eval matches training. describe_table acquires schema.
    return _catalog(h)


def new_ctx() -> dict:
    """Online harness state threaded across one trajectory's actions (history + provenance)."""
    return {"history": {}, "handle_to_step": {}}


def execute_tool(h: Harness, tool: str, args: dict, ctx: dict, step_id: str):
    """Run one tool call, threading online provenance in `ctx`. Returns (output, created|None).

    V2b: a predicate's `value_ref` cites the producing step_id directly (no add_to_memory); the
    harness grounds the scalar from `ctx["history"]` with strict validation. An illegal value_ref
    (unknown step / non-scalar source) raises ScalarGroundingError -> surfaced as an execution_error."""
    if tool not in TOOLS or tool == "answer_from_context":
        raise ProtocolError(f"tool {tool!r} not executable here")

    def resolve_step(ref):
        if ref in ctx["handle_to_step"]:
            return ctx["handle_to_step"][ref]
        if ref in ctx["history"]:            # already a step_id (a value_ref)
            return ref
        return None
    references = build_references(tool, args, resolve_step)

    if tool in ("describe_table", "inspect_column", "read_subtable"):   # read-only perception; no table
        out = getattr(h, tool)(**args)
        output = out if isinstance(out, dict) else {"rows": [list(r) for r in out], "row_count": len(out)}
        ctx["history"][step_id] = {"tool": tool, "arguments": args, "output": output, "references": references}
        return output, None

    exec_args = dict(args)
    if tool == "condition_filter":
        # resolve each value_ref (a producing step_id) to its grounded scalar; a bad reference raises
        values = {ref: extract_scalar(ctx["history"], ref)
                  for ref in _value_ref_ids(exec_args.get("conditions"))}
        exec_args["conditions"] = resolve_cond(exec_args.get("conditions"), {}, values)
    out = getattr(h, tool)(**exec_args)
    if isinstance(out, dict) and "table_name" in out:
        output = {"table": out["table_name"], "kind": out["kind"],     # V2-ctx: metadata-only handle
                  "columns": out["columns"], "row_count": out["row_count"]}
        if out["row_count"] == 1 and len(out["columns"]) == 1:         # scalar-shaped result keeps its cell
            output["rows"] = [list(r) for r in h.rows(out["table_name"])]
        ctx["handle_to_step"][out["table_name"]] = step_id
        created = out["table_name"]
    else:
        rows = out if isinstance(out, list) else [(out,)]
        output = {"result_sample": [list(r) for r in rows[:5]], "row_count": len(rows)}
        created = None
    ctx["history"][step_id] = {"tool": tool, "arguments": args, "output": output, "references": references}
    return output, created


def score(h: Harness, gold_sql: str, answer_args: dict, created: set) -> tuple[bool, list, list]:
    gold = h.gold(gold_sql)
    ev = (answer_args.get("evidence") or {}).get("table")
    pred = None
    if ev and ev in created:
        try:
            pred = h.rows(ev)
        except Exception:
            pred = None
    if pred is None:
        pred = answer_args.get("answer") or []
    try:
        ok = rows_equal(pred, gold)
    except Exception:
        ok = False
    return ok, pred[:5], gold[:5]


# ---------------- live mode ----------------
def chat(base_url: str, model: str, messages: list[dict], max_tokens: int = 2048) -> str:
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    # Qwen3.5 is a thinking model. EVAL_ENABLE_THINKING=0 disables chain-of-thought via the chat
    # template so the baseline is directly comparable to the non-thinking Qwen2.5 runs and stays
    # within max_tokens; =1 forces it on; unset leaves the model/template default.
    think = os.environ.get("EVAL_ENABLE_THINKING")
    if think is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": think == "1"}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def fewshot_text(trajectory_ids: list[str]) -> str:
    """Render fixed train trajectories as example transcripts appended to the system prompt."""
    if not trajectory_ids:
        return ""
    wanted = set(trajectory_ids)
    found = {}
    with open(os.path.join(ROOT, "data", "trajectories", "spider_train_v2.jsonl")) as f:
        for line in f:
            t = json.loads(line)
            if t["trajectory_id"] in wanted:
                found[t["trajectory_id"]] = t
    missing = [trajectory_id for trajectory_id in trajectory_ids if trajectory_id not in found]
    if missing:
        raise ValueError(f"few-shot trajectories not found: {missing}")
    blocks = []
    for trajectory_id in trajectory_ids:
        t = found[trajectory_id]
        ov = t["initial_state"]["dataset_overview"]
        lines = [f"USER: {first_user_message(ov, t['question'])}"]
        for i, s in enumerate(t["steps"]):
            tc = s["tool_call"]
            lines.append(
                f"ASSISTANT: "
                f"{assistant_message(s.get('think', ''), tc['tool'], tc['arguments'])}"
            )
            if i < len(t["steps"]) - 1:
                lines.append(f"USER: {tool_output_message(s['step_id'], s['tool_output'])}")
        blocks.append("\n".join(lines))
    return "\n\nEXAMPLE SESSIONS\n" + "\n\n---\n\n".join(blocks)


def run_live(
    ex: dict, example_index: int, base_url: str, model: str, system: str, max_steps: int
) -> dict:
    h = Harness(db_path(ex["db_id"]))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": first_user_message(overview(h), ex["question"])}]
    initial_messages = deepcopy(messages)
    created: set[str] = set()
    ctx = new_ctx()
    steps = errors = consecutive = 0
    text = ""
    turns = []
    started = time.time()
    rec = {
        "example_index": example_index,
        "db_id": ex["db_id"],
        "question": ex["question"],
        "gold_sql": ex["query"],
        "initial_model_input": initial_messages,
        "turns": turns,
        "correct": False,
        "legal": False,
        "steps": 0,
        "errors": 0,
        "failure_type": None,
        "fail": None,
    }

    while steps < max_steps:
        turn = {"turn_index": len(turns), "model_input": deepcopy(messages)}
        try:
            text = chat(base_url, model, messages)
        except Exception as e:
            rec["failure_type"] = "api_error"
            rec["fail"] = f"api: {type(e).__name__}: {e}"
            turn["api_error"] = rec["fail"]
            turns.append(turn)
            rec["elapsed_seconds"] = round(time.time() - started, 3)
            return rec
        turn["model_output"] = text
        messages.append({"role": "assistant", "content": text})
        try:
            think, tool, args = parse_assistant(text)
            turn["parsed"] = {"think": think, "tool": tool, "arguments": args}
            if tool == "answer_from_context":
                rec["legal"] = True
                rec["steps"] = steps + 1
                rec["errors"] = errors
                rec["correct"], rec["pred_sample"], rec["gold_sample"] = score(h, ex["query"], args, created)
                if not rec["correct"]:
                    rec["failure_type"] = "wrong_answer"
                turns.append(turn)
                rec["final_messages"] = messages
                rec["elapsed_seconds"] = round(time.time() - started, 3)
                return rec
            step_id = f"step_{steps + 1}"
            out, tname = execute_tool(h, tool, args, ctx, step_id)
            turn["tool_output"] = out
        except (ProtocolError, Exception) as e:  # noqa: BLE001 — every failure becomes feedback
            errors += 1
            consecutive += 1
            error = f"{type(e).__name__}: {e}"
            turn["execution_error"] = error
            turn["execution_error_type"] = (
                "protocol_error" if isinstance(e, ProtocolError) else "execution_error"
            )
            turns.append(turn)
            if consecutive >= MAX_CONSECUTIVE_ERRORS:
                rec["failure_type"] = turn["execution_error_type"]
                rec["fail"] = f"aborted after {consecutive} consecutive errors: {error}"
                rec["errors"] = errors
                rec["steps"] = steps
                rec["final_messages"] = messages
                rec["elapsed_seconds"] = round(time.time() - started, 3)
                return rec
            messages.append({"role": "user", "content": json.dumps(
                {"step_id": f"step_{steps + 1}", "status": "error",
                 "error": {"type": turn["execution_error_type"], "message": error}})})
            continue
        turns.append(turn)
        consecutive = 0
        steps += 1
        if tname:
            created.add(tname)
        messages.append({"role": "user", "content": tool_output_message(step_id, out)})

    rec["failure_type"] = "max_steps"
    rec["fail"] = "max_steps"
    rec["steps"] = steps
    rec["errors"] = errors
    rec["final_messages"] = messages
    rec["elapsed_seconds"] = round(time.time() - started, 3)
    return rec


# ---------------- replay mode (no model) ----------------
def run_replay(n: int, path: str = "") -> int:
    path = path or os.path.join(ROOT, "data", "trajectories", "spider_dev_v2ctx.jsonl")
    total = ok_exec = ok_score = 0
    with open(path) as f:
        for line in f:
            if total >= n:
                break
            t = json.loads(line)
            total += 1
            h = Harness(db_path(t["source"]["db_id"]))
            created: set[str] = set()
            ctx = new_ctx()
            try:
                for s in t["steps"][:-1]:
                    tc = s["tool_call"]
                    _, tname = execute_tool(h, tc["tool"], tc["arguments"], ctx, s["step_id"])
                    if tname:
                        created.add(tname)
                ok_exec += 1
            except Exception as e:
                print(f"  REPLAY EXEC FAIL {t['trajectory_id']}: {type(e).__name__}: {e}")
                continue
            args = t["steps"][-1]["tool_call"]["arguments"]
            correct, _, _ = score(h, t["source"]["gold_sql"], args, created)
            ok_score += correct
            if not correct:
                print(f"  REPLAY SCORE FAIL {t['trajectory_id']}")
    print(f"replay: {total} trajectories | executed {ok_exec} | scored correct {ok_score} "
          f"({100 * ok_score / max(1, total):.1f}%)  — expect ~100% on verified data")
    return 0 if ok_score == total else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=int, default=0, help="replay N dev trajectories (no model)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--n", type=int, default=50, help="number of dev questions (live mode)")
    ap.add_argument("--few-shot", type=int, default=0)
    ap.add_argument("--few-shot-ids", nargs="*", default=DEFAULT_FEWSHOT_IDS)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent questions (vLLM batches requests; each worker owns its Harness/sqlite)")
    ap.add_argument("--result-dir", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.replay:
        return run_replay(args.replay)

    indexed_dev = list(enumerate(json.load(open(os.path.join(SPIDER, "dev.json")))[: args.n]))
    indexed_dev = [(i, ex) for i, ex in indexed_dev if os.path.exists(db_path(ex["db_id"]))]
    fewshot_ids = args.few_shot_ids[:args.few_shot] if args.few_shot else []
    system = SYSTEM_PROMPT + fewshot_text(fewshot_ids)
    writer = None
    if args.result_dir:
        writer = ArtifactWriter(args.result_dir, {
            "runner": "tool_rollout",
            "model": args.model,
            "base_url": args.base_url,
            "dev_size": args.n,
            "few_shot_ids": fewshot_ids,
            "max_steps": args.max_steps,
            "max_consecutive_errors": MAX_CONSECUTIVE_ERRORS,
            "temperature": 0,
            "max_tokens": 2048,
            "system_prompt": system,
        }, args.resume)
        indexed_dev = [(i, ex) for i, ex in indexed_dev if i not in writer.completed]
    results = []
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_live, ex, i, args.base_url, args.model, system, args.max_steps)
                    for i, ex in indexed_dev]
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                if writer:
                    writer.append(r)
                flag = "OK " if r["correct"] else ("LEG" if r["legal"] else "ERR")
                print(f"[{len(results)}/{len(indexed_dev)}] {flag} steps={r['steps']} errs={r['errors']} {r['question'][:60]}")
    else:
        for position, (i, ex) in enumerate(indexed_dev, 1):
            r = run_live(ex, i, args.base_url, args.model, system, args.max_steps)
            results.append(r)
            if writer:
                writer.append(r)
            flag = "OK " if r["correct"] else ("LEG" if r["legal"] else "ERR")
            print(f"[{position}/{len(indexed_dev)}] {flag} steps={r['steps']} errs={r['errors']} {ex['question'][:60]}")

    summary = writer.summarize() if writer else {
        "total": len(results),
        "correct": sum(r["correct"] for r in results),
        "accuracy": sum(r["correct"] for r in results) / max(1, len(results)),
    }
    print(f"\nEXEC-ACC {summary['correct']}/{summary['total']} "
          f"({100 * summary['accuracy']:.1f}%)")
    if writer:
        print(f"-> {writer.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
