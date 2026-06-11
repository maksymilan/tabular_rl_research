#!/usr/bin/env python3
"""Closed-loop rollout runner: a model converses with the harness through the shared protocol
(scripts/sft/protocol.py) until answer_from_context; the answer is scored against the gold
SQL executed on the real DB. One runner serves three purposes: (a) few-shot baseline eval of
the base model, (b) post-SFT eval, (c) later, the RL environment interaction loop.

Modes
  live   : .venv/bin/python scripts/eval/rollout.py --base-url http://127.0.0.1:8000/v1 \
               --model Qwen/Qwen2.5-3B-Instruct [--n 50] [--few-shot 2] [--out results.jsonl]
           (vLLM on the GPU server; reach it from the Mac via `ssh -L 8000:127.0.0.1:8000 NewGNN`)
  replay : .venv/bin/python scripts/eval/rollout.py --replay 50
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
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "scripts", "harness"))
sys.path.insert(0, os.path.join(ROOT, "scripts", "sft"))

from executor import Harness                                    # noqa: E402
from plan import resolve_cond                                  # noqa: E402
from protocol import (SYSTEM_PROMPT, ProtocolError, TOOLS,      # noqa: E402
                      assistant_message, first_user_message, parse_assistant,
                      rows_equal, tool_output_message)

SPIDER = os.path.join(ROOT, "data", "spider_data")
MAX_CONSECUTIVE_ERRORS = 3   # error feedback turns allowed before aborting the trajectory


def db_path(db_id: str) -> str:
    return os.path.join(SPIDER, "database", db_id, f"{db_id}.sqlite")


def overview(h: Harness) -> dict:
    tables = []
    for (name,) in h.conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        cols = [{"name": r[1], "type": (r[2] or "text").lower()}
                for r in h.conn.execute(f'PRAGMA table_info("{name}")')]
        n = h.conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        tables.append({"table_name": name, "num_rows": n, "columns": cols})
    return {"tables": tables}


def execute_tool(h: Harness, tool: str, args: dict, memory: dict | None = None):
    """Run one tool call; return (tool_output dict in the emitter's exact shape, created_table|None)."""
    if tool not in TOOLS or tool == "answer_from_context":
        raise ProtocolError(f"tool {tool!r} not executable here")
    exec_args = dict(args)
    if tool == "condition_filter":
        exec_args["conditions"] = resolve_cond(
            exec_args.get("conditions"), {}, memory if memory is not None else {}
        )
    out = getattr(h, tool)(**exec_args)
    if isinstance(out, dict) and "table_name" in out:
        return {"table": out["table_name"], "kind": out["kind"], **h.preview(out["table_name"])}, out["table_name"]
    if tool == "add_to_memory":
        if memory is not None:
            memory[args["key"]] = args.get("value")
        return {"memory": out["memory"]}, None
    rows = out if isinstance(out, list) else [(out,)]
    return {"result_sample": [list(r) for r in rows[:5]], "row_count": len(rows)}, None


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
    body = json.dumps({"model": model, "messages": messages,
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def fewshot_text(k: int) -> str:
    """Render k short train trajectories as example transcripts appended to the system prompt
    (used for the pre-SFT baseline; pass --few-shot 0 for the SFT'd model)."""
    if not k:
        return ""
    blocks, seen = [], 0
    with open(os.path.join(ROOT, "data", "trajectories", "spider_train.jsonl")) as f:
        for line in f:
            t = json.loads(line)
            ov = t["initial_state"]["dataset_overview"]
            if not (3 <= len(t["steps"]) <= 4) or len(json.dumps(ov)) > 1600:
                continue
            lines = [f"USER: {first_user_message(ov, t['question'])}"]
            for i, s in enumerate(t["steps"]):
                tc = s["tool_call"]
                lines.append(f"ASSISTANT: {assistant_message(s.get('think', ''), tc['tool'], tc['arguments'])}")
                if i < len(t["steps"]) - 1:
                    lines.append(f"USER: {tool_output_message(s['tool_output'])}")
            blocks.append("\n".join(lines))
            seen += 1
            if seen == k:
                break
    return "\n\nEXAMPLE SESSIONS\n" + "\n\n---\n\n".join(blocks)


def run_live(ex: dict, base_url: str, model: str, system: str, max_steps: int) -> dict:
    h = Harness(db_path(ex["db_id"]))
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": first_user_message(overview(h), ex["question"])}]
    created: set[str] = set()
    memory: dict = {}
    steps = errors = consecutive = 0
    text = ""
    rec = {"db_id": ex["db_id"], "question": ex["question"], "correct": False, "legal": False,
           "steps": 0, "errors": 0, "fail": None}

    while steps < max_steps:
        try:
            text = chat(base_url, model, messages)
        except Exception as e:
            rec["fail"] = f"api: {type(e).__name__}: {e}"
            return rec
        messages.append({"role": "assistant", "content": text})
        try:
            _, tool, args = parse_assistant(text)
            if tool == "answer_from_context":
                rec["legal"] = True
                rec["steps"] = steps + 1
                rec["errors"] = errors
                rec["correct"], rec["pred_sample"], rec["gold_sample"] = score(h, ex["query"], args, created)
                return rec
            out, tname = execute_tool(h, tool, args, memory)
        except (ProtocolError, Exception) as e:  # noqa: BLE001 — every failure becomes feedback
            errors += 1
            consecutive += 1
            if consecutive >= MAX_CONSECUTIVE_ERRORS:
                rec["fail"] = f"aborted after {consecutive} consecutive errors: {e}"
                rec["errors"] = errors
                rec["steps"] = steps
                rec["last_text"] = text[:300]  # raw model output, for failure attribution
                return rec
            messages.append({"role": "user",
                             "content": json.dumps({"error": f"{type(e).__name__}: {e}"})})
            continue
        consecutive = 0
        steps += 1
        if tname:
            created.add(tname)
        messages.append({"role": "user", "content": tool_output_message(out)})

    rec["fail"] = "max_steps"
    rec["steps"] = steps
    rec["errors"] = errors
    rec["last_text"] = text[:300]
    return rec


# ---------------- replay mode (no model) ----------------
def run_replay(n: int) -> int:
    path = os.path.join(ROOT, "data", "trajectories", "spider_dev.jsonl")
    total = ok_exec = ok_score = 0
    with open(path) as f:
        for line in f:
            if total >= n:
                break
            t = json.loads(line)
            total += 1
            h = Harness(db_path(t["source"]["db_id"]))
            created: set[str] = set()
            memory: dict = {}
            try:
                for s in t["steps"][:-1]:
                    tc = s["tool_call"]
                    _, tname = execute_tool(h, tc["tool"], tc["arguments"], memory)
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
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent questions (vLLM batches requests; each worker owns its Harness/sqlite)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.replay:
        return run_replay(args.replay)

    dev = json.load(open(os.path.join(SPIDER, "dev.json")))[: args.n]
    dev = [ex for ex in dev if os.path.exists(db_path(ex["db_id"]))]
    system = SYSTEM_PROMPT + fewshot_text(args.few_shot)
    results = []
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(run_live, ex, args.base_url, args.model, system, args.max_steps)
                    for ex in dev]
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                flag = "OK " if r["correct"] else ("LEG" if r["legal"] else "ERR")
                print(f"[{len(results)}/{len(dev)}] {flag} steps={r['steps']} errs={r['errors']} {r['question'][:60]}")
    else:
        for i, ex in enumerate(dev):
            r = run_live(ex, args.base_url, args.model, system, args.max_steps)
            results.append(r)
            flag = "OK " if r["correct"] else ("LEG" if r["legal"] else "ERR")
            print(f"[{i + 1}/{len(dev)}] {flag} steps={r['steps']} errs={r['errors']} {ex['question'][:60]}")

    n = len(results)
    acc = sum(r["correct"] for r in results)
    legal = sum(r["legal"] for r in results)
    print(f"\nEXEC-ACC {acc}/{n} ({100 * acc / max(1, n):.1f}%) | "
          f"legal-finish {legal}/{n} ({100 * legal / max(1, n):.1f}%) | "
          f"avg steps {sum(r['steps'] for r in results) / max(1, n):.1f} | "
          f"avg errors {sum(r['errors'] for r in results) / max(1, n):.1f}")
    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
