#!/usr/bin/env python3
"""Add model-authored plan steps to verified relational-backbone trajectories.

This pass deliberately runs after raw SQL->trajectory emission. The emitter stays a
verified relational backbone; an external model supplies natural task plans and plan
updates, then the harness replays the sequence so every plan operation and tool output is
checked by the same environment used at eval/RL time.

Smoke without API:
  .venv/bin/python src/sft/enrich_plan.py --input data/trajectories/spider_train.jsonl \
    --out /tmp/plan_smoke.jsonl --limit 5 --dry-run-template

External model:
  .venv/bin/python src/sft/enrich_plan.py --input data/trajectories/spider_train.jsonl \
    --out data/trajectories/spider_train_plan.jsonl --limit 50 --model deepseek-v4-pro
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "eval"))
sys.path.insert(0, os.path.join(ROOT, "src", "harness"))
sys.path.insert(0, HERE)

from fill_think import call, load_api  # noqa: E402
from protocol import ProtocolError  # noqa: E402
from rollout import db_path, execute_tool, new_ctx, score  # noqa: E402
from executor import Harness  # noqa: E402


SYSTEM = (
    "You write task plans for a table-tool agent. The raw trajectory is an execution-verified "
    "relational backbone, but it has no planning steps. Add a natural first-person task plan and "
    "a small number of plan updates that reflect progress after important backbone steps.\n\n"
    "The plan is control state only. It must not claim unsupported facts, leak final answers, or "
    "invent table values. It can say what needs to be checked or computed, and later mark a "
    "subgoal done when a corresponding tool step has executed.\n\n"
    "Output ONLY JSON:\n"
    "{\n"
    "  \"initial_think\": \"first-person reason for creating the plan\",\n"
    "  \"initial_plan\": [\n"
    "    {\"id\":\"p1\", \"goal\":\"...\", \"status\":\"pending\", \"depends_on\":[]}\n"
    "  ],\n"
    "  \"updates\": [\n"
    "    {\"after_step\": 1, \"think\":\"first-person reason for updating the plan\", "
    "\"ops\":[{\"op\":\"update\", \"id\":\"p1\", \"status\":\"done\", "
    "\"evidence_step_id\":\"step_1\", \"notes\":\"...\"}]}\n"
    "  ]\n"
    "}\n\n"
    "`after_step` is the 1-based index of the raw backbone step after which to insert the update. "
    "Do not insert updates after the final answer_from_context step; the final answer must remain "
    "terminal. Use at most 4 initial goals and at most 6 update blocks. Use ids p1, p2, ... ."
)


def compact(obj: Any, limit: int = 900) -> str:
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    return text if len(text) <= limit else text[:limit] + "..."


def step_summary(traj: dict) -> list[dict]:
    out = []
    for i, step in enumerate(traj.get("steps", []), 1):
        call = step.get("tool_call") or {}
        output = step.get("tool_output") or {}
        item = {
            "index": i,
            "step_id": step.get("step_id"),
            "tool": call.get("tool"),
            "arguments": call.get("arguments"),
        }
        if "table" in output:
            item["output"] = {
                "table": output.get("table"),
                "kind": output.get("kind"),
                "row_count": output.get("row_count"),
                "columns": output.get("columns", [])[:16],
            }
        elif "result_sample" in output:
            item["output"] = {
                "row_count": output.get("row_count"),
                "result_sample": output.get("result_sample"),
            }
        elif "final_answer" in output:
            item["output"] = {"final_answer": output.get("final_answer")}
        out.append(item)
    return out


def build_messages(traj: dict) -> list[dict]:
    user = (
        f"QUESTION:\n{traj.get('question')}\n\n"
        f"LAZY CATALOG:\n{compact(traj.get('initial_state', {}).get('dataset_overview', {}), 4000)}\n\n"
        f"RAW VERIFIED BACKBONE STEPS:\n{compact(step_summary(traj), 12000)}\n\n"
        "Write the JSON plan enrichment. Keep goals abstract: mention needed joins, filters, "
        "aggregates, set operations, and observations to perform, but do not reveal final answer "
        "values unless they are already in the raw step outputs above."
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_json_object(text: str) -> dict | None:
    stripped = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        obj = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, re.S)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def call_retry(base: str, key: str, model: str, messages: list[dict],
               tries: int, timeout: int) -> tuple[str, dict]:
    last = None
    for attempt in range(tries):
        try:
            return call(base, key, model, messages, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt + 1 < tries:
                time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"plan enrichment API failed: {type(last).__name__}: {last}")


def template_plan(traj: dict) -> dict:
    """Deterministic fallback for tests/smoke only; not intended for final training data."""
    tools = [s.get("tool_call", {}).get("tool") for s in traj.get("steps", [])]
    initial = [
        {"id": "p1", "goal": "identify the relevant tables and build the needed intermediate result", "status": "pending"},
        {"id": "p2", "goal": "apply filters, aggregations, ordering, or set operations required by the question", "status": "pending"},
        {"id": "p3", "goal": "produce the final answer from the computed evidence", "status": "pending"},
    ]
    updates = []
    if tools:
        updates.append({
            "after_step": max(1, len(tools) // 2),
            "think": "The intermediate computation is underway, so I update the plan to track progress.",
            "ops": [{"op": "update", "id": "p1", "status": "done", "evidence_step_id": "step_1"}],
        })
        updates.append({
            "after_step": max(1, len(tools) - 1),
            "think": "The final answer step has been reached, so I close the remaining plan items.",
            "ops": [
                {"op": "update", "id": "p2", "status": "done"},
                {"op": "update", "id": "p3", "status": "done"},
            ],
        })
    return {
        "initial_think": "I first turn the question into a short task plan so each tool call has a clear subgoal.",
        "initial_plan": initial,
        "updates": updates,
    }


def normalize_plan_payload(payload: dict, n_steps: int) -> dict:
    initial_plan = payload.get("initial_plan")
    if not isinstance(initial_plan, list) or not initial_plan:
        raise ValueError("initial_plan must be a non-empty list")
    if len(initial_plan) > 4:
        initial_plan = initial_plan[:4]
    create_ops = []
    seen_ids = set()
    for i, item in enumerate(initial_plan, 1):
        if not isinstance(item, dict):
            raise ValueError("initial_plan items must be objects")
        item_id = str(item.get("id") or f"p{i}").strip()
        if not item_id or item_id in seen_ids:
            item_id = f"p{i}"
        seen_ids.add(item_id)
        goal = str(item.get("goal") or "").strip()
        if not goal:
            raise ValueError(f"plan item {item_id} has no goal")
        op = {"op": "create", "id": item_id, "goal": goal,
              "status": item.get("status") or "pending"}
        for key in ("depends_on", "notes"):
            if key in item:
                op[key] = item[key]
        create_ops.append(op)

    updates = []
    for block in payload.get("updates", []) if isinstance(payload.get("updates"), list) else []:
        if not isinstance(block, dict):
            continue
        after = int(block.get("after_step") or 0)
        # answer_from_context must stay terminal, so plan updates may only appear before the final
        # raw backbone step.
        if after < 1 or after >= n_steps:
            continue
        ops = block.get("ops")
        if not isinstance(ops, list) or not ops:
            continue
        clean_ops = []
        for op in ops:
            if not isinstance(op, dict):
                continue
            action = op.get("op") or "update"
            if action not in {"add", "create", "update", "delete"}:
                continue
            item_id = op.get("id")
            if not isinstance(item_id, str) or not item_id.strip():
                continue
            clean = {"op": action, "id": item_id.strip()}
            for key in ("goal", "status", "depends_on", "notes", "reason", "evidence_step_id"):
                if key in op:
                    clean[key] = op[key]
            clean_ops.append(clean)
        if clean_ops:
            updates.append({
                "after_step": after,
                "think": str(block.get("think") or "Update the task plan based on the latest tool result.").strip(),
                "ops": clean_ops,
            })
        if len(updates) >= 6:
            break

    return {
        "initial_think": str(payload.get("initial_think") or "Create a task plan before using tools.").strip(),
        "initial_ops": create_ops,
        "updates": updates,
    }


def remap_step_refs(obj: Any, old_to_new: dict[str, str]) -> Any:
    if isinstance(obj, list):
        return [remap_step_refs(x, old_to_new) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out = {}
    for key, value in obj.items():
        if key in {"value_ref", "evidence_step_id"} and isinstance(value, str):
            out[key] = old_to_new.get(value, value)
        elif key == "in_table" and isinstance(value, str):
            out[key] = old_to_new.get(value, value)
        else:
            out[key] = remap_step_refs(value, old_to_new)
    return out


def replay_with_plan(traj: dict, plan_payload: dict) -> dict:
    h = Harness(db_path(traj["source"]["db_id"]))
    ctx = new_ctx(traj.get("initial_state", {}).get("dataset_overview"))
    created: set[str] = set()
    old_to_new: dict[str, str] = {}
    steps_out: list[dict] = []
    updates_by_after: dict[int, list[dict]] = {}
    for block in plan_payload["updates"]:
        updates_by_after.setdefault(block["after_step"], []).append(block)

    def append_plan(think: str, ops: list[dict]) -> None:
        sid = f"step_{len(steps_out) + 1}"
        mapped_ops = remap_step_refs(copy.deepcopy(ops), old_to_new)
        out, _ = execute_tool(h, "plan", {"ops": mapped_ops}, ctx, sid)
        steps_out.append({
            "step_id": sid,
            "think": think,
            "tool_call": {"tool": "plan", "arguments": {"ops": mapped_ops}},
            "tool_output": out,
            "produces": {"kind": "plan"},
            "references": [],
            "environment_state": ctx["environment"].snapshot(),
        })

    append_plan(plan_payload["initial_think"], plan_payload["initial_ops"])

    for raw_index, raw_step in enumerate(traj.get("steps", []), 1):
        call = raw_step.get("tool_call") or {}
        tool = call.get("tool")
        args = remap_step_refs(copy.deepcopy(call.get("arguments") or {}), old_to_new)
        sid = f"step_{len(steps_out) + 1}"
        if tool == "answer_from_context":
            correct, pred, gold = score(h, traj["source"]["gold_sql"], args, created)
            if not correct:
                raise ValueError(f"final answer mismatch after plan insertion: pred={pred[:3]} gold={gold[:3]}")
            step_out = {
                "step_id": sid,
                "think": raw_step.get("think", ""),
                "rationale": raw_step.get("rationale"),
                "tool_call": {"tool": tool, "arguments": args},
                "tool_output": {"final_answer": args.get("answer")},
                "produces": raw_step.get("produces", {"kind": "answer"}),
                "references": raw_step.get("references", []),
                "environment_state": ctx["environment"].snapshot(),
            }
        else:
            out, table_name = execute_tool(h, tool, args, ctx, sid)
            if table_name:
                created.add(table_name)
            step_out = {
                "step_id": sid,
                "think": raw_step.get("think", ""),
                "rationale": raw_step.get("rationale"),
                "tool_call": {"tool": tool, "arguments": args},
                "tool_output": out,
                "produces": raw_step.get("produces"),
                "references": raw_step.get("references", []),
                "environment_state": ctx["environment"].snapshot(),
            }
            if raw_step.get("perception"):
                step_out["perception"] = raw_step["perception"]
        steps_out.append({k: v for k, v in step_out.items() if v is not None})
        old_to_new[raw_step.get("step_id", f"step_{raw_index}")] = sid

        for update in updates_by_after.get(raw_index, []):
            append_plan(update["think"], update["ops"])

    if not steps_out or steps_out[-1].get("tool_call", {}).get("tool") != "answer_from_context":
        raise ValueError("plan-enriched sequence does not end with answer_from_context")

    out = copy.deepcopy(traj)
    out["schema_version"] = f"{traj.get('schema_version', 'v3')}-plan"
    out["steps"] = steps_out
    out["label_status"] = "verified"
    out["final_answer"] = steps_out[-1]["tool_call"]["arguments"].get("answer", traj.get("final_answer"))
    out["plan_enrichment"] = {
        "status": "enriched",
        "initial_plan_items": len(plan_payload["initial_ops"]),
        "update_blocks": len(plan_payload["updates"]),
    }
    return out


def enrich_one(traj: dict, *, base: str, key: str, model: str, api_timeout: int,
               api_retries: int, dry_run_template: bool = False) -> tuple[dict, dict]:
    if dry_run_template:
        raw_payload = template_plan(traj)
        usage = {}
    else:
        text, usage = call_retry(base, key, model, build_messages(traj), api_retries, api_timeout)
        raw_payload = parse_json_object(text)
        if raw_payload is None:
            raise ValueError(f"model did not return JSON object: {text[:200]!r}")
    plan_payload = normalize_plan_payload(raw_payload, len(traj.get("steps", [])))
    enriched = replay_with_plan(traj, plan_payload)
    enriched["plan_enrichment"]["generator"] = {
        "model": "dry_run_template" if dry_run_template else model,
        "usage": usage,
    }
    return enriched, usage


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(ROOT) / p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input trajectory JSONL")
    parser.add_argument("--out", required=True, help="output plan-enriched JSONL")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--api-timeout", type=int, default=180)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--dry-run-template", action="store_true",
                        help="use deterministic placeholder plans for local smoke tests")
    args = parser.parse_args()

    source = resolve_path(args.input)
    out_path = resolve_path(args.out)
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        rows = rows[:args.limit]
    key = base = ""
    if not args.dry_run_template:
        key, base = load_api()
        if not key or not base:
            parser.error("api.md must define API_KEY and BASE_URL unless --dry-run-template is used")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    usage_total = {}

    def process(item: tuple[int, dict]):
        i, traj = item
        enriched, usage = enrich_one(
            traj,
            base=base,
            key=key,
            model=args.model,
            api_timeout=args.api_timeout,
            api_retries=args.api_retries,
            dry_run_template=args.dry_run_template,
        )
        return i, enriched, usage

    results: dict[int, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(process, item) for item in enumerate(rows)]
        for future in as_completed(futures):
            try:
                i, enriched, usage = future.result()
                results[i] = enriched
                for k, v in (usage or {}).items():
                    if isinstance(v, (int, float)):
                        usage_total[k] = usage_total.get(k, 0) + v
                print(f"[{len(results)}/{len(rows)}] ok {enriched.get('trajectory_id')}")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{type(exc).__name__}: {exc}")
                print(f"[fail] {type(exc).__name__}: {exc}")

    with out_path.open("w", encoding="utf-8") as out:
        for i in range(len(rows)):
            if i in results:
                out.write(json.dumps(results[i], ensure_ascii=False, default=str) + "\n")

    manifest = {
        "input": os.path.relpath(source, ROOT),
        "output": os.path.relpath(out_path, ROOT),
        "source": len(rows),
        "kept": len(results),
        "failed": len(failures),
        "failures": failures[:20],
        "model": "dry_run_template" if args.dry_run_template else args.model,
        "usage_total": usage_total,
    }
    out_path.with_suffix(out_path.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
