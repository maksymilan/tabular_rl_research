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
from protocol import ProtocolError, TOOL_SPECS  # noqa: E402
from rollout import db_path, execute_tool, new_ctx, score  # noqa: E402
from executor import Harness  # noqa: E402


SYSTEM = (
    "You write task plans for a table-tool agent. The raw trajectory is an execution-verified "
    "relational backbone, but it has no planning steps. Add a natural first-person task plan and "
    "a small number of plan updates that reflect progress after important backbone steps.\n\n"
    "Base every plan on the available tool interface supplied in the user message. Use tool-level "
    "language such as describe_table, inspect_column, condition_filter, join_tables, aggregate, "
    "project, extreme_value_select, read_subtable, and answer_from_context. Do not tell the agent "
    "to execute a SQL query or use hidden SQL; the agent can only call the listed tools.\n\n"
    "The plan is control state only. It must not claim unsupported facts, leak final answers, or "
    "invent table values. It can say what needs to be checked or computed, and later mark a "
    "subgoal done when a corresponding tool step has executed. A done or blocked subgoal should "
    "also record `result.summary`: a short text conclusion grounded by `evidence_step_id`. Do NOT "
    "write factual `result.value` fields, lists, scalars, booleans, or final answer values into the "
    "plan; factual values belong to tool outputs, not model-authored plan state. `status` says "
    "whether the subtask is complete; `result.summary` says what changed at the task-state level.\n\n"
    "Plan updates should be sparse and state-change-based, not mechanical. Do NOT update the plan after every tool call. "
    "Update only when a meaningful milestone happens: a branch/subgoal starts, completes, becomes "
    "blocked, gets revised by an observation, or obtains a reusable conclusion. Pure schema reads, "
    "row reads, and intermediate handle creation do not need plan updates unless they change a "
    "subgoal's status or result. Do not spend an update block merely to say that schemas, columns, "
    "or sample rows have been inspected; put that progress in the next step's think instead. "
    "Tool types are only clues: update the plan when the task state changes, not because a specific "
    "tool was called.\n\n"
    "Output ONLY JSON:\n"
    "{\n"
    "  \"initial_think\": \"first-person reason for creating the plan\",\n"
    "  \"initial_plan\": [\n"
    "    {\"id\":\"p1\", \"goal\":\"...\", \"status\":\"pending\", \"depends_on\":[]}\n"
    "  ],\n"
    "  \"updates\": [\n"
    "    {\"after_step\": 1, \"think\":\"first-person reason for updating the plan\", "
    "\"ops\":[{\"op\":\"update\", \"id\":\"p1\", \"status\":\"done\", "
    "\"evidence_step_id\":\"step_1\", \"result\":{\"type\":\"text\", \"summary\":\"...\"}, "
    "\"notes\":\"...\"}]}\n"
    "  ]\n"
    "}\n\n"
    "`after_step` is the 1-based index of the raw backbone step after which to insert the update. "
    "Do not insert updates after the final answer_from_context step; the final answer must remain "
    "terminal. Use ids p1, p2, ... . Initial plan items must be pending only: no initial item may "
    "be done, blocked, or contain result/conclusion. Do not create plan items whose only purpose is "
    "describe_table, inspect_column, schema discovery, or row reading; those observations belong in "
    "think text, while plan items should be semantic task goals. Do not write natural-language "
    "step numbers like step_1 or step 2 in plan think text; use evidence_step_id fields instead. "
    "For `result`, prefer {\"type\":\"text\",\"summary\":\"...\"}. If the evidence step produced a "
    "scalar/list/boolean, summarize the conclusion without copying the value."
)


PERCEPTION_TOOLS = {"describe_table", "inspect_column", "read_subtable"}
STEP_REF_RE = re.compile(r"\bstep[_ ]?\d+\b", re.I)
SCHEMA_ONLY_GOAL_RE = re.compile(
    r"\b(describe|inspect|schema|schemas|column spelling|exact spelling|table structure|"
    r"explore schema|understand (?:the )?(?:available )?columns|read rows|sample rows)\b",
    re.I,
)
TASK_GOAL_RE = re.compile(
    r"\b(join|filter|aggregate|average|count|sum|min|max|mean|sort|order|top|lowest|highest|"
    r"intersect|except|union|set|project|deduplicate|group|compute|compare|combine|threshold|"
    r"identify .* set|find .* set|answer)\b",
    re.I,
)


def tool_specs_text() -> str:
    return "\n".join(f"- {name}: {spec}" for name, spec in TOOL_SPECS.items())


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


def _walk_conditions(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_conditions(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_conditions(item)


def plan_requirements(traj: dict) -> dict:
    """Dynamic update budget based on trajectory complexity.

    This is deliberately a soft complexity estimate, not a fixed tool-to-update template. The model
    still decides which task-state changes deserve updates; these bounds only prevent degenerate
    no-update plans and every-step update rituals.
    """
    steps = traj.get("steps", [])
    tools = [(s.get("tool_call") or {}).get("tool") for s in steps]
    work_tools = [t for t in tools if t and t not in PERCEPTION_TOOLS and t != "answer_from_context"]
    non_perception = len(work_tools)
    set_ops = sum(1 for t in work_tools if t == "set_op")
    aggregates = sum(1 for t in work_tools if t == "aggregate")
    group_or_extreme = sum(1 for t in work_tools if t in {"group_aggregate", "extreme_value_select"})
    value_refs = 0
    for step in steps:
        call = step.get("tool_call") or {}
        if call.get("tool") != "condition_filter":
            continue
        for node in _walk_conditions((call.get("arguments") or {}).get("conditions")):
            if isinstance(node, dict) and node.get("value_ref"):
                value_refs += 1

    # The initial plan already provides control structure.  Short linear trajectories should not be
    # forced to add middle updates; otherwise plan calls become a ritual rather than state changes.
    min_updates = 0
    if non_perception <= 5:
        min_updates = 0
    elif value_refs or set_ops:
        min_updates = 1
    if set_ops >= 2 or (set_ops and (aggregates or group_or_extreme)) or non_perception >= 10:
        min_updates = 2

    loose_by_length = max(0, (non_perception + 5) // 6)
    max_updates = min(4, max(min_updates + 1, loose_by_length))
    return {
        "non_perception_tools": non_perception,
        "set_ops": set_ops,
        "aggregates": aggregates,
        "group_or_extreme": group_or_extreme,
        "value_refs": value_refs,
        "min_updates": min_updates,
        "max_updates": max_updates,
    }


def build_messages(traj: dict, feedback: str = "") -> list[dict]:
    requirements = plan_requirements(traj)
    user = (
        f"QUESTION:\n{traj.get('question')}\n\n"
        f"LAZY CATALOG:\n{compact(traj.get('initial_state', {}).get('dataset_overview', {}), 4000)}\n\n"
        f"AVAILABLE TOOLS AND HOW TO USE THEM:\n{tool_specs_text()}\n\n"
        f"PLAN UPDATE REQUIREMENTS:\n{compact(requirements, 1200)}\n"
        "Use this as a dynamic budget, not a template. The number of updates must be within "
        "`min_updates` and `max_updates`. If `min_updates` is 0, it is valid to produce no middle "
        "updates. Short linear tasks often need only the initial plan. Longer branchy, set-comparison, "
        "aggregate-threshold, or final-combine tasks should update only when task state actually "
        "changes.\n\n"
        f"RAW VERIFIED BACKBONE STEPS:\n{compact(step_summary(traj), 12000)}\n\n"
        "Write the JSON plan enrichment. Keep goals grounded in the tool interface above: mention "
        "needed semantic operations such as finding a branch result, computing a scalar threshold, "
        "combining sets, aggregating, ordering/extreme selection, projecting final evidence, and "
        "answering. Do not make standalone plan items for schema description, column inspection, or "
        "row reading; those are observation details inside think. "
        "Because this enrichment pass only inserts plan steps, the plan must be compatible with the "
        "raw backbone: do not create a subgoal that requires a tool absent from the backbone unless "
        "it remains explicitly pending or blocked, and never mark such a subgoal done. Prefer "
        "subgoals that summarize the actual backbone operations. "
        "Do not reveal final answer values unless they are already in the raw step outputs above. "
        "Even then, do not copy those values into plan result fields; summarize the milestone only."
    )
    if feedback:
        user += (
            "\n\nPREVIOUS PLAN FAILED VALIDATION:\n"
            f"{feedback}\n"
            "Revise only the plan JSON. Keep the same raw backbone semantics and satisfy the validation."
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
        first_after = 1
        mid_after = max(1, len(tools) // 2)
        final_after = max(1, len(tools) - 1)
        updates.append({
            "after_step": first_after,
            "think": "The intermediate computation is underway, so I update the plan to track progress.",
            "ops": [{"op": "update", "id": "p1", "status": "done", "evidence_step_id": "step_1",
                     "result": {"type": "text", "summary": "The relevant intermediate result has been started."}}],
        })
        updates.append({
            "after_step": mid_after,
            "think": "The main relational operations have produced a useful intermediate state.",
            "ops": [{"op": "update", "id": "p2", "status": "done", "evidence_step_id": f"step_{mid_after}",
                     "result": {"type": "text", "summary": "The required relational operations are complete."}}],
        })
        updates.append({
            "after_step": final_after,
            "think": "The final answer step has been reached, so I close the remaining plan items.",
            "ops": [
                {"op": "update", "id": "p3", "status": "done", "evidence_step_id": f"step_{final_after}",
                 "result": {"type": "text", "summary": "The final answer is ready to submit."}},
            ],
        })
    return {
        "initial_think": "I first turn the question into a short task plan so each tool call has a clear subgoal.",
        "initial_plan": initial,
        "updates": updates,
    }


def sanitize_plan_result(result: Any) -> Any:
    """Keep model-authored plan results as summaries, not factual value stores."""
    if not isinstance(result, dict):
        return result
    clean = {k: copy.deepcopy(v) for k, v in result.items() if k != "value"}
    clean_type = str(clean.get("type") or "text")
    clean["type"] = clean_type if clean_type in {"text", "structured", "boolean", "scalar", "list"} else "text"
    summary = clean.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        clean["summary"] = "The cited evidence step contains the subgoal result."
    return clean


def normalize_plan_payload(
    payload: dict,
    n_steps: int,
    raw_steps: list[dict] | None = None,
    requirements: dict | None = None,
) -> dict:
    requirements = requirements or {}
    max_updates = int(requirements.get("max_updates") or 6)
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
        op = {"op": "create", "id": item_id, "goal": goal, "status": "pending"}
        for key in ("depends_on", "notes"):
            if key in item:
                op[key] = item[key]
        create_ops.append(op)

    updates = []
    known_ids = set(seen_ids)
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
            item_id = item_id.strip()
            if action in {"update", "delete"} and item_id not in known_ids:
                continue
            if action in {"add", "create"}:
                if item_id in known_ids:
                    continue
                goal = op.get("goal")
                if not isinstance(goal, str) or not goal.strip():
                    continue
            clean = {"op": action, "id": item_id}
            for key in ("goal", "status", "depends_on", "notes", "reason", "evidence_step_id",
                        "result", "conclusion"):
                if key in op:
                    if key == "depends_on" and isinstance(op[key], list):
                        clean[key] = [dep for dep in op[key] if dep in known_ids]
                    elif key == "result":
                        clean[key] = sanitize_plan_result(op[key])
                    elif key == "conclusion":
                        clean[key] = sanitize_plan_result(op[key])
                    else:
                        clean[key] = op[key]
            clean_ops.append(clean)
            if action in {"add", "create"}:
                known_ids.add(item_id)
        if clean_ops:
            updates.append({
                "after_step": after,
                "think": str(block.get("think") or "Update the task plan based on the latest tool result.").strip(),
                "ops": clean_ops,
            })
        if len(updates) >= max_updates:
            break

    updates = sorted(updates, key=lambda b: b["after_step"])[:max_updates]

    return {
        "initial_think": str(payload.get("initial_think") or "Create a task plan before using tools.").strip(),
        "initial_ops": create_ops,
        "updates": updates,
    }


def is_observation_only_goal(goal: str) -> bool:
    text = goal.strip().lower()
    return bool(SCHEMA_ONLY_GOAL_RE.search(text)) and not bool(TASK_GOAL_RE.search(text))


def validate_plan_payload(plan_payload: dict, traj: dict, requirements: dict) -> list[str]:
    """Return validation issues for model-authored plan structure.

    The checks are about plan semantics, not SQL/tool correctness. Tool correctness is still owned by
    replay_with_plan. These constraints keep plan state from becoming an answer leak or an
    observation log.
    """
    issues: list[str] = []
    min_updates = int(requirements.get("min_updates") or 0)
    max_updates = int(requirements.get("max_updates") or 6)
    updates = plan_payload.get("updates", [])
    initial_ops = plan_payload.get("initial_ops", [])

    if STEP_REF_RE.search(plan_payload.get("initial_think", "")):
        issues.append("initial_think must not mention natural-language step numbers such as step_1")

    if len(updates) < min_updates:
        issues.append(f"too few plan updates: got {len(updates)}, need at least {min_updates}")
    if len(updates) > max_updates:
        issues.append(f"too many plan updates: got {len(updates)}, max is {max_updates}")

    def check_no_step_refs(obj: Any, label: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key == "evidence_step_id":
                    continue
                if key == "value":
                    issues.append(f"{label} must not contain model-authored factual result.value")
                check_no_step_refs(value, f"{label}.{key}")
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                check_no_step_refs(value, f"{label}[{i}]")
        elif isinstance(obj, str) and STEP_REF_RE.search(obj):
            issues.append(f"{label} must not mention natural-language step numbers; use evidence_step_id")

    plan_state: dict[str, dict] = {}
    for op in initial_ops:
        item_id = op.get("id")
        if not item_id:
            continue
        check_no_step_refs(op, f"initial op {item_id}")
        goal = str(op.get("goal") or "")
        if is_observation_only_goal(goal):
            issues.append(
                f"initial plan item {item_id} is only schema/inspection/reading; fold that into think "
                "and replace it with a semantic task subgoal"
            )
        if op.get("status") not in {"pending", "in_progress"}:
            issues.append(f"initial plan item {item_id} must start pending/in_progress, not {op.get('status')}")
        if op.get("result") is not None or op.get("conclusion") is not None:
            issues.append(f"initial plan item {item_id} must not contain result/conclusion")
        plan_state[item_id] = {
            "goal": goal,
            "status": op.get("status") or "pending",
            "result": op.get("result") or op.get("conclusion"),
        }

    for block in updates:
        think = str(block.get("think") or "")
        if STEP_REF_RE.search(think):
            issues.append("plan update think must not mention natural-language step numbers; use evidence_step_id only")
        if SCHEMA_ONLY_GOAL_RE.search(think) and not TASK_GOAL_RE.search(think):
            issues.append("plan update think is only about schema/inspection progress, not a task-state change")
        after = int(block.get("after_step") or 0)
        for op in block.get("ops", []):
            action = op.get("op") or "update"
            item_id = op.get("id")
            check_no_step_refs(op, f"update op {item_id}")
            if action in {"add", "create"}:
                goal = str(op.get("goal") or "")
                if is_observation_only_goal(goal):
                    issues.append(f"added plan item {item_id} is only schema/inspection/reading")
                plan_state[item_id] = {
                    "goal": goal,
                    "status": op.get("status") or "pending",
                    "result": op.get("result") or op.get("conclusion"),
                }
            elif action == "update" and item_id in plan_state:
                if op.get("status"):
                    plan_state[item_id]["status"] = op.get("status")
                if op.get("result") is not None or op.get("conclusion") is not None:
                    plan_state[item_id]["result"] = op.get("result") or op.get("conclusion")
                    result_obj = op.get("result") or op.get("conclusion")
                    if isinstance(result_obj, dict) and "value" in result_obj:
                        issues.append(f"update for {item_id} must not contain model-authored result.value")
                if op.get("status") in {"done", "blocked"}:
                    if op.get("result") is None and op.get("conclusion") is None:
                        issues.append(f"done/blocked update for {item_id} should include result/conclusion")
                    ev = op.get("evidence_step_id")
                    if not isinstance(ev, str) or not ev.strip():
                        issues.append(f"done/blocked update for {item_id} should include evidence_step_id")
            elif action == "delete":
                plan_state.pop(item_id, None)

        if after >= len(traj.get("steps", [])):
            issues.append("plan update cannot be inserted after the final answer step")

    return issues


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
               api_retries: int, plan_attempts: int = 3,
               dry_run_template: bool = False) -> tuple[dict, dict]:
    requirements = plan_requirements(traj)
    feedback = ""
    last_error = ""
    usage_total: dict[str, int | float] = {}
    if dry_run_template:
        raw_payload = template_plan(traj)
        usage = {}
        plan_payload = normalize_plan_payload(
            raw_payload, len(traj.get("steps", [])), traj.get("steps", []), requirements
        )
        issues = validate_plan_payload(plan_payload, traj, requirements)
        if issues:
            raise ValueError("; ".join(issues[:6]))
        enriched = replay_with_plan(traj, plan_payload)
        enriched["plan_enrichment"]["generator"] = {
            "model": "dry_run_template",
            "usage": usage,
            "requirements": requirements,
            "attempts": 1,
        }
        return enriched, usage

    for attempt in range(1, max(1, plan_attempts) + 1):
        text, usage = call_retry(base, key, model, build_messages(traj, feedback), api_retries, api_timeout)
        for k, v in (usage or {}).items():
            if isinstance(v, (int, float)):
                usage_total[k] = usage_total.get(k, 0) + v
        raw_payload = parse_json_object(text)
        if raw_payload is None:
            last_error = f"model did not return JSON object: {text[:200]!r}"
            feedback = last_error
            continue
        try:
            plan_payload = normalize_plan_payload(
                raw_payload, len(traj.get("steps", [])), traj.get("steps", []), requirements
            )
            issues = validate_plan_payload(plan_payload, traj, requirements)
            if issues:
                last_error = "; ".join(issues[:8])
                feedback = last_error
                continue
            enriched = replay_with_plan(traj, plan_payload)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            feedback = last_error
            continue
        enriched["plan_enrichment"]["generator"] = {
            "model": model,
            "usage": usage_total,
            "requirements": requirements,
            "attempts": attempt,
        }
        return enriched, usage_total
    raise ValueError(f"plan validation failed after {plan_attempts} attempts: {last_error}")


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(ROOT) / p


def review_path_for(path: Path) -> Path:
    if path.suffix == ".jsonl" and not path.stem.endswith("_review"):
        return path.with_name(f"{path.stem}_review{path.suffix}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input trajectory JSONL")
    parser.add_argument("--out", required=True, help="output plan-enriched JSONL")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=16,
                        help="parallel plan-enrichment workers; external API is assumed to tolerate concurrency")
    parser.add_argument("--api-timeout", type=int, default=180)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--plan-attempts", type=int, default=3,
                        help="outer retries for invalid plan payloads without rerunning trajectory enrichment")
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
            plan_attempts=args.plan_attempts,
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
    review_path = review_path_for(out_path)
    if review_path != out_path:
        review_path.write_text(out_path.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "input": os.path.relpath(source, ROOT),
        "output": os.path.relpath(out_path, ROOT),
        "review_output": os.path.relpath(review_path, ROOT),
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
