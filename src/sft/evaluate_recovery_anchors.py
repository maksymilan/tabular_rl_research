#!/usr/bin/env python3
"""Use an external evaluator to select real recovery anchors without exposing gold SQL.

Evaluator rationale is audit-only and is never included in teacher continuation prompts or SFT
targets.  If the evaluator rejects all anchors, the full task is routed to the teacher-from-scratch
lane. API/transport or malformed-response failures remain unresolved rather than being silently
treated as semantic decisions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(HERE),
]

from generate_teacher_rollouts import chat_with_retries  # noqa: E402
from provider_adapter import DEEPSEEK_CARRIER_JSON_OUTPUT  # noqa: E402
from provider_client import load_api_config  # noqa: E402


EVALUATOR_SYSTEM_PROMPT = """\
You are an audit-only recovery-anchor selector for a relational tool-use trajectory.
You are not solving the task and must not propose an answer or a new tool action.

Every candidate is a real harness error whose visible environment state did not change. Select a
candidate only when its legal prefix retains useful grounded state and its exact LAST TOOL ERROR
provides enough feedback for a capable teacher to continue causally. Prefer the earliest suitable
candidate. If every prefix is confused, excessively repetitive, or lacks useful grounded progress,
route the task to a teacher that starts from scratch.

Return one strict JSON object with exactly:
{"decision":"select_candidate"|"route_to_teacher_from_scratch",
 "candidate_id":<string or null>,
 "rationale":<brief string>}
For select_candidate, candidate_id must exactly match an offered candidate. For the scratch
decision, candidate_id must be null. Your rationale is audit-only and will not be shown to the
teacher or used as an SFT target.
"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def evaluator_messages(package: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "task": package["task"],
        "student_rollout": {
            key: package["student_rollout"].get(key)
            for key in ("sample_index", "failure_type", "legal", "steps", "errors")
        },
        "candidates": package["candidates"],
    }
    return [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Select one recovery anchor or route this task to teacher-from-scratch.\n\n"
                + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            ),
        },
    ]


def parse_decision(text: str, package: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"evaluator response is not JSON: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {
        "decision",
        "candidate_id",
        "rationale",
    }:
        raise ValueError("evaluator response must contain exactly decision/candidate_id/rationale")
    decision = value["decision"]
    candidate_id = value["candidate_id"]
    rationale = value["rationale"]
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("evaluator rationale must be a non-empty string")
    offered = {
        str(candidate["candidate_id"]): candidate
        for candidate in package.get("candidates") or []
    }
    if decision == "select_candidate":
        if not isinstance(candidate_id, str) or candidate_id not in offered:
            raise ValueError("evaluator selected a candidate_id that was not offered")
    elif decision == "route_to_teacher_from_scratch":
        if candidate_id is not None:
            raise ValueError("scratch decision requires candidate_id=null")
    else:
        raise ValueError(f"unsupported evaluator decision: {decision!r}")
    return {
        "decision": decision,
        "candidate_id": candidate_id,
        "rationale": rationale.strip(),
    }


def selected_candidate(
    package: dict[str, Any],
    decision: dict[str, Any],
    *,
    evaluator_model: str,
    usage: dict[str, Any],
) -> dict[str, Any] | None:
    if decision["decision"] != "select_candidate":
        return None
    candidate = next(
        item
        for item in package["candidates"]
        if item["candidate_id"] == decision["candidate_id"]
    )
    return {
        "task": package["task"],
        "student_rollout": package["student_rollout"],
        "selected_candidate": candidate,
        "selection_audit": {
            "model": evaluator_model,
            "decision": decision["decision"],
            "rationale": decision["rationale"],
            "usage": usage,
            "rationale_is_teacher_visible": False,
            "rationale_is_sft_target": False,
            "gold_sql_visible": False,
        },
    }


def task_identity(task: dict[str, Any]) -> str:
    value = task.get("example_id") or task.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("task is missing example_id")
    return value


def run(
    candidates_path: Path,
    tasks_path: Path,
    immediate_scratch_path: Path,
    selected_out: Path,
    scratch_out: Path,
    audit_out: Path,
    failures_out: Path,
    manifest_path: Path,
    *,
    model: str,
    workers: int,
    max_tokens: int,
    api_timeout: int,
    api_retries: int,
    resume: bool,
) -> dict[str, Any]:
    packages = read_jsonl(candidates_path)
    tasks = read_jsonl(tasks_path)
    tasks_by_id = {task_identity(task): task for task in tasks}
    if len(tasks_by_id) != len(tasks):
        raise ValueError("task cohort contains duplicate example ids")
    immediate_scratch = read_jsonl(immediate_scratch_path)
    immediate_ids = {task_identity(task) for task in immediate_scratch}
    package_ids = {package["task"]["example_id"] for package in packages}
    if immediate_ids & package_ids:
        raise ValueError("immediate scratch and recovery candidate lanes overlap")
    missing = package_ids - set(tasks_by_id)
    if missing:
        raise ValueError(f"recovery candidates missing from task cohort: {sorted(missing)[:5]}")

    api_key, base_url = load_api_config()
    if not api_key or not base_url:
        raise ValueError("api.md must define API_KEY and BASE_URL")

    def evaluate(package: dict[str, Any]) -> dict[str, Any]:
        text, usage, reasoning = chat_with_retries(
            base_url=base_url,
            api_key=api_key,
            model=model,
            messages=evaluator_messages(package),
            max_tokens=max_tokens,
            timeout=api_timeout,
            retries=api_retries,
            deepseek_carrier=DEEPSEEK_CARRIER_JSON_OUTPUT,
        )
        decision = parse_decision(text, package)
        return {
            "example_id": package["task"]["example_id"],
            "decision": decision,
            "selected": selected_candidate(
                package,
                decision,
                evaluator_model=model,
                usage=usage,
            ),
            "usage": usage,
            "provider_reasoning_present": bool(reasoning.strip()),
        }

    if audit_out.exists() and not resume:
        raise ValueError(f"{audit_out} exists; use --resume or choose new outputs")
    previous_results = read_jsonl(audit_out) if resume and audit_out.exists() else []
    previous_ids = [row["example_id"] for row in previous_results]
    if len(previous_ids) != len(set(previous_ids)):
        raise ValueError("existing evaluator audit contains duplicate example ids")
    unknown_previous = set(previous_ids) - package_ids
    if unknown_previous:
        raise ValueError(
            f"existing evaluator audit contains tasks outside candidates: "
            f"{sorted(unknown_previous)[:5]}"
        )
    if not audit_out.exists():
        write_jsonl_atomic(audit_out, [])

    work = [
        package
        for package in packages
        if package["task"]["example_id"] not in set(previous_ids)
    ]
    new_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate, package): package for package in work}
        for future in as_completed(futures):
            package = futures[future]
            identity = package["task"]["example_id"]
            try:
                result = future.result()
                append_jsonl(audit_out, result)
                new_results.append(result)
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "example_id": identity,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    results = [*previous_results, *new_results]
    results.sort(key=lambda row: row["example_id"])
    failures.sort(key=lambda row: row["example_id"])
    selected = [
        row["selected"] for row in results if row["selected"] is not None
    ]
    evaluator_scratch_ids = {
        row["example_id"]
        for row in results
        if row["decision"]["decision"] == "route_to_teacher_from_scratch"
    }
    scratch = [
        *immediate_scratch,
        *(tasks_by_id[identity] for identity in sorted(evaluator_scratch_ids)),
    ]
    if failures:
        # Keep unresolved tasks out of both semantic lanes so retries cannot duplicate work.
        resolved_ids = {
            row["example_id"] for row in results
        }
    else:
        resolved_ids = package_ids
    selected_ids = {row["task"]["example_id"] for row in selected}
    scratch_ids = {task_identity(task) for task in scratch}
    if selected_ids & scratch_ids:
        raise RuntimeError("selected recovery and teacher-from-scratch outputs overlap")
    if (selected_ids | evaluator_scratch_ids) != resolved_ids:
        raise RuntimeError("resolved evaluator decisions do not partition candidate tasks")

    write_jsonl_atomic(selected_out, selected)
    write_jsonl_atomic(scratch_out, scratch)
    write_jsonl_atomic(failures_out, failures)
    decisions = Counter(row["decision"]["decision"] for row in results)
    manifest = {
        "method": "external_model_real_error_anchor_selection",
        "model": model,
        "candidate_tasks": len(packages),
        "resumed_completed_tasks": len(previous_results),
        "evaluated_this_run": len(new_results),
        "decisions": dict(sorted(decisions.items())),
        "unresolved_api_or_format_failures": len(failures),
        "selected_recovery": len(selected),
        "teacher_from_scratch": len(scratch),
        "immediate_teacher_from_scratch": len(immediate_scratch),
        "safety": {
            "gold_sql_visible_to_evaluator": False,
            "evaluator_rationale_visible_to_teacher": False,
            "evaluator_rationale_is_sft_target": False,
            "teacher_receives_future_student_suffix": False,
            "rejected_student_action_is_sft_target": False,
        },
        "outputs": {
            "selected_recovery": str(selected_out),
            "teacher_from_scratch": str(scratch_out),
            "audit": str(audit_out),
            "unresolved_failures": str(failures_out),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--immediate-scratch", type=Path, required=True)
    parser.add_argument("--selected-out", type=Path, required=True)
    parser.add_argument("--scratch-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--failures-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    manifest = run(
        args.candidates.resolve(),
        args.tasks.resolve(),
        args.immediate_scratch.resolve(),
        args.selected_out.resolve(),
        args.scratch_out.resolve(),
        args.audit_out.resolve(),
        args.failures_out.resolve(),
        args.manifest.resolve(),
        model=args.model,
        workers=args.workers,
        max_tokens=args.max_tokens,
        api_timeout=args.api_timeout,
        api_retries=args.api_retries,
        resume=args.resume,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
