#!/usr/bin/env python3
"""Pair verified teacher fallback branches with failed student pass@K trajectories."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src" / "eval"), str(ROOT / "src" / "harness")]

from bird_sft1_teacher import replay_success_trajectory  # noqa: E402
from build_rolling_sft_data import convert_step, digest  # noqa: E402
from build_bird_sft2_dataset import (  # noqa: E402
    read_jsonl,
    replay_sample,
    tasks_by_index,
    write_jsonl,
)
from protocol import SYSTEM_PROMPT, rolling_system_prompt  # noqa: E402


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def state_hash(value: Any) -> str:
    return hashlib.sha256(compact(value).encode("utf-8")).hexdigest()


def semantic_action(call: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize only tool fields whose order is explicitly non-semantic."""
    normalized = json.loads(json.dumps(call))
    if normalized.get("tool") == "describe_table":
        tables = (normalized.get("arguments") or {}).get("tables")
        if isinstance(tables, list):
            normalized["arguments"]["tables"] = sorted(tables)
    return normalized


def same_step(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("step_id") == right.get("step_id")
        and compact(semantic_action(left.get("tool_call") or {}))
        == compact(semantic_action(right.get("tool_call") or {}))
        and state_hash(left.get("environment_state_before")) == state_hash(right.get("environment_state_before"))
        and state_hash(left.get("environment_state")) == state_hash(right.get("environment_state"))
    )


def referenced_base_tables(call: dict[str, Any], base_tables: set[str]) -> set[str]:
    arguments = call.get("arguments") or {}
    values: list[Any] = []
    if "table" in arguments:
        values.append(arguments["table"])
    if isinstance(arguments.get("tables"), list):
        values.extend(arguments["tables"])
    return {str(value) for value in values if str(value) in base_tables}


def deterministic_diagnosis(
    bad: dict[str, Any],
    teacher_steps: list[dict[str, Any]],
    base_tables: set[str],
) -> str | None:
    if bad.get("tool_call", {}).get("tool") == "answer_from_context":
        return "wrong_terminal_answer"
    output = bad.get("tool_output") or {}
    if isinstance(output, dict) and output.get("row_count") == 0:
        return "empty_result_branch"
    teacher_tables = set().union(*(
        referenced_base_tables(step.get("tool_call") or {}, base_tables)
        for step in teacher_steps
    )) if teacher_steps else set()
    extra = referenced_base_tables(bad.get("tool_call") or {}, base_tables) - teacher_tables
    if extra:
        return "extraneous_base_table:" + ",".join(sorted(extra))
    return None


def passk_records(paths: list[Path]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            index = int(row["example_index"])
            if index in result:
                raise ValueError(f"duplicate pass@K task {index}")
            result[index] = row
    return result


def teacher_example_index(trajectory: dict[str, Any], tasks: dict[int, dict[str, Any]]) -> int:
    source = trajectory.get("source") or {}
    if source.get("example_index") is not None:
        return int(source["example_index"])
    example_id = source.get("example_id")
    matches = [index for index, task in tasks.items() if task.get("example_id") == example_id]
    if len(matches) != 1:
        raise ValueError(f"cannot resolve teacher example_id {example_id!r}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--passk-all", type=Path, action="append", required=True)
    parser.add_argument("--teacher-success", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    task_map = tasks_by_index(read_jsonl(args.tasks))
    student_records = passk_records(args.passk_all)
    teacher_rows = read_jsonl(args.teacher_success)
    system = rolling_system_prompt(SYSTEM_PROMPT, compact=False)
    records: list[dict[str, Any]] = []
    indices: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for teacher in teacher_rows:
        replay_ok, replay_error = replay_success_trajectory(teacher)
        if not replay_ok:
            skipped.append({"teacher_trajectory_id": teacher.get("trajectory_id"), "reason": replay_error})
            continue
        example_index = teacher_example_index(teacher, task_map)
        task = task_map[example_index]
        student_record = student_records.get(example_index)
        if not student_record or student_record.get("correct"):
            skipped.append({
                "teacher_trajectory_id": teacher.get("trajectory_id"),
                "reason": "not_a_pass4_failed_student_task",
            })
            continue
        teacher_steps = teacher.get("steps") or []
        base_tables = {
            str(item["table_name"])
            for item in teacher["initial_state"]["dataset_overview"].get("tables", [])
        }
        candidates = []
        for failed_sample in student_record.get("samples") or []:
            if failed_sample.get("correct"):
                continue
            try:
                student = replay_sample(
                    task,
                    failed_sample,
                    require_correct=False,
                    trajectory_id=(
                        f"bird_sft2_student_failure_{example_index}_sample_"
                        f"{failed_sample['sample_index']}"
                    ),
                )
            except Exception:
                continue
            student_steps = student.get("steps") or []
            common = 0
            while common < min(len(teacher_steps), len(student_steps)) and same_step(
                teacher_steps[common], student_steps[common]
            ):
                common += 1
            if common >= len(teacher_steps) or common >= len(student_steps):
                continue
            better = teacher_steps[common]
            bad = student_steps[common]
            if state_hash(better.get("environment_state_before")) != state_hash(
                bad.get("environment_state_before")
            ):
                continue
            if compact(semantic_action(better.get("tool_call") or {})) == compact(
                semantic_action(bad.get("tool_call") or {})
            ):
                continue
            diagnosis = deterministic_diagnosis(bad, teacher_steps, base_tables)
            if diagnosis is None:
                continue
            candidates.append((
                common, int(failed_sample.get("sample_index", 0)), diagnosis, failed_sample, student,
            ))
        if not candidates:
            skipped.append({
                "teacher_trajectory_id": teacher.get("trajectory_id"),
                "reason": "no_semantically_divergent_action_at_shared_state",
            })
            continue
        common, _, diagnosis, failed_sample, student = sorted(
            candidates, key=lambda item: (-item[0], item[1])
        )[0]
        student_steps = student["steps"]
        better = teacher_steps[common]
        bad = student_steps[common]

        synthetic = {
            "trajectory_id": f"bird_sft2_correction_{example_index}",
            "source": teacher["source"],
            "question": teacher["question"],
            "difficulty": teacher.get("difficulty") or (task.get("metadata") or {}).get("difficulty_proxy"),
            "initial_state": teacher["initial_state"],
            "steps": student_steps[:common] + [better],
        }
        record, index = convert_step(
            synthetic, common, history_turns=4, system=system, compact_observations=True,
        )
        correction_id = f"bird_sft2_correction_{example_index}_{better['step_id']}"
        record["metadata"].update({
            "record_id": correction_id,
            "transition_type": "decision_correction",
            "origin": "flash_teacher_after_student_pass4_failure",
            "student_bad_action_sha256": digest(bad["tool_call"]),
            "teacher_branch_trajectory_id": teacher["trajectory_id"],
            "deterministic_diagnosis": diagnosis,
        })
        index.update(record["metadata"])
        index.update({
            "source_episode_id": teacher["trajectory_id"],
            "student_failure_sample_index": int(failed_sample.get("sample_index", 0)),
            "student_failure_type": failed_sample.get("failure_type"),
            "common_prefix_steps": common,
            "deterministic_diagnosis": diagnosis,
            "state_before_sha256": state_hash(better["environment_state_before"]),
        })
        records.append(record)
        indices.append(index)
        audits.append({
            "record_id": correction_id,
            "example_index": example_index,
            "example_id": task.get("example_id"),
            "difficulty": synthetic["difficulty"],
            "student_passk_failed": True,
            "student_sample_index": int(failed_sample.get("sample_index", 0)),
            "student_failure_type": failed_sample.get("failure_type"),
            "common_prefix_steps": common,
            "deterministic_diagnosis": diagnosis,
            "state_before_sha256": state_hash(better["environment_state_before"]),
            "student_bad_action": bad["tool_call"],
            "teacher_better_action": better["tool_call"],
            "teacher_full_branch_replay_verified": True,
            "bad_action_is_sft_target": False,
            "teacher_target_is_sft_target": True,
        })

    write_jsonl(args.out, records)
    write_jsonl(args.index_out, indices)
    write_jsonl(args.audit_out, audits)
    manifest = {
        "protocol": "draft/bird_sft2_onpolicy_data_protocol.md",
        "teacher_success_input": str(args.teacher_success),
        "teacher_verified_branches": len(teacher_rows),
        "decision_correction_records": len(records),
        "skipped": skipped,
        "context_mode": "rolling-legal-history",
        "history_turns": 4,
        "rolling_prompt_variant": "full",
        "rolling_observation_style": "resident",
        "loss_policy": "last_assistant_turn_only",
        "error_or_bad_actions_are_sft_targets": False,
        "validation": {
            "teacher_full_branch_replay": True,
            "exact_shared_prefix": True,
            "same_state_before_divergent_action": True,
            "gold_sql_model_input": False,
        },
        "outputs": {
            "sft": str(args.out),
            "index": str(args.index_out),
            "audit": str(args.audit_out),
        },
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
