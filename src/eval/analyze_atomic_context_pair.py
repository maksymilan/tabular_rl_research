#!/usr/bin/env python3
"""Audit a paired atomic database-context experiment from persisted rollout records."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
from pathlib import Path


MATCHED_CONFIG_FIELDS = (
    "tool_scheme",
    "tool_scheme_registry_version",
    "assistant_carrier",
    "protocol_version",
    "model",
    "split",
    "examples_file",
    "source_count",
    "max_steps",
    "workers",
    "max_errors_per_type",
    "attempts_per_example",
    "max_tokens",
    "table_output_rows",
    "context_mode",
    "history_turns",
    "rolling_prompt_variant",
    "policy_prompt_variant",
    "plan_policy",
    "deepseek_carrier",
    "denotation_comparison",
    "api_transport_retries_per_request",
    "provider_request_options",
    "training_admission",
    "sft_export_eligible",
    "tool_schema_sha256",
    "model_visible_tool_schema_sha256",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> dict[str, dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_id = {str(record["trajectory_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError(f"{path}: duplicate trajectory_id records")
    return by_id


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_two_sided_sign_p(gains: int, regressions: int) -> float:
    discordant = gains + regressions
    if not discordant:
        return 1.0
    lower = min(gains, regressions)
    probability = (
        2
        * sum(math.comb(discordant, value) for value in range(lower + 1))
        / (2**discordant)
    )
    return min(1.0, probability)


def error_histogram(records: dict[str, dict]) -> dict[str, int]:
    counter = collections.Counter(
        event.get("error_type", "unknown")
        for record in records.values()
        for event in record.get("error_events", [])
    )
    return dict(sorted(counter.items()))


def attempted_tool_histogram(records: dict[str, dict]) -> dict[str, int]:
    counter = collections.Counter()
    for record in records.values():
        for turn in record.get("turns", []):
            tool = (turn.get("parsed") or {}).get("tool")
            if tool:
                counter[tool] += 1
    return dict(counter.most_common())


def arm_summary(records: dict[str, dict]) -> dict:
    values = list(records.values())
    steps = [int(record.get("steps") or 0) for record in values]
    return {
        "count": len(values),
        "correct": sum(bool(record.get("correct")) for record in values),
        "accuracy": (
            sum(bool(record.get("correct")) for record in values) / len(values)
            if values
            else 0
        ),
        "legal": sum(bool(record.get("legal")) for record in values),
        "legal_rate": (
            sum(bool(record.get("legal")) for record in values) / len(values)
            if values
            else 0
        ),
        "process_errors": sum(int(record.get("errors") or 0) for record in values),
        "error_types": error_histogram(records),
        "total_steps": sum(steps),
        "mean_steps": statistics.mean(steps) if steps else 0,
        "total_tokens": sum(
            int((record.get("usage") or {}).get("total_tokens") or 0)
            for record in values
        ),
        "prompt_tokens": sum(
            int((record.get("usage") or {}).get("prompt_tokens") or 0)
            for record in values
        ),
        "completion_tokens": sum(
            int((record.get("usage") or {}).get("completion_tokens") or 0)
            for record in values
        ),
        "attempted_tool_calls": attempted_tool_histogram(records),
    }


def enrichment_summary(records: dict[str, dict]) -> dict:
    calls = collections.Counter()
    enriched_calls = collections.Counter()
    fields = collections.Counter()
    tasks = collections.Counter()
    task_patterns = collections.Counter()
    visible_inspect_fields = collections.Counter()
    tasks_with_visible_inspect_field = collections.Counter()
    for record in records.values():
        task_tools = set()
        task_inspect_fields = set()
        audit = record.get("database_context_audit") or {}
        for event in audit.get("perception_enrichment_events", []):
            tool = str(event.get("tool"))
            field_count = int(event.get("enriched_field_count") or 0)
            calls[tool] += 1
            fields[tool] += field_count
            if field_count:
                enriched_calls[tool] += 1
                task_tools.add(tool)
        for tool in task_tools:
            tasks[tool] += 1
        task_patterns["+".join(sorted(task_tools)) or "none"] += 1
        for turn in record.get("turns", []):
            if (turn.get("parsed") or {}).get("tool") != "inspect_column":
                continue
            output = turn.get("tool_output")
            if not isinstance(output, dict):
                continue
            has_name = bool(output.get("semantic_name"))
            has_description = bool(output.get("column_description"))
            visible_inspect_fields["outputs"] += 1
            visible_inspect_fields["semantic_name"] += int(has_name)
            visible_inspect_fields["column_description"] += int(
                has_description
            )
            visible_inspect_fields["both"] += int(
                has_name and has_description
            )
            visible_inspect_fields["neither"] += int(
                not has_name and not has_description
            )
            if has_name:
                task_inspect_fields.add("semantic_name")
            if has_description:
                task_inspect_fields.add("column_description")
        for field in task_inspect_fields:
            tasks_with_visible_inspect_field[field] += 1
    return {
        "calls": dict(calls),
        "enriched_calls": dict(enriched_calls),
        "enriched_fields": dict(fields),
        "tasks_with_enrichment": dict(tasks),
        "task_coverage_patterns": dict(task_patterns),
        "visible_inspect_field_calls": dict(visible_inspect_fields),
        "tasks_with_visible_inspect_field": dict(
            tasks_with_visible_inspect_field
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--treatment", required=True)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--treatment-manifest", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    treatment_path = Path(args.treatment)
    baseline_manifest_path = Path(args.baseline_manifest)
    treatment_manifest_path = Path(args.treatment_manifest)
    dataset_path = Path(args.dataset)
    baseline = read_jsonl(baseline_path)
    treatment = read_jsonl(treatment_path)
    baseline_manifest = read_json(baseline_manifest_path)
    treatment_manifest = read_json(treatment_manifest_path)
    if set(baseline) != set(treatment):
        raise ValueError("baseline and treatment trajectory IDs do not match")

    config_mismatches = {
        field: {
            "baseline": baseline_manifest.get(field),
            "treatment": treatment_manifest.get(field),
        }
        for field in MATCHED_CONFIG_FIELDS
        if baseline_manifest.get(field) != treatment_manifest.get(field)
    }
    if config_mismatches:
        raise ValueError(f"matched configuration differs: {config_mismatches}")

    ids = sorted(baseline)
    gains = [
        trajectory_id
        for trajectory_id in ids
        if not baseline[trajectory_id].get("correct")
        and treatment[trajectory_id].get("correct")
    ]
    regressions = [
        trajectory_id
        for trajectory_id in ids
        if baseline[trajectory_id].get("correct")
        and not treatment[trajectory_id].get("correct")
    ]
    stable_correct = [
        trajectory_id
        for trajectory_id in ids
        if baseline[trajectory_id].get("correct")
        and treatment[trajectory_id].get("correct")
    ]
    stable_wrong = [
        trajectory_id
        for trajectory_id in ids
        if not baseline[trajectory_id].get("correct")
        and not treatment[trajectory_id].get("correct")
    ]
    changed = []
    for trajectory_id in gains + regressions:
        events = (
            treatment[trajectory_id]
            .get("database_context_audit", {})
            .get("perception_enrichment_events", [])
        )
        changed.append({
            "trajectory_id": trajectory_id,
            "direction": "gain" if trajectory_id in gains else "regression",
            "db_id": treatment[trajectory_id].get("db_id"),
            "question": treatment[trajectory_id].get("question"),
            "baseline_steps": baseline[trajectory_id].get("steps"),
            "treatment_steps": treatment[trajectory_id].get("steps"),
            "baseline_errors": baseline[trajectory_id].get("errors"),
            "treatment_errors": treatment[trajectory_id].get("errors"),
            "enrichment_events": [
                {
                    "tool": event.get("tool"),
                    "enriched_field_count": event.get("enriched_field_count"),
                }
                for event in events
            ],
        })

    baseline_summary = arm_summary(baseline)
    treatment_summary = arm_summary(treatment)
    result = {
        "analysis": "paired-atomic-database-context-v1",
        "dataset": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
        "baseline_records": str(baseline_path),
        "baseline_records_sha256": sha256(baseline_path),
        "treatment_records": str(treatment_path),
        "treatment_records_sha256": sha256(treatment_path),
        "baseline_manifest": str(baseline_manifest_path),
        "baseline_manifest_sha256": sha256(baseline_manifest_path),
        "treatment_manifest": str(treatment_manifest_path),
        "treatment_manifest_sha256": sha256(treatment_manifest_path),
        "matched_config": {
            field: baseline_manifest.get(field)
            for field in MATCHED_CONFIG_FIELDS
        },
        "intentional_differences": {
            "database_context_profile": {
                "baseline": baseline_manifest.get("database_context_profile"),
                "treatment": treatment_manifest.get("database_context_profile"),
            },
            "student_runtime_prompt_sha256": {
                "baseline": baseline_manifest.get("student_runtime_prompt_sha256"),
                "treatment": treatment_manifest.get("student_runtime_prompt_sha256"),
            },
            "teacher_provider_prompt_sha256": {
                "baseline": baseline_manifest.get("teacher_provider_prompt_sha256"),
                "treatment": treatment_manifest.get("teacher_provider_prompt_sha256"),
            },
        },
        "baseline": baseline_summary,
        "treatment": treatment_summary,
        "delta": {
            "correct": treatment_summary["correct"] - baseline_summary["correct"],
            "accuracy_pp": 100
            * (treatment_summary["accuracy"] - baseline_summary["accuracy"]),
            "legal": treatment_summary["legal"] - baseline_summary["legal"],
            "process_errors": (
                treatment_summary["process_errors"]
                - baseline_summary["process_errors"]
            ),
            "total_steps": (
                treatment_summary["total_steps"] - baseline_summary["total_steps"]
            ),
            "mean_steps": (
                treatment_summary["mean_steps"] - baseline_summary["mean_steps"]
            ),
            "total_tokens": (
                treatment_summary["total_tokens"]
                - baseline_summary["total_tokens"]
            ),
            "total_tokens_percent": (
                100
                * (
                    treatment_summary["total_tokens"]
                    / baseline_summary["total_tokens"]
                    - 1
                )
                if baseline_summary["total_tokens"]
                else None
            ),
        },
        "paired_outcomes": {
            "gains": gains,
            "regressions": regressions,
            "stable_correct_count": len(stable_correct),
            "stable_wrong_count": len(stable_wrong),
            "discordant_count": len(gains) + len(regressions),
            "exact_two_sided_sign_p": exact_two_sided_sign_p(
                len(gains),
                len(regressions),
            ),
        },
        "enrichment": enrichment_summary(treatment),
        "changed_tasks": changed,
    }
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
