#!/usr/bin/env python3
"""Build a no-leak paired summary for Direct-SQL-search versus atomic.

The emitted artifact contains only public outcome, protocol, action, error, usage, and task-id
metadata. It never emits questions, SQL text, database rows, model reasoning, or verifier values.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


USAGE_KEYS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "api_request_attempts",
    "api_transport_retries",
    "api_context_retries",
    "api_completion_retries",
    "api_carrier_retries",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_mcnemar(direct_only: int, atomic_only: int) -> float:
    discordant = direct_only + atomic_only
    if not discordant:
        return 1.0
    lower = min(direct_only, atomic_only)
    tail = sum(
        math.comb(discordant, index) * (0.5 ** discordant)
        for index in range(lower + 1)
    )
    return min(1.0, 2.0 * tail)


def _record_id(record: dict, id_field: str) -> str:
    value = record.get(id_field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing non-empty {id_field}")
    return value


def _error_type(event: dict) -> str:
    return str(event.get("error_type") or event.get("type") or "unknown")


def summarize_arm(records: list[dict], *, id_field: str) -> dict:
    by_id: dict[str, dict] = {}
    tools: collections.Counter[str] = collections.Counter()
    errors_by_type: collections.Counter[str] = collections.Counter()
    failure_types: collections.Counter[str] = collections.Counter()
    usage: collections.Counter[str] = collections.Counter()

    for record in records:
        task_id = _record_id(record, id_field)
        if task_id in by_id:
            raise ValueError(f"duplicate task id {task_id}")
        by_id[task_id] = record
        if not record.get("correct"):
            failure_types[str(record.get("failure_type") or "unknown")] += 1
        for key in USAGE_KEYS:
            usage[key] += int((record.get("usage") or {}).get(key) or 0)
        for event in record.get("error_events") or []:
            if isinstance(event, dict):
                errors_by_type[_error_type(event)] += 1
        for turn in record.get("turns") or []:
            parsed = turn.get("parsed")
            if isinstance(parsed, dict) and isinstance(parsed.get("tool"), str):
                tools[parsed["tool"]] += 1

    total = len(records)
    actions = sum(int(record.get("steps") or 0) for record in records)
    correct = sum(bool(record.get("correct")) for record in records)
    legal = sum(bool(record.get("legal")) for record in records)
    return {
        "tasks": total,
        "correct": correct,
        "legal": legal,
        "clean_success": sum(record.get("outcome") == "clean_success" for record in records),
        "recovered_success": sum(
            record.get("outcome") == "recovered_success" for record in records
        ),
        "actions": actions,
        "mean_actions": round(actions / total, 4) if total else 0,
        "process_errors": sum(int(record.get("errors") or 0) for record in records),
        "max_step_failures": sum(
            record.get("failure_type") == "max_steps" for record in records
        ),
        "failure_types": dict(sorted(failure_types.items())),
        "errors_by_type": dict(sorted(errors_by_type.items())),
        "tool_counts": dict(sorted(tools.items())),
        "search_calls": int(tools.get("search_values", 0)),
        "no_progress_errors": int(errors_by_type.get("no_progress_error", 0)),
        "usage": {key: int(usage[key]) for key in USAGE_KEYS},
        "protocol_version": sorted({str(record.get("protocol_version")) for record in records}),
        "tool_scheme": sorted({str(record.get("tool_scheme")) for record in records}),
        "outcomes": {task_id: bool(record.get("correct")) for task_id, record in by_id.items()},
    }


def build_summary(
    direct_records: list[dict],
    atomic_records: list[dict],
    direct_audit: dict,
    atomic_audit: dict,
) -> dict:
    direct = summarize_arm(direct_records, id_field="instance_id")
    atomic = summarize_arm(atomic_records, id_field="trajectory_id")
    direct_ids = set(direct["outcomes"])
    atomic_ids = set(atomic["outcomes"])
    if direct_ids != atomic_ids:
        raise ValueError(
            "paired task ids differ: "
            f"direct_only={sorted(direct_ids - atomic_ids)}, "
            f"atomic_only={sorted(atomic_ids - direct_ids)}"
        )
    task_ids = sorted(direct_ids)
    both = [
        task_id for task_id in task_ids
        if direct["outcomes"][task_id] and atomic["outcomes"][task_id]
    ]
    direct_only = [
        task_id for task_id in task_ids
        if direct["outcomes"][task_id] and not atomic["outcomes"][task_id]
    ]
    atomic_only = [
        task_id for task_id in task_ids
        if not direct["outcomes"][task_id] and atomic["outcomes"][task_id]
    ]
    neither = [
        task_id for task_id in task_ids
        if not direct["outcomes"][task_id] and not atomic["outcomes"][task_id]
    ]

    direct["retained_diagnostic_success"] = int(
        direct_audit.get("retained_diagnostic_success") or 0
    )
    atomic["retained_diagnostic_success"] = int(
        atomic_audit.get("verified_episodes") or 0
    )
    direct.pop("outcomes")
    atomic.pop("outcomes")
    atomic_tokens = atomic["usage"]["total_tokens"]
    atomic_actions = atomic["actions"]
    return {
        "schema_version": "direct-sql-atomic-gate-summary-v1",
        "information_boundary": {
            "emits_sql": False,
            "emits_rows": False,
            "emits_question": False,
            "emits_reasoning": False,
            "emits_gold_or_reference_values": False,
        },
        "direct": direct,
        "atomic": atomic,
        "paired": {
            "tasks": len(task_ids),
            "both_correct": len(both),
            "direct_only": len(direct_only),
            "atomic_only": len(atomic_only),
            "neither_correct": len(neither),
            "direct_only_ids": direct_only,
            "atomic_only_ids": atomic_only,
            "exact_mcnemar_p": exact_mcnemar(len(direct_only), len(atomic_only)),
        },
        "efficiency": {
            "direct_token_fraction_of_atomic": round(
                direct["usage"]["total_tokens"] / atomic_tokens, 6
            ) if atomic_tokens else None,
            "direct_token_reduction_vs_atomic": round(
                1 - direct["usage"]["total_tokens"] / atomic_tokens, 6
            ) if atomic_tokens else None,
            "direct_action_fraction_of_atomic": round(
                direct["actions"] / atomic_actions, 6
            ) if atomic_actions else None,
            "direct_action_reduction_vs_atomic": round(
                1 - direct["actions"] / atomic_actions, 6
            ) if atomic_actions else None,
        },
        "preregistered_gates": {
            "engineering_stability": (
                direct["legal"] >= direct["tasks"] - 1
                and direct["max_step_failures"] <= 1
            ),
            "worth_larger_fair_comparison": (
                direct["correct"] >= atomic["correct"] - 1
            ),
            "no_observed_accuracy_deficit": direct["correct"] >= atomic["correct"],
            "direct_is_at_least_two_lower": (
                direct["correct"] <= atomic["correct"] - 2
            ),
        },
        "audit_gates": {
            "direct_structural_pass": (
                int(direct_audit.get("structural_pass") or 0) == direct["tasks"]
            ),
            "direct_replay_pass": (
                int(direct_audit.get("replay_correct") or 0) == direct["correct"]
            ),
            "atomic_structural_gate": atomic_audit.get("structural_gate"),
            "atomic_prompt_variant_gate": atomic_audit.get("prompt_variant_gate"),
        },
        "training_admission": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct-all", required=True)
    parser.add_argument("--direct-audit", required=True)
    parser.add_argument("--atomic-all", required=True)
    parser.add_argument("--atomic-audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    direct_path = Path(args.direct_all)
    direct_audit_path = Path(args.direct_audit)
    atomic_path = Path(args.atomic_all)
    atomic_audit_path = Path(args.atomic_audit)
    summary = build_summary(
        read_jsonl(direct_path),
        read_jsonl(atomic_path),
        read_json(direct_audit_path),
        read_json(atomic_audit_path),
    )
    summary["source_artifacts"] = {
        "direct_all": str(direct_path),
        "direct_all_sha256": file_sha256(direct_path),
        "direct_audit": str(direct_audit_path),
        "direct_audit_sha256": file_sha256(direct_audit_path),
        "atomic_all": str(atomic_path),
        "atomic_all_sha256": file_sha256(atomic_path),
        "atomic_audit": str(atomic_audit_path),
        "atomic_audit_sha256": file_sha256(atomic_audit_path),
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
