#!/usr/bin/env python3
"""Paired analysis for action-block episodes versus recorded atomic-tool controls."""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path


def read_records(paths: list[Path]) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            record = json.loads(line)
            trajectory_id = str(record.get("trajectory_id"))
            if not trajectory_id or trajectory_id == "None":
                raise ValueError(f"{path}:{line_number}: missing trajectory_id")
            if trajectory_id in records:
                raise ValueError(
                    f"duplicate trajectory_id {trajectory_id!r} across input records"
                )
            records[trajectory_id] = record
    return records


def exact_two_sided_binomial_p(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k)
        for k in range(0, min(a_only, b_only) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2 * tail)


def numeric_usage(records: list[dict], key: str) -> float:
    return sum(
        float((record.get("usage") or {}).get(key) or 0)
        for record in records
    )


def error_hist(records: list[dict]) -> dict[str, int]:
    counts: collections.Counter = collections.Counter()
    for record in records:
        for event in record.get("error_events") or []:
            counts[str(event.get("error_type") or "unknown")] += 1
    return dict(counts.most_common())


def failure_hist(records: list[dict]) -> dict[str, int]:
    counts = collections.Counter(
        str(record.get("failure_type") or "none")
        for record in records
    )
    return dict(counts.most_common())


def interface_resolution_hist(records: list[dict]) -> dict[str, int]:
    counts: collections.Counter = collections.Counter()
    for record in records:
        counts.update(record.get("interface_resolution_hist") or {})
    return dict(counts.most_common())


def summarize_new(records: list[dict]) -> dict:
    n = len(records)
    batch_sizes: list[int] = []
    provider_fingerprints: collections.Counter = collections.Counter()
    for record in records:
        for turn in record.get("turns") or []:
            parsed = turn.get("parsed") or {}
            if parsed.get("tool") in {"plan", "action_block"}:
                calls = (parsed.get("arguments") or {}).get("calls") or []
                batch_sizes.append(len(calls))
            fingerprint = (
                turn.get("provider_response_metadata") or {}
            ).get("system_fingerprint")
            if fingerprint:
                provider_fingerprints[str(fingerprint)] += 1
    return {
        "tasks": n,
        "correct": sum(bool(record.get("correct")) for record in records),
        "legal": sum(bool(record.get("legal")) for record in records),
        "model_turns_total": sum(int(record.get("model_turns") or 0) for record in records),
        "model_turns_mean": (
            sum(int(record.get("model_turns") or 0) for record in records) / n
            if n else None
        ),
        "action_blocks_total": sum(
            int(record.get("action_blocks") or record.get("plan_rounds") or 0)
            for record in records
        ),
        "action_blocks_mean": (
            sum(
                int(record.get("action_blocks") or record.get("plan_rounds") or 0)
                for record in records
            ) / n
            if n else None
        ),
        "atomic_actions_total": sum(int(record.get("atomic_actions") or 0) for record in records),
        "atomic_actions_mean": (
            sum(int(record.get("atomic_actions") or 0) for record in records) / n
            if n else None
        ),
        "process_errors": sum(int(record.get("errors") or 0) for record in records),
        "interface_resolutions_total": sum(
            int(record.get("interface_resolutions") or 0)
            for record in records
        ),
        "tasks_with_interface_resolution": sum(
            int(record.get("interface_resolutions") or 0) > 0
            for record in records
        ),
        "correct_with_interface_resolution": sum(
            (
                int(record.get("interface_resolutions") or 0) > 0
                and bool(record.get("correct"))
            )
            for record in records
        ),
        "interface_resolution_hist": interface_resolution_hist(records),
        "submitted_calls_total": sum(
            int(record.get("submitted_calls") or record.get("planned_nodes") or 0)
            for record in records
        ),
        "blocked_nodes_total": sum(
            int(record.get("blocked_nodes") or 0) for record in records
        ),
        "batch_size_mean": sum(batch_sizes) / len(batch_sizes) if batch_sizes else None,
        "batch_size_hist": dict(collections.Counter(batch_sizes).most_common()),
        "api_requests": numeric_usage(records, "api_request_attempts"),
        "prompt_tokens": numeric_usage(records, "prompt_tokens"),
        "completion_tokens": numeric_usage(records, "completion_tokens"),
        "total_tokens": numeric_usage(records, "total_tokens"),
        "error_hist": error_hist(records),
        "failure_hist": failure_hist(records),
        "provider_system_fingerprints": dict(provider_fingerprints),
    }


def summarize_baseline(records: list[dict]) -> dict:
    n = len(records)
    return {
        "tasks": n,
        "correct": sum(bool(record.get("correct")) for record in records),
        "legal": sum(bool(record.get("legal")) for record in records),
        "model_turns_total": sum(len(record.get("turns") or []) for record in records),
        "model_turns_mean": (
            sum(len(record.get("turns") or []) for record in records) / n
            if n else None
        ),
        "semantic_actions_total": sum(int(record.get("steps") or 0) for record in records),
        "semantic_actions_mean": (
            sum(int(record.get("steps") or 0) for record in records) / n
            if n else None
        ),
        "process_errors": sum(int(record.get("errors") or 0) for record in records),
        "api_requests": numeric_usage(records, "api_request_attempts"),
        "prompt_tokens": numeric_usage(records, "prompt_tokens"),
        "completion_tokens": numeric_usage(records, "completion_tokens"),
        "total_tokens": numeric_usage(records, "total_tokens"),
        "error_hist": error_hist(records),
        "failure_hist": failure_hist(records),
    }


def paired_summary(
    ids: list[str],
    new_by_id: dict[str, dict],
    baseline_by_id: dict[str, dict],
) -> dict:
    both_correct = []
    both_wrong = []
    new_only = []
    baseline_only = []
    for trajectory_id in ids:
        new = new_by_id[trajectory_id]
        baseline = baseline_by_id[trajectory_id]
        pair = (bool(new.get("correct")), bool(baseline.get("correct")))
        if pair == (True, True):
            both_correct.append(trajectory_id)
        elif pair == (False, False):
            both_wrong.append(trajectory_id)
        elif pair == (True, False):
            new_only.append(trajectory_id)
        else:
            baseline_only.append(trajectory_id)
    return {
        "tasks": len(ids),
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        "new_only_correct": len(new_only),
        "baseline_only_correct": len(baseline_only),
        "net_correct_delta": len(new_only) - len(baseline_only),
        "exact_two_sided_p": exact_two_sided_binomial_p(
            len(new_only), len(baseline_only)
        ),
        "new_correct": sum(
            bool(new_by_id[trajectory_id].get("correct"))
            for trajectory_id in ids
        ),
        "baseline_correct": sum(
            bool(baseline_by_id[trajectory_id].get("correct"))
            for trajectory_id in ids
        ),
        "new_only_ids": new_only,
        "baseline_only_ids": baseline_only,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--baseline", type=Path, nargs="+", required=True
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    new_by_id = read_records(args.new)
    baseline_by_id = read_records(args.baseline)
    missing = sorted(set(new_by_id) - set(baseline_by_id))
    if missing:
        raise ValueError(
            f"{len(missing)} new task ids are absent from baseline: {missing[:10]}"
        )
    ids = sorted(new_by_id)
    new_records = [new_by_id[trajectory_id] for trajectory_id in ids]
    baseline_records = [baseline_by_id[trajectory_id] for trajectory_id in ids]

    resolved_ids = [
        trajectory_id
        for trajectory_id in ids
        if int(new_by_id[trajectory_id].get("interface_resolutions") or 0) > 0
    ]
    resolved_id_set = set(resolved_ids)
    unresolved_ids = [
        trajectory_id
        for trajectory_id in ids
        if trajectory_id not in resolved_id_set
    ]

    analysis = {
        "task_ids": ids,
        "new_paths": [str(path) for path in args.new],
        "baseline_paths": [str(path) for path in args.baseline],
        "new": summarize_new(new_records),
        "baseline": summarize_baseline(baseline_records),
        "paired": paired_summary(ids, new_by_id, baseline_by_id),
        "paired_by_interface_resolution": {
            "resolved": paired_summary(
                resolved_ids,
                new_by_id,
                baseline_by_id,
            ),
            "not_resolved": paired_summary(
                unresolved_ids,
                new_by_id,
                baseline_by_id,
            ),
        },
    }
    text = json.dumps(analysis, ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
