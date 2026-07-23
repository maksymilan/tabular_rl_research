#!/usr/bin/env python3
"""Audit a join-interface pilot at the local-action and replayed-reference boundaries."""
from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for subdir in ("harness", "sft", "eval", "rl"):
    sys.path.insert(0, str(ROOT / "src" / subdir))

from process_credit import replay_step_features  # noqa: E402
from protocol import ProtocolError, validate_model_arguments  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def join_arity(arguments: dict) -> int:
    if isinstance(arguments.get("joins"), list):
        return 1 + len(arguments["joins"])
    if isinstance(arguments.get("tables"), list):
        return len(arguments["tables"])
    if arguments.get("left") is not None and arguments.get("right") is not None:
        return 2
    return 0


def summarize_attempts(records: list[dict], *, current_protocol: bool) -> dict:
    join_turns: list[tuple[dict, dict]] = []
    for record in records:
        for turn in record.get("turns") or []:
            if (turn.get("parsed") or {}).get("tool") == "join_tables":
                join_turns.append((record, turn))
    executed = [
        turn for _, turn in join_turns
        if isinstance(turn.get("tool_output"), dict)
        and turn["tool_output"].get("kind") == "join"
    ]
    widths = [len(turn["tool_output"].get("columns") or []) for turn in executed]
    column_name_chars = [
        sum(len(str(column)) for column in turn["tool_output"].get("columns") or [])
        for turn in executed
    ]
    all_columns = [
        column
        for turn in executed
        for column in turn["tool_output"].get("columns") or []
    ]
    errors = [
        event
        for record in records
        for event in record.get("error_events") or []
        if event.get("attempted_tool") == "join_tables"
    ]
    canonical = 0
    if current_protocol:
        for _, turn in join_turns:
            try:
                validate_model_arguments("join_tables", turn["parsed"]["arguments"])
            except ProtocolError:
                continue
            canonical += 1
    arity_hist = collections.Counter(
        join_arity(turn["parsed"].get("arguments") or {}) for _, turn in join_turns
    )
    failure_hist = collections.Counter(
        str(record.get("failure_type") or "none") for record in records
    )
    lengths = [int(record.get("steps") or 0) for record in records]
    join_task_ids = {record.get("trajectory_id") for record, _ in join_turns}
    return {
        "episodes": len(records),
        "correct": sum(bool(record.get("correct")) for record in records),
        "fully_legal": sum(bool(record.get("legal")) for record in records),
        "failure_histogram": dict(sorted(failure_hist.items())),
        "trajectory_actions": {
            "mean": round(statistics.mean(lengths), 3) if lengths else None,
            "median": statistics.median(lengths) if lengths else None,
            "p90": percentile(lengths, 0.9),
            "max": max(lengths) if lengths else None,
        },
        "episodes_with_join_attempt": len(join_task_ids),
        "join_attempts": len(join_turns),
        "canonical_current_shape": canonical if current_protocol else None,
        "join_executed": len(executed),
        "join_local_success_rate": round(len(executed) / len(join_turns), 4) if join_turns else None,
        "join_local_errors": len(errors),
        "join_error_histogram": dict(sorted(collections.Counter(
            str(event.get("error_type") or "unknown") for event in errors
        ).items())),
        "join_arity_histogram": {str(key): value for key, value in sorted(arity_hist.items())},
        "multiway_join_attempts": sum(
            count for arity, count in arity_hist.items() if arity > 2
        ),
        "semantic_role_calls": sum(
            "base_role" in (turn["parsed"].get("arguments") or {})
            or any(
                isinstance(item, dict) and "role" in item
                for item in (turn["parsed"].get("arguments") or {}).get("joins") or []
            )
            for _, turn in join_turns
        ),
        "join_output_width": {
            "median": statistics.median(widths) if widths else None,
            "p90": percentile(widths, 0.9),
            "max": max(widths) if widths else None,
        },
        "join_output_column_name_chars": {
            "median": statistics.median(column_name_chars) if column_name_chars else None,
            "p90": percentile(column_name_chars, 0.9),
            "max": max(column_name_chars) if column_name_chars else None,
        },
        "flat_namespace_outputs": sum(
            all(
                isinstance(column, str)
                and "." in column
                and "__" not in column
                and not re.match(r"^join_[0-9]+\.", column)
                for column in turn["tool_output"].get("columns") or []
            )
            for turn in executed
        ),
        "output_columns_total": len(all_columns),
    }


def reference_audit(trajectories: list[dict]) -> dict:
    replay_correct = 0
    deterministic_complete = 0
    final_value_complete = 0
    action_literal_complete = 0
    join_steps = 0
    exact_join_input_edges = 0
    joins_in_back_slice = 0
    replay_errors: list[dict] = []
    replay_incorrect_ids: list[str] = []
    incomplete_final_value_ids: list[str] = []
    incomplete_join_edge_ids: list[str] = []
    joins_outside_back_slice_ids: list[str] = []

    for trajectory in trajectories:
        try:
            features, diagnostics = replay_step_features(trajectory)
        except Exception as exc:  # noqa: BLE001
            replay_errors.append({
                "trajectory_id": trajectory.get("trajectory_id"),
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue
        is_replay_correct = bool(diagnostics.get("replay_correct"))
        replay_correct += is_replay_correct
        if not is_replay_correct:
            replay_incorrect_ids.append(str(trajectory.get("trajectory_id")))
        deterministic_complete += bool(diagnostics.get("deterministic_grounding_complete"))
        final_value_complete += bool(diagnostics.get("final_value_grounding_complete"))
        if not diagnostics.get("final_value_grounding_complete"):
            incomplete_final_value_ids.append(str(trajectory.get("trajectory_id")))
        action_literal_complete += bool(diagnostics.get("action_literal_grounding_complete"))
        by_step = {step.get("step_id"): step for step in trajectory.get("steps") or []}
        back_slice = set(diagnostics.get("back_slice_step_ids") or [])
        for feature in features:
            if feature.tool != "join_tables" or not feature.legal_success:
                continue
            join_steps += 1
            joins_in_back_slice += feature.step_id in back_slice
            if feature.step_id not in back_slice:
                joins_outside_back_slice_ids.append(
                    f"{trajectory.get('trajectory_id')}:{feature.step_id}"
                )
            arguments = (by_step.get(feature.step_id) or {}).get("tool_call", {}).get("arguments") or {}
            expected = [arguments.get("base")] + [
                item.get("table")
                for item in arguments.get("joins") or []
                if isinstance(item, dict)
            ]
            observed = []
            for ref in feature.references:
                if ref.get("type") != "data":
                    continue
                target = ref.get("target") or {}
                observed.append(target.get("handle", target.get("table", ref.get("source"))))
            exact = collections.Counter(expected) == collections.Counter(observed)
            exact_join_input_edges += exact
            if not exact:
                incomplete_join_edge_ids.append(
                    f"{trajectory.get('trajectory_id')}:{feature.step_id}"
                )

    return {
        "verified_trajectories": len(trajectories),
        "replay_correct": replay_correct,
        "replay_incorrect_ids": replay_incorrect_ids,
        "replay_errors": replay_errors,
        "join_steps_replayed": join_steps,
        "join_steps_with_exact_per_input_data_edges": exact_join_input_edges,
        "join_steps_with_incomplete_input_edges": incomplete_join_edge_ids,
        "join_steps_in_final_backward_slice": joins_in_back_slice,
        "join_steps_outside_final_backward_slice": joins_outside_back_slice_ids,
        "action_literal_grounding_complete": action_literal_complete,
        "final_value_grounding_complete": final_value_complete,
        "final_value_grounding_incomplete_ids": incomplete_final_value_ids,
        "deterministic_grounding_complete": deterministic_complete,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", type=Path, required=True)
    parser.add_argument("--verified", type=Path, required=True)
    parser.add_argument("--baseline-all", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--protocol", default="current")
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--cohort", default="all supplied trajectories")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.all)
    verified = read_jsonl(args.verified)
    report = {
        "protocol": args.protocol,
        "model": args.model,
        "cohort": args.cohort,
        "current": summarize_attempts(records, current_protocol=True),
        "verified_reference_replay": reference_audit(verified),
    }
    if args.baseline_all and args.selection:
        selected_ids = {
            item["trajectory_id"]
            for item in json.loads(args.selection.read_text(encoding="utf-8"))["selected"]
        }
        baseline = [
            record for record in read_jsonl(args.baseline_all)
            if record.get("trajectory_id") in selected_ids
        ]
        report["descriptive_version4_baseline_same_ids"] = summarize_attempts(
            baseline, current_protocol=False,
        )
        report["baseline_caveat"] = (
            "Selection was conditioned on these prior trajectories, and baseline used bounded "
            "rolling context; any treatment context mismatch makes this descriptive rather than "
            "a causal A/B estimate."
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
