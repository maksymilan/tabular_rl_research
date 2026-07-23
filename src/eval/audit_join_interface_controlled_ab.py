#!/usr/bin/env python3
"""Paired audit for two closed-loop join-interface runs on the same frozen tasks."""
from __future__ import annotations

import argparse
import collections
import copy
import json
import math
import statistics
from pathlib import Path


CONTROL_FIELDS = (
    "model", "split", "examples_file", "source_count", "max_steps",
    "max_errors_per_type", "attempts_per_example", "max_tokens", "table_output_rows",
    "context_mode", "history_turns", "rolling_prompt_variant", "teacher_parser",
    "error_actions_are_sft_targets", "api_transport_retries_per_request",
    "provider_request_options",
)


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


def input_chars(turn: dict) -> int:
    return sum(
        len(str(message.get("content") or ""))
        for message in turn.get("model_input") or []
        if isinstance(message, dict)
    )


def state_json_from_turn(turn: dict) -> dict | None:
    marker = "CURRENT ENVIRONMENT STATE\n"
    for message in reversed(turn.get("model_input") or []):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or marker not in content:
            continue
        state_text = content.rsplit(marker, 1)[-1]
        state_text = state_text.split("\n\nLAST TOOL ERROR\n", 1)[0]
        try:
            state = json.loads(state_text)
        except json.JSONDecodeError:
            continue
        return state if isinstance(state, dict) else None
    return None


def expand_column_namespaces(value):
    if isinstance(value, list):
        return [expand_column_namespaces(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {key: expand_column_namespaces(item) for key, item in value.items()}
    namespaces = out.pop("column_namespaces", None)
    if isinstance(namespaces, dict):
        out["columns"] = [
            f"{namespace}.{column}"
            for namespace, columns in namespaces.items()
            for column in columns
        ]
    return out


def namespace_savings(records: list[dict]) -> dict:
    savings = []
    states = 0
    for record in records:
        for turn in record.get("turns") or []:
            state = state_json_from_turn(turn)
            if state is None:
                continue
            compact = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
            expanded_state = expand_column_namespaces(copy.deepcopy(state))
            expanded = json.dumps(expanded_state, ensure_ascii=False, separators=(",", ":"))
            delta = len(expanded) - len(compact)
            if delta > 0:
                states += 1
                savings.append(delta)
    return {
        "turns_with_namespaced_state": states,
        "saved_characters_total": sum(savings),
        "saved_characters_mean_when_used": round(statistics.mean(savings), 2) if savings else 0,
        "saved_characters_p90_when_used": percentile(savings, 0.9) or 0,
        "saved_characters_max": max(savings) if savings else 0,
    }


def classify_join_error(message: str) -> str:
    text = str(message).lower()
    if "return_columns" in text:
        return "join_local_projection_identifier"
    if "not an introduced logical column" in text:
        return "derived_handle_prefixed_logical_column"
    if "sql aliases" in text or "no such column" in text or "ambiguous column" in text:
        return "join_edge_identifier"
    return "other"


def classify_protocol_error(message: str) -> str:
    text = str(message).lower()
    if "native reasoning_content was empty" in text:
        return "deepseek_empty_native_reasoning"
    if "visible content was empty" in text:
        return "deepseek_empty_visible_content"
    if "visible content was not exactly one valid json action object" in text:
        return "deepseek_visible_content_not_json"
    if "visible json must contain exactly the top-level keys" in text:
        return "deepseek_visible_json_wrong_shape"
    if "visible content contained reasoning/prose" in text:
        return "deepseek_reasoning_in_visible_content"
    return "other"


def completion_order_buckets(records: list[dict], width: int = 20) -> list[dict]:
    buckets = []
    for start in range(0, len(records), width):
        group = records[start:start + width]
        protocol_events = [
            event
            for record in group
            for event in record.get("error_events") or []
            if event.get("error_type") == "protocol_error"
        ]
        buckets.append({
            "completion_positions": [start + 1, start + len(group)],
            "correct": sum(bool(record.get("correct")) for record in group),
            "final_protocol_error": sum(
                record.get("failure_type") == "protocol_error" for record in group
            ),
            "protocol_error_events": len(protocol_events),
        })
    return buckets


def summarize(records: list[dict], manifest: dict) -> dict:
    turns = [turn for record in records for turn in record.get("turns") or []]
    join_pairs = [
        (record, turn)
        for record in records
        for turn in record.get("turns") or []
        if (turn.get("parsed") or {}).get("tool") == "join_tables"
    ]
    executed = [
        turn for _, turn in join_pairs
        if isinstance(turn.get("tool_output"), dict)
        and turn["tool_output"].get("kind") == "join"
    ]
    join_errors = [
        event
        for record in records
        for event in record.get("error_events") or []
        if event.get("attempted_tool") == "join_tables"
    ]
    all_errors = [
        event for record in records for event in record.get("error_events") or []
    ]
    lengths = [int(record.get("steps") or 0) for record in records]
    chars = [input_chars(turn) for turn in turns if turn.get("model_input")]
    arity_hist = collections.Counter(
        join_arity((turn.get("parsed") or {}).get("arguments") or {})
        for _, turn in join_pairs
    )
    execution_tool_hist = collections.Counter(
        str(event.get("attempted_tool") or "unparsed")
        for event in all_errors
        if event.get("error_type") in {"execution_error", "nonrecoverable_execution_error"}
    )
    failure_hist = collections.Counter(
        str(record.get("failure_type") or "none") for record in records
    )
    error_hist = collections.Counter(
        str(event.get("error_type") or "unknown") for event in all_errors
    )
    parsed_tool_hist = collections.Counter(
        str((turn.get("parsed") or {}).get("tool") or "unparsed") for turn in turns
    )
    protocol_error_categories = collections.Counter(
        classify_protocol_error(event.get("message") or "")
        for event in all_errors
        if event.get("error_type") == "protocol_error"
    )
    join_error_categories = collections.Counter(
        classify_join_error(event.get("message") or "") for event in join_errors
    )
    join_ids = {record.get("trajectory_id") for record, _ in join_pairs}
    usage = manifest.get("usage_total") or {}
    api_attempts = int(usage.get("api_request_attempts") or 0)
    non_final_protocol = [
        record for record in records if record.get("failure_type") != "protocol_error"
    ]
    return {
        "episodes": len(records),
        "correct": sum(bool(record.get("correct")) for record in records),
        "legal_terminal": sum(bool(record.get("legal")) for record in records),
        "descriptive_excluding_final_protocol_error": {
            "episodes": len(non_final_protocol),
            "correct": sum(bool(record.get("correct")) for record in non_final_protocol),
            "accuracy": round(
                sum(bool(record.get("correct")) for record in non_final_protocol)
                / len(non_final_protocol),
                4,
            ) if non_final_protocol else None,
        },
        "failure_histogram": dict(sorted(failure_hist.items())),
        "actions": {
            "total": sum(lengths),
            "mean": round(statistics.mean(lengths), 3),
            "median": statistics.median(lengths),
            "p90": percentile(lengths, 0.9),
            "max": max(lengths),
        },
        "semantic_turns_recorded": len(turns),
        "input_characters": {
            "mean_per_turn": round(statistics.mean(chars), 2),
            "median_per_turn": statistics.median(chars),
            "p90_per_turn": percentile(chars, 0.9),
        },
        "usage": {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
            "api_request_attempts": api_attempts,
            "prompt_tokens_per_api_attempt": (
                round(usage.get("prompt_tokens", 0) / api_attempts, 2) if api_attempts else None
            ),
        },
        "error_events": len(all_errors),
        "error_event_histogram": dict(sorted(error_hist.items())),
        "protocol_error_categories": dict(sorted(protocol_error_categories.items())),
        "completion_order_buckets": completion_order_buckets(records),
        "parsed_tool_attempt_histogram": dict(sorted(parsed_tool_hist.items())),
        "execution_error_tool_histogram": dict(sorted(execution_tool_hist.items())),
        "execution_error_rate_by_parsed_tool": {
            tool: round(count / parsed_tool_hist[tool], 4)
            for tool, count in sorted(execution_tool_hist.items())
            if parsed_tool_hist[tool]
        },
        "episodes_with_join_attempt": len(join_ids),
        "join_attempts": len(join_pairs),
        "join_executed": len(executed),
        "join_local_success_rate": round(len(executed) / len(join_pairs), 4) if join_pairs else None,
        "join_local_errors": len(join_errors),
        "join_error_categories": dict(sorted(join_error_categories.items())),
        "join_arity_histogram": {str(key): count for key, count in sorted(arity_hist.items())},
        "multiway_join_attempts": sum(
            count for arity, count in arity_hist.items() if arity > 2
        ),
        "binary_equivalent_join_actions": sum(
            max(1, arity - 1) * count for arity, count in arity_hist.items()
        ),
        "namespace_rendering": namespace_savings(records),
    }


def exact_mcnemar_p(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(left_only, right_only) + 1)
    ) / (2 ** discordant)
    return min(1.0, 2 * tail)


def paired(base: dict[str, dict], treatment: dict[str, dict], ids: set[str]) -> dict:
    both = base_only = treatment_only = neither = 0
    for trajectory_id in ids:
        left = bool(base[trajectory_id].get("correct"))
        right = bool(treatment[trajectory_id].get("correct"))
        if left and right:
            both += 1
        elif left:
            base_only += 1
        elif right:
            treatment_only += 1
        else:
            neither += 1
    return {
        "n": len(ids),
        "both_correct": both,
        "baseline_only": base_only,
        "treatment_only": treatment_only,
        "neither": neither,
        "treatment_minus_baseline": treatment_only - base_only,
        "exact_mcnemar_p": round(exact_mcnemar_p(base_only, treatment_only), 6),
    }


def join_episode_ids(records: list[dict]) -> set[str]:
    return {
        str(record.get("trajectory_id"))
        for record in records
        if any(
            (turn.get("parsed") or {}).get("tool") == "join_tables"
            for turn in record.get("turns") or []
        )
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-all", type=Path, required=True)
    parser.add_argument("--baseline-manifest", type=Path, required=True)
    parser.add_argument("--treatment-all", type=Path, required=True)
    parser.add_argument("--treatment-manifest", type=Path, required=True)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--treatment-label", default="treatment")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline_records = read_jsonl(args.baseline_all)
    treatment_records = read_jsonl(args.treatment_all)
    baseline_manifest = json.loads(args.baseline_manifest.read_text(encoding="utf-8"))
    treatment_manifest = json.loads(args.treatment_manifest.read_text(encoding="utf-8"))
    baseline = {str(record["trajectory_id"]): record for record in baseline_records}
    treatment = {str(record["trajectory_id"]): record for record in treatment_records}
    if set(baseline) != set(treatment):
        raise ValueError("baseline and treatment trajectory ids differ")

    controls = {
        field: {
            "baseline": baseline_manifest.get(field),
            "treatment": treatment_manifest.get(field),
            "match": baseline_manifest.get(field) == treatment_manifest.get(field),
        }
        for field in CONTROL_FIELDS
    }
    mismatches = [field for field, values in controls.items() if not values["match"]]
    baseline_join_ids = join_episode_ids(baseline_records)
    treatment_join_ids = join_episode_ids(treatment_records)
    all_ids = set(baseline)
    both_non_final_protocol_ids = {
        trajectory_id
        for trajectory_id in all_ids
        if baseline[trajectory_id].get("failure_type") != "protocol_error"
        and treatment[trajectory_id].get("failure_type") != "protocol_error"
    }
    report = {
        "comparison": f"{args.baseline_label} vs {args.treatment_label}",
        "controlled_fields": controls,
        "control_mismatches": mismatches,
        "intended_differences": {
            "baseline_protocol_hash": baseline_manifest.get("protocol_hash"),
            "treatment_protocol_hash": treatment_manifest.get("protocol_hash"),
            "protocol_and_tool_semantics": True,
        },
        "baseline": summarize(baseline_records, baseline_manifest),
        "treatment": summarize(treatment_records, treatment_manifest),
        "paired_all_tasks": paired(baseline, treatment, all_ids),
        "paired_descriptive_both_non_final_protocol": paired(
            baseline, treatment, both_non_final_protocol_ids,
        ),
        "outcome_conditioning_caveat": (
            "Excluding final protocol failures is descriptive only: it conditions on a "
            "post-treatment outcome and must not replace the all-task causal estimate."
        ),
        "paired_union_join_attempt_tasks": paired(
            baseline, treatment, baseline_join_ids | treatment_join_ids,
        ),
        "join_attempt_id_overlap": {
            "baseline": len(baseline_join_ids),
            "treatment": len(treatment_join_ids),
            "intersection": len(baseline_join_ids & treatment_join_ids),
            "union": len(baseline_join_ids | treatment_join_ids),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
