#!/usr/bin/env python3
"""Unified analysis for deterministic single-sample evaluation JSONL files.

This is the only implementation of evaluation-arm metrics, paired outcome
statistics, legal-termination statistics, and closed-loop policy shift.  Older
experiment-specific summarizers are compatibility frontends over this module.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


PREFERRED_DIFFICULTIES = ("simple", "moderate", "challenging")
_MISSING = object()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def load_indices(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["indices"] if isinstance(payload, dict) else payload
    indices = [int(value) for value in values]
    if len(indices) != len(set(indices)):
        raise ValueError("indices contain duplicates")
    return indices


def only_sample(row: dict[str, Any]) -> dict[str, Any]:
    samples = row.get("samples") or []
    if len(samples) != 1:
        raise ValueError(
            f"expected exactly one sample for example {row.get('example_index')}"
        )
    return samples[0]


def actions(row: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    sequence = []
    for turn in only_sample(row).get("turns") or []:
        parsed = turn.get("parsed") or {}
        tool = parsed.get("tool")
        arguments = parsed.get("arguments")
        if tool is None or arguments is None:
            continue
        sequence.append(
            (
                str(tool),
                json.dumps(
                    arguments,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            )
        )
    return tuple(sequence)


def edit_distance(left: Sequence[Any], right: Sequence[Any]) -> int:
    if len(left) > len(right):
        left, right = right, left
    previous = list(range(len(left) + 1))
    for right_index, right_value in enumerate(right, start=1):
        current = [right_index]
        for left_index, left_value in enumerate(left, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[left_index] + 1,
                    previous[left_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def common_prefix_length(left: Sequence[Any], right: Sequence[Any]) -> int:
    count = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        count += 1
    return count


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "median": statistics.median(values) if values else 0.0,
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "max": max(values, default=0.0),
    }


def exact_mcnemar_p(gains: int, regressions: int) -> float:
    discordant = gains + regressions
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(gains, regressions) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def js_divergence_bits(left: Counter[str], right: Counter[str]) -> float:
    # Stable summation order makes the serialized float reproducible across
    # processes instead of depending on hash randomization of a set.
    keys = sorted(set(left) | set(right))
    left_total = sum(left.values())
    right_total = sum(right.values())
    result = 0.0
    for key in keys:
        p = left[key] / left_total if left_total else 0.0
        q = right[key] / right_total if right_total else 0.0
        midpoint = (p + q) / 2.0
        if p:
            result += 0.5 * p * math.log2(p / midpoint)
        if q:
            result += 0.5 * q * math.log2(q / midpoint)
    return result


def selected_rows(path: Path, indices: list[int]) -> dict[int, dict[str, Any]]:
    wanted = set(indices)
    selected: dict[int, dict[str, Any]] = {}
    duplicates = set()
    for row in load_jsonl(path):
        index = int(row["example_index"])
        if index not in wanted:
            continue
        if index in selected:
            duplicates.add(index)
        selected[index] = row
    if duplicates:
        raise ValueError(f"{path} has duplicate examples: {sorted(duplicates)}")
    if set(selected) != wanted:
        missing = sorted(wanted - set(selected))
        extra = sorted(set(selected) - wanted)
        raise ValueError(
            f"{path} does not contain exact selected cohort; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    for row in selected.values():
        only_sample(row)
    return selected


def _contract_values(rows: dict[int, dict[str, Any]]) -> dict[str, list[Any]]:
    return {
        field: sorted(
            {row.get(field) for row in rows.values()},
            key=lambda value: str(value),
        )
        for field in (
            "protocol_version",
            "protocol_hash",
            "temperature",
            "top_p",
            "denotation_comparison",
        )
    }


def evaluation_identity(
    result_path: Path,
    identity_path: Path | None = None,
) -> dict[str, Any] | None:
    """Load an arm identity without guessing across directory boundaries.

    Historical result directories put the sidecar beside ``all.jsonl``.  Some
    newer remote launchers put it one level above ``result/all.jsonl``.  The
    latter must be passed explicitly: walking ancestors can silently bind a
    result to an unrelated launcher directory.
    """

    path = identity_path or (result_path.parent / "evaluation_identity.json")
    if not path.exists():
        if identity_path is not None:
            raise ValueError(f"evaluation identity does not exist: {path}")
        return None
    with path.open(encoding="utf-8") as source:
        identity = json.load(source)
    if not isinstance(identity, dict):
        raise ValueError(f"evaluation identity must be a JSON object: {path}")
    return identity


def _identity_value(identity: dict[str, Any], field: str) -> Any:
    """Return one identity value, with one canonical base-model alias."""

    if field == "base_model":
        for name in ("base_model", "base_model_path"):
            if name in identity:
                return identity[name]
        return _MISSING
    value: Any = identity
    for component in field.split("."):
        if not isinstance(value, dict) or component not in value:
            return _MISSING
        value = value[component]
    return value


def validate_identity_contract(
    *,
    arm_paths: dict[str, Path],
    identity_paths: dict[str, Path],
    evaluation_contract: dict[str, list[Any]],
    require_identities: bool,
    matched_identity_fields: Sequence[str],
    expected_adapter_sha256: dict[str, str],
    require_distinct_adapters: bool,
) -> tuple[dict[str, dict[str, Any] | None], dict[str, Any]]:
    """Bind result arms to model identities and validate matched-run fields."""

    unknown_identity_labels = sorted(set(identity_paths) - set(arm_paths))
    if unknown_identity_labels:
        raise ValueError(
            f"identity paths reference unknown arms: {unknown_identity_labels}"
        )
    unknown_adapter_labels = sorted(set(expected_adapter_sha256) - set(arm_paths))
    if unknown_adapter_labels:
        raise ValueError(
            "expected adapter SHA values reference unknown arms: "
            f"{unknown_adapter_labels}"
        )

    identities = {
        name: evaluation_identity(arm_paths[name], identity_paths.get(name))
        for name in arm_paths
    }
    missing = sorted(name for name, value in identities.items() if value is None)
    if require_identities and missing:
        raise ValueError(
            f"evaluation identity is required for every arm; missing={missing}"
        )

    # An identity that is present may never contradict the row-level executable
    # contract, even for a historical non-strict analysis.
    row_identity_fields = {
        "protocol_version": "protocol_version",
        "protocol_hash": "protocol_hash",
        "temperature": "decode.temperature",
        "top_p": "decode.top_p",
        "denotation_comparison": "agent.denotation_comparison",
    }
    for name, identity in identities.items():
        if identity is None:
            continue
        for contract_field, identity_field in row_identity_fields.items():
            identity_value = _identity_value(identity, identity_field)
            if identity_value is _MISSING:
                continue
            observed = evaluation_contract[contract_field]
            if observed != [identity_value]:
                raise ValueError(
                    f"evaluation identity/row contract mismatch for {name}: "
                    f"identity {identity_field}={identity_value!r}, "
                    f"rows {contract_field}={observed!r}"
                )

    matched_values: dict[str, Any] = {}
    for field in matched_identity_fields:
        values: dict[str, Any] = {}
        for name, identity in identities.items():
            if identity is None:
                raise ValueError(
                    f"cannot match identity field {field!r}; {name} has no identity"
                )
            value = _identity_value(identity, field)
            if value is _MISSING:
                raise ValueError(
                    f"evaluation identity for {name} is missing matched field {field!r}"
                )
            values[name] = value
        reference_name = next(iter(values))
        reference = values[reference_name]
        mismatches = {
            name: value for name, value in values.items() if value != reference
        }
        if mismatches:
            raise ValueError(
                f"evaluation identity field mismatch for {field!r}: "
                f"{reference_name}={reference!r}, mismatches={mismatches!r}"
            )
        matched_values[field] = reference

    adapter_hashes: dict[str, str | None] = {}
    for name, identity in identities.items():
        value = None if identity is None else identity.get("adapter_sha256")
        adapter_hashes[name] = str(value) if value else None
        expected = expected_adapter_sha256.get(name)
        if expected is not None and value != expected:
            raise ValueError(
                f"adapter SHA mismatch for {name}: expected={expected!r}, "
                f"observed={value!r}"
            )
    if require_distinct_adapters:
        if any(value is None for value in adapter_hashes.values()):
            raise ValueError(
                "distinct-adapter guard requires adapter_sha256 for every arm"
            )
        values = list(adapter_hashes.values())
        if len(values) != len(set(values)):
            raise ValueError(
                "evaluation arms do not have distinct adapter SHA values: "
                f"{adapter_hashes}"
            )

    return identities, {
        "required": require_identities,
        "all_present": not missing,
        "missing_arms": missing,
        "identity_paths": {
            name: str(
                identity_paths.get(
                    name, arm_paths[name].parent / "evaluation_identity.json"
                )
            )
            for name in arm_paths
        },
        "matched_fields": matched_values,
        "adapter_sha256": adapter_hashes,
        "distinct_adapters_required": require_distinct_adapters,
    }


def validate_contract(
    arms: dict[str, dict[int, dict[str, Any]]],
    *,
    protocol_version: str | None = None,
    protocol_hash: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    denotation_comparison: str | None = None,
) -> dict[str, list[Any]]:
    contracts = {name: _contract_values(rows) for name, rows in arms.items()}
    reference_name = next(iter(contracts))
    reference = contracts[reference_name]
    for name, contract in contracts.items():
        if contract != reference:
            raise ValueError(
                f"evaluation contract mismatch: {reference_name}={reference}, "
                f"{name}={contract}"
            )
    expected = {
        "protocol_version": protocol_version,
        "protocol_hash": protocol_hash,
        "temperature": temperature,
        "top_p": top_p,
        "denotation_comparison": denotation_comparison,
    }
    for field, value in expected.items():
        if value is not None and reference[field] != [value]:
            raise ValueError(
                f"expected {field}={value!r}, observed {reference[field]!r}"
            )
    return reference


def arm_metrics(
    rows: dict[int, dict[str, Any]],
    indices: list[int],
    difficulties: dict[int, str],
) -> dict[str, Any]:
    samples = {index: only_sample(rows[index]) for index in indices}
    sequences = {index: actions(rows[index]) for index in indices}
    tool_counts: Counter[str] = Counter(
        tool for sequence in sequences.values() for tool, _ in sequence
    )
    difficulty_order = list(PREFERRED_DIFFICULTIES)
    difficulty_order.extend(
        sorted(set(difficulties.values()) - set(PREFERRED_DIFFICULTIES))
    )
    correct = sum(bool(value["correct"]) for value in samples.values())
    legal = sum(bool(value["legal"]) for value in samples.values())
    return {
        "correct": correct,
        "total": len(indices),
        "accuracy": correct / len(indices),
        "legal": legal,
        "legal_rate": legal / len(indices),
        "mean_steps": statistics.fmean(
            float(value["steps"]) for value in samples.values()
        ),
        "difficulty": {
            level: {
                "correct": sum(
                    bool(samples[index]["correct"])
                    for index in indices
                    if difficulties[index] == level
                ),
                "legal": sum(
                    bool(samples[index]["legal"])
                    for index in indices
                    if difficulties[index] == level
                ),
                "total": sum(difficulties[index] == level for index in indices),
            }
            for level in difficulty_order
            if level in set(difficulties.values())
        },
        "action_count": sum(len(sequence) for sequence in sequences.values()),
        "adjacent_exact_repeat_calls": sum(
            sum(left == right for left, right in zip(sequence, sequence[1:]))
            for sequence in sequences.values()
        ),
        "questions_with_adjacent_exact_repeat": sum(
            any(left == right for left, right in zip(sequence, sequence[1:]))
            for sequence in sequences.values()
        ),
        "tool_counts": dict(sorted(tool_counts.items())),
    }


def _group_shift(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "total": 0,
            "sequence_changed": 0,
            "first_action_changed": 0,
            "mean_exact_edit_distance": 0.0,
            "mean_normalized_exact_edit_distance": 0.0,
            "mean_step_delta": 0.0,
        }
    return {
        "total": len(rows),
        "sequence_changed": sum(row["exact_action_sequence_changed"] for row in rows),
        "first_action_changed": sum(row["first_exact_action_changed"] for row in rows),
        "mean_exact_edit_distance": statistics.fmean(
            float(row["exact_action_edit_distance"]) for row in rows
        ),
        "mean_normalized_exact_edit_distance": statistics.fmean(
            float(row["normalized_exact_action_edit_distance"]) for row in rows
        ),
        "mean_step_delta": statistics.fmean(int(row["step_delta"]) for row in rows),
    }


def compare_arms(
    baseline: dict[int, dict[str, Any]],
    candidate: dict[int, dict[str, Any]],
    indices: list[int],
    *,
    include_per_example: bool = False,
) -> dict[str, Any]:
    per_example = []
    baseline_tools: Counter[str] = Counter()
    candidate_tools: Counter[str] = Counter()
    for index in indices:
        baseline_sample = only_sample(baseline[index])
        candidate_sample = only_sample(candidate[index])
        baseline_actions = actions(baseline[index])
        candidate_actions = actions(candidate[index])
        baseline_tool_sequence = tuple(tool for tool, _ in baseline_actions)
        candidate_tool_sequence = tuple(tool for tool, _ in candidate_actions)
        baseline_tools.update(baseline_tool_sequence)
        candidate_tools.update(candidate_tool_sequence)
        exact_edit = edit_distance(baseline_actions, candidate_actions)
        tool_edit = edit_distance(baseline_tool_sequence, candidate_tool_sequence)
        denominator = max(len(baseline_actions), len(candidate_actions), 1)
        per_example.append(
            {
                "example_index": index,
                "baseline_correct": bool(baseline_sample["correct"]),
                "candidate_correct": bool(candidate_sample["correct"]),
                "baseline_legal": bool(baseline_sample["legal"]),
                "candidate_legal": bool(candidate_sample["legal"]),
                "baseline_steps": int(baseline_sample["steps"]),
                "candidate_steps": int(candidate_sample["steps"]),
                "step_delta": int(candidate_sample["steps"])
                - int(baseline_sample["steps"]),
                "baseline_action_count": len(baseline_actions),
                "candidate_action_count": len(candidate_actions),
                "action_count_delta": len(candidate_actions) - len(baseline_actions),
                "exact_action_edit_distance": exact_edit,
                "normalized_exact_action_edit_distance": exact_edit / denominator,
                "tool_edit_distance": tool_edit,
                "normalized_tool_edit_distance": tool_edit / denominator,
                "exact_common_prefix_length": common_prefix_length(
                    baseline_actions, candidate_actions
                ),
                "first_exact_action_changed": (
                    baseline_actions[:1] != candidate_actions[:1]
                ),
                "exact_action_sequence_changed": baseline_actions != candidate_actions,
                "tool_sequence_changed": (
                    baseline_tool_sequence != candidate_tool_sequence
                ),
            }
        )

    gains = [
        row["example_index"]
        for row in per_example
        if row["candidate_correct"] and not row["baseline_correct"]
    ]
    regressions = [
        row["example_index"]
        for row in per_example
        if row["baseline_correct"] and not row["candidate_correct"]
    ]
    legal_gains = [
        row["example_index"]
        for row in per_example
        if row["candidate_legal"] and not row["baseline_legal"]
    ]
    legal_regressions = [
        row["example_index"]
        for row in per_example
        if row["baseline_legal"] and not row["candidate_legal"]
    ]
    changed = [row for row in per_example if row["exact_action_sequence_changed"]]
    first_changed = [
        row["example_index"] for row in per_example if row["first_exact_action_changed"]
    ]
    exact_edits = [float(row["exact_action_edit_distance"]) for row in per_example]
    normalized_exact_edits = [
        float(row["normalized_exact_action_edit_distance"]) for row in per_example
    ]
    tool_edits = [float(row["tool_edit_distance"]) for row in per_example]
    normalized_tool_edits = [
        float(row["normalized_tool_edit_distance"]) for row in per_example
    ]
    step_deltas = [int(row["step_delta"]) for row in per_example]
    total = len(indices)
    baseline_correct = sum(row["baseline_correct"] for row in per_example)
    candidate_correct = sum(row["candidate_correct"] for row in per_example)
    baseline_legal = sum(row["baseline_legal"] for row in per_example)
    candidate_legal = sum(row["candidate_legal"] for row in per_example)
    conditions = {
        "gain": lambda row: row["candidate_correct"] and not row["baseline_correct"],
        "regression": lambda row: row["baseline_correct"] and not row["candidate_correct"],
        "both_correct": lambda row: row["baseline_correct"] and row["candidate_correct"],
        "both_wrong": lambda row: not row["baseline_correct"] and not row["candidate_correct"],
    }
    comparison = {
        "accuracy": {
            "baseline_rate": baseline_correct / total if total else 0.0,
            "candidate_rate": candidate_correct / total if total else 0.0,
            "gains": len(gains),
            "regressions": len(regressions),
            "net": len(gains) - len(regressions),
            "net_rate": (len(gains) - len(regressions)) / total if total else 0.0,
            "net_percentage_points": (
                100.0 * (len(gains) - len(regressions)) / total
                if total
                else 0.0
            ),
            "exact_mcnemar_p": exact_mcnemar_p(len(gains), len(regressions)),
            "gain_example_indices": gains,
            "regression_example_indices": regressions,
        },
        "legal": {
            "baseline_rate": baseline_legal / total if total else 0.0,
            "candidate_rate": candidate_legal / total if total else 0.0,
            "gains": len(legal_gains),
            "regressions": len(legal_regressions),
            "net": len(legal_gains) - len(legal_regressions),
            "net_rate": (
                (len(legal_gains) - len(legal_regressions)) / total
                if total
                else 0.0
            ),
            "net_percentage_points": (
                100.0 * (len(legal_gains) - len(legal_regressions)) / total
                if total
                else 0.0
            ),
            "exact_mcnemar_p": exact_mcnemar_p(
                len(legal_gains), len(legal_regressions)
            ),
            "gain_example_indices": legal_gains,
            "regression_example_indices": legal_regressions,
        },
        "sequence_change": {
            "exact_action_sequence_changed": len(changed),
            "exact_action_sequence_unchanged": len(indices) - len(changed),
            "exact_action_sequence_changed_indices": [
                row["example_index"] for row in changed
            ],
            "tool_sequence_changed": sum(
                row["tool_sequence_changed"] for row in per_example
            ),
            "arguments_only_changed": sum(
                row["exact_action_sequence_changed"]
                and not row["tool_sequence_changed"]
                for row in per_example
            ),
            "first_exact_action_changed": len(first_changed),
            "first_exact_action_changed_indices": first_changed,
            "exact_edit_distance": distribution(exact_edits),
            "exact_edit_distance_changed_only": distribution(
                [float(row["exact_action_edit_distance"]) for row in changed]
            ),
            "normalized_exact_edit_distance": distribution(normalized_exact_edits),
            "tool_edit_distance": distribution(tool_edits),
            "normalized_tool_edit_distance": distribution(normalized_tool_edits),
        },
        "trajectory_length_change": {
            "mean_step_delta": statistics.fmean(step_deltas),
            "median_step_delta": statistics.median(step_deltas),
            "mean_absolute_step_delta": statistics.fmean(
                abs(value) for value in step_deltas
            ),
            "shorter": sum(value < 0 for value in step_deltas),
            "same": sum(value == 0 for value in step_deltas),
            "longer": sum(value > 0 for value in step_deltas),
            "total_action_count_delta": sum(
                int(row["action_count_delta"]) for row in per_example
            ),
        },
        "marginal_tools": {
            "baseline": dict(sorted(baseline_tools.items())),
            "candidate": dict(sorted(candidate_tools.items())),
            "candidate_minus_baseline": {
                tool: candidate_tools[tool] - baseline_tools[tool]
                for tool in sorted(set(baseline_tools) | set(candidate_tools))
            },
            "jensen_shannon_divergence_bits": js_divergence_bits(
                baseline_tools, candidate_tools
            ),
        },
        "outcome_groups": {
            name: sum(condition(row) for row in per_example)
            for name, condition in conditions.items()
        },
        "outcome_group_shift": {
            name: _group_shift(
                [row for row in per_example if condition(row)]
            )
            for name, condition in conditions.items()
        },
    }
    if include_per_example:
        comparison["per_example"] = per_example
    return comparison


def analyze_results(
    *,
    examples_path: Path | None,
    arm_paths: dict[str, Path],
    comparisons: list[tuple[str, str]],
    indices_path: Path | None = None,
    expected_count: int | None = None,
    protocol_version: str | None = None,
    protocol_hash: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    denotation_comparison: str | None = None,
    identity_paths: dict[str, Path] | None = None,
    require_identities: bool = False,
    matched_identity_fields: Sequence[str] = (),
    expected_adapter_sha256: dict[str, str] | None = None,
    require_distinct_adapters: bool = False,
    include_per_example: bool = False,
) -> dict[str, Any]:
    if examples_path is None and indices_path is None:
        raise ValueError("examples_path or indices_path is required")
    example_rows = load_jsonl(examples_path) if examples_path else []
    example_indices = [int(row["example_index"]) for row in example_rows]
    if len(example_indices) != len(set(example_indices)):
        raise ValueError("examples contain duplicate example_index values")
    indices = load_indices(indices_path) if indices_path else example_indices
    if expected_count is not None and len(indices) != expected_count:
        raise ValueError(f"expected {expected_count} examples, observed {len(indices)}")
    example_by_index = {int(row["example_index"]): row for row in example_rows}
    missing_examples = sorted(set(indices) - set(example_by_index))
    if examples_path is not None and missing_examples:
        raise ValueError(f"indices missing from examples: {missing_examples[:10]}")
    difficulties = {
        index: str(
            (example_by_index.get(index, {}).get("metadata") or {}).get(
                "difficulty", "unknown"
            )
        )
        for index in indices
    }
    arms = {name: selected_rows(path, indices) for name, path in arm_paths.items()}
    contract = validate_contract(
        arms,
        protocol_version=protocol_version,
        protocol_hash=protocol_hash,
        temperature=temperature,
        top_p=top_p,
        denotation_comparison=denotation_comparison,
    )
    identities, identity_contract = validate_identity_contract(
        arm_paths=arm_paths,
        identity_paths=identity_paths or {},
        evaluation_contract=contract,
        require_identities=require_identities,
        matched_identity_fields=matched_identity_fields,
        expected_adapter_sha256=expected_adapter_sha256 or {},
        require_distinct_adapters=require_distinct_adapters,
    )
    for candidate, baseline in comparisons:
        if candidate not in arms or baseline not in arms:
            raise ValueError(
                f"comparison {candidate}:{baseline} references unknown arm"
            )
        if candidate == baseline:
            raise ValueError("candidate and baseline labels must differ")
    return {
        "schema_version": "unified-evaluation-analysis-v1",
        "cohort": {
            "total": len(indices),
            "indices": indices,
            "difficulty_counts": dict(sorted(Counter(difficulties.values()).items())),
            "examples": str(examples_path) if examples_path else None,
            "indices_file": str(indices_path) if indices_path else None,
        },
        "evaluation_contract": contract,
        "identity_contract": identity_contract,
        "arms": {
            name: {
                "result": str(arm_paths[name]),
                "evaluation_identity": identities[name],
                **arm_metrics(rows, indices, difficulties),
            }
            for name, rows in arms.items()
        },
        "comparisons": {
            f"{candidate}_vs_{baseline}": {
                "candidate": candidate,
                "baseline": baseline,
                **compare_arms(
                    arms[baseline],
                    arms[candidate],
                    indices,
                    include_per_example=include_per_example,
                ),
            }
            for candidate, baseline in comparisons
        },
    }


def _named_path(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("arm must use LABEL=PATH")
    return label, Path(path)


def _comparison(value: str) -> tuple[str, str]:
    candidate, separator, baseline = value.partition(":")
    if not separator or not candidate or not baseline:
        raise argparse.ArgumentTypeError("comparison must use CANDIDATE:BASELINE")
    return candidate, baseline


def _named_value(value: str) -> tuple[str, str]:
    label, separator, item = value.partition("=")
    if not separator or not label or not item:
        raise argparse.ArgumentTypeError("value must use LABEL=VALUE")
    return label, item


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze one or more paired single-sample evaluation arms."
    )
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--indices", type=Path)
    parser.add_argument("--arm", action="append", type=_named_path, required=True)
    parser.add_argument("--compare", action="append", type=_comparison, default=[])
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--protocol-version")
    parser.add_argument("--protocol-hash")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--denotation-comparison")
    parser.add_argument(
        "--identity",
        action="append",
        type=_named_path,
        default=[],
        help="explicit LABEL=PATH identity sidecar; never inferred from ancestors",
    )
    parser.add_argument(
        "--require-identities",
        action="store_true",
        help="fail unless every arm has an evaluation identity",
    )
    parser.add_argument(
        "--match-identity-field",
        action="append",
        default=[],
        help="require an exact cross-arm match for a dotted identity field",
    )
    parser.add_argument(
        "--adapter-sha",
        action="append",
        type=_named_value,
        default=[],
        help="require LABEL=SHA256 for an evaluated adapter",
    )
    parser.add_argument(
        "--require-distinct-adapters",
        action="store_true",
        help="fail if any two arms bind the same adapter SHA256",
    )
    parser.add_argument("--include-per-example", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite analysis: {args.output}")
    arm_paths = dict(args.arm)
    if len(arm_paths) != len(args.arm):
        raise SystemExit("arm labels must be unique")
    identity_paths = dict(args.identity)
    if len(identity_paths) != len(args.identity):
        raise SystemExit("identity labels must be unique")
    expected_adapter_sha256 = dict(args.adapter_sha)
    if len(expected_adapter_sha256) != len(args.adapter_sha):
        raise SystemExit("adapter SHA labels must be unique")
    labels = list(arm_paths)
    comparisons = args.compare or [
        (candidate, labels[0]) for candidate in labels[1:]
    ]
    result = analyze_results(
        examples_path=args.examples,
        arm_paths=arm_paths,
        comparisons=comparisons,
        indices_path=args.indices,
        expected_count=args.expected_count,
        protocol_version=args.protocol_version,
        protocol_hash=args.protocol_hash,
        temperature=args.temperature,
        top_p=args.top_p,
        denotation_comparison=args.denotation_comparison,
        identity_paths=identity_paths,
        require_identities=args.require_identities,
        matched_identity_fields=args.match_identity_field,
        expected_adapter_sha256=expected_adapter_sha256,
        require_distinct_adapters=args.require_distinct_adapters,
        include_per_example=args.include_per_example,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "total": result["cohort"]["total"],
                "arms": {
                    name: {
                        "correct": values["correct"],
                        "legal": values["legal"],
                    }
                    for name, values in result["arms"].items()
                },
                "comparisons": {
                    name: {
                        "accuracy_net": values["accuracy"]["net"],
                        "legal_net": values["legal"]["net"],
                        "action_sequence_changed": values["sequence_change"][
                            "exact_action_sequence_changed"
                        ],
                    }
                    for name, values in result["comparisons"].items()
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
