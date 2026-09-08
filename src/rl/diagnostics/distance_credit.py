"""Gold-path-free distance and error-class diagnostics for paired rollouts.

This module is deliberately diagnostic-only.  It never reads ``gold_sql`` and does not
turn a terminal denotation comparison into an intermediate target path.  A pair consists
of one verified-correct and one incorrect rollout for the same question.  The correct
rollouts provide a set of observed successful frontiers; the implementation uses
Harness-owned derivation metadata, output schemas, row grain, and structured errors to
compare the paths.

All outputs are hypotheses, not semantic certificates.  In particular source overlap is
not understanding, a zero feature distance is not state equivalence, and a persistent
increase is not a causal proof.  Real-data counterexamples are retained in the report.
No category exposes a trainable reward or calibrated confidence.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from rl.diagnostics.io import canonical_json, read_jsonl


TERMINAL_TOOL = "answer_from_context"


def _stable(value: Any) -> str:
    return canonical_json(value)


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _scalar(value: Any) -> Any:
    """Apply only conservative terminal-display normalization.

    This is not an answer scorer.  It is used only to detect obvious Boolean display
    variants when the underlying Harness lineage is also the same.
    """

    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "yes"}:
            return True
        if normalized in {"false", "no"}:
            return False
        return value.strip()
    return value


def _row_cells(row: Any) -> list[Any] | None:
    if not isinstance(row, (list, tuple)):
        return None
    return [_scalar(value) for value in row]


def _row_subset(needle: list[Any], haystack: list[Any]) -> bool:
    """Return whether cells in one row occur in another row, ignoring column position."""

    remaining = list(haystack)
    for value in needle:
        for index, candidate in enumerate(remaining):
            if type(candidate) is type(value) and candidate == value:
                remaining.pop(index)
                break
        else:
            return False
    return True


def _sample_compatible(positive: Any, negative: Any, *, allow_concat: bool = False) -> bool:
    """Detect equal values, extra columns, or a conservative concatenation variant."""

    p_rows = [_row_cells(row) for row in (positive or [])]
    n_rows = [_row_cells(row) for row in (negative or [])]
    p_rows = [row for row in p_rows if row is not None]
    n_rows = [row for row in n_rows if row is not None]
    if not p_rows or not n_rows:
        return False
    if all(any(_row_subset(row, other) for other in n_rows) for row in p_rows):
        return True
    # Full-name expressions are not generally invertible.  Only accept this diagnostic
    # hint when one negative cell is exactly the whitespace-joined positive row.
    return allow_concat and len(p_rows) == len(n_rows) and all(
        len(n_row) == 1
        and all(isinstance(value, str) for value in p_row)
        and n_row[0] == " ".join(p_row)
        for p_row, n_row in zip(p_rows, n_rows)
    )


def _terminal_value_columns(record: dict[str, Any]) -> tuple[list[Any], list[str], int | None]:
    terminal = None
    for turn in reversed(record.get("turns") or []):
        parsed = turn.get("parsed") or {}
        if parsed.get("tool") == TERMINAL_TOOL:
            terminal = turn
            break
    if terminal is None:
        return [], [], None
    parsed = terminal.get("parsed") or {}
    evidence = (parsed.get("arguments") or {}).get("evidence") or {}
    handle = evidence.get("table")
    output_by_handle = {
        (turn.get("tool_output") or {}).get("table"): turn.get("tool_output") or {}
        for turn in record.get("turns") or []
        if (turn.get("tool_output") or {}).get("table")
    }
    output = output_by_handle.get(handle, {})
    return (
        list(terminal.get("pred_sample") or []),
        list(output.get("columns") or []),
        output.get("row_count"),
    )


def _derivation_index(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for turn in record.get("turns") or []:
        output = turn.get("tool_output") or {}
        handle = output.get("table")
        if handle:
            result[str(handle)] = output
    return result


def _semantic_node(
    ref: Any,
    outputs: dict[str, dict[str, Any]],
    stack: tuple[str, ...] = (),
) -> Any:
    """Canonicalize an anonymous derived handle into its Harness lineage facts."""

    if not isinstance(ref, str) or ref not in outputs:
        return {"source": str(ref)}
    if ref in stack:
        return {"cycle": ref}
    output = outputs[ref]
    derivation = output.get("derivation") or {}
    semantics = derivation.get("semantics") or {}
    inputs = []
    for item in derivation.get("inputs") or []:
        if item.get("kind") == "table":
            inputs.append({"role": item.get("role"), "value": _semantic_node(item.get("ref"), outputs, stack + (ref,))})
        elif item.get("kind") == "value":
            inputs.append({"role": item.get("role"), "value": {"value_ref": item.get("ref")}})
        else:
            inputs.append({"role": item.get("role"), "value": item.get("value")})
    return {
        "operator": derivation.get("operator"),
        "inputs": inputs,
        "semantics": semantics,
    }


def _leaves(node: Any) -> set[str]:
    if isinstance(node, dict):
        if set(node) == {"source"}:
            return {str(node["source"])}
        values: set[str] = set()
        # aggregation.source is a column/expression, not a base relation.
        # Traverse only the input DAG, never semantic metadata field names.
        for value in node.get("inputs", []):
            value = value.get("value") if isinstance(value, dict) else value
            values |= _leaves(value)
        return values
    if isinstance(node, list):
        values: set[str] = set()
        for value in node:
            values |= _leaves(value)
        return values
    return set()


def _grain(node: Any) -> str:
    if not isinstance(node, dict):
        return "unknown"
    semantics = node.get("semantics") or {}
    for key in ("row_grain", "row_operation", "layout"):
        if semantics.get(key) is not None:
            return str(semantics[key])
    return "unknown"


@dataclass(frozen=True)
class StateFingerprint:
    turn_index: int
    tool: str
    lineage: str
    leaves: frozenset[str]
    columns: frozenset[str]
    row_count: int | None
    grain: str
    has_error: bool


def state_sequence(record: dict[str, Any]) -> list[StateFingerprint]:
    outputs = _derivation_index(record)
    states: list[StateFingerprint] = []
    for index, turn in enumerate(record.get("turns") or []):
        parsed = turn.get("parsed") or {}
        tool = parsed.get("tool")
        output = turn.get("tool_output") or {}
        if not tool or not output.get("table"):
            continue
        node = _semantic_node(output["table"], outputs)
        error = bool(turn.get("error_event") or turn.get("error_events") or turn.get("execution_error_type"))
        states.append(
            StateFingerprint(
                turn_index=index,
                tool=str(tool),
                lineage=_stable(node),
                leaves=frozenset(_leaves(node)),
                columns=frozenset(str(column) for column in output.get("columns") or []),
                row_count=int(output["row_count"]) if isinstance(output.get("row_count"), int) else None,
                grain=_grain(node),
                has_error=error,
            )
        )
    return states


def _terminal_state(record: dict[str, Any], states: Sequence[StateFingerprint]) -> StateFingerprint | None:
    evidence = None
    for turn in record.get("turns") or []:
        parsed = turn.get("parsed") or {}
        if parsed.get("tool") == TERMINAL_TOOL:
            evidence = ((parsed.get("arguments") or {}).get("evidence") or {}).get("table")
    if evidence is None:
        return None
    for state in states:
        if (record["turns"][state.turn_index].get("tool_output") or {}).get("table") == evidence:
            return state
    return None


def state_distance(left: StateFingerprint, right: StateFingerprint) -> float:
    """Distance over Harness facts, not over intermediate result-table equality."""

    if left.lineage == right.lineage:
        return 0.0
    leaf = _jaccard(left.leaves, right.leaves)
    columns = _jaccard(left.columns, right.columns)
    operator = 1.0 if left.tool == right.tool else 0.0
    grain = 1.0 if left.grain == right.grain else 0.0
    return round(1.0 - (0.50 * leaf + 0.20 * columns + 0.20 * operator + 0.10 * grain), 6)


def _state_distance_profile(negative: Sequence[StateFingerprint], positive: Sequence[StateFingerprint]) -> list[float]:
    if not positive:
        return [1.0 for _ in negative]
    return [min(state_distance(state, candidate) for candidate in positive) for state in negative]


def _persistent_increases(distances: Sequence[float], threshold: float = 0.15) -> list[int]:
    result = []
    for index in range(1, len(distances)):
        previous = distances[index - 1]
        if distances[index] <= previous + threshold:
            continue
        if min(distances[index:]) > previous + 0.10:
            result.append(index)
    return result


def _hint_field_candidates(record: dict[str, Any]) -> set[str]:
    text = str(record.get("external_knowledge") or "")
    if not text:
        for message in record.get("initial_model_input") or []:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = str(message.get("content") or "")
            marker = "EXTERNAL KNOWLEDGE\n"
            if marker in content:
                text = content.split(marker, 1)[1].split("\n", 1)[0]
                break
    fields = set()
    for match in re.finditer(r"refer(?:s)?\s+to\s+([A-Za-z][A-Za-z0-9_.]*)", text, flags=re.I):
        fields.add(match.group(1).strip().casefold().replace(" ", ""))
    return fields


def _contract_conflict_candidate(positive: dict[str, Any], negative: dict[str, Any]) -> bool:
    hints = _hint_field_candidates(negative)
    if not hints:
        return False
    p_sample, p_columns, _ = _terminal_value_columns(positive)
    n_sample, n_columns, _ = _terminal_value_columns(negative)
    # Do not match arbitrary substrings or terminal cell text: a hint that names
    # birthCountry as a predicate is not evidence that it is an output field.
    p_fields = {str(value).split(".")[-1].casefold() for value in p_columns}
    n_fields = {str(value).split(".")[-1].casefold() for value in n_columns}
    return bool(hints & n_fields - p_fields)


def _tie_candidate(positive: dict[str, Any], negative: dict[str, Any]) -> bool:
    p_sample, _, p_count = _terminal_value_columns(positive)
    n_sample, _, n_count = _terminal_value_columns(negative)
    if not isinstance(p_count, int) or not isinstance(n_count, int) or n_count == p_count:
        return False
    if not p_sample or not n_sample:
        return False
    # A shared first row is not evidence of ties (missing predicates also do that).
    has_top_one = any(
        (t.get("parsed") or {}).get("tool") == "extreme_value_select"
        and ((t.get("parsed") or {}).get("arguments") or {}).get("top_k") == 1
        for r in (positive, negative) for t in r.get("turns") or []
    )
    has_extreme_aggregate = any(
        agg.get("op") in {"min", "max"}
        for r in (positive, negative) for t in r.get("turns") or []
        for agg in (((t.get("parsed") or {}).get("arguments") or {}).get("aggregations") or [])
    )
    return has_top_one and has_extreme_aggregate


@dataclass(frozen=True)
class PairClassification:
    label: str
    reason_codes: tuple[str, ...]
    negative_turns: tuple[int, ...]
    distances: tuple[float, ...]
    positive_terminal_columns: tuple[str, ...]
    negative_terminal_columns: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": None,
            "penalty_coefficient": None,
            "actor_update_allowed": False,
            "admission_status": "diagnostic_only",
            "reason_codes": list(self.reason_codes),
            "candidate_turn_indices_zero_based": list(self.negative_turns),
            "causal_attribution_certified": False,
            "negative_state_distances": list(self.distances),
            "distance_is_value_or_equivalence": False,
            "positive_terminal_columns": list(self.positive_terminal_columns),
            "negative_terminal_columns": list(self.negative_terminal_columns),
        }


def classify_pair(positive: dict[str, Any], negative: dict[str, Any]) -> PairClassification:
    """Classify one pair without topic/example-id rules.

    ``manual_review`` is intentional: uncertainty is safer than assigning a strong negative
    coefficient to a semantically valid alternative.
    """

    p_sample, p_columns, p_count = _terminal_value_columns(positive)
    n_sample, n_columns, n_count = _terminal_value_columns(negative)
    p_states, n_states = state_sequence(positive), state_sequence(negative)
    distances = _state_distance_profile(n_states, p_states)
    increases = [n_states[i].turn_index for i in _persistent_increases(distances)]
    reasons: list[str] = []

    if negative.get("failure_type") in {"generation_length", "generation_oom", "context_overflow", "nonrecoverable_execution_error"}:
        reasons.append("incomplete_or_runtime_failure")
        return PairClassification("manual_review", tuple(reasons), tuple(increases), tuple(distances), tuple(p_columns), tuple(n_columns))
    p_terminal, n_terminal = _terminal_state(positive, p_states), _terminal_state(negative, n_states)
    if p_terminal is None or n_terminal is None:
        return PairClassification("manual_review", ("missing_terminal_lineage",), (), tuple(distances), tuple(p_columns), tuple(n_columns))
    if _contract_conflict_candidate(positive, negative):
        reasons.append("prompt_output_contract_conflict_candidate")
    if _tie_candidate(positive, negative):
        reasons.append("multiple_maximum_or_cardinality_candidate")
    if reasons:
        return PairClassification("manual_review", tuple(reasons), tuple(increases), tuple(distances), tuple(p_columns), tuple(n_columns))

    leaves_same = _jaccard(p_terminal.leaves, n_terminal.leaves) >= 0.80
    terminal_action = (negative["turns"][n_terminal.turn_index].get("parsed") or {})
    concatenation_declared = terminal_action.get("tool") == "project" and any(
        "||" in str(expr) for expr in (terminal_action.get("arguments") or {}).get("expressions") or []
    )
    sample_compatible = _sample_compatible(p_sample, n_sample, allow_concat=concatenation_declared)
    shape_diff = p_columns != n_columns or p_count != n_count
    full_terminal_coverage = (
        p_count is not None and p_count == n_count and p_count > 0
        and p_count == len(p_sample) == len(n_sample)
    )
    if leaves_same and shape_diff and sample_compatible and full_terminal_coverage:
        reasons.append("same_terminal_values_or_conservative_display_variant")
        reasons.append("terminal_schema_or_column_shape_diff")
        return PairClassification("format_candidate", tuple(reasons), tuple(increases), tuple(distances), tuple(p_columns), tuple(n_columns))

    if shape_diff and sample_compatible and not full_terminal_coverage:
        reasons.append("partial_terminal_samples_do_not_certify_format")
        return PairClassification("manual_review", tuple(reasons), tuple(increases), tuple(distances), tuple(p_columns), tuple(n_columns))

    if leaves_same:
        reasons.append("shared_terminal_source_lineage")
        reasons.append("terminal_value_or_operator_mismatch")
        return PairClassification("local_structure_candidate", tuple(reasons), tuple(increases), tuple(distances), tuple(p_columns), tuple(n_columns))

    reasons.append("terminal_source_lineage_diverged")
    reasons.append("persistent_state_distance" if increases else "no_recoverable_shared_frontier")
    return PairClassification("source_divergence_candidate", tuple(reasons), tuple(increases), tuple(distances), tuple(p_columns), tuple(n_columns))


def load_smc_pairs(audit_path: str, rollouts_path: str) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]]:
    """Load selected SMC pairs; the audit file supplies only linkage and selection."""

    audits = read_jsonl(Path(audit_path))
    rollouts = read_jsonl(Path(rollouts_path))
    indexed = {(row["policy_global_step"], row["trajectory_id"], row["example_index"]): row for row in rollouts}
    if len(indexed) != len(rollouts):
        raise ValueError("duplicate rollout identity")
    pairs = []
    seen = set()
    for audit in audits:
        p, n = audit.get("selected_positive"), audit.get("selected_negative")
        if not p or not n:
            if bool(p) != bool(n):
                raise ValueError("incomplete selected positive/negative pair")
            continue
        pair_key = (audit["policy_global_step"], audit["example_index"])
        if pair_key in seen:
            raise ValueError("duplicate selected question group")
        seen.add(pair_key)
        positive = indexed[(audit["policy_global_step"], p["trajectory_id"], audit["example_index"])]
        negative = indexed[(audit["policy_global_step"], n["trajectory_id"], audit["example_index"])]
        if positive.get("correct") is not True or negative.get("correct") is not False:
            raise ValueError("selection and rollout correctness disagree")
        if (positive.get("db_id"), positive.get("question")) != (negative.get("db_id"), negative.get("question")):
            raise ValueError("selected pair is not the same task")
        pairs.append((audit, positive, negative))
    return pairs


def summarize_classifications(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["classification"]["label"] for row in rows)
    return {"pairs": len(rows), "labels": dict(sorted(counts.items()))}


def evaluate_reference_labels(
    predictions: Sequence[dict[str, Any]], reference: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate only after inference; reference labels cannot affect classification."""

    labels = {(r["example_index"], r["policy_global_step"]): r for r in reference["rows"]}
    if len(labels) != len(reference["rows"]):
        raise ValueError("duplicate manual reference identity")
    predicted_keys = {(r["example_index"], r["policy_global_step"]) for r in predictions}
    if len(predicted_keys) != len(predictions) or predicted_keys != set(labels):
        raise ValueError("manual reference and prediction coverage do not match")
    confusion: dict[str, Counter] = {}
    comparisons = []
    tp = fp = actual = predicted = site_cases = site_hits = 0
    for row in predictions:
        ref = labels[(row["example_index"], row["policy_global_step"])]
        result = row["classification"]
        truth, guess = ref["manual_category"], result["label"]
        confusion.setdefault(truth, Counter())[guess] += 1
        actual += truth == "format"
        predicted += guess == "format_candidate"
        tp += truth == "format" and guess == "format_candidate"
        fp += truth != "format" and guess == "format_candidate"
        expected_sites = ref.get("difference_turn_indices")
        hit = None
        if expected_sites is not None:
            site_cases += 1
            hit = bool(set(expected_sites) & set(result["candidate_turn_indices_zero_based"]))
            site_hits += hit
        comparisons.append({
            "example_index": row["example_index"], "policy_global_step": row["policy_global_step"],
            "manual_category": truth, "predicted_label": guess,
            "reference_difference_sites": expected_sites, "difference_site_hit": hit,
        })
    return {
        "annotation_provenance": reference["annotation_provenance"],
        "evaluation_scope": "development audit, not independent or held-out validation",
        "confusion": {k: dict(sorted(v.items())) for k, v in sorted(confusion.items())},
        "format": {"true_positive": tp, "false_positive": fp, "reference_count": actual,
                   "prediction_count": predicted, "precision": tp / predicted if predicted else None,
                   "recall": tp / actual if actual else None},
        "difference_site": {"cases": site_cases, "hits": site_hits,
                            "hit_rate": site_hits / site_cases if site_cases else None,
                            "is_causal_accuracy": False},
        "comparisons": comparisons,
        "reward_admission": "not_admitted",
    }


__all__ = [
    "PairClassification",
    "classify_pair",
    "evaluate_reference_labels",
    "load_smc_pairs",
    "state_distance",
    "state_sequence",
    "summarize_classifications",
]
