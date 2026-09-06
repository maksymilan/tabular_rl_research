#!/usr/bin/env python3
"""Replay SAAM identities using Harness relation lineage instead of local handles.

This is an offline audit for raw online rollout records.  It never reads ``gold_sql``.  Raw
records contain allocation handles such as ``filter_003``; those handles are replaced by a
recursive, fact-only identity made from relation-derivation-v1.  The audit reports both the old
literal collisions and lineage-aware collisions, and emits one compact validation row per
trajectory so every episode is checked for unresolved or malformed lineage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from rl.frameworks.trl.transition_batch import standardized_group_advantages  # noqa: E402


SCHEMA_VERSION = "saam-lineage-replay-v1"
ANONYMOUS_HANDLE = re.compile(
    r"^(?:filter|join|project|group_aggregate|scalar_compute|extreme_value_select|set_op|pivot)_\d+$",
    re.IGNORECASE,
)
STEP_REF = re.compile(r"^step_(\d+)$", re.IGNORECASE)
PRODUCING_TOOLS = {
    "condition_filter": "condition_filter",
    "project": "project",
    "scalar_compute": "scalar_compute",
    "join_tables": "join_tables",
    "group_aggregate": "group_aggregate",
    "extreme_value_select": "extreme_value_select",
    "set_op": "set_op",
    "pivot": "pivot",
}
TABLE_FIELDS = {
    "condition_filter": {"table"},
    "project": {"table"},
    "scalar_compute": set(),
    "join_tables": {"base"},
    "group_aggregate": {"table"},
    "extreme_value_select": {"table"},
    "set_op": {"left", "right"},
    "read_subtable": {"table"},
    "inspect_rows": {"table"},
    "inspect_column": {"table"},
    "search_values": {"table"},
    "answer_from_context": set(),
}
COLUMN_KEYS = {
    "column",
    "column_value",
    "columns",
    "return_columns",
    "group_by",
    "order_by",
    "expressions",
    "passthrough",
}
LITERAL_KEYS = {"value", "values", "low", "high", "category_values", "reason", "think"}


def _stable(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_stable(value).encode("utf-8")).hexdigest()


def _is_anonymous(value: str) -> bool:
    return bool(ANONYMOUS_HANDLE.fullmatch(value))


def _norm_source(value: str) -> str:
    return value.strip().casefold()


def _step_number(value: str) -> int | None:
    match = STEP_REF.fullmatch(value.strip())
    return int(match.group(1)) if match else None


def _replace_lineage_prefix(value: str, handles: dict[str, Any]) -> tuple[Any, bool]:
    """Replace a handle or a dotted handle prefix while preserving ordinary literals."""
    if value in handles:
        return handles[value], True
    for handle in sorted(handles, key=len, reverse=True):
        prefix = handle + "."
        if value.startswith(prefix):
            return {"kind": "column", "relation": handles[handle], "column": value[len(prefix):]}, True
    return value, True


class LineageReplay:
    """Resolve one raw trajectory's local allocation graph."""

    def __init__(self, row: dict[str, Any]):
        self.row = row
        self.handles: dict[str, Any] = {}
        self.handle_resolved: dict[str, bool] = {}
        self.step_lineage: dict[int, Any] = {}
        self.step_resolved: dict[int, bool] = {}
        self.issues: list[dict[str, Any]] = []
        self.produced: list[dict[str, Any]] = []
        self.used_anonymous: Counter[str] = Counter()
        self._seen_output_handles: set[str] = set()

    def issue(self, kind: str, depth: int, **details: Any) -> None:
        self.issues.append({"kind": kind, "depth": depth, **details})

    def resolve_table(self, value: Any, depth: int, *, field: str) -> tuple[Any, bool]:
        if not isinstance(value, str):
            self.issue("non_string_table_ref", depth, field=field, value=value)
            return {"kind": "unresolved", "value": value}, False
        if value in self.handles:
            self.used_anonymous[value] += int(_is_anonymous(value))
            return self.handles[value], self.handle_resolved.get(value, False)
        if _is_anonymous(value):
            self.used_anonymous[value] += 1
            self.issue("unknown_anonymous_handle", depth, field=field, handle=value)
            return {"kind": "unresolved_handle", "handle": value}, False
        return {"kind": "source", "table": _norm_source(value)}, True

    def resolve_step(self, value: Any, depth: int, *, field: str) -> tuple[Any, bool]:
        if not isinstance(value, str):
            self.issue("non_string_step_ref", depth, field=field, value=value)
            return {"kind": "unresolved", "value": value}, False
        number = _step_number(value)
        if number is None or number not in self.step_lineage:
            self.issue("unknown_step_ref", depth, field=field, step=value)
            return {"kind": "unresolved_step", "step": value}, False
        return self.step_lineage[number], self.step_resolved.get(number, False)

    def _normalize_semantics(self, value: Any, depth: int, *, key: str | None = None) -> tuple[Any, bool]:
        if isinstance(value, list):
            items = []
            ok = True
            for item in value:
                normalized, item_ok = self._normalize_semantics(item, depth, key=key)
                items.append(normalized)
                ok = ok and item_ok
            return items, ok
        if not isinstance(value, dict):
            if isinstance(value, str) and key not in LITERAL_KEYS:
                replaced, ok = _replace_lineage_prefix(value, self.handles)
                if replaced is not value:
                    return replaced, self.handle_resolved.get(value, ok)
            return value, True
        result: dict[str, Any] = {}
        ok = True
        for child_key, child in value.items():
            normalized, child_ok = self._normalize_semantics(child, depth, key=child_key)
            result[child_key] = normalized
            ok = ok and child_ok
        return result, ok

    def derive_output(self, depth: int, tool: str, output: dict[str, Any], step_number: int) -> None:
        handle = output.get("table")
        derivation = output.get("derivation")
        if not isinstance(handle, str):
            self.issue("producer_missing_output_handle", depth, tool=tool)
            return
        if handle in self._seen_output_handles:
            self.issue("duplicate_output_handle", depth, handle=handle, tool=tool)
        self._seen_output_handles.add(handle)
        expected_operator = PRODUCING_TOOLS.get(tool)
        if expected_operator is not None and not isinstance(derivation, dict):
            self.issue("missing_derivation", depth, handle=handle, tool=tool)
            self.handles[handle] = {"kind": "unresolved_handle", "handle": handle}
            self.handle_resolved[handle] = False
            self.step_lineage[step_number] = self.handles[handle]
            self.step_resolved[step_number] = False
            return
        if not isinstance(derivation, dict):
            return
        operator = derivation.get("operator")
        if expected_operator is not None and operator != expected_operator:
            self.issue(
                "operator_mismatch",
                depth,
                tool=tool,
                operator=operator,
                expected=expected_operator,
            )
        inputs = derivation.get("inputs")
        input_nodes: list[Any] = []
        resolved = isinstance(inputs, list) and operator is not None
        if not isinstance(inputs, list):
            self.issue("invalid_derivation_inputs", depth, handle=handle)
            inputs = []
            resolved = False
        for index, item in enumerate(inputs):
            if not isinstance(item, dict):
                self.issue("invalid_derivation_input", depth, index=index, handle=handle)
                input_nodes.append({"kind": "unresolved_input", "index": index})
                resolved = False
                continue
            kind = item.get("kind")
            node: dict[str, Any] = {"kind": kind, "role": item.get("role")}
            if kind == "table":
                ref_node, ref_ok = self.resolve_table(item.get("ref"), depth, field=f"derivation.inputs[{index}].ref")
                node["ref"] = ref_node
                resolved = resolved and ref_ok
            elif kind == "value":
                ref_node, ref_ok = self.resolve_step(item.get("ref"), depth, field=f"derivation.inputs[{index}].ref")
                node["ref"] = ref_node
                if "column" in item:
                    node["column"] = item["column"]
                resolved = resolved and ref_ok
            elif kind == "constant":
                node["value"] = item.get("value")
            else:
                self.issue("unknown_derivation_input_kind", depth, index=index, kind=kind)
                resolved = False
            for extra in ("namespace", "operand_index"):
                if extra in item:
                    node[extra] = item[extra]
            input_nodes.append(node)
        semantics, semantic_ok = self._normalize_semantics(derivation.get("semantics"), depth)
        resolved = resolved and semantic_ok and isinstance(derivation.get("semantics"), dict)
        lineage = {"kind": "relation", "operator": operator, "inputs": input_nodes, "semantics": semantics}
        self.handles[handle] = lineage
        self.handle_resolved[handle] = resolved
        self.step_lineage[step_number] = lineage
        self.step_resolved[step_number] = resolved
        self.produced.append(
            {
                "depth": depth,
                "handle": handle,
                "operator": operator,
                "resolved": resolved,
                "lineage_digest": _digest(lineage),
            }
        )

    def _normalize_action_value(
        self,
        value: Any,
        depth: int,
        *,
        key: str | None = None,
        tool: str | None = None,
        normalize_describe_tables: bool = True,
    ) -> tuple[Any, bool]:
        if isinstance(value, list):
            result = []
            ok = True
            for item in value:
                normalized, item_ok = self._normalize_action_value(
                    item,
                    depth,
                    key=key,
                    tool=tool,
                    normalize_describe_tables=normalize_describe_tables,
                )
                result.append(normalized)
                ok = ok and item_ok
            if (
                normalize_describe_tables
                and key == "tables"
                and tool == "describe_table"
            ):
                result.sort(key=_stable)
            return result, ok
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            ok = True
            for child_key, child in value.items():
                normalized, child_ok = self._normalize_action_value(
                    child,
                    depth,
                    key=child_key,
                    tool=tool,
                    normalize_describe_tables=normalize_describe_tables,
                )
                result[child_key] = normalized
                ok = ok and child_ok
            return result, ok
        if not isinstance(value, str):
            return value, True
        if key in LITERAL_KEYS:
            return value, True
        if key == "value_ref":
            return self.resolve_step(value, depth, field="action.value_ref")
        if key in TABLE_FIELDS.get(tool or "", set()):
            return self.resolve_table(value, depth, field=f"action.{key}")
        if key == "table" and tool == "answer_from_context":
            return self.resolve_table(value, depth, field="action.evidence.table")
        if key in {"in_table", "input_table"}:
            return self.resolve_table(value, depth, field=f"action.{key}")
        if key in COLUMN_KEYS or key in {"left", "right", "on", "source", "expression", "output", "namespace"}:
            replaced, _ = _replace_lineage_prefix(value, self.handles)
            return replaced, True
        # Join items use ``table`` for their right input, including nested joins.
        if key == "table" and tool == "join_tables":
            return self.resolve_table(value, depth, field="action.joins.table")
        return value, True

    def action_identity(
        self,
        depth: int,
        parsed: Any,
        *,
        normalize_describe_tables: bool = True,
    ) -> tuple[dict[str, Any] | None, bool]:
        if not isinstance(parsed, dict):
            self.issue("unparsed_action", depth)
            return None, False
        tool = parsed.get("tool")
        arguments = parsed.get("arguments")
        if not isinstance(tool, str) or not isinstance(arguments, dict):
            self.issue("malformed_action", depth, tool=tool)
            return None, False
        normalized, resolved = self._normalize_action_value(
            arguments,
            depth,
            tool=tool,
            normalize_describe_tables=normalize_describe_tables,
        )
        return {"tool": tool, "arguments": normalized}, resolved

    def canonical_outcome(self, depth: int, turn: dict[str, Any]) -> Any:
        """Normalize only Harness-owned fields; rows and literal values remain untouched."""
        if "tool_output" in turn and isinstance(turn["tool_output"], dict):
            output = turn["tool_output"]
            result: dict[str, Any] = {}
            for key, value in output.items():
                if key == "table" and isinstance(value, str):
                    result[key], _ = self.resolve_table(value, depth, field="output.table")
                elif key == "derivation" and isinstance(value, dict):
                    # The derived relation has already been recursively compiled.  Avoid keeping
                    # allocation refs in the state identity.
                    handle = output.get("table")
                    if isinstance(handle, str) and handle in self.handles:
                        result[key] = self.handles[handle]
                    else:
                        result[key] = value
                elif key in {"columns", "observed_columns"} and isinstance(value, list):
                    result[key] = [
                        _replace_lineage_prefix(item, self.handles)[0] if isinstance(item, str) else item
                        for item in value
                    ]
                else:
                    result[key] = value
            return {"kind": "tool_output", "value": result}
        event = turn.get("error_event")
        if isinstance(event, dict) or "execution_error_type" in turn:
            event = event if isinstance(event, dict) else {}
            return {
                "kind": "tool_error",
                "error_type": turn.get("execution_error_type") or event.get("error_type"),
                "error_code": event.get("error_code"),
                "details": event.get("details"),
                "message": event.get("message") or turn.get("execution_error"),
            }
        return {"kind": "no_observation"}

    @staticmethod
    def literal_outcome(depth: int, turn: dict[str, Any]) -> Any:
        """The pre-fix state representation, retaining allocation handles verbatim."""
        if "tool_output" in turn and isinstance(turn["tool_output"], dict):
            return {"kind": "tool_output", "value": turn["tool_output"]}
        event = turn.get("error_event")
        if isinstance(event, dict) or "execution_error_type" in turn:
            event = event if isinstance(event, dict) else {}
            return {
                "kind": "tool_error",
                "error_type": turn.get("execution_error_type") or event.get("error_type"),
                "error_code": event.get("error_code"),
                "details": event.get("details"),
                "message": event.get("message") or turn.get("execution_error"),
            }
        return {"kind": "no_observation"}

    def replay(self) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        prior: list[Any] = []
        literal_prior: list[Any] = []
        prefix_resolved = True
        events: list[dict[str, Any]] = []
        turns = self.row.get("turns")
        if not isinstance(turns, list):
            self.issue("missing_turns", -1)
            turns = []
        for depth, turn in enumerate(turns):
            if not isinstance(turn, dict):
                self.issue("invalid_turn", depth)
                prefix_resolved = False
                continue
            parsed = turn.get("parsed")
            action, action_resolved = self.action_identity(depth, parsed)
            state_action, state_action_resolved = self.action_identity(
                depth,
                parsed,
                normalize_describe_tables=False,
            )
            state_digest = _digest(prior)
            literal_action = None
            if isinstance(parsed, dict) and isinstance(parsed.get("tool"), str) and isinstance(parsed.get("arguments"), dict):
                literal_action = {
                    "tool": parsed["tool"],
                    "arguments": json.loads(_stable(parsed["arguments"])),
                }
                if literal_action["tool"] == "describe_table" and isinstance(literal_action["arguments"].get("tables"), list):
                    literal_action["arguments"]["tables"] = sorted(literal_action["arguments"]["tables"])
            action_digest = _digest(action) if action is not None else None
            action_matchable = action_resolved and action is not None
            matched = prefix_resolved and action_matchable
            events.append(
                {
                    "trajectory_id": str(self.row.get("trajectory_id") or ""),
                    "example_index": int(self.row.get("example_index", -1)),
                    "policy_global_step": int(self.row.get("policy_global_step", -1)),
                    "depth": depth,
                    "correct": bool(self.row.get("correct")),
                    "matched": matched,
                    "action_matchable": action_matchable,
                    "action": action,
                    "action_digest": action_digest,
                    "state_digest": state_digest if matched else None,
                    "literal_state_digest": _digest(literal_prior) if matched else None,
                    "literal_action": parsed if isinstance(parsed, dict) else None,
                }
            )
            tool = parsed.get("tool") if isinstance(parsed, dict) else None
            output = turn.get("tool_output")
            if isinstance(tool, str) and isinstance(output, dict) and output.get("table") is not None:
                self.derive_output(depth, tool, output, depth + 1)
            outcome = self.canonical_outcome(depth, turn)
            literal_outcome = self.literal_outcome(depth, turn)
            prior.append(
                {
                    "action": (
                        state_action
                        if state_action is not None
                        else {"unmatchable_turn": depth}
                    ),
                    "outcome": outcome,
                }
            )
            literal_prior.append({"action": literal_action if literal_action is not None else {"unmatchable_turn": depth}, "outcome": literal_outcome})
            prefix_resolved = (
                prefix_resolved
                and action_resolved
                and state_action_resolved
                and action is not None
                and state_action is not None
            )
        summary = {
            "trajectory_id": str(self.row.get("trajectory_id") or ""),
            "example_index": int(self.row.get("example_index", -1)),
            "policy_global_step": int(self.row.get("policy_global_step", -1)),
            "correct": bool(self.row.get("correct")),
            "turns": len(turns),
            "produced_tables": len(self.produced),
            "resolved_tables": sum(int(item["resolved"]) for item in self.produced),
            "anonymous_handles_used": sum(self.used_anonymous.values()),
            "lineage_events": sum(int(event["matched"]) for event in events),
            "lineage_action_events": sum(int(event["action_matchable"]) for event in events),
            "issues": len(self.issues),
            "issue_kinds": dict(sorted(Counter(item["kind"] for item in self.issues).items())),
            "produced": self.produced,
            "issues_detail": self.issues[:20],
        }
        return events, summary


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield row


def _process_update(row: dict[str, Any]) -> bool:
    explicit = row.get("process_update")
    if isinstance(explicit, bool):
        return explicit
    if row.get("optimization_exclusion") is not None:
        return False
    reward = row.get("result_reward")
    return isinstance(reward, dict) and "value" in reward


def _literal_action_signature(event: dict[str, Any]) -> str | None:
    parsed = event.get("literal_action")
    if not isinstance(parsed, dict):
        return None
    tool = parsed.get("tool")
    arguments = parsed.get("arguments")
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        return None
    arguments = json.loads(_stable(arguments))
    if tool == "describe_table" and isinstance(arguments.get("tables"), list):
        arguments["tables"] = sorted(arguments["tables"])
    return _stable({"tool": tool, "arguments": arguments})


def _mixed_stats(events_by_key: dict[Any, list[dict[str, Any]]]) -> dict[str, Any]:
    mixed_groups = 0
    mixed_events = 0
    positive = 0.0
    negative = 0.0
    conflict = 0.0
    for values in events_by_key.values():
        if {bool(item["correct"]) for item in values} != {False, True}:
            continue
        mixed_groups += 1
        mixed_events += len(values)
        pos = sum(float(item["advantage"]) for item in values if item["advantage"] > 0)
        neg = sum(-float(item["advantage"]) for item in values if item["advantage"] < 0)
        positive += pos
        negative += neg
        conflict += 2.0 * min(pos, neg)
    return {
        "mixed_groups": mixed_groups,
        "mixed_events": mixed_events,
        "positive_mass": positive,
        "negative_mass": negative,
        "direct_conflict_mass": conflict,
    }


def _temporal_stats(
    events: list[dict[str, Any]],
    *,
    key_fields: tuple[str, ...],
    match_field: str = "matched",
) -> dict[str, Any]:
    """Replay batch-local SAAM and count sign changes of the surviving key sums."""
    by_update_key: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if not event.get(match_field):
            continue
        key = tuple(event[field] for field in key_fields)
        by_update_key[(event["policy_global_step"], *key)].append(event)
    applied: dict[tuple[Any, ...], float] = {}
    local_conflict = 0.0
    mixed_groups = 0
    mixed_events = 0
    for update_key, values in by_update_key.items():
        mixed = {bool(item["correct"]) for item in values} == {False, True}
        if mixed:
            mixed_groups += 1
            mixed_events += len(values)
        current = [0.0 if mixed else float(item["advantage"]) for item in values]
        positive = sum(value for value in current if value > 0.0)
        negative = sum(-value for value in current if value < 0.0)
        local_conflict += 2.0 * min(positive, negative)
        applied[update_key] = sum(current)
    appearances: dict[tuple[Any, ...], list[tuple[int, float]]] = defaultdict(list)
    for update_key, value in applied.items():
        policy_step = int(update_key[0])
        appearances[update_key[1:]].append((policy_step, value))
    recurrent = 0
    comparable = 0
    flips = 0
    touches_zero = 0
    for sequence in appearances.values():
        if len(sequence) < 2:
            continue
        recurrent += 1
        sequence.sort()
        for (_, previous), (_, current) in zip(sequence, sequence[1:]):
            if previous == 0.0 or current == 0.0:
                touches_zero += 1
                continue
            comparable += 1
            flips += int((previous > 0.0) != (current > 0.0))
    return {
        "key_fields": list(key_fields),
        "match_field": match_field,
        "mixed_groups": mixed_groups,
        "mixed_events": mixed_events,
        "batch_local_direct_conflict_mass": local_conflict,
        "recurrent_keys": recurrent,
        "comparable_nonzero_transitions": comparable,
        "sign_flips": flips,
        "sign_flip_rate": flips / comparable if comparable else 0.0,
        "transitions_touching_zero": touches_zero,
    }


def _pair_stats(events: list[dict[str, Any]], *, include_policy_step: bool) -> dict[str, Any]:
    literal_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    lineage_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if not event.get("action_matchable"):
            continue
        literal = _literal_action_signature(event)
        lineage = event.get("action_digest")
        prefix = (
            (event["policy_global_step"], event["example_index"])
            if include_policy_step
            else (event["example_index"],)
        )
        if literal is not None:
            literal_groups[(*prefix, literal)].append(event)
        if isinstance(lineage, str):
            lineage_groups[(*prefix, lineage)].append(event)
    cross_depth_pairs = 0
    false_literal_pairs = 0
    literal_groups_with_false = 0
    missed_lineage_pairs = 0
    lineage_groups_with_missed = 0
    false_examples: list[dict[str, Any]] = []
    missed_examples: list[dict[str, Any]] = []
    false_by_tool: Counter[str] = Counter()
    missed_by_tool: Counter[str] = Counter()
    for values in literal_groups.values():
        false_here = False
        for left_index, left in enumerate(values):
            for right in values[left_index + 1:]:
                if left["correct"] == right["correct"] or left["depth"] == right["depth"]:
                    continue
                cross_depth_pairs += 1
                if left.get("action_digest") != right.get("action_digest"):
                    false_literal_pairs += 1
                    false_here = True
                    false_by_tool[str(left.get("action", {}).get("tool", "__UNKNOWN__"))] += 1
                    if len(false_examples) < 12:
                        false_examples.append(
                            {
                                "policy_global_step": left["policy_global_step"],
                                "example_index": left["example_index"],
                                "literal_action": json.loads(_literal_action_signature(left) or "null"),
                                "left": {
                                    "trajectory_id": left["trajectory_id"],
                                    "depth": left["depth"],
                                    "correct": left["correct"],
                                    "lineage_action": left["action"],
                                    "lineage_digest": left["action_digest"],
                                },
                                "right": {
                                    "trajectory_id": right["trajectory_id"],
                                    "depth": right["depth"],
                                    "correct": right["correct"],
                                    "lineage_action": right["action"],
                                    "lineage_digest": right["action_digest"],
                                },
                            }
                        )
        literal_groups_with_false += int(false_here)
    for values in lineage_groups.values():
        missed_here = False
        for left_index, left in enumerate(values):
            for right in values[left_index + 1:]:
                if left["correct"] == right["correct"] or left["depth"] == right["depth"]:
                    continue
                if _literal_action_signature(left) != _literal_action_signature(right):
                    missed_lineage_pairs += 1
                    missed_here = True
                    missed_by_tool[str(left.get("action", {}).get("tool", "__UNKNOWN__"))] += 1
                    if len(missed_examples) < 12:
                        missed_examples.append(
                            {
                                "policy_global_step": left["policy_global_step"],
                                "example_index": left["example_index"],
                                "left": {
                                    "trajectory_id": left["trajectory_id"],
                                    "depth": left["depth"],
                                    "literal_action": json.loads(_literal_action_signature(left) or "null"),
                                },
                                "right": {
                                    "trajectory_id": right["trajectory_id"],
                                    "depth": right["depth"],
                                    "literal_action": json.loads(_literal_action_signature(right) or "null"),
                                },
                                "lineage_action": left["action"],
                                "lineage_digest": left["action_digest"],
                            }
                        )
        lineage_groups_with_missed += int(missed_here)
    return {
        "scope": "within_update" if include_policy_step else "across_updates_same_example",
        "literal_cross_depth_mixed_pairs": cross_depth_pairs,
        "literal_pairs_with_different_lineage": false_literal_pairs,
        "literal_false_pair_rate": false_literal_pairs / cross_depth_pairs if cross_depth_pairs else 0.0,
        "literal_groups_with_false_lineage": literal_groups_with_false,
        "lineage_pairs_missed_by_literal_name": missed_lineage_pairs,
        "lineage_groups_missed_by_literal_name": lineage_groups_with_missed,
        "false_pairs_by_tool": dict(sorted(false_by_tool.items())),
        "missed_pairs_by_tool": dict(sorted(missed_by_tool.items())),
        "false_literal_examples": false_examples,
        "missed_literal_examples": missed_examples,
    }


def audit(path: Path, *, expected_group_size: int) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    all_events: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    for row in _rows(path):
        key = (int(row.get("policy_global_step", -1)), int(row["example_index"]))
        grouped[key].append(row)
    if not grouped:
        raise ValueError("no eligible groups")
    for key, rows in sorted(grouped.items()):
        if len(rows) != expected_group_size:
            raise ValueError(f"group {key} has {len(rows)} records, expected {expected_group_size}")
        rewards = [1.0 if row.get("correct") is True else 0.0 for row in rows]
        eligible = [_process_update(row) for row in rows]
        advantages = standardized_group_advantages(rewards, eligible)
        group_events: list[dict[str, Any]] = []
        for row, advantage in zip(rows, advantages, strict=True):
            if not _process_update(row):
                continue
            replay = LineageReplay(row)
            events, summary = replay.replay()
            for event in events:
                event["advantage"] = float(advantage)
            group_events.extend(events)
            trajectory_rows.append(summary)
        all_events.extend(group_events)
    strict_literal: dict[tuple[int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    strict_lineage: dict[tuple[int, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in all_events:
        if not event.get("matched"):
            continue
        literal = _literal_action_signature(event)
        if literal is not None:
            strict_literal[(event["policy_global_step"], event["example_index"], event["state_digest"], literal)].append(event)
        strict_lineage[(event["policy_global_step"], event["example_index"], event["state_digest"], event["action_digest"])].append(event)
    literal_strict_events_for_stats = []
    lineage_strict_events_for_stats = []
    literal_action_events_for_stats = []
    lineage_action_events_for_stats = []
    for event in all_events:
        if event.get("matched"):
            literal = _literal_action_signature(event)
            if literal is not None:
                copy = dict(event)
                copy["key"] = literal
                copy["state_digest"] = event.get("literal_state_digest")
                literal_strict_events_for_stats.append(copy)
            copy = dict(event)
            copy["key"] = event["action_digest"]
            lineage_strict_events_for_stats.append(copy)
        if event.get("action_matchable"):
            literal = _literal_action_signature(event)
            if literal is not None:
                copy = dict(event)
                copy["key"] = literal
                literal_action_events_for_stats.append(copy)
            copy = dict(event)
            copy["key"] = event["action_digest"]
            lineage_action_events_for_stats.append(copy)
    literal_batch_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    lineage_batch_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    literal_action_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    lineage_action_groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for event in literal_strict_events_for_stats:
        literal_batch_groups[(event["policy_global_step"], event["example_index"], event["state_digest"], event["key"])].append(event)
    for event in lineage_strict_events_for_stats:
        lineage_batch_groups[(event["policy_global_step"], event["example_index"], event["state_digest"], event["key"])].append(event)
    for event in lineage_action_events_for_stats:
        lineage_action_groups[(event["policy_global_step"], event["example_index"], event["key"])].append(event)
    for event in literal_action_events_for_stats:
        literal_action_groups[(event["policy_global_step"], event["example_index"], event["key"])].append(event)
    issue_counts = Counter()
    for summary in trajectory_rows:
        issue_counts.update(summary["issue_kinds"])
    return {
        "schema_version": SCHEMA_VERSION,
        "input": str(path),
        "gold_sql_read": False,
        "groups": len(grouped),
        "episodes_in_groups": sum(len(rows) for rows in grouped.values()),
        "eligible_episodes": sum(
            int(_process_update(row)) for rows in grouped.values() for row in rows
        ),
        "eligible_trajectories_checked": len(trajectory_rows),
        "events": len(all_events),
        "lineage_matchable_events": sum(int(event.get("matched")) for event in all_events),
        "lineage_action_matchable_events": sum(int(event.get("action_matchable")) for event in all_events),
        "trajectory_issue_counts": dict(sorted(issue_counts.items())),
        "trajectories_with_any_issue": sum(int(summary["issues"] > 0) for summary in trajectory_rows),
        "trajectories_with_unresolved_lineage": sum(
            int(any(kind in summary["issue_kinds"] for kind in ("unknown_anonymous_handle", "unknown_step_ref", "missing_derivation", "invalid_derivation_inputs")))
            for summary in trajectory_rows
        ),
        "literal_strict_state_action": _mixed_stats(literal_batch_groups),
        "lineage_strict_state_action": _mixed_stats(lineage_batch_groups),
        "literal_action_only": _mixed_stats(literal_action_groups),
        "lineage_action_only": _mixed_stats(lineage_action_groups),
        "literal_temporal_strict": _temporal_stats(
            literal_strict_events_for_stats, key_fields=("example_index", "literal_state_digest", "key")
        ),
        "lineage_temporal_strict": _temporal_stats(
            all_events, key_fields=("example_index", "state_digest", "action_digest")
        ),
        "literal_temporal_action_only": _temporal_stats(
            literal_action_events_for_stats,
            key_fields=("example_index", "key"),
            match_field="action_matchable",
        ),
        "lineage_temporal_action_only": _temporal_stats(
            lineage_action_events_for_stats,
            key_fields=("example_index", "key"),
            match_field="action_matchable",
        ),
        "literal_vs_lineage_action_pairs_within_update": _pair_stats(all_events, include_policy_step=True),
        "literal_vs_lineage_action_pairs_across_updates": _pair_stats(all_events, include_policy_step=False),
        "trajectory_audits": trajectory_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rollouts", type=Path)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.rollouts, expected_group_size=args.group_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "trajectory_audits"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
