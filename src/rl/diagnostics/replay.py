"""Reusable replay preparation helpers for rollout diagnostics.

Scenario modules retain their experiment-specific credit rules, while this
module owns the mechanical boundary work shared by lineage and branch replays:
strict JSONL loading, deterministic K-grouping, and local Harness error
classification.  The functions intentionally return ordinary dictionaries so
historical report schemas can be preserved by callers.
"""

from __future__ import annotations

import collections
import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .io import read_jsonl


def stable_action_digest(value: Any) -> str:
    """Return a deterministic SHA-256 digest for a JSON-compatible action."""

    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _lineage_node(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value)
    kind = value.get("kind")
    if kind == "source":
        return str(value.get("table", "?"))
    if kind == "relation":
        inputs = [_lineage_node(item.get("ref")) for item in value.get("inputs", []) if isinstance(item, dict)]
        return f"{value.get('operator', '?')}({','.join(inputs)})"
    if kind == "column":
        return f"{_lineage_node(value.get('relation'))}.{value.get('column', '?')}"
    if kind in {"unresolved", "unresolved_handle", "unresolved_step"}:
        return "UNRES"
    return str(kind or "?")


def _lineage_condition(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value)
    if "and" in value:
        return "AND[" + ",".join(_lineage_condition(item) for item in value["and"]) + "]"
    if "or" in value:
        return "OR[" + ",".join(_lineage_condition(item) for item in value["or"]) + "]"
    if "not" in value:
        return "NOT[" + _lineage_condition(value["not"]) + "]"
    column = value.get("column", value.get("column_value", "?"))
    op = value.get("op", "?")
    raw = value.get("value", value.get("values", value.get("in_table", "")))
    return f"{column}{op}{'REF' if isinstance(raw, dict) else raw}"


def format_action(event: Mapping[str, Any]) -> str:
    """Format a lineage replay event for compact human-readable casebooks."""

    action = event.get("action") or {}
    tool = action.get("tool", "?")
    args = action.get("arguments", {})
    if not isinstance(args, dict):
        return str(tool)
    if tool == "describe_table":
        return "D(" + ",".join(map(str, args.get("tables", []))) + ")"
    if tool == "condition_filter":
        return f"F({_lineage_node(args.get('table'))};{_lineage_condition(args.get('conditions'))};ret={args.get('return_columns')})"
    if tool == "inspect_column":
        return f"I({_lineage_node(args.get('table'))}.{args.get('column')})"
    if tool == "read_subtable":
        return f"R({_lineage_node(args.get('table'))};cols={args.get('columns')};n={args.get('limit')})"
    if tool == "join_tables":
        joins = []
        for item in args.get("joins", []):
            edges = []
            for edge in item.get("on", []):
                left = edge.get("left")
                edges.append(f"{_lineage_node(left) if isinstance(left, dict) else left}={edge.get('right')}")
            joins.append(f"{_lineage_node(item.get('table'))}[{','.join(edges)}]")
        return f"J({_lineage_node(args.get('base'))};{'|'.join(joins)})"
    if tool == "project":
        return f"P({_lineage_node(args.get('table'))};expr={args.get('expressions')};d={args.get('distinct')})"
    if tool == "group_aggregate":
        aggs = ",".join(f"{item.get('op')}:{item.get('column')}" for item in args.get("aggregations", []) if isinstance(item, dict))
        return f"G({_lineage_node(args.get('table'))};gb={args.get('group_by')};{aggs})"
    if tool == "extreme_value_select":
        return f"E({_lineage_node(args.get('table'))};ord={args.get('order_by')};k={args.get('top_k')};ret={args.get('return_columns')})"
    if tool == "scalar_compute":
        return f"C({args.get('operation')};n={len(args.get('operands', []))})"
    if tool == "set_op":
        return f"S({_lineage_node(args.get('left'))},{_lineage_node(args.get('right'))};{args.get('op')})"
    if tool == "answer_from_context":
        return f"A({_lineage_node((args.get('evidence') or {}).get('table'))})"
    return str(tool)


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a rollout JSONL artifact and require object rows."""

    rows = read_jsonl(Path(path), require_object=True)
    return [dict(row) for row in rows]


def group_rollouts(
    rows: Iterable[Mapping[str, Any]],
    *,
    step_field: str = "policy_global_step",
    example_field: str = "example_index",
    trajectory_field: str = "trajectory_id",
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Group rollout rows by ``(policy step, example)`` in stable order."""

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        key = (int(row[step_field]), int(row[example_field]))
        grouped[key].append(dict(row))
    for values in grouped.values():
        values.sort(key=lambda row: str(row.get(trajectory_field, "")))
    return dict(grouped)


def local_error_statuses(
    row: Mapping[str, Any],
    *,
    error_signature: Callable[[Mapping[str, Any], Mapping[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    """Classify each turn's Harness event for deterministic replay credit.

    ``error_signature`` may be supplied by a lineage-aware caller.  The
    default canonicalizes the attempted action and is sufficient for detecting
    repeated no-progress errors in ordinary rollout records.
    """

    signature_fn = error_signature or (
        lambda turn, event: stable_action_digest(
            {
                "tool": (turn.get("parsed") or {}).get("tool")
                if isinstance(turn.get("parsed"), Mapping)
                else event.get("attempted_tool"),
                "arguments": (turn.get("parsed") or {}).get("arguments")
                if isinstance(turn.get("parsed"), Mapping)
                else event.get("attempted_arguments"),
            }
        )
    )
    statuses: list[dict[str, Any]] = []
    prior_error: dict[str, Any] | None = None
    turns_value = row.get("turns")
    turns = [] if turns_value is None else turns_value
    if not isinstance(turns, list):
        raise ValueError(f"trajectory {row.get('trajectory_id')!r} has non-list turns")
    for depth, turn_value in enumerate(turns):
        if not isinstance(turn_value, Mapping):
            raise ValueError(f"trajectory {row.get('trajectory_id')!r} has non-object turn")
        turn = turn_value
        event = turn.get("error_event")
        if not isinstance(event, Mapping):
            statuses.append({"kind": "success", "error_type": None, "local_error": False})
            prior_error = None
            continue
        error_type = str(event.get("error_type") or turn.get("execution_error_type") or "unknown_error")
        error_code = str(event.get("error_code") or "")
        message = str(event.get("message") or turn.get("execution_error") or "")
        signature = signature_fn(turn, event)
        before, after = event.get("state_before_hash"), event.get("state_after_hash")
        unchanged = bool(before and after and before == after)
        explicit_no_progress = (
            error_type == "no_progress_error"
            or error_code == "no_progress_error"
            or "no_progress" in message.lower()
        )
        repeated_no_progress = bool(
            prior_error
            and unchanged
            and prior_error.get("unchanged")
            and prior_error.get("signature") == signature
            and prior_error.get("after") == before
        )
        if error_type == "timeout_error" or error_code == "tool_execution_timeout":
            kind, local_error = "infrastructure_timeout", False
        elif explicit_no_progress or repeated_no_progress:
            kind, local_error = "no_progress_repeat", True
        elif error_type == "protocol_error":
            kind, local_error = "protocol_error", True
        elif error_type == "argument_validation_error":
            kind, local_error = "argument_validation_error", True
        else:
            kind, local_error = "execution_error", True
        statuses.append(
            {
                "kind": kind,
                "error_type": error_type,
                "error_code": error_code or None,
                "local_error": local_error,
                "state_unchanged": unchanged,
                "message": message,
            }
        )
        prior_error = {
            "signature": signature,
            "unchanged": unchanged,
            "before": before,
            "after": after,
            "depth": depth,
        }
    return statuses


__all__ = ["format_action", "group_rollouts", "local_error_statuses", "read_rows", "stable_action_digest"]
