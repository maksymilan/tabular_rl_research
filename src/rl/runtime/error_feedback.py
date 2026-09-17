"""Bounded, factual rejected-action observations; never repair or execute an action.

Only caller-provided public state, successful-step metadata and the rejected action
are used. The module has no database/model access and does not classify rewards.
"""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


FEEDBACK_VERSION = "actionable-error-v1"
LEGACY_FEEDBACK_VERSION = "legacy"
MAX_ITEMS = 32
MAX_OPERANDS = 8
MAX_ACTION_BYTES = 8192
_PUBLIC_DETAIL_KEYS = {
    "argument_path", "received", "received_type", "requested_column",
    "available_columns", "valid_table_handles", "expected_arguments",
    "required_keys", "received_keys", "legal_tools", "execution_engine",
    "timeout_seconds", "state_preserved",
}


def rejected_action(exc: Exception, parsed: dict | None = None) -> tuple[str | None, dict | None]:
    """Use only a structurally parsed action; no permissive JSON fallback."""
    parsed = parsed or {}
    tool = parsed.get("tool") or getattr(exc, "attempted_tool", None)
    arguments = (parsed.get("arguments") if parsed.get("tool")
                 else getattr(exc, "attempted_arguments", None))
    return (tool if isinstance(tool, str) else None,
            deepcopy(arguments) if isinstance(arguments, dict) else None)


def attach_strict_rejected_action(exc: Exception, text: str, protocol) -> None:
    """Recover diagnostics with the SAME strict carrier, never a repair parser.

    Older pinned parsers do not attach the syntactically parsed action when its
    argument validation fails. This does not retry validation or execute anything.
    """
    if getattr(exc, "attempted_tool", None):
        return
    try:
        _, call = protocol.parse_action_carrier(text)
    except protocol.ActionCarrierError:
        return
    if not isinstance(call, dict) or set(call) != {"tool", "arguments"}:
        return
    if not isinstance(call["tool"], str) or not isinstance(call["arguments"], dict):
        return
    exc.attempted_tool = call["tool"]
    exc.attempted_arguments = deepcopy(call["arguments"])


def _columns(table: dict) -> list[str]:
    columns = table.get("columns")
    if isinstance(columns, list):
        return [item for item in columns if isinstance(item, str)]
    namespaces = table.get("column_namespaces") or {}
    if isinstance(namespaces, dict) and namespaces:
        return [f"{namespace}.{column}" for namespace, names in namespaces.items()
                if isinstance(names, list) for column in names if isinstance(column, str)]
    schema = table.get("schema") or {}
    return [item["name"] for item in schema.get("columns", [])
            if isinstance(item, dict) and isinstance(item.get("name"), str)]


def _table_metadata(state: dict, handle: Any) -> dict:
    if not isinstance(handle, str):
        return {}
    table = (state.get("tables") or {}).get(handle)
    if not isinstance(table, dict):
        return {"table": handle, "metadata_visible": False}
    result = {"table": handle, "metadata_visible": True}
    for key in ("kind", "created_by", "row_count"):
        if key in table:
            result[key] = deepcopy(table[key])
    columns = _columns(table)
    if columns:
        result["available_columns"] = columns[:MAX_ITEMS]
        result["column_count"] = len(columns)
        if len(columns) > MAX_ITEMS:
            result["columns_truncated"] = True
    return result


def _reference_metadata(state: dict, history: dict, ref: Any) -> dict:
    result = {"step_id": deepcopy(ref)}
    if not isinstance(ref, str):
        return result
    step = history.get(ref)
    if not isinstance(step, dict):
        result["successful_step_exists"] = False
        return result
    result["successful_step_exists"] = True
    result["tool"] = step.get("tool")
    output = step.get("output") or {}
    handle = output.get("table")
    if not handle and step.get("tool") == "read_subtable":
        handle = (step.get("arguments") or {}).get("table")
    if handle:
        result.update(_table_metadata(state, handle))
    # Never copy output rows, result_sample, gold, or model reasoning from history.
    return result


def feedback_details(exc: Exception, tool: str | None, arguments: dict | None,
                     *, state: dict, history: dict) -> dict:
    """Explain contracts using known metadata, not message regexes or SQL guesses."""
    original = getattr(exc, "details", None) or {}
    details = {key: deepcopy(value) for key, value in original.items()
               if key in _PUBLIC_DETAIL_KEYS}
    details["feedback_version"] = FEEDBACK_VERSION
    arguments = arguments or {}
    if tool == "scalar_compute":
        operation = arguments.get("operation")
        operands = arguments.get("operands")
        if isinstance(operands, list):
            facts = []
            for index, operand in enumerate(operands[:MAX_OPERANDS]):
                if not isinstance(operand, dict):
                    continue
                path = f"scalar_compute.operands[{index}]"
                if "value_ref" in operand:
                    fact = {
                        "argument_path": path + ".value_ref",
                        "received": deepcopy(operand["value_ref"]),
                        "expected": "a producing step resolving to one non-NULL scalar; "
                                    "a named column requires exactly one row",
                        "source": _reference_metadata(state, history, operand["value_ref"]),
                    }
                    if "column" in operand:
                        fact["requested_column"] = deepcopy(operand["column"])
                    facts.append(fact)
                elif "value" in operand:
                    value = operand["value"]
                    facts.append({
                        "argument_path": path + ".value",
                        "received_type": type(value).__name__,
                        "expected": ("an ISO date/time string" if operation == "date_diff_days"
                                     else "a finite numeric task constant, not a container or boolean"),
                    })
            details["operand_context"] = facts
            if len(operands) > MAX_OPERANDS:
                details["operands_truncated"] = True
        has_references = isinstance(operands, list) and any(
            isinstance(item, dict) and "value_ref" in item for item in operands
        )
        details["hint"] = (
            "read_subtable observes a table; it does not produce a scalar. "
            "Changing to created_by alone is insufficient if that table has multiple rows. "
            if has_references else ""
        ) + "No implicit row selection, aggregation, value unwrapping or NULL-to-zero conversion is performed."
        if operation in ("divide", "percent", "percent_change"):
            details["operation_constraint"] = {
                "argument_path": "scalar_compute.operands[1]",
                "expected": "a non-zero denominator",
            }
    elif tool == "join_tables":
        refs = [arguments.get("base")]
        edges = []
        joins = arguments.get("joins")
        if isinstance(joins, list):
            for index, item in enumerate(joins[:MAX_OPERANDS]):
                if not isinstance(item, dict):
                    continue
                refs.append(item.get("table"))
                on = item.get("on")
                if not isinstance(on, list):
                    continue
                for edge_index, edge in enumerate(on[:MAX_OPERANDS]):
                    if isinstance(edge, dict) and isinstance(edge.get("right"), str) and "." in edge["right"]:
                        edges.append({
                            "argument_path": f"join_tables.joins[{index}].on[{edge_index}].right",
                            "received": edge["right"],
                            "expected": "a bare column of this newly attached table, without a qualifier",
                        })
        if edges:
            details["argument_constraints"] = edges[:MAX_OPERANDS]
        details["referenced_tables"] = [_table_metadata(state, ref) for ref in refs
                                         if isinstance(ref, str)][:MAX_OPERANDS]
        details["hint"] = (
            "left uses an exact already-introduced logical relation.column; "
            "right uses a bare column of the newly attached table. "
            "Available columns describe names, not which join key answers the question."
        )
    # Bound public error metadata too; never duplicate full schemas in every error.
    for key in ("available_columns", "valid_table_handles", "legal_tools"):
        if isinstance(details.get(key), list) and len(details[key]) > MAX_ITEMS:
            details[key] = details[key][:MAX_ITEMS]
            details[key + "_truncated"] = True
    return details


def error_feedback_payload(exc: Exception, *, parsed: dict | None = None,
                           state: dict, history: dict) -> dict:
    """Payload merged into LAST TOOL ERROR without changing step/type/status."""
    tool, arguments = rejected_action(exc, parsed)
    payload = {"error": {
        "message": f"{type(exc).__name__}: {exc}",
        "code": getattr(exc, "code", type(exc).__name__),
        "details": feedback_details(exc, tool, arguments, state=state, history=history),
    }}
    if tool:
        action = {"tool": tool, "arguments": arguments or {}}
        if len(json.dumps(action, ensure_ascii=False).encode("utf-8")) > MAX_ACTION_BYTES:
            action = {"tool": tool, "arguments_omitted_due_to_size": True,
                      "argument_keys": sorted(arguments or {})[:MAX_ITEMS]}
        payload["attempted_action"] = action
    return payload


def merge_error_feedback(observation: dict, payload: dict) -> dict:
    """Do not overwrite the caller's error type or terminal/step bookkeeping."""
    result = deepcopy(observation)
    result["error"] = {**result["error"], **deepcopy(payload["error"])}
    if "attempted_action" in payload:
        result["attempted_action"] = deepcopy(payload["attempted_action"])
    return result
