#!/usr/bin/env python3
"""Evaluate the sequential action-block interface.

Each block executes a few consecutive atomic operations. Local references only name earlier
results in the submitted list; the harness discovers dependencies and returns one complete result
per submitted operation. The active public ``join`` is one edge and is deterministically lowered
to the frozen executor. ``answer_from_context`` is a separate top-level terminal action citing one
grounded resident table. Gold SQL is hidden from the model and used only for terminal scoring.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "sft"))

from tool_modules.action_block.protocol import (  # noqa: E402
    BATCH_CARRIERS,
    BATCH_CARRIER_INLINE_THINK,
    BATCH_CARRIER_PROVIDER_NATIVE,
    BATCH_PLAN_TOOL,
    BATCH_PLAN_PROTOCOL_VERSION,
    LOCAL_COLUMN_REF_RE,
    LOCAL_REF_RE,
    LOW_FRICTION_INTERFACE_PROTOCOL_VERSION,
    MAX_ACTION_BLOCK_CALLS,
    SAFE_LOW_FRICTION_INTERFACE_PROTOCOL_VERSION,
    SEQUENTIAL_ACTION_BLOCK_PROTOCOL_VERSION,
    SIMPLE_SCALAR_CELL_PROTOCOL_VERSION,
    SIMPLE_SEQUENTIAL_ACTION_BLOCK_PROTOCOL_VERSION,
    STRUCTURED_ERROR_FEEDBACK_PROTOCOL_VERSION,
    TERMINAL_TOOL,
    UNIFIED_ACTION_BLOCK_PROTOCOL_VERSION,
    BatchPlanProtocolError,
    LocalReferenceError,
    batch_plan_protocol_hash,
    build_batch_plan_messages,
    build_batch_plan_system_prompt,
    build_sequential_messages,
    local_reference_ids,
    lower_sequential_atomic_call,
    parse_legacy_batch_plan_action,
    parse_legacy_batch_plan_assistant,
    parse_batch_plan_assistant,
    parse_batch_plan_action,
    prepare_simple_scalar_cell_arguments,
    publicize_simple_scalar_error,
    publicize_sequential_error,
    render_batch_observation,
    render_sequential_observation,
    resolve_local_references,
    validate_atomic_call,
    validate_sequential_atomic_call,
)
from denotation import add_denotation_comparison_argument  # noqa: E402
from executor import Harness  # noqa: E402
from generate_teacher_rollouts import (  # noqa: E402
    ProviderCarrierError,
    add_usage,
    append_jsonl,
    chat_with_retries,
    compact_json,
    load_examples,
    trajectory_id,
    write_manifest,
)
from provider_adapter import (  # noqa: E402
    DEEPSEEK_CARRIER_JSON_OUTPUT,
    is_deepseek_split_model,
    provider_request_options,
)
from provider_client import load_api_config  # noqa: E402
from protocol import ProtocolError, validate_model_arguments  # noqa: E402
from tool_modules.registry import (  # noqa: E402
    ACTION_BLOCK_TOOL_SCHEME,
    TOOL_SCHEME_REGISTRY_VERSION,
)
from rollout import (  # noqa: E402
    ContextOverflowError,
    TERMINAL_ANSWER_CONTRACT,
    execute_tool,
    format_tool_error,
    new_ctx,
    overview,
    score,
    task_db_path,
    task_gold_sql,
)


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_ACTION_BLOCKS = 40
DEFAULT_MAX_BATCH_CALLS = MAX_ACTION_BLOCK_CALLS
DEFAULT_HISTORY_TURNS = 4
DEFAULT_MAX_TOKENS = 2048
DEFAULT_MAX_ERRORS_PER_TYPE = 3


def _error_type(exc: Exception) -> str:
    if isinstance(exc, LocalReferenceError):
        return "argument_validation_error"
    if isinstance(exc, BatchPlanProtocolError):
        return "protocol_error"
    if isinstance(exc, ProtocolError):
        text = str(exc).lower()
        if any(marker in text for marker in (
            "arguments", "unexpected", "requires", "missing", "must be",
            "must contain", "must resolve", "unknown or non-nestable",
        )):
            return "argument_validation_error"
        return "protocol_error"
    if exc.__class__.__name__ == "EnvironmentStateError":
        return "argument_validation_error"
    return "execution_error"


def _state_hash(state: dict) -> str:
    return hashlib.sha256(compact_json(state).encode("utf-8")).hexdigest()


def _error_event(
    *,
    atomic_index: int | None,
    model_turn: int,
    batch_index: int | None,
    call_id: str | None,
    tool: str | None,
    arguments: dict | None,
    error_type: str,
    message: str,
    state_before: dict,
    state_after: dict,
    facts: dict | None = None,
) -> dict:
    event = {
        "atomic_index": atomic_index,
        "step_id": (
            f"step_{atomic_index}" if atomic_index is not None else None
        ),
        "model_turn": model_turn,
        "batch_index": batch_index,
        "call_id": call_id,
        "attempted_tool": tool,
        "attempted_arguments": deepcopy(arguments or {}),
        "error_type": error_type,
        "message": message,
        "state_before_hash": _state_hash(state_before),
        "state_after_hash": _state_hash(state_after),
    }
    if facts:
        event["facts"] = deepcopy(facts)
    return event


def _top_level_error_message(
    *,
    block_action_index: int,
    error_type: str,
    message: str,
) -> dict:
    return {
        "action_block_index": block_action_index,
        "status": "error",
        "error": {"type": error_type, "message": message},
    }


def _canonical_output(reasoning: str, raw_content: str) -> str:
    return f"<think>{reasoning.strip()}</think>\n{raw_content.strip()}"


def _resolve_terminal_columns(h: Harness, table: str, requested: list[str]) -> list[str]:
    """Resolve declared answer columns without allowing expressions or silent ambiguity."""
    try:
        available = h.table_columns(table)
    except Exception as exc:  # noqa: BLE001
        raise ProtocolError(
            f"answer_from_context: evidence table {table!r} is not available"
        ) from exc
    resolved: list[str] = []
    for column in requested:
        exact = [
            candidate
            for candidate in available
            if candidate.casefold() == column.casefold()
        ]
        suffix = [
            candidate
            for candidate in available
            if candidate.rsplit(".", 1)[-1].casefold() == column.casefold()
        ]
        matches = exact or suffix
        if len(matches) != 1:
            raise ProtocolError(
                f"answer_from_context: evidence column {column!r} must resolve to exactly one "
                f"existing column; available columns: {available}"
            )
        resolved.append(matches[0])
    return resolved


def _materialize_terminal_evidence(
    *,
    h: Harness,
    ctx: dict,
    arguments: dict,
    created: set[str],
    step_id: str,
) -> tuple[dict, dict]:
    """Ground the model-declared terminal column selection as one harness projection."""
    evidence = arguments["evidence"]
    source_table = evidence["table"]
    if source_table not in created and source_table not in getattr(h, "views", {}):
        raise ProtocolError(
            f"answer_from_context: evidence table {source_table!r} is not a resident derived "
            "result handle"
        )
    selected_columns = _resolve_terminal_columns(
        h, source_table, evidence["columns"]
    )
    output, projected_table = execute_tool(
        h,
        "project",
        {
            "table": source_table,
            "expressions": selected_columns,
        },
        ctx,
        step_id,
        table_output_rows=0,
    )
    if not projected_table:
        raise ProtocolError(
            "answer_from_context: terminal column selection did not produce a grounded table"
        )
    created.add(projected_table)
    score_arguments = {
        "evidence": {"table": projected_table},
    }
    if "reason" in arguments:
        score_arguments["reason"] = arguments["reason"]
    return score_arguments, {
        "source_table": source_table,
        "source_columns": h.table_columns(source_table),
        "selected_columns": selected_columns,
        "projected_table": projected_table,
        "projection_output": deepcopy(output),
    }


def _walk_argument_strings(value, path=()):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_argument_strings(child, (*path, str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_argument_strings(child, (*path, str(index)))
    elif isinstance(value, str):
        yield ".".join(path), value


def _binding_fact(binding: dict | None) -> dict | None:
    if not isinstance(binding, dict):
        return None
    return {
        key: deepcopy(binding.get(key))
        for key in ("block_index", "step_id", "table", "columns")
        if binding.get(key) is not None
    }


def _reference_error_facts(
    arguments: dict,
    *,
    declared_ids: set[str],
    bindings: dict[str, dict],
    prior_bindings: dict[str, dict],
) -> list[dict]:
    facts = []
    table_reference_keys = {"table", "base", "in_table", "left", "right"}
    for path, value in _walk_argument_strings(arguments):
        parent_key = path.rsplit(".", 1)[-1]
        if value.startswith("$"):
            local_id = value[1:].split(".", 1)[0]
            if local_id not in declared_ids:
                fact = {
                    "path": path,
                    "provided": value,
                    "status": "not_declared_in_current_block",
                    "current_block_ids": sorted(declared_ids),
                }
                prior = _binding_fact(prior_bindings.get(local_id))
                if prior:
                    fact["prior_binding"] = prior
                facts.append(fact)
        elif parent_key in table_reference_keys and value in declared_ids:
            fact = {
                "path": path,
                "provided": value,
                "status": "matches_current_call_id_without_local_reference",
                "local_reference": f"${value}",
            }
            current = _binding_fact(bindings.get(value))
            if current:
                fact["current_binding"] = current
            facts.append(fact)
    return facts


def _fact_input_table(
    tool: str,
    arguments: dict,
    *,
    bindings: dict[str, dict],
    prior_bindings: dict[str, dict],
) -> str | None:
    key = "base" if tool == "join_tables" else "table"
    value = arguments.get(key)
    if not isinstance(value, str):
        return None
    if value.startswith("$"):
        local_id = value[1:].split(".", 1)[0]
        binding = bindings.get(local_id) or prior_bindings.get(local_id)
        return binding.get("table") if isinstance(binding, dict) else None
    if value in bindings and bindings[value].get("table"):
        return bindings[value]["table"]
    return value


def _strip_order_direction(value: str) -> str:
    parts = value.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].upper() in {"ASC", "DESC"}:
        return parts[0]
    return value


def _requested_columns_for_error(tool: str, arguments: dict) -> list[dict]:
    requested = []
    if tool == "join_tables":
        for join_index, join in enumerate(arguments.get("joins") or []):
            for on_index, pair in enumerate(join.get("on") or []):
                if isinstance(pair.get("left"), str):
                    requested.append({
                        "path": f"joins.{join_index}.on.{on_index}.left",
                        "provided": pair["left"],
                    })
    elif tool == "extreme_value_select":
        for index, column in enumerate(arguments.get("order_by") or []):
            if isinstance(column, str):
                requested.append({
                    "path": f"order_by.{index}",
                    "provided": _strip_order_direction(column),
                })
        for index, column in enumerate(arguments.get("return_columns") or []):
            if isinstance(column, str):
                requested.append({
                    "path": f"return_columns.{index}",
                    "provided": column,
                })
    elif tool == "read_subtable":
        for index, column in enumerate(arguments.get("columns") or []):
            if isinstance(column, str):
                requested.append({
                    "path": f"columns.{index}",
                    "provided": column,
                })
    return requested


def _column_error_facts(
    h: Harness,
    tool: str,
    arguments: dict,
    *,
    bindings: dict[str, dict],
    prior_bindings: dict[str, dict],
) -> dict | None:
    input_table = _fact_input_table(
        tool,
        arguments,
        bindings=bindings,
        prior_bindings=prior_bindings,
    )
    if not input_table:
        return None
    resolutions = []

    def append_resolution(requested: dict, available: list[str]) -> None:
        provided = requested["provided"]
        exact = [
            column
            for column in available
            if column.casefold() == provided.casefold()
        ]
        suffix = [
            column
            for column in available
            if column.rsplit(".", 1)[-1].casefold()
            == provided.rsplit(".", 1)[-1].casefold()
        ]
        if exact:
            return
        if len(suffix) == 1:
            status = "unique_suffix_only"
        elif len(suffix) > 1:
            status = "ambiguous_suffix"
        else:
            status = "missing"
        resolutions.append({
            **requested,
            "status": status,
            "candidates": suffix,
        })

    try:
        available = h.table_columns(input_table)
    except Exception:  # noqa: BLE001
        return None

    if tool == "join_tables":
        base_namespace = arguments.get("base_role") or input_table
        available = [
            column if "." in column else f"{base_namespace}.{column}"
            for column in available
        ]
        for join_index, join in enumerate(arguments.get("joins") or []):
            for on_index, pair in enumerate(join.get("on") or []):
                left = pair.get("left")
                if isinstance(left, str):
                    append_resolution({
                        "path": f"joins.{join_index}.on.{on_index}.left",
                        "provided": left,
                    }, available)
            table = join.get("table")
            if not isinstance(table, str):
                break
            try:
                introduced = h.table_columns(table)
            except Exception:  # noqa: BLE001
                break
            namespace = join.get("role") or table
            available.extend(
                column if "." in column else f"{namespace}.{column}"
                for column in introduced
            )
    else:
        for requested in _requested_columns_for_error(tool, arguments):
            append_resolution(requested, available)

    if not resolutions:
        return None
    return {
        "input_table": input_table,
        "column_resolution": resolutions,
    }


def _scalar_source_facts(ctx: dict, tool: str, arguments: dict) -> list[dict]:
    if tool != "scalar_compute":
        return []
    facts = []
    for index, operand in enumerate(arguments.get("operands") or []):
        if not isinstance(operand, dict):
            continue
        value_ref = operand.get("value_ref")
        record = (ctx.get("history") or {}).get(value_ref)
        if not isinstance(record, dict):
            continue
        output = record.get("output") or {}
        fact = {
            "operand_index": index,
            "value_ref": value_ref,
        }
        for key in ("table", "row_count", "columns"):
            if output.get(key) is not None:
                fact[key] = deepcopy(output[key])
        facts.append(fact)
    return facts


def _structured_error_facts(
    *,
    h: Harness,
    ctx: dict,
    tool: str,
    original_arguments: dict,
    resolved_arguments: dict | None,
    declared_ids: set[str],
    bindings: dict[str, dict],
    prior_bindings: dict[str, dict],
) -> dict:
    facts = {}
    references = _reference_error_facts(
        original_arguments,
        declared_ids=declared_ids,
        bindings=bindings,
        prior_bindings=prior_bindings,
    )
    if references:
        facts["reference_resolution"] = references
    effective_arguments = resolved_arguments or original_arguments
    columns = _column_error_facts(
        h,
        tool,
        effective_arguments,
        bindings=bindings,
        prior_bindings=prior_bindings,
    )
    if columns:
        facts.update(columns)
    scalar_sources = _scalar_source_facts(ctx, tool, effective_arguments)
    if scalar_sources:
        facts["scalar_sources"] = scalar_sources
    return facts


def _is_table_or_value_reference_path(tool: str, path: tuple[str, ...]) -> str | None:
    if not path:
        return None
    key = path[-1]
    if key in {"table", "base", "in_table"}:
        return "table"
    if tool == "set_op" and len(path) == 1 and key in {"left", "right"}:
        return "table"
    if key == "value_ref":
        return "step"
    return None


def _implicit_local_reference_ids(
    tool: str,
    arguments: dict,
    declared_ids: set[str],
) -> list[str]:
    found = []

    def visit(value, path=()):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, (*path, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, (*path, str(index)))
        elif (
            isinstance(value, str)
            and not value.startswith("$")
            and value in declared_ids
            and _is_table_or_value_reference_path(tool, path)
            and value not in found
        ):
            found.append(value)

    visit(arguments)
    return found


def _resolve_implicit_local_references(
    tool: str,
    value,
    *,
    bindings: dict[str, dict],
    declared_ids: set[str],
    resolutions: list[dict],
    path=(),
):
    if isinstance(value, dict):
        return {
            key: _resolve_implicit_local_references(
                tool,
                child,
                bindings=bindings,
                declared_ids=declared_ids,
                resolutions=resolutions,
                path=(*path, str(key)),
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_implicit_local_references(
                tool,
                child,
                bindings=bindings,
                declared_ids=declared_ids,
                resolutions=resolutions,
                path=(*path, str(index)),
            )
            for index, child in enumerate(value)
        ]
    reference_kind = _is_table_or_value_reference_path(tool, path)
    if (
        not isinstance(value, str)
        or value.startswith("$")
        or value not in declared_ids
        or reference_kind is None
    ):
        return value
    binding = bindings.get(value)
    if binding is None:
        raise LocalReferenceError(
            f"implicit local reference {value!r} is forward; only earlier calls may be "
            "referenced"
        )
    if binding.get("status") != "success":
        raise LocalReferenceError(
            f"implicit local reference {value!r} depends on a "
            f"{binding.get('status')} call"
        )
    resolved = (
        binding.get("step_id")
        if reference_kind == "step"
        else binding.get("table")
    )
    if not resolved:
        raise LocalReferenceError(
            f"implicit local reference {value!r} has no {reference_kind} output"
        )
    resolutions.append({
        "path": ".".join(path),
        "provided": value,
        "resolved": resolved,
        "rule": f"implicit_same_block_{reference_kind}_reference",
    })
    return resolved


def _resolve_resident_runtime_references(
    tool: str,
    value,
    *,
    declared_ids: set[str],
    resident_handles: set[str],
    handle_to_step: dict[str, str],
    resolutions: list[dict],
    path=(),
):
    """Resolve action-block-only handle spellings that the harness owns deterministically."""
    if isinstance(value, dict):
        # Models naturally extend the action-block "$call.column" shorthand to persistent
        # handles.  For scalar operands this is exactly equivalent to the public
        # value_ref+column form, so normalize it before resolving the handle to its producer step.
        # A current-block declaration still shadows an identically named resident handle.
        resident_scalar = value.get("value_ref") if tool == "scalar_compute" else None
        if isinstance(resident_scalar, str) and "column" not in value:
            sigiled = resident_scalar.startswith("$")
            candidate = resident_scalar[1:] if sigiled else resident_scalar
            matching_handles = sorted(
                (
                    handle
                    for handle in resident_handles
                    if handle not in declared_ids
                    and candidate.startswith(f"{handle}.")
                    and handle_to_step.get(handle)
                ),
                key=len,
                reverse=True,
            )
            if matching_handles:
                handle = matching_handles[0]
                column = candidate[len(handle) + 1 :]
                resolved = handle_to_step[handle]
                normalized = deepcopy(value)
                normalized["value_ref"] = resolved
                normalized["column"] = column
                resolutions.append({
                    "path": ".".join((*path, "value_ref")),
                    "provided": resident_scalar,
                    "resolved": {
                        "value_ref": resolved,
                        "column": column,
                    },
                    "rule": (
                        "sigiled_resident_handle_column_to_producing_step"
                        if sigiled
                        else "resident_handle_column_to_producing_step"
                    ),
                })
                return {
                    key: _resolve_resident_runtime_references(
                        tool,
                        child,
                        declared_ids=declared_ids,
                        resident_handles=resident_handles,
                        handle_to_step=handle_to_step,
                        resolutions=resolutions,
                        path=(*path, str(key)),
                    )
                    for key, child in normalized.items()
                }
        return {
            key: _resolve_resident_runtime_references(
                tool,
                child,
                declared_ids=declared_ids,
                resident_handles=resident_handles,
                handle_to_step=handle_to_step,
                resolutions=resolutions,
                path=(*path, str(key)),
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_resident_runtime_references(
                tool,
                child,
                declared_ids=declared_ids,
                resident_handles=resident_handles,
                handle_to_step=handle_to_step,
                resolutions=resolutions,
                path=(*path, str(index)),
            )
            for index, child in enumerate(value)
        ]
    reference_kind = _is_table_or_value_reference_path(tool, path)
    if not isinstance(value, str) or reference_kind is None:
        return value
    sigiled = value.startswith("$")
    candidate = value[1:] if sigiled else value
    if candidate in declared_ids or candidate not in resident_handles:
        return value
    if reference_kind == "step":
        resolved = handle_to_step.get(candidate)
        if not resolved:
            return value
        rule = (
            "sigiled_resident_handle_to_producing_step"
            if sigiled
            else "resident_handle_to_producing_step"
        )
    else:
        if not sigiled:
            return value
        resolved = candidate
        rule = "remove_sigil_from_resident_handle"
    resolutions.append({
        "path": ".".join(path),
        "provided": value,
        "resolved": resolved,
        "rule": rule,
    })
    return resolved


def _normalize_action_block_predicates(
    tool: str,
    arguments: dict,
    resolutions: list[dict],
) -> dict:
    """Normalize an unambiguous infix boolean list without changing predicate meaning.

    The public predicate tree remains canonical.  This action-block-only adapter accepts the
    common equivalent ``[predicate, {"op":"and"}, predicate]`` spelling when every connector is
    the same boolean operator. Mixed connectors are deliberately left for validation because
    precedence would require a semantic guess.
    """
    if tool != "condition_filter":
        return arguments
    conditions = arguments.get("conditions")
    if not isinstance(conditions, list) or len(conditions) < 3 or len(conditions) % 2 == 0:
        return arguments
    predicates = conditions[::2]
    connectors = conditions[1::2]
    if (
        not all(isinstance(predicate, dict) and predicate for predicate in predicates)
        or not all(
            isinstance(connector, dict)
            and set(connector) == {"op"}
            and connector.get("op") in {"and", "or"}
            for connector in connectors
        )
    ):
        return arguments
    operators = {connector["op"] for connector in connectors}
    if len(operators) != 1:
        return arguments
    operator = next(iter(operators))
    resolved = deepcopy(arguments)
    resolved["conditions"] = {operator: deepcopy(predicates)}
    resolutions.append({
        "path": "conditions",
        "provided": deepcopy(conditions),
        "resolved": deepcopy(resolved["conditions"]),
        "rule": "uniform_infix_boolean_predicate",
    })
    return resolved


def _normalize_action_block_order_by(
    tool: str,
    arguments: dict,
    resolutions: list[dict],
) -> dict:
    """Accept the unambiguous structured spelling of an ordered column."""
    if tool != "extreme_value_select":
        return arguments
    order_by = arguments.get("order_by")
    if not isinstance(order_by, list):
        return arguments
    normalized = []
    changed = False
    for index, item in enumerate(order_by):
        if not (
            isinstance(item, dict)
            and set(item) == {"column", "direction"}
            and isinstance(item.get("column"), str)
            and item["column"].strip()
            and isinstance(item.get("direction"), str)
            and item["direction"].upper() in {"ASC", "DESC"}
        ):
            normalized.append(item)
            continue
        rendered = f"{item['column']} {item['direction'].upper()}"
        normalized.append(rendered)
        changed = True
        resolutions.append({
            "path": f"order_by.{index}",
            "provided": deepcopy(item),
            "resolved": rendered,
            "rule": "structured_order_by_to_string",
        })
    if not changed:
        return arguments
    resolved = deepcopy(arguments)
    resolved["order_by"] = normalized
    return resolved


def _normalize_predicate_scalar_references(
    tool: str,
    arguments: dict,
    *,
    ctx: dict,
    bindings: dict[str, dict],
    prior_bindings: dict[str, dict],
    resolutions: list[dict],
) -> dict:
    """Normalize named references to a verified one-cell predicate scalar.

    Predicate ``value_ref`` has no separate source-column field. The adapter accepts named
    ``value_ref`` spellings and the common cross-result ``column_value`` spelling only when the
    cited successful source is factually one row and one column and the supplied column names that
    sole output. Nothing is guessed for wider sources; unresolved local ``column_value`` references
    are rejected later instead of becoming same-table self-comparisons.
    """
    if tool not in {"condition_filter", "group_aggregate"}:
        return arguments

    merged_bindings = dict(prior_bindings)
    merged_bindings.update(bindings)
    history = ctx.get("history") or {}
    handle_to_step = ctx.get("handle_to_step") or {}

    def source(raw):
        provided = deepcopy(raw)
        requested_column = None
        reference = raw
        if isinstance(raw, dict):
            if (
                set(raw) != {"value_ref", "column"}
                or not isinstance(raw.get("value_ref"), str)
                or not isinstance(raw.get("column"), str)
            ):
                return None
            reference = raw["value_ref"]
            requested_column = raw["column"]
        if not isinstance(reference, str):
            return None

        base = reference
        if "." in reference:
            base, dotted_column = reference.split(".", 1)
            requested_column = requested_column or dotted_column

        step_id = None
        if base.startswith("$"):
            binding = merged_bindings.get(base[1:])
            if isinstance(binding, dict) and binding.get("status") == "success":
                step_id = binding.get("step_id")
        elif base in history:
            step_id = base
        elif base in handle_to_step:
            step_id = handle_to_step[base]
        if not isinstance(step_id, str) or not requested_column:
            return None

        record = history.get(step_id)
        output = record.get("output") if isinstance(record, dict) else None
        columns = output.get("columns") if isinstance(output, dict) else None
        if (
            not isinstance(columns, list)
            or len(columns) != 1
            or output.get("row_count") != 1
        ):
            return None
        only_column = columns[0]
        if not isinstance(only_column, str) or (
            only_column.casefold() != requested_column.casefold()
            and only_column.rsplit(".", 1)[-1].casefold()
            != requested_column.rsplit(".", 1)[-1].casefold()
        ):
            return None
        return provided, step_id, only_column

    def visit(value, path=()):
        if isinstance(value, dict):
            normalized = deepcopy(value)
            if (
                "column_value" in value
                and "value_ref" not in value
                and "value" not in value
            ):
                match = source(value["column_value"])
                if match is not None:
                    provided, step_id, column = match
                    normalized.pop("column_value")
                    normalized["value_ref"] = step_id
                    resolutions.append({
                        "path": ".".join((*path, "column_value")),
                        "provided": provided,
                        "resolved": step_id,
                        "verified_source_column": column,
                        "rule": "one_cell_column_value_to_value_ref",
                    })
            if "value_ref" in normalized:
                match = source(normalized["value_ref"])
                if match is not None:
                    provided, step_id, column = match
                    normalized["value_ref"] = step_id
                    resolutions.append({
                        "path": ".".join((*path, "value_ref")),
                        "provided": provided,
                        "resolved": step_id,
                        "verified_source_column": column,
                        "rule": "named_one_cell_predicate_value_ref",
                    })
            return {
                key: visit(child, (*path, str(key)))
                for key, child in normalized.items()
            }
        if isinstance(value, list):
            return [
                visit(child, (*path, str(index)))
                for index, child in enumerate(value)
            ]
        return value

    return visit(arguments)


def _resolution_bindings(
    *,
    bindings: dict[str, dict],
    prior_bindings: dict[str, dict],
    declared_ids: set[str],
) -> tuple[dict[str, dict], set[str]]:
    # A declaration in the current block shadows an older id, including before the new call runs.
    merged = {
        call_id: binding
        for call_id, binding in prior_bindings.items()
        if call_id not in declared_ids
    }
    merged.update(bindings)
    return merged, declared_ids | set(prior_bindings)


def _record_persistent_reference_resolutions(
    arguments: dict,
    *,
    declared_ids: set[str],
    prior_bindings: dict[str, dict],
    resolutions: list[dict],
) -> None:
    for path, value in _walk_argument_strings(arguments):
        if not value.startswith("$"):
            continue
        match = LOCAL_COLUMN_REF_RE.fullmatch(value) or LOCAL_REF_RE.fullmatch(value)
        if not match:
            continue
        call_id = match.group(1)
        if call_id in declared_ids or call_id not in prior_bindings:
            continue
        binding = prior_bindings[call_id]
        resolved = None
        column_match = LOCAL_COLUMN_REF_RE.fullmatch(value)
        if column_match:
            requested = column_match.group(2)
            columns = binding.get("columns") or []
            exact, suffix = _column_candidates(columns, requested)
            matches = exact or suffix
            if len(matches) == 1:
                resolved = matches[0]
                if (
                    path.rsplit(".", 1)[-1] == "left"
                    and "." not in resolved
                    and binding.get("table")
                ):
                    resolved = f"{binding['table']}.{resolved}"
        elif path.rsplit(".", 1)[-1] == "value_ref":
            resolved = binding.get("step_id")
        else:
            resolved = binding.get("table")
        resolutions.append({
            "path": path,
            "provided": value,
            "resolved": resolved,
            "rule": "persistent_prior_block_call_reference",
            "prior_block_index": binding.get("block_index"),
        })


def _reject_local_column_literal_references(arguments: dict) -> None:
    literal_fields = {"value", "low", "high"}
    for path, value in _walk_argument_strings(arguments):
        if not LOCAL_COLUMN_REF_RE.fullmatch(value):
            continue
        named_parts = [
            part for part in path.split(".")
            if not part.isdigit()
        ]
        if named_parts and named_parts[-1] in literal_fields:
            raise LocalReferenceError(
                f"local column reference {value!r} at {path} cannot supply a literal value; "
                "use value_ref for a one-row grounded value or read/inspect the literal first"
            )


def _column_candidates(available: list[str], provided: str) -> tuple[list[str], list[str]]:
    exact = [
        column
        for column in available
        if column.casefold() == provided.casefold()
    ]
    suffix = [
        column
        for column in available
        if column.rsplit(".", 1)[-1].casefold()
        == provided.rsplit(".", 1)[-1].casefold()
    ]
    return exact, suffix


def _join_equivalence_groups(state: dict, table: str) -> list[set[str]]:
    table_entry = next(
        (
            entry
            for name, entry in (state.get("tables") or {}).items()
            if name.casefold() == table.casefold()
        ),
        None,
    )
    derivation = (table_entry or {}).get("derivation") or {}
    if derivation.get("operator") != "join_tables":
        return []

    groups: list[set[str]] = []
    for edge in (derivation.get("semantics") or {}).get("edges") or []:
        if edge.get("join_type") != "inner":
            continue
        namespace = edge.get("namespace")
        if not isinstance(namespace, str):
            continue
        for pair in edge.get("on") or []:
            left, right = pair.get("left"), pair.get("right")
            if not isinstance(left, str) or not isinstance(right, str):
                continue
            pair_group = {left.casefold(), f"{namespace}.{right}".casefold()}
            overlapping = [group for group in groups if group & pair_group]
            if overlapping:
                merged = set().union(pair_group, *overlapping)
                groups = [group for group in groups if group not in overlapping]
                groups.append(merged)
            else:
                groups.append(pair_group)
    return groups


def _merge_equivalent_columns(
    groups: list[set[str]],
    left: str,
    right: str,
) -> None:
    pair = {left.casefold(), right.casefold()}
    overlapping = [group for group in groups if group & pair]
    if overlapping:
        merged = set().union(pair, *overlapping)
        groups[:] = [group for group in groups if group not in overlapping]
        groups.append(merged)
    else:
        groups.append(pair)


def _resolve_equivalent_local_column_references(
    tool: str,
    value,
    *,
    bindings: dict[str, dict],
    state: dict,
    resolutions: list[dict],
    path=(),
):
    if isinstance(value, dict):
        return {
            key: _resolve_equivalent_local_column_references(
                tool,
                child,
                bindings=bindings,
                state=state,
                resolutions=resolutions,
                path=(*path, str(key)),
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _resolve_equivalent_local_column_references(
                tool,
                child,
                bindings=bindings,
                state=state,
                resolutions=resolutions,
                path=(*path, str(index)),
            )
            for index, child in enumerate(value)
        ]
    if not isinstance(value, str):
        return value
    match = LOCAL_COLUMN_REF_RE.fullmatch(value)
    if not match:
        return value
    call_id, requested = match.groups()
    binding = bindings.get(call_id)
    if not isinstance(binding, dict) or binding.get("status") != "success":
        return value
    columns = binding.get("columns")
    table = binding.get("table")
    if not isinstance(columns, list) or not isinstance(table, str):
        return value
    exact, suffix = _column_candidates(columns, requested)
    if exact or len(suffix) < 2:
        return value
    groups = _join_equivalence_groups(state, table)
    folded = {column.casefold() for column in suffix}
    if not any(folded <= group for group in groups):
        return value
    resolved = suffix[0]
    resolutions.append({
        "path": ".".join(path),
        "provided": value,
        "resolved": resolved,
        "rule": "local_inner_join_equivalent_columns",
    })
    return resolved


def _canonical_column(
    provided: str,
    *,
    available: list[str],
    equivalent_groups: list[set[str]] | None = None,
) -> tuple[str, str | None]:
    exact, suffix = _column_candidates(available, provided)
    if exact:
        return exact[0], None
    if len(suffix) == 1:
        return suffix[0], "unique_column_suffix"
    if len(suffix) > 1 and equivalent_groups:
        folded = {column.casefold() for column in suffix}
        if any(folded <= group for group in equivalent_groups):
            return suffix[0], "inner_join_equivalent_columns"
    return provided, None


def _canonicalize_low_friction_columns(
    h: Harness,
    ctx: dict,
    tool: str,
    arguments: dict,
    resolutions: list[dict],
    *,
    progressive_join_equivalence: bool = False,
) -> dict:
    resolved = deepcopy(arguments)
    if tool == "join_tables":
        base = resolved.get("base")
        if not isinstance(base, str):
            return resolved
        try:
            base_columns = h.table_columns(base)
        except Exception:  # noqa: BLE001
            return resolved
        namespace = resolved.get("base_role") or base
        available = [
            column if "." in column else f"{namespace}.{column}"
            for column in base_columns
        ]
        equivalent_groups: list[set[str]] = []
        for join_index, join in enumerate(resolved.get("joins") or []):
            for on_index, pair in enumerate(join.get("on") or []):
                provided = pair.get("left")
                if not isinstance(provided, str):
                    continue
                canonical, rule = _canonical_column(
                    provided,
                    available=available,
                    equivalent_groups=(
                        equivalent_groups
                        if progressive_join_equivalence
                        else None
                    ),
                )
                if rule:
                    pair["left"] = canonical
                    resolutions.append({
                        "path": f"joins.{join_index}.on.{on_index}.left",
                        "provided": provided,
                        "resolved": canonical,
                        "rule": rule,
                    })
            table = join.get("table")
            if not isinstance(table, str):
                break
            try:
                introduced = h.table_columns(table)
            except Exception:  # noqa: BLE001
                break
            namespace = join.get("role") or table
            if progressive_join_equivalence and join.get("type", "inner") == "inner":
                for pair in join.get("on") or []:
                    left, right = pair.get("left"), pair.get("right")
                    if isinstance(left, str) and isinstance(right, str):
                        _merge_equivalent_columns(
                            equivalent_groups,
                            left,
                            f"{namespace}.{right}",
                        )
            available.extend(
                column if "." in column else f"{namespace}.{column}"
                for column in introduced
            )
        return resolved

    table = resolved.get("table")
    if not isinstance(table, str):
        return resolved
    try:
        available = h.table_columns(table)
    except Exception:  # noqa: BLE001
        return resolved
    equivalent_groups = _join_equivalence_groups(
        ctx["environment"].snapshot(),
        table,
    )

    def canonicalize_list(key: str, *, ordered: bool = False) -> None:
        values = resolved.get(key)
        if not isinstance(values, list):
            return
        for index, value in enumerate(values):
            if not isinstance(value, str):
                continue
            column = _strip_order_direction(value) if ordered else value
            direction = value[len(column):] if ordered else ""
            canonical, rule = _canonical_column(
                column,
                available=available,
                equivalent_groups=equivalent_groups,
            )
            if rule:
                values[index] = canonical + direction
                resolutions.append({
                    "path": f"{key}.{index}",
                    "provided": value,
                    "resolved": values[index],
                    "rule": rule,
                })

    if tool == "extreme_value_select":
        canonicalize_list("order_by", ordered=True)
        canonicalize_list("return_columns")
    elif tool == "read_subtable":
        canonicalize_list("columns")
    elif tool == "project":
        canonicalize_list("expressions")
    return resolved


def _schedule_action_block_calls(
    calls: list[dict],
    *,
    low_friction_interface: bool,
) -> list[dict]:
    """Topologically order calls from harness-derived result references.

    The submitted list is only a set of requested tool calls. The model does not provide or
    maintain a DAG. Unknown references and cycles remain executable error roots so the harness can
    return precise factual feedback instead of rejecting unrelated independent calls.
    """
    if not low_friction_interface:
        return list(calls)

    declared_ids = {call["id"] for call in calls}
    dependencies: dict[str, list[str]] = {}
    for call in calls:
        refs = local_reference_ids(call["arguments"])
        if low_friction_interface:
            for dependency in _implicit_local_reference_ids(
                call["tool"],
                call["arguments"],
                declared_ids,
            ):
                if dependency not in refs:
                    refs.append(dependency)
        dependencies[call["id"]] = refs

    pending = list(calls)
    scheduled_ids: set[str] = set()
    scheduled: list[dict] = []
    while pending:
        nonterminal_pending = [
            (index, call)
            for index, call in enumerate(pending)
            if call.get("tool") != TERMINAL_TOOL
        ]
        candidate_items = nonterminal_pending or list(enumerate(pending))
        ready_index = next(
            (
                index
                for index, call in candidate_items
                if all(
                    dependency not in declared_ids
                    or dependency in scheduled_ids
                    for dependency in dependencies[call["id"]]
                )
            ),
            None,
        )
        # A cycle or self-reference has no topologically ready node. Execute one root so normal
        # local-reference validation reports the actual bad reference; its descendants then become
        # blocked without being counted as additional errors.
        if ready_index is None:
            ready_index = candidate_items[0][0]
        call = pending.pop(ready_index)
        scheduled.append(call)
        scheduled_ids.add(call["id"])
    return scheduled


def _execute_action_block(
    *,
    h: Harness,
    ctx: dict,
    arguments: dict,
    created: set[str],
    atomic_count: int,
    model_turn: int,
    batch_index: int,
    table_output_rows: int,
    error_counts: collections.Counter,
    error_events: list[dict],
    prior_bindings: dict[str, dict] | None = None,
    structured_error_feedback: bool = False,
    low_friction_interface: bool = False,
    safe_low_friction_interface: bool = False,
    interface_resolution_events: list[dict] | None = None,
    validate_call=validate_atomic_call,
    prepare_call_arguments=None,
    lower_call=None,
    publicize_error=None,
) -> tuple[int, list[dict], list[dict], bool]:
    """Execute one block and isolate root errors from blocked descendants."""
    low_friction_interface = (
        low_friction_interface or safe_low_friction_interface
    )
    submitted_calls = arguments.get("calls") or []
    terminal_calls = [
        call for call in submitted_calls
        if call.get("tool") == TERMINAL_TOOL
    ]
    if terminal_calls and (
        len(terminal_calls) != 1 or len(submitted_calls) != 1
    ):
        raise BatchPlanProtocolError(
            "answer_from_context must be the one and only call in its action_block"
        )
    results: list[dict] = []
    atomic_events: list[dict] = []
    nonrecoverable = False
    calls = _schedule_action_block_calls(
        submitted_calls,
        low_friction_interface=low_friction_interface,
    )
    declared_ids = {call["id"] for call in calls}
    bindings: dict[str, dict] = {}
    prior_bindings = prior_bindings if prior_bindings is not None else {}

    for call_index, call in enumerate(calls):
        call_id = call["id"]
        tool = call["tool"]
        original_args = deepcopy(call["arguments"])
        dependencies = local_reference_ids(original_args)
        if low_friction_interface:
            for dependency in _implicit_local_reference_ids(
                tool,
                original_args,
                declared_ids,
            ):
                if dependency not in dependencies:
                    dependencies.append(dependency)
        failed_dependencies = [
            dependency
            for dependency in dependencies
            if (
                dependency in bindings
                and bindings[dependency].get("status") != "success"
            )
        ]
        if failed_dependencies:
            state = ctx["environment"].snapshot()
            root_causes = sorted({
                root
                for dependency in failed_dependencies
                for root in (
                    bindings[dependency].get("root_causes") or [dependency]
                )
            })
            blocked = {
                "call_id": call_id,
                "step_id": None,
                "tool": tool,
                "status": "blocked",
                "dependencies": dependencies,
                "blocked_by": failed_dependencies,
                "root_causes": root_causes,
                "reason": "not executed because a required earlier call did not succeed",
                "arguments": original_args,
                "environment_state_before": state,
                "environment_state": state,
            }
            bindings[call_id] = {
                "status": "blocked",
                "tool": tool,
                "step_id": None,
                "table": None,
                "columns": None,
                "row_count": None,
                "source_reference": original_args.get("table"),
                "root_causes": root_causes,
            }
            results.append(blocked)
            atomic_events.append(deepcopy(blocked))
            continue

        atomic_count += 1
        step_id = f"step_{atomic_count}"
        state_before = ctx["environment"].snapshot()
        resolved_args = None
        execution_tool = tool
        execution_args = None
        terminal_score_arguments = None
        interface_resolutions: list[dict] = []
        try:
            reference_args = original_args
            if prepare_call_arguments is not None:
                reference_args = prepare_call_arguments(
                    tool,
                    reference_args,
                    bindings=bindings,
                    declared_ids=declared_ids,
                    ctx=ctx,
                    resolutions=interface_resolutions,
                )
            resolution_bindings = bindings
            resolution_declared_ids = declared_ids
            if low_friction_interface:
                if safe_low_friction_interface:
                    _reject_local_column_literal_references(original_args)
                reference_args = _normalize_action_block_predicates(
                    tool,
                    reference_args,
                    interface_resolutions,
                )
                reference_args = _normalize_action_block_order_by(
                    tool,
                    reference_args,
                    interface_resolutions,
                )
                reference_args = _normalize_predicate_scalar_references(
                    tool,
                    reference_args,
                    ctx=ctx,
                    bindings=bindings,
                    prior_bindings=prior_bindings,
                    resolutions=interface_resolutions,
                )
                reference_args = _resolve_implicit_local_references(
                    tool,
                    reference_args,
                    bindings=bindings,
                    declared_ids=declared_ids,
                    resolutions=interface_resolutions,
                )
                reference_args = _resolve_resident_runtime_references(
                    tool,
                    reference_args,
                    declared_ids=declared_ids,
                    resident_handles=set(
                        (ctx["environment"].snapshot().get("tables") or {})
                    ),
                    handle_to_step=ctx["handle_to_step"],
                    resolutions=interface_resolutions,
                )
                _record_persistent_reference_resolutions(
                    original_args,
                    declared_ids=declared_ids,
                    prior_bindings=prior_bindings,
                    resolutions=interface_resolutions,
                )
                resolution_bindings, resolution_declared_ids = _resolution_bindings(
                    bindings=bindings,
                    prior_bindings=prior_bindings,
                    declared_ids=declared_ids,
                )
                if safe_low_friction_interface:
                    reference_args = _resolve_equivalent_local_column_references(
                        tool,
                        reference_args,
                        bindings=resolution_bindings,
                        state=ctx["environment"].snapshot(),
                        resolutions=interface_resolutions,
                    )
            resolved_args = resolve_local_references(
                reference_args,
                resolution_bindings,
                resolution_declared_ids,
            )
            if low_friction_interface:
                resolved_args = _canonicalize_low_friction_columns(
                    h,
                    ctx,
                    tool,
                    resolved_args,
                    interface_resolutions,
                    progressive_join_equivalence=safe_low_friction_interface,
                )
            if tool == TERMINAL_TOOL:
                validate_model_arguments(TERMINAL_TOOL, resolved_args)
                terminal_score_arguments = deepcopy(resolved_args)
                output = {
                    "terminal_ready": True,
                    "evidence_table": resolved_args["evidence"]["table"],
                }
                table = None
            else:
                validate_call(tool, resolved_args)
                if lower_call is not None:
                    execution_tool, execution_args = lower_call(
                        tool,
                        resolved_args,
                    )
                else:
                    execution_args = deepcopy(resolved_args)
                output, table = execute_tool(
                    h,
                    execution_tool,
                    execution_args,
                    ctx,
                    step_id,
                    table_output_rows=table_output_rows,
                )
            if table:
                created.add(table)
            result = {
                "call_id": call_id,
                "step_id": step_id,
                "tool": tool,
                "status": "success",
                "dependencies": dependencies,
                "arguments": original_args,
                "resolved_arguments": deepcopy(resolved_args),
                "output": output,
                "table": table,
                "environment_state_before": state_before,
                "environment_state": ctx["environment"].snapshot(),
            }
            if execution_tool != tool:
                result["execution_tool"] = execution_tool
                result["execution_arguments"] = deepcopy(execution_args)
            if terminal_score_arguments is not None:
                result["terminal_score_arguments"] = deepcopy(
                    terminal_score_arguments
                )
                result["resolved_evidence_arguments"] = deepcopy(resolved_args)
            if interface_resolutions:
                result["interface_resolutions"] = deepcopy(interface_resolutions)
            bindings[call_id] = {
                "status": "success",
                "tool": tool,
                "step_id": step_id,
                "table": table,
                "columns": deepcopy(
                    output.get("columns")
                    if isinstance(output, dict)
                    else None
                ),
                "row_count": (
                    output.get("row_count")
                    if isinstance(output, dict)
                    else None
                ),
                "source_reference": original_args.get("table"),
                "root_causes": [],
            }
        except Exception as exc:  # noqa: BLE001
            state_after = ctx["environment"].snapshot()
            error_type = _error_type(exc)
            if compact_json(state_after) != compact_json(state_before):
                error_type = "nonrecoverable_execution_error"
                nonrecoverable = True
            if isinstance(exc, LocalReferenceError):
                message = str(exc)
            else:
                message = format_tool_error(
                    exc,
                    h,
                    execution_tool,
                    execution_args or original_args,
                )
            if publicize_error is not None:
                message = publicize_error(message)
            facts = {}
            if structured_error_feedback:
                facts = _structured_error_facts(
                    h=h,
                    ctx=ctx,
                    tool=tool,
                    original_arguments=original_args,
                    resolved_arguments=resolved_args,
                    declared_ids=declared_ids,
                    bindings=bindings,
                    prior_bindings=prior_bindings,
                )
            error_counts[error_type] += 1
            event = _error_event(
                atomic_index=atomic_count,
                model_turn=model_turn,
                batch_index=batch_index,
                call_id=call_id,
                tool=tool,
                arguments=original_args,
                error_type=error_type,
                message=message,
                state_before=state_before,
                state_after=state_after,
                facts=facts,
            )
            error_events.append(event)
            result = {
                "call_id": call_id,
                "step_id": step_id,
                "tool": tool,
                "status": "error",
                "dependencies": dependencies,
                "arguments": original_args,
                "error": {
                    "type": error_type,
                    "message": message,
                    **({"facts": facts} if facts else {}),
                },
                "environment_state_before": state_before,
                "environment_state": state_after,
            }
            if interface_resolutions:
                result["interface_resolutions"] = deepcopy(interface_resolutions)
            bindings[call_id] = {
                "status": "error",
                "tool": tool,
                "step_id": step_id,
                "table": None,
                "columns": None,
                "row_count": None,
                "source_reference": original_args.get("table"),
                "root_causes": [call_id],
            }
        results.append(result)
        atomic_events.append(deepcopy(result))
        if interface_resolutions and interface_resolution_events is not None:
            interface_resolution_events.extend({
                "model_turn": model_turn,
                "block_index": batch_index,
                "call_id": call_id,
                "step_id": step_id,
                "tool": tool,
                **deepcopy(resolution),
            } for resolution in interface_resolutions)

        if nonrecoverable:
            remaining = calls[call_index + 1:]
            for pending in remaining:
                state_after = ctx["environment"].snapshot()
                blocked = {
                    "call_id": pending["id"],
                    "step_id": None,
                    "tool": pending["tool"],
                    "status": "blocked",
                    "dependencies": local_reference_ids(pending["arguments"]),
                    "blocked_by": [call_id],
                    "root_causes": [call_id],
                    "reason": "not executed after a nonrecoverable root error",
                    "arguments": deepcopy(pending["arguments"]),
                    "environment_state_before": state_after,
                    "environment_state": state_after,
                }
                results.append(blocked)
                atomic_events.append(deepcopy(blocked))
            break

    if structured_error_feedback or low_friction_interface:
        for local_id, binding in bindings.items():
            if binding.get("status") != "success":
                continue
            prior_bindings[local_id] = {
                "status": "success",
                "tool": binding.get("tool"),
                "block_index": batch_index,
                "step_id": binding.get("step_id"),
                "table": binding.get("table"),
                "columns": deepcopy(binding.get("columns")),
                "row_count": binding.get("row_count"),
                "source_reference": binding.get("source_reference"),
            }
    return atomic_count, results, atomic_events, nonrecoverable


# Explicit replay compatibility for v4-v11 audit code. Active callers use
# ``_execute_action_block`` so the retired planning name does not define current semantics.
_execute_plan_action = _execute_action_block


def run_episode(
    *,
    example_index: int,
    ex: dict,
    split: str,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    protocol_hash: str,
    max_action_blocks: int,
    max_batch_calls: int,
    max_tokens: int,
    api_retries: int,
    api_timeout: int,
    max_errors_per_type: int,
    table_output_rows: int,
    history_turns: int,
    denotation_comparison: str,
    protocol_version: str = UNIFIED_ACTION_BLOCK_PROTOCOL_VERSION,
    structured_error_feedback: bool = False,
    low_friction_interface: bool = False,
    safe_low_friction_interface: bool = False,
    assistant_carrier: str = BATCH_CARRIER_PROVIDER_NATIVE,
    tool_scheme: str = ACTION_BLOCK_TOOL_SCHEME,
    build_messages=build_batch_plan_messages,
    parse_provider_action=parse_batch_plan_action,
    parse_inline_action=parse_batch_plan_assistant,
    prepare_work_action=None,
    render_work_observation=render_batch_observation,
    validate_call=validate_atomic_call,
    prepare_call_arguments=None,
    lower_call=None,
    publicize_error=None,
) -> dict:
    low_friction_interface = (
        low_friction_interface or safe_low_friction_interface
    )
    task_path = task_db_path(ex)
    gold_sql = task_gold_sql(ex)
    if not gold_sql:
        raise ValueError(
            f"task has no gold SQL: {ex.get('db_id')} / {ex.get('question')}"
        )
    h = Harness(task_path)
    dataset_overview = overview(h)
    external_knowledge = ex.get("external_knowledge") or None
    ctx = new_ctx(dataset_overview)
    created: set[str] = set()
    legal_history: list[dict] = []
    last_error: dict | None = None
    turns: list[dict] = []
    atomic_events: list[dict] = []
    error_events: list[dict] = []
    interface_resolution_events: list[dict] = []
    error_counts: collections.Counter = collections.Counter()
    usage: collections.Counter = collections.Counter()
    atomic_count = 0
    model_turn_count = 0
    batch_count = 0
    submitted_call_count = 0
    blocked_node_count = 0
    prior_call_bindings: dict[str, dict] = {}
    pending_feedback_recovery = False
    started = time.time()

    rec = {
        "tool_scheme": tool_scheme,
        "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
        "assistant_carrier": assistant_carrier,
        "example_index": example_index,
        "trajectory_id": trajectory_id(split, example_index, ex),
        "db_id": ex["db_id"],
        "question": ex["question"],
        "gold_sql": gold_sql,
        "difficulty": (ex.get("metadata") or {}).get("difficulty_proxy"),
        "correct": False,
        "legal": False,
        "model_turns": 0,
        "action_blocks": 0,
        "executed_action_blocks": 0,
        "atomic_actions": 0,
        "submitted_calls": 0,
        "blocked_nodes": 0,
        "errors": 0,
        "failure_type": None,
        "fail": None,
        "turns": turns,
        "atomic_events": atomic_events,
        "error_events": error_events,
        "interface_resolution_events": interface_resolution_events,
        "denotation_comparison": denotation_comparison,
        "terminal_answer_contract": TERMINAL_ANSWER_CONTRACT,
        "protocol_version": protocol_version,
        "protocol_hash": protocol_hash,
        "sft_export_eligible": False,
    }
    legacy_action_shape = (
        tool_scheme == ACTION_BLOCK_TOOL_SCHEME
        and protocol_version not in {
            UNIFIED_ACTION_BLOCK_PROTOCOL_VERSION,
            SEQUENTIAL_ACTION_BLOCK_PROTOCOL_VERSION,
            SIMPLE_SEQUENTIAL_ACTION_BLOCK_PROTOCOL_VERSION,
            SIMPLE_SCALAR_CELL_PROTOCOL_VERSION,
        }
    )

    try:
        while model_turn_count < max_action_blocks:
            model_turn_count += 1
            state_before = ctx["environment"].snapshot()
            model_input = build_messages(
                system_prompt=system_prompt,
                overview=dataset_overview,
                question=ex["question"],
                external_knowledge=external_knowledge,
                state=state_before,
                last_error=last_error,
                legal_history=legal_history,
                history_turns=history_turns,
            )
            turn = {
                "turn_index": model_turn_count - 1,
                "model_input": deepcopy(model_input),
                "feedback_recovery": pending_feedback_recovery,
                "provider_request_options": provider_request_options(
                    model, carrier=DEEPSEEK_CARRIER_JSON_OUTPUT
                ),
            }
            try:
                raw_content, call_usage, reasoning = chat_with_retries(
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    messages=model_input,
                    max_tokens=max_tokens,
                    timeout=api_timeout,
                    retries=api_retries,
                    deepseek_carrier=DEEPSEEK_CARRIER_JSON_OUTPUT,
                )
                add_usage(usage, call_usage)
                turn["api_finish_reason"] = call_usage.get("api_finish_reason")
                if call_usage.get("provider_response_metadata"):
                    turn["provider_response_metadata"] = deepcopy(
                        call_usage["provider_response_metadata"]
                    )
                if call_usage.get("api_retry_events"):
                    turn["provider_retry_events"] = deepcopy(
                        call_usage["api_retry_events"]
                    )
            except ProviderCarrierError as exc:
                add_usage(usage, exc.usage)
                turn["api_error_type"] = "provider_carrier_error"
                turn["api_error"] = str(exc)
                turns.append(turn)
                rec["failure_type"] = "provider_carrier_error"
                rec["fail"] = str(exc)
                break
            except ContextOverflowError as exc:
                turn["api_error_type"] = "context_overflow"
                turn["api_error"] = str(exc)
                turns.append(turn)
                rec["failure_type"] = "context_overflow"
                rec["fail"] = str(exc)
                break
            except Exception as exc:  # noqa: BLE001
                turn["api_error_type"] = "api_error"
                turn["api_error"] = f"{type(exc).__name__}: {exc}"
                turns.append(turn)
                rec["failure_type"] = "api_error"
                rec["fail"] = turn["api_error"]
                break

            turn["raw_model_output"] = raw_content
            try:
                if assistant_carrier == BATCH_CARRIER_PROVIDER_NATIVE:
                    if not reasoning.strip():
                        raise BatchPlanProtocolError(
                            "provider native reasoning field must be non-empty"
                        )
                    parser = (
                        parse_legacy_batch_plan_action
                        if legacy_action_shape
                        else parse_provider_action
                    )
                    tool, arguments = parser(
                        raw_content,
                        max_batch_calls=max_batch_calls,
                    )
                    authored_reasoning = reasoning.strip()
                    canonical_output = _canonical_output(
                        authored_reasoning,
                        raw_content,
                    )
                else:
                    parser = (
                        parse_legacy_batch_plan_assistant
                        if legacy_action_shape
                        else parse_inline_action
                    )
                    authored_reasoning, tool, arguments = (
                        parser(
                            raw_content,
                            max_batch_calls=max_batch_calls,
                        )
                    )
                    canonical_output = raw_content.strip()
                turn["provider_reasoning_content"] = authored_reasoning
                turn["canonical_model_output"] = canonical_output
                turn["parsed"] = {
                    "tool": tool,
                    "arguments": deepcopy(arguments),
                }

                if tool == TERMINAL_TOOL:
                    terminal_arguments = arguments
                    terminal_call_id = "__answer__"
                    step_id = f"step_{atomic_count + 1}"
                    terminal_projection = None
                    if legacy_action_shape:
                        # Retired column-selecting terminal actions are materialized only for
                        # explicitly named historical reproduction. The active protocol cites its
                        # exact resident result without any hidden execution.
                        score_arguments, terminal_projection = (
                            _materialize_terminal_evidence(
                                h=h,
                                ctx=ctx,
                                arguments=terminal_arguments,
                                created=created,
                                step_id=step_id,
                            )
                        )
                    else:
                        score_arguments = deepcopy(terminal_arguments)
                        submitted_call_count += 1
                    atomic_count += 1
                    correct, pred_sample, gold_sample = score(
                        h,
                        gold_sql,
                        score_arguments,
                        created,
                        denotation_comparison=denotation_comparison,
                    )
                    rec["legal"] = True
                    rec["correct"] = correct
                    rec["pred_sample"] = pred_sample
                    rec["gold_sample"] = gold_sample
                    rec["failure_type"] = None if correct else "wrong_answer"
                    terminal_event = {
                        "call_id": terminal_call_id,
                        "step_id": step_id,
                        "tool": TERMINAL_TOOL,
                        "status": "success",
                        "arguments": deepcopy(terminal_arguments),
                        "resolved_arguments": deepcopy(score_arguments),
                        "resolved_evidence_arguments": deepcopy(score_arguments),
                        "output": {
                            "correct": correct,
                            "pred_sample": pred_sample,
                            "gold_sample": gold_sample,
                        },
                        "environment_state_before": state_before,
                        "environment_state": ctx["environment"].snapshot(),
                    }
                    if terminal_projection is not None:
                        terminal_event["terminal_projection"] = deepcopy(
                            terminal_projection
                        )
                    atomic_events.append(terminal_event)
                    turn["terminal_result"] = deepcopy(terminal_event["output"])
                    turns.append(turn)
                    break

                executable_arguments = arguments
                if prepare_work_action is not None:
                    executable_arguments, work_graph = prepare_work_action(
                        tool,
                        arguments,
                    )
                    turn["work_graph"] = deepcopy(work_graph)
                batch_count += 1
                submitted_call_count += len(executable_arguments["calls"])
                atomic_count, results, events, nonrecoverable = _execute_action_block(
                    h=h,
                    ctx=ctx,
                    arguments=executable_arguments,
                    created=created,
                    atomic_count=atomic_count,
                    model_turn=model_turn_count,
                    batch_index=batch_count,
                    table_output_rows=table_output_rows,
                    error_counts=error_counts,
                    error_events=error_events,
                    prior_bindings=prior_call_bindings,
                    structured_error_feedback=structured_error_feedback,
                    low_friction_interface=low_friction_interface,
                    safe_low_friction_interface=safe_low_friction_interface,
                    interface_resolution_events=interface_resolution_events,
                    validate_call=validate_call,
                    prepare_call_arguments=prepare_call_arguments,
                    lower_call=lower_call,
                    publicize_error=publicize_error,
                )
                terminal_result = next(
                    (
                        result
                        for result in results
                        if result.get("tool") == TERMINAL_TOOL
                    ),
                    None,
                )
                if (
                    terminal_result is not None
                    and terminal_result.get("status") == "success"
                ):
                    score_arguments = terminal_result["terminal_score_arguments"]
                    correct, pred_sample, gold_sample = score(
                        h,
                        gold_sql,
                        score_arguments,
                        created,
                        denotation_comparison=denotation_comparison,
                    )
                    rec["legal"] = True
                    rec["correct"] = correct
                    rec["pred_sample"] = pred_sample
                    rec["gold_sample"] = gold_sample
                    rec["failure_type"] = None if correct else "wrong_answer"
                    scored_output = {
                        "correct": correct,
                        "pred_sample": pred_sample,
                        "gold_sample": gold_sample,
                    }
                    terminal_result["output"] = deepcopy(scored_output)
                    for event in events:
                        if event.get("call_id") == terminal_result.get("call_id"):
                            event["output"] = deepcopy(scored_output)
                            event["resolved_evidence_arguments"] = deepcopy(
                                terminal_result["resolved_evidence_arguments"]
                            )
                            break
                    atomic_events.extend(events)
                    turn["batch_index"] = batch_count
                    turn["batch_results"] = deepcopy(results)
                    turn["terminal_result"] = deepcopy(scored_output)
                    turns.append(turn)
                    break

                atomic_events.extend(events)
                blocked_in_block = sum(
                    result.get("status") == "blocked" for result in results
                )
                root_errors_in_block = sum(
                    result.get("status") == "error" for result in results
                )
                blocked_node_count += blocked_in_block
                observation = render_work_observation(
                    batch_count,
                    results,
                    structured_error_feedback=structured_error_feedback,
                )
                turn["batch_index"] = batch_count
                turn["batch_results"] = deepcopy(results)
                turn["root_error_count"] = root_errors_in_block
                turn["blocked_count"] = blocked_in_block
                turn["observation"] = observation
                turns.append(turn)
                legal_history.append({
                    "assistant": raw_content.strip(),
                    "observation": observation,
                })
                last_error = None
                pending_feedback_recovery = bool(
                    root_errors_in_block or blocked_in_block
                )
                if nonrecoverable:
                    rec["failure_type"] = "nonrecoverable_execution_error"
                    rec["fail"] = "a batch call changed resident state before failing"
                    break
                exhausted = [
                    error_type
                    for error_type, count in error_counts.items()
                    if count >= max_errors_per_type
                ]
                if exhausted:
                    rec["failure_type"] = sorted(exhausted)[0]
                    rec["fail"] = (
                        "aborted after error budget was exhausted: "
                        + ", ".join(
                            f"{name}={error_counts[name]}"
                            for name in sorted(exhausted)
                        )
                    )
                    break
            except Exception as exc:  # noqa: BLE001
                # Every model turn spends one action-block budget unit. A rejected top-level
                # response contains no attempted primitive call, so it does not increment the
                # primitive-action audit counter and is never inserted into legal history.
                state_after = ctx["environment"].snapshot()
                error_type = _error_type(exc)
                if compact_json(state_after) != compact_json(state_before):
                    error_type = "nonrecoverable_execution_error"
                message = str(exc)
                error_counts[error_type] += 1
                event = _error_event(
                    atomic_index=None,
                    model_turn=model_turn_count,
                    batch_index=None,
                    call_id=None,
                    tool=(turn.get("parsed") or {}).get("tool"),
                    arguments=(turn.get("parsed") or {}).get("arguments"),
                    error_type=error_type,
                    message=message,
                    state_before=state_before,
                    state_after=state_after,
                )
                error_events.append(event)
                turn["execution_error_type"] = error_type
                turn["execution_error"] = message
                turn["error_event"] = event
                turns.append(turn)
                last_error = _top_level_error_message(
                    block_action_index=model_turn_count,
                    error_type=error_type,
                    message=message,
                )
                pending_feedback_recovery = True
                if (
                    error_type == "nonrecoverable_execution_error"
                    or error_counts[error_type] >= max_errors_per_type
                ):
                    rec["failure_type"] = error_type
                    rec["fail"] = (
                        f"aborted after {error_counts[error_type]} "
                        f"{error_type} events: {message}"
                    )
                    break
        else:
            rec["failure_type"] = "max_action_blocks"
            rec["fail"] = "max_action_blocks"
    finally:
        try:
            h.conn.close()
        except Exception:  # noqa: BLE001
            pass

    rec["model_turns"] = model_turn_count
    rec["action_blocks"] = model_turn_count
    rec["work_actions"] = batch_count
    rec["executed_action_blocks"] = batch_count
    rec["atomic_actions"] = atomic_count
    rec["submitted_calls"] = submitted_call_count
    rec["blocked_nodes"] = blocked_node_count
    rec["errors"] = len(error_events)
    rec["interface_resolutions"] = len(interface_resolution_events)
    rec["interface_resolution_hist"] = dict(collections.Counter(
        str(event.get("rule"))
        for event in interface_resolution_events
    ))
    rec["error_counts"] = dict(error_counts)
    rec["usage"] = dict(usage)
    rec["final_environment_state"] = ctx["environment"].snapshot()
    rec["elapsed_seconds"] = round(time.time() - started, 3)
    return rec


def read_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                completed.add(str(json.loads(line).get("trajectory_id")))
    return completed


def summarize_records(path: Path) -> dict:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts: collections.Counter = collections.Counter()
    usage: collections.Counter = collections.Counter()
    tool_hist: collections.Counter = collections.Counter()
    status_hist: collections.Counter = collections.Counter()
    total_turns = total_blocks = total_atomic = total_errors = 0
    total_submitted = total_blocked = total_interface_resolutions = 0
    interface_resolution_hist: collections.Counter = collections.Counter()
    fingerprints: collections.Counter = collections.Counter()
    for record in records:
        counts["total"] += 1
        counts["correct" if record.get("correct") else "failed"] += 1
        if record.get("legal"):
            counts["legal"] += 1
        if record.get("failure_type"):
            counts[f"failure:{record['failure_type']}"] += 1
        total_turns += int(record.get("model_turns") or 0)
        total_blocks += int(
            record.get("action_blocks") or record.get("plan_rounds") or 0
        )
        total_atomic += int(record.get("atomic_actions") or 0)
        total_submitted += int(
            record.get("submitted_calls") or record.get("planned_nodes") or 0
        )
        total_blocked += int(record.get("blocked_nodes") or 0)
        total_errors += int(record.get("errors") or 0)
        total_interface_resolutions += int(record.get("interface_resolutions") or 0)
        interface_resolution_hist.update(
            record.get("interface_resolution_hist") or {}
        )
        add_usage(usage, record.get("usage") or {})
        for event in record.get("atomic_events") or []:
            tool_hist[str(event.get("tool"))] += 1
            status_hist[str(event.get("status"))] += 1
        for turn in record.get("turns") or []:
            metadata = turn.get("provider_response_metadata") or {}
            fingerprint = metadata.get("system_fingerprint")
            if fingerprint:
                fingerprints[str(fingerprint)] += 1
    n = len(records)
    return {
        "records": n,
        "counts": dict(counts),
        "accuracy": counts.get("correct", 0) / n if n else None,
        "legal_rate": counts.get("legal", 0) / n if n else None,
        "total_model_turns": total_turns,
        "mean_model_turns": total_turns / n if n else None,
        "total_action_blocks": total_blocks,
        "mean_action_blocks": total_blocks / n if n else None,
        "total_atomic_actions": total_atomic,
        "mean_atomic_actions": total_atomic / n if n else None,
        "total_submitted_calls": total_submitted,
        "mean_submitted_calls": total_submitted / n if n else None,
        "total_blocked_nodes": total_blocked,
        "mean_blocked_nodes": total_blocked / n if n else None,
        "total_process_errors": total_errors,
        "total_interface_resolutions": total_interface_resolutions,
        "mean_interface_resolutions": (
            total_interface_resolutions / n if n else None
        ),
        "interface_resolution_hist": dict(interface_resolution_hist.most_common()),
        "tool_hist": dict(tool_hist.most_common()),
        "atomic_status_hist": dict(status_hist.most_common()),
        "usage_total": dict(usage),
        "provider_system_fingerprints": dict(fingerprints),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--examples-file", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--assistant-carrier",
        choices=BATCH_CARRIERS,
        default=BATCH_CARRIER_PROVIDER_NATIVE,
        help=(
            "provider-native is the DeepSeek split response; think-json-v1 "
            "lets a local/student model emit the complete canonical turn directly"
        ),
    )
    parser.add_argument("--out", required=True, help="all episode records JSONL")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-action-blocks",
        type=int,
        default=DEFAULT_MAX_ACTION_BLOCKS,
        help=(
            "maximum model action blocks; each block costs one budget unit "
            "regardless of its number of primitive calls"
        ),
    )
    parser.add_argument(
        "--max-batch-calls", type=int, default=DEFAULT_MAX_BATCH_CALLS
    )
    parser.add_argument("--history-turns", type=int, default=DEFAULT_HISTORY_TURNS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--trajectory-id",
        action="append",
        default=[],
        help=(
            "evaluate only this exact trajectory id; repeat for a frozen "
            "targeted gate (exact ids override --start/--limit)"
        ),
    )
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument(
        "--max-errors-per-type",
        type=int,
        default=DEFAULT_MAX_ERRORS_PER_TYPE,
    )
    parser.add_argument("--table-output-rows", type=int, default=0)
    parser.add_argument(
        "--structured-error-feedback",
        action="store_true",
        help=(
            "experimental action-block-v9 fact-only feedback on failed blocks; "
            "retained only for artifact reproduction"
        ),
    )
    parser.add_argument(
        "--low-friction-interface",
        action="store_true",
        help=(
            "evaluated action-block-v10 deterministic reference and namespace "
            "resolution, retained for artifact reproduction"
        ),
    )
    parser.add_argument(
        "--safe-low-friction-interface",
        action="store_true",
        help=(
            "experimental action-block-v11: v10 plus literal-column guards and "
            "fact-proven progressive inner-join equivalence"
        ),
    )
    add_denotation_comparison_argument(parser)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if (
        args.assistant_carrier == BATCH_CARRIER_PROVIDER_NATIVE
        and not is_deepseek_split_model(args.model)
    ):
        parser.error(
            "provider-native action-block carrier requires deepseek-v4-flash or "
            "deepseek-v4-pro; use --assistant-carrier think-json-v1 "
            "for a local/student model"
        )
    if (
        args.assistant_carrier == BATCH_CARRIER_INLINE_THINK
        and is_deepseek_split_model(args.model)
    ):
        parser.error(
            "DeepSeek split-response models must use the provider-native action-block carrier"
        )
    if args.denotation_comparison != "bird-set":
        parser.error("new BIRD evaluations must use --denotation-comparison bird-set")
    if args.max_action_blocks < 2:
        parser.error("--max-action-blocks must be at least 2")
    if not 1 <= args.max_batch_calls <= MAX_ACTION_BLOCK_CALLS:
        parser.error(
            f"--max-batch-calls must be between 1 and {MAX_ACTION_BLOCK_CALLS} "
            "for the active action-block protocol"
        )
    selected_ablations = sum(bool(item) for item in (
        args.structured_error_feedback,
        args.low_friction_interface,
        args.safe_low_friction_interface,
    ))
    if selected_ablations > 1:
        parser.error(
            "structured feedback, v10 low-friction, and v11 safe low-friction "
            "are separate ablations and cannot be combined"
        )
    if selected_ablations == 0 and args.history_turns != 4:
        parser.error(
            "active action-block-v35 requires --history-turns 4"
        )

    api_key, base_url = load_api_config()
    if not api_key or not base_url:
        parser.error("api.md must define API_KEY and BASE_URL")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(".manifest.json")
    if out_path.exists() and not args.resume:
        parser.error(f"{out_path} already exists; use a new path or --resume")

    requested_ids = list(dict.fromkeys(args.trajectory_id))
    if requested_ids:
        load_args = deepcopy(args)
        load_args.start = 0
        load_args.limit = None
        examples = load_examples(load_args)
    else:
        examples = load_examples(args)
    if requested_ids:
        requested_id_set = set(requested_ids)
        available_ids = {
            trajectory_id(args.split, index, ex)
            for index, ex in examples
        }
        missing_ids = sorted(requested_id_set - available_ids)
        if missing_ids:
            parser.error(
                "requested trajectory ids are absent from the selected examples: "
                + ", ".join(missing_ids)
            )
        examples = [
            (index, ex)
            for index, ex in examples
            if trajectory_id(args.split, index, ex) in requested_id_set
        ]
    completed = read_completed(out_path) if args.resume else set()
    work = [
        (index, ex)
        for index, ex in examples
        if trajectory_id(args.split, index, ex) not in completed
    ]
    protocol_version = (
        STRUCTURED_ERROR_FEEDBACK_PROTOCOL_VERSION
        if args.structured_error_feedback
        else (
            SAFE_LOW_FRICTION_INTERFACE_PROTOCOL_VERSION
            if args.safe_low_friction_interface
            else (
                LOW_FRICTION_INTERFACE_PROTOCOL_VERSION
                if args.low_friction_interface
                else SIMPLE_SCALAR_CELL_PROTOCOL_VERSION
            )
        )
    )
    system_prompt = build_batch_plan_system_prompt(
        args.max_batch_calls,
        assistant_carrier=args.assistant_carrier,
        protocol_version=protocol_version,
    )
    protocol_hash = batch_plan_protocol_hash(
        system_prompt,
        args.max_batch_calls,
        protocol_version=protocol_version,
    )
    started = time.time()

    def process(item: tuple[int, dict]) -> dict:
        index, ex = item
        try:
            return run_episode(
                example_index=index,
                ex=ex,
                split=args.split,
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                system_prompt=system_prompt,
                protocol_hash=protocol_hash,
                max_action_blocks=args.max_action_blocks,
                max_batch_calls=args.max_batch_calls,
                max_tokens=args.max_tokens,
                api_retries=args.api_retries,
                api_timeout=args.api_timeout,
                max_errors_per_type=args.max_errors_per_type,
                table_output_rows=args.table_output_rows,
                history_turns=args.history_turns,
                denotation_comparison=args.denotation_comparison,
                protocol_version=protocol_version,
                structured_error_feedback=args.structured_error_feedback,
                low_friction_interface=(
                    args.low_friction_interface
                    or args.safe_low_friction_interface
                ),
                safe_low_friction_interface=args.safe_low_friction_interface,
                assistant_carrier=args.assistant_carrier,
                build_messages=(
                    build_batch_plan_messages
                    if selected_ablations
                    else build_sequential_messages
                ),
                render_work_observation=(
                    render_batch_observation
                    if selected_ablations
                    else render_sequential_observation
                ),
                validate_call=(
                    validate_atomic_call
                    if selected_ablations
                    else validate_sequential_atomic_call
                ),
                prepare_call_arguments=(
                    None
                    if selected_ablations
                    else prepare_simple_scalar_cell_arguments
                ),
                lower_call=(
                    None
                    if selected_ablations
                    else lower_sequential_atomic_call
                ),
                publicize_error=(
                    None
                    if selected_ablations
                    else publicize_simple_scalar_error
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "tool_scheme": ACTION_BLOCK_TOOL_SCHEME,
                "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
                "assistant_carrier": args.assistant_carrier,
                "example_index": index,
                "trajectory_id": trajectory_id(args.split, index, ex),
                "db_id": ex.get("db_id"),
                "question": ex.get("question"),
                "correct": False,
                "legal": False,
                "model_turns": 0,
                "action_blocks": 0,
                "executed_action_blocks": 0,
                "atomic_actions": 0,
                "submitted_calls": 0,
                "blocked_nodes": 0,
                "errors": 0,
                "failure_type": "runner_error",
                "fail": f"{type(exc).__name__}: {exc}",
                "turns": [],
                "atomic_events": [],
                "error_events": [],
                "interface_resolution_events": [],
                "interface_resolutions": 0,
                "interface_resolution_hist": {},
                "usage": {},
                "denotation_comparison": args.denotation_comparison,
                "terminal_answer_contract": TERMINAL_ANSWER_CONTRACT,
                "protocol_version": protocol_version,
                "protocol_hash": protocol_hash,
                "sft_export_eligible": False,
            }

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(process, item) for item in work]
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            append_jsonl(out_path, record)
            status = "OK " if record.get("correct") else "ERR"
            print(
                f"[{done}/{len(work)}] {status} turns={record.get('model_turns')} "
                f"blocks={record.get('action_blocks')} atomic={record.get('atomic_actions')} "
                f"blocked={record.get('blocked_nodes')} "
                f"errors={record.get('errors')} type={record.get('failure_type')} "
                f"{record.get('trajectory_id')}",
                flush=True,
            )

    summary = summarize_records(out_path)
    manifest = {
        "generator": "src/tool_modules/action_block/evaluator.py",
        "tool_scheme": ACTION_BLOCK_TOOL_SCHEME,
        "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
        "assistant_carrier": args.assistant_carrier,
        "method": (
            "ordered_variable_width_action_block_"
            + (
                "and_error_only_structured_facts"
                if args.structured_error_feedback
                else (
                    "and_safe_deterministic_low_friction_resolution"
                    if args.safe_low_friction_interface
                    else (
                        "and_deterministic_low_friction_resolution"
                        if args.low_friction_interface
                        else "v35_one_edge_join_and_simple_scalar_cells"
                    )
                )
            )
        ),
        "model": args.model,
        "split": args.split,
        "examples_file": args.examples_file,
        "source_count": len(examples),
        "requested_trajectory_ids": requested_ids,
        "attempted_this_run": len(work),
        "resume": args.resume,
        "output": str(out_path),
        "protocol_version": protocol_version,
        "protocol_hash": protocol_hash,
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "system_prompt_characters": len(system_prompt),
        "system_prompt": system_prompt,
        "max_action_blocks": args.max_action_blocks,
        "max_batch_calls": args.max_batch_calls,
        "max_errors_per_type": args.max_errors_per_type,
        "history_turns": args.history_turns,
        "rolling_observation_style": "full-atomic-results-plus-resident-state",
        "max_tokens": args.max_tokens,
        "workers": max(1, args.workers),
        "table_output_rows": args.table_output_rows,
        "structured_error_feedback": args.structured_error_feedback,
        "low_friction_interface": args.low_friction_interface,
        "safe_low_friction_interface": args.safe_low_friction_interface,
        "ordered_execution": not (
            args.low_friction_interface or args.safe_low_friction_interface
        ),
        "automatic_argument_rewrites": bool(
            args.low_friction_interface or args.safe_low_friction_interface
        ),
        "deterministic_public_lowering": (
            None
            if selected_ablations
            else {
                "join": "join_tables with exactly one edge",
                "scalar_cell": (
                    '"$id.column" or "step_id.column" to verified '
                    "one-row value_ref+column"
                ),
                "semantic_guessing": False,
            }
        ),
        "temperature": 0,
        "thinking": (
            "enabled"
            if args.assistant_carrier == BATCH_CARRIER_PROVIDER_NATIVE
            else "inline"
        ),
        "reasoning_effort": (
            "high"
            if args.assistant_carrier == BATCH_CARRIER_PROVIDER_NATIVE
            else None
        ),
        "deepseek_carrier": (
            DEEPSEEK_CARRIER_JSON_OUTPUT
            if args.assistant_carrier == BATCH_CARRIER_PROVIDER_NATIVE
            else None
        ),
        "provider_request_options": provider_request_options(
            args.model, carrier=DEEPSEEK_CARRIER_JSON_OUTPUT
        ),
        "denotation_comparison": args.denotation_comparison,
        "terminal_answer_contract": TERMINAL_ANSWER_CONTRACT,
        "sft_export_eligible": False,
        "gold_sql_visible_to_model": False,
        "summary": summary,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
