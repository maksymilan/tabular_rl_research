#!/usr/bin/env python3
"""Evaluate the hybrid action-block interface.

Each block executes multiple existing atomic tools. Local references induce dependencies; root
errors are attempted failures while dependency-blocked descendants are not executed or counted as
additional process errors. The terminal action selects exact answer columns from a grounded
resident table without changing its rows. Gold SQL is hidden from the model and used only for
terminal scoring.
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "sft"))

from batch_plan_protocol import (  # noqa: E402
    BATCH_PLAN_PROTOCOL_VERSION,
    LOCAL_COLUMN_REF_RE,
    LOCAL_REF_RE,
    LOW_FRICTION_INTERFACE_PROTOCOL_VERSION,
    SAFE_LOW_FRICTION_INTERFACE_PROTOCOL_VERSION,
    STRUCTURED_ERROR_FEEDBACK_PROTOCOL_VERSION,
    TERMINAL_TOOL,
    BatchPlanProtocolError,
    LocalReferenceError,
    batch_plan_protocol_hash,
    build_batch_plan_messages,
    build_batch_plan_system_prompt,
    local_reference_ids,
    parse_batch_plan_action,
    render_batch_observation,
    resolve_local_references,
    validate_atomic_call,
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
from protocol import ProtocolError  # noqa: E402
from rollout import (  # noqa: E402
    ContextOverflowError,
    execute_tool,
    format_tool_error,
    new_ctx,
    overview,
    score,
    task_db_path,
    task_gold_sql,
)


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_ATOMIC_ACTIONS = 30
DEFAULT_MAX_MODEL_TURNS = 30
DEFAULT_MAX_BATCH_CALLS = 8
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
    atomic_index: int,
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
        "step_id": f"step_{atomic_index}",
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
    atomic_index: int,
    error_type: str,
    message: str,
) -> dict:
    return {
        "step_id": f"step_{atomic_index}",
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


def _execute_plan_action(
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
) -> tuple[int, list[dict], list[dict], bool]:
    """Best-effort execute a block while separating root errors from blocked descendants."""
    low_friction_interface = (
        low_friction_interface or safe_low_friction_interface
    )
    results: list[dict] = []
    atomic_events: list[dict] = []
    nonrecoverable = False
    calls = arguments["calls"]
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
                "step_id": None,
                "table": None,
                "columns": None,
                "root_causes": root_causes,
            }
            results.append(blocked)
            atomic_events.append(deepcopy(blocked))
            continue

        atomic_count += 1
        step_id = f"step_{atomic_count}"
        state_before = ctx["environment"].snapshot()
        resolved_args = None
        interface_resolutions: list[dict] = []
        try:
            reference_args = original_args
            resolution_bindings = bindings
            resolution_declared_ids = declared_ids
            if low_friction_interface:
                if safe_low_friction_interface:
                    _reject_local_column_literal_references(original_args)
                reference_args = _resolve_implicit_local_references(
                    tool,
                    reference_args,
                    bindings=bindings,
                    declared_ids=declared_ids,
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
            validate_atomic_call(tool, resolved_args)
            output, table = execute_tool(
                h,
                tool,
                resolved_args,
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
            if interface_resolutions:
                result["interface_resolutions"] = deepcopy(interface_resolutions)
            bindings[call_id] = {
                "status": "success",
                "step_id": step_id,
                "table": table,
                "columns": deepcopy(
                    output.get("columns")
                    if isinstance(output, dict)
                    else None
                ),
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
                message = format_tool_error(exc, h, tool, original_args)
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
                "step_id": step_id,
                "table": None,
                "columns": None,
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
                "block_index": batch_index,
                "step_id": binding.get("step_id"),
                "table": binding.get("table"),
                "columns": deepcopy(binding.get("columns")),
            }
    return atomic_count, results, atomic_events, nonrecoverable


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
    max_atomic_actions: int,
    max_model_turns: int,
    max_batch_calls: int,
    max_tokens: int,
    api_retries: int,
    api_timeout: int,
    max_errors_per_type: int,
    table_output_rows: int,
    history_turns: int,
    denotation_comparison: str,
    protocol_version: str = BATCH_PLAN_PROTOCOL_VERSION,
    structured_error_feedback: bool = False,
    low_friction_interface: bool = False,
    safe_low_friction_interface: bool = False,
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
    planned_node_count = 0
    blocked_node_count = 0
    prior_call_bindings: dict[str, dict] = {}
    started = time.time()

    rec = {
        "example_index": example_index,
        "trajectory_id": trajectory_id(split, example_index, ex),
        "db_id": ex["db_id"],
        "question": ex["question"],
        "gold_sql": gold_sql,
        "difficulty": (ex.get("metadata") or {}).get("difficulty_proxy"),
        "correct": False,
        "legal": False,
        "model_turns": 0,
        "plan_rounds": 0,
        "action_blocks": 0,
        "atomic_actions": 0,
        "planned_nodes": 0,
        "blocked_nodes": 0,
        "errors": 0,
        "failure_type": None,
        "fail": None,
        "turns": turns,
        "atomic_events": atomic_events,
        "error_events": error_events,
        "interface_resolution_events": interface_resolution_events,
        "denotation_comparison": denotation_comparison,
        "protocol_version": protocol_version,
        "protocol_hash": protocol_hash,
        "sft_export_eligible": False,
    }

    try:
        while (
            model_turn_count < max_model_turns
            and atomic_count < max_atomic_actions
        ):
            model_turn_count += 1
            state_before = ctx["environment"].snapshot()
            model_input = build_batch_plan_messages(
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
            turn["provider_reasoning_content"] = reasoning
            turn["canonical_model_output"] = _canonical_output(
                reasoning, raw_content
            )
            try:
                if not reasoning.strip():
                    raise BatchPlanProtocolError(
                        "provider native reasoning field must be non-empty"
                    )
                tool, arguments = parse_batch_plan_action(
                    raw_content,
                    max_batch_calls=max_batch_calls,
                )
                turn["parsed"] = {
                    "tool": tool,
                    "arguments": deepcopy(arguments),
                }

                if tool == TERMINAL_TOOL:
                    step_id = f"step_{atomic_count + 1}"
                    score_arguments, terminal_projection = (
                        _materialize_terminal_evidence(
                            h=h,
                            ctx=ctx,
                            arguments=arguments,
                            created=created,
                            step_id=step_id,
                        )
                    )
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
                        "call_id": "__answer__",
                        "step_id": step_id,
                        "tool": tool,
                        "status": "success",
                        "arguments": deepcopy(arguments),
                        "resolved_arguments": deepcopy(score_arguments),
                        "terminal_projection": deepcopy(terminal_projection),
                        "output": {
                            "correct": correct,
                            "pred_sample": pred_sample,
                            "gold_sample": gold_sample,
                        },
                        "environment_state_before": state_before,
                        "environment_state": ctx["environment"].snapshot(),
                    }
                    atomic_events.append(terminal_event)
                    turn["terminal_result"] = deepcopy(terminal_event["output"])
                    turns.append(turn)
                    break

                proposed = len(arguments["calls"])
                remaining = max_atomic_actions - atomic_count
                if proposed > remaining:
                    raise BatchPlanProtocolError(
                        f"action_block plans {proposed} atomic calls, but only {remaining} "
                        "primitive-action budget slots remain"
                    )

                batch_count += 1
                planned_node_count += len(arguments["calls"])
                atomic_count, results, events, nonrecoverable = _execute_plan_action(
                    h=h,
                    ctx=ctx,
                    arguments=arguments,
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
                )
                atomic_events.extend(events)
                blocked_in_block = sum(
                    result.get("status") == "blocked" for result in results
                )
                blocked_node_count += blocked_in_block
                observation = render_batch_observation(
                    batch_count,
                    results,
                    structured_error_feedback=structured_error_feedback,
                )
                turn["batch_index"] = batch_count
                turn["batch_results"] = deepcopy(results)
                turn["root_error_count"] = sum(
                    result.get("status") == "error" for result in results
                )
                turn["blocked_count"] = blocked_in_block
                turn["observation"] = observation
                turns.append(turn)
                legal_history.append({
                    "assistant": raw_content.strip(),
                    "observation": observation,
                })
                last_error = None
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
                # A rejected top-level response spends one atomic action, as in the production
                # protocol, and is never inserted into legal assistant history.
                atomic_count += 1
                state_after = ctx["environment"].snapshot()
                error_type = _error_type(exc)
                if compact_json(state_after) != compact_json(state_before):
                    error_type = "nonrecoverable_execution_error"
                message = str(exc)
                error_counts[error_type] += 1
                event = _error_event(
                    atomic_index=atomic_count,
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
                    atomic_index=atomic_count,
                    error_type=error_type,
                    message=message,
                )
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
            if atomic_count >= max_atomic_actions:
                rec["failure_type"] = "max_atomic_actions"
                rec["fail"] = "max_atomic_actions"
            else:
                rec["failure_type"] = "max_model_turns"
                rec["fail"] = "max_model_turns"
    finally:
        try:
            h.conn.close()
        except Exception:  # noqa: BLE001
            pass

    rec["model_turns"] = model_turn_count
    rec["plan_rounds"] = batch_count
    rec["action_blocks"] = batch_count
    rec["atomic_actions"] = atomic_count
    rec["planned_nodes"] = planned_node_count
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
    total_planned = total_blocked = total_interface_resolutions = 0
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
        total_planned += int(record.get("planned_nodes") or 0)
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
        "total_planned_nodes": total_planned,
        "mean_planned_nodes": total_planned / n if n else None,
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
    parser.add_argument("--out", required=True, help="all episode records JSONL")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--max-atomic-actions", type=int, default=DEFAULT_MAX_ATOMIC_ACTIONS
    )
    parser.add_argument(
        "--max-model-turns", type=int, default=DEFAULT_MAX_MODEL_TURNS
    )
    parser.add_argument(
        "--max-batch-calls", type=int, default=DEFAULT_MAX_BATCH_CALLS
    )
    parser.add_argument("--history-turns", type=int, default=DEFAULT_HISTORY_TURNS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
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
            "the action-block-v4 default remains unchanged"
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

    if not is_deepseek_split_model(args.model):
        parser.error("this experiment currently requires deepseek-v4-flash or deepseek-v4-pro")
    if args.denotation_comparison != "bird-set":
        parser.error("new BIRD evaluations must use --denotation-comparison bird-set")
    if args.max_atomic_actions < 2:
        parser.error("--max-atomic-actions must be at least 2")
    if args.max_batch_calls < 1:
        parser.error("--max-batch-calls must be positive")
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

    api_key, base_url = load_api_config()
    if not api_key or not base_url:
        parser.error("api.md must define API_KEY and BASE_URL")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = out_path.with_suffix(".manifest.json")
    if out_path.exists() and not args.resume:
        parser.error(f"{out_path} already exists; use a new path or --resume")

    examples = load_examples(args)
    completed = read_completed(out_path) if args.resume else set()
    work = [
        (index, ex)
        for index, ex in examples
        if trajectory_id(args.split, index, ex) not in completed
    ]
    system_prompt = build_batch_plan_system_prompt(args.max_batch_calls)
    protocol_version = (
        STRUCTURED_ERROR_FEEDBACK_PROTOCOL_VERSION
        if args.structured_error_feedback
        else (
            SAFE_LOW_FRICTION_INTERFACE_PROTOCOL_VERSION
            if args.safe_low_friction_interface
            else (
                LOW_FRICTION_INTERFACE_PROTOCOL_VERSION
                if args.low_friction_interface
                else BATCH_PLAN_PROTOCOL_VERSION
            )
        )
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
                max_atomic_actions=args.max_atomic_actions,
                max_model_turns=args.max_model_turns,
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
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "example_index": index,
                "trajectory_id": trajectory_id(args.split, index, ex),
                "db_id": ex.get("db_id"),
                "question": ex.get("question"),
                "correct": False,
                "legal": False,
                "model_turns": 0,
                "plan_rounds": 0,
                "action_blocks": 0,
                "atomic_actions": 0,
                "planned_nodes": 0,
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
        "generator": "src/eval/evaluate_batch_plan.py",
        "method": (
            "hybrid_action_block_with_branch_local_recovery_terminal_column_selection_"
            + (
                "and_error_only_structured_facts"
                if args.structured_error_feedback
                else (
                    "and_safe_deterministic_low_friction_resolution"
                    if args.safe_low_friction_interface
                    else (
                        "and_deterministic_low_friction_resolution"
                        if args.low_friction_interface
                        else "v4_default"
                    )
                )
            )
        ),
        "model": args.model,
        "split": args.split,
        "examples_file": args.examples_file,
        "source_count": len(examples),
        "attempted_this_run": len(work),
        "resume": args.resume,
        "output": str(out_path),
        "protocol_version": protocol_version,
        "protocol_hash": protocol_hash,
        "system_prompt_sha256": hashlib.sha256(
            system_prompt.encode("utf-8")
        ).hexdigest(),
        "system_prompt_characters": len(system_prompt),
        "max_atomic_actions": args.max_atomic_actions,
        "max_model_turns": args.max_model_turns,
        "max_batch_calls": args.max_batch_calls,
        "max_errors_per_type": args.max_errors_per_type,
        "history_turns": args.history_turns,
        "max_tokens": args.max_tokens,
        "workers": max(1, args.workers),
        "table_output_rows": args.table_output_rows,
        "structured_error_feedback": args.structured_error_feedback,
        "low_friction_interface": args.low_friction_interface,
        "safe_low_friction_interface": args.safe_low_friction_interface,
        "temperature": 0,
        "thinking": "enabled",
        "reasoning_effort": "high",
        "deepseek_carrier": DEEPSEEK_CARRIER_JSON_OUTPUT,
        "provider_request_options": provider_request_options(
            args.model, carrier=DEEPSEEK_CARRIER_JSON_OUTPUT
        ),
        "denotation_comparison": args.denotation_comparison,
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
