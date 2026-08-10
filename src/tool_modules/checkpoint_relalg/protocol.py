#!/usr/bin/env python3
"""Public contract for the checkpointed relational-algebra tool module.

This module deliberately owns its schema compiler and structural validator.  The
``TOOL_DEFINITIONS`` mapping is the only model-visible tool source: provider tools,
runtime validation, hashes, and capability manifests are all derived from it.
Environment-aware validation and execution live outside this module.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .checkpoint_store import (
    CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
    CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1,
    CHECKPOINT_GOAL_POLICY_VERSION,
    CHECKPOINT_MILESTONE_PRODUCER_TOOLS,
    FIRST_COMMIT_MIN_MILESTONE_PRODUCERS,
    LATER_COMMIT_MIN_MILESTONE_PRODUCERS,
    MAX_CHECKPOINTS,
    normalize_checkpoint_commit_eligibility_policy,
)


PROTOCOL_VERSION = "checkpoint-relalg-v1"
SCHEME = "checkpoint-relalg"
TOOL_SCHEME = SCHEME
MODES = ("direct", "atomic", "hybrid")
MAX_EXPRESSION_DEPTH = 5
MAX_AST_DEPTH = MAX_EXPRESSION_DEPTH
NATIVE_ASSISTANT_CARRIER = "provider-native-single-tool-call-v1"
TEXT_JSON_ASSISTANT_CARRIER = "provider-thinking-raw-text-json-single-action-v1"
CARRIER_NATIVE_TOOL_CALLS = "native-tool-calls"
CARRIER_TEXT_JSON = "text-json"
CARRIERS = (CARRIER_NATIVE_TOOL_CALLS, CARRIER_TEXT_JSON)
DEFAULT_CARRIER = CARRIER_NATIVE_TOOL_CALLS
CARRIER_ABLATION_PROTOCOL_VERSION = "checkpoint-relalg-carrier-ab-v1"
CARRIER_POLICY_VERSION = "checkpoint-relalg-carrier-policy-v1"
ADMISSION_STATUS = "diagnostic-only"
BACKEND = "sqlite"
DIALECT = "sqlite"
ENVIRONMENT_RENDERER_VERSION = "checkpoint-relalg-environment-renderer-v1"
CHECKPOINT_POLICY_VERSION = "checkpoint-relalg-semantic-checkpoint-v2-distinct-goals"
CHECKPOINT_GUIDANCE_PROFILE_STANDARD = "adaptive-v1"
CHECKPOINT_GUIDANCE_PROFILE_STRESS = "checkpoint-stress-v1"
CHECKPOINT_GUIDANCE_PROFILE_RESTORE_TRIGGER = "restore-trigger-v1"
CHECKPOINT_GUIDANCE_PROFILE_RESTORE_TARGET = "restore-target-v2"
CHECKPOINT_GUIDANCE_PROFILE_RESTORE_PROBE = "restore-probe-v3"
CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE = "semantic-milestone-v1"
CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V2 = "semantic-milestone-v2"
CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V3 = "semantic-milestone-v3"
CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V4 = "semantic-milestone-v4"
CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V5 = "semantic-milestone-v5"
CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V6 = "semantic-milestone-v6"
CHECKPOINT_GUIDANCE_PROFILES = (
    CHECKPOINT_GUIDANCE_PROFILE_STANDARD,
    CHECKPOINT_GUIDANCE_PROFILE_STRESS,
    CHECKPOINT_GUIDANCE_PROFILE_RESTORE_TRIGGER,
    CHECKPOINT_GUIDANCE_PROFILE_RESTORE_TARGET,
    CHECKPOINT_GUIDANCE_PROFILE_RESTORE_PROBE,
    CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE,
    CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V2,
    CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V3,
    CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V4,
    CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V5,
    CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V6,
)
DEFAULT_CHECKPOINT_GUIDANCE_PROFILE = CHECKPOINT_GUIDANCE_PROFILE_STANDARD
EXECUTOR_VERSION = "checkpoint-relalg-sqlite-executor-v1"
ATOMIC_OPERATOR_PROFILE_MICRO = "micro-v1"
ATOMIC_OPERATOR_PROFILE_SEMANTIC = "semantic-v2"
ATOMIC_OPERATOR_PROFILES = (
    ATOMIC_OPERATOR_PROFILE_MICRO,
    ATOMIC_OPERATOR_PROFILE_SEMANTIC,
)
DEFAULT_ATOMIC_OPERATOR_PROFILE = ATOMIC_OPERATOR_PROFILE_MICRO

CANONICAL_TYPES = (
    "NULL",
    "BOOLEAN",
    "INTEGER",
    "REAL",
    "TEXT",
    "DATE",
    "DATETIME",
    "BLOB",
)
COMPARISON_OPERATORS = ("=", "!=", ">", ">=", "<", "<=")
BINARY_EXPRESSION_OPERATORS = (
    "add",
    "subtract",
    "multiply",
    "divide",
    "modulo",
    "date_diff_days",
)
UNARY_EXPRESSION_OPERATORS = (
    "abs",
    "lower",
    "upper",
    "trim",
    "length",
    "extract_year",
    "extract_month",
    "extract_day",
)
VARIABLE_EXPRESSION_ARITY = {
    "round": (1, 2),
    "concat": (2, 8),
    "coalesce": (2, 8),
}
PREDICATE_OPERATORS = (
    *COMPARISON_OPERATORS,
    "and",
    "or",
    "not",
    "between",
    "in",
    "not_in",
    "like",
    "not_like",
    "contains",
    "not_contains",
    "is_null",
    "is_not_null",
)


class ProtocolValidationError(ValueError):
    """A provider-visible call does not satisfy the public structural contract."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_arguments",
        path: str = "$",
    ) -> None:
        super().__init__(f"{path}: {message}")
        self.code = code
        self.path = path
        self.message = message


@dataclass(frozen=True)
class ToolDefinition:
    """One compact public tool definition."""

    description: str
    parameters: Mapping[str, Any]
    capabilities: frozenset[str]


def _object(
    properties: Mapping[str, Any],
    *,
    required: Sequence[str] = (),
    one_of: Sequence[Mapping[str, Any]] = (),
    all_of: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": False,
    }
    if required:
        schema["required"] = list(required)
    if one_of:
        schema["oneOf"] = list(one_of)
    if all_of:
        schema["allOf"] = list(all_of)
    return schema


def _array(
    items: Mapping[str, Any],
    *,
    minimum: int | None = None,
    maximum: int | None = None,
    unique: bool = False,
    default: Any | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": dict(items)}
    if minimum is not None:
        schema["minItems"] = minimum
    if maximum is not None:
        schema["maxItems"] = maximum
    if unique:
        schema["uniqueItems"] = True
    if default is not None:
        schema["default"] = default
    return schema


NONEMPTY_STRING: dict[str, Any] = {"type": "string", "minLength": 1}
NONBLANK_STRING: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "pattern": r"[\s\S]*\S[\s\S]*",
}
IDENTIFIER: dict[str, Any] = {
    "type": "string",
    "pattern": r"^[A-Za-z_][A-Za-z0-9_]*$",
}
OUTPUT_IDENTIFIER: dict[str, Any] = {
    "type": "string",
    # Double-underscore names are reserved for Harness physical metadata.
    # Keeping this rule provider-visible prevents late materialization drift.
    "pattern": r"^(?!__)[A-Za-z_][A-Za-z0-9_]*$",
}
COLUMN_NAME = NONEMPTY_STRING
TABLE_NAME = NONEMPTY_STRING
LITERAL_VALUE: dict[str, Any] = {
    "type": ["null", "boolean", "number", "string"],
}


def _expression_function_schema(
    operators: Sequence[str],
    minimum: int,
    maximum: int,
) -> dict[str, Any]:
    return _object(
        {
            "op": {"type": "string", "enum": list(operators)},
            "args": _array(
                {"$ref": "#/$defs/expression"},
                minimum=minimum,
                maximum=maximum,
            ),
        },
        required=("op", "args"),
    )


SHARED_DEFS: dict[str, Any] = {
    "column_expression": _object(
        {"column": COLUMN_NAME},
        required=("column",),
    ),
    "literal_expression": _object(
        {"value": LITERAL_VALUE},
        required=("value",),
    ),
}

SHARED_DEFS["function_expression"] = {
    "oneOf": [
        _expression_function_schema(BINARY_EXPRESSION_OPERATORS, 2, 2),
        _expression_function_schema(UNARY_EXPRESSION_OPERATORS, 1, 1),
        _expression_function_schema(("round",), 1, 2),
        _expression_function_schema(("concat", "coalesce"), 2, 8),
    ]
}
SHARED_DEFS["cast_expression"] = _object(
    {
        "op": {"const": "cast"},
        "args": _array({"$ref": "#/$defs/expression"}, minimum=1, maximum=1),
        "to": {"type": "string", "enum": list(CANONICAL_TYPES)},
    },
    required=("op", "args", "to"),
)
SHARED_DEFS["case_branch"] = _object(
    {
        "when": {"$ref": "#/$defs/predicate"},
        "then": {"$ref": "#/$defs/expression"},
    },
    required=("when", "then"),
)
SHARED_DEFS["case_expression"] = _object(
    {
        "op": {"const": "case_when"},
        "branches": _array({"$ref": "#/$defs/case_branch"}, minimum=1),
        "else": {"$ref": "#/$defs/expression"},
    },
    required=("op", "branches", "else"),
)
SHARED_DEFS["computed_expression"] = {
    "oneOf": [
        {"$ref": "#/$defs/function_expression"},
        {"$ref": "#/$defs/cast_expression"},
        {"$ref": "#/$defs/case_expression"},
    ]
}
SHARED_DEFS["expression"] = {
    "oneOf": [
        {"$ref": "#/$defs/column_expression"},
        {"$ref": "#/$defs/literal_expression"},
        {"$ref": "#/$defs/computed_expression"},
    ]
}

_PREDICATE_VARIANTS = [
    _object(
        {
            "op": {"type": "string", "enum": list(COMPARISON_OPERATORS)},
            "left": {"$ref": "#/$defs/expression"},
            "right": {"$ref": "#/$defs/expression"},
        },
        required=("op", "left", "right"),
    ),
    _object(
        {
            "op": {"type": "string", "enum": ["and", "or"]},
            "args": _array({"$ref": "#/$defs/predicate"}, minimum=2),
        },
        required=("op", "args"),
    ),
    _object(
        {"op": {"const": "not"}, "arg": {"$ref": "#/$defs/predicate"}},
        required=("op", "arg"),
    ),
    _object(
        {
            "op": {"const": "between"},
            "value": {"$ref": "#/$defs/expression"},
            "lower": {"$ref": "#/$defs/expression"},
            "upper": {"$ref": "#/$defs/expression"},
        },
        required=("op", "value", "lower", "upper"),
    ),
    _object(
        {
            "op": {"type": "string", "enum": ["in", "not_in"]},
            "value": {"$ref": "#/$defs/expression"},
            "values": _array(
                {"$ref": "#/$defs/literal_expression"},
                minimum=1,
                maximum=50,
            ),
        },
        required=("op", "value", "values"),
    ),
    _object(
        {
            "op": {
                "type": "string",
                "enum": ["like", "not_like", "contains", "not_contains"],
            },
            "value": {"$ref": "#/$defs/expression"},
            "pattern": {"$ref": "#/$defs/expression"},
        },
        required=("op", "value", "pattern"),
    ),
    _object(
        {
            "op": {"type": "string", "enum": ["is_null", "is_not_null"]},
            "value": {"$ref": "#/$defs/expression"},
        },
        required=("op", "value"),
    ),
]
SHARED_DEFS["predicate"] = {"oneOf": _PREDICATE_VARIANTS}


def _impossible_schema() -> dict[str, Any]:
    return {"not": {}}


def _depth_ref(kind: str, depth: int) -> dict[str, str]:
    return {"$ref": f"#/$defs/{kind}_d{depth}"}


def _finite_expression_schema(depth: int) -> dict[str, Any]:
    leaves = [
        {"$ref": "#/$defs/column_expression"},
        {"$ref": "#/$defs/literal_expression"},
    ]
    if depth >= MAX_EXPRESSION_DEPTH:
        return {"oneOf": leaves}
    return {"oneOf": [*leaves, _depth_ref("computed_expression", depth)]}


def _finite_computed_expression_schema(depth: int) -> dict[str, Any]:
    if depth >= MAX_EXPRESSION_DEPTH:
        return _impossible_schema()
    child_expression = _depth_ref("expression", depth + 1)

    def function_variant(operators: Sequence[str], minimum: int, maximum: int) -> dict[str, Any]:
        return _object(
            {
                "op": {"type": "string", "enum": list(operators)},
                "args": _array(
                    child_expression,
                    minimum=minimum,
                    maximum=maximum,
                ),
            },
            required=("op", "args"),
        )

    return {
        "oneOf": [
            function_variant(BINARY_EXPRESSION_OPERATORS, 2, 2),
            function_variant(UNARY_EXPRESSION_OPERATORS, 1, 1),
            function_variant(("round",), 1, 2),
            function_variant(("concat", "coalesce"), 2, 8),
            _object(
                {
                    "op": {"const": "cast"},
                    "args": _array(child_expression, minimum=1, maximum=1),
                    "to": {"type": "string", "enum": list(CANONICAL_TYPES)},
                },
                required=("op", "args", "to"),
            ),
            _object(
                {
                    "op": {"const": "case_when"},
                    "branches": _array(
                        _object(
                            {
                                "when": _depth_ref("predicate", depth + 1),
                                "then": child_expression,
                            },
                            required=("when", "then"),
                        ),
                        minimum=1,
                    ),
                    "else": child_expression,
                },
                required=("op", "branches", "else"),
            ),
        ]
    }


def _finite_predicate_schema(depth: int) -> dict[str, Any]:
    if depth >= MAX_EXPRESSION_DEPTH:
        return _impossible_schema()
    child_expression = _depth_ref("expression", depth + 1)
    child_predicate = _depth_ref("predicate", depth + 1)
    return {
        "oneOf": [
            _object(
                {
                    "op": {"type": "string", "enum": list(COMPARISON_OPERATORS)},
                    "left": child_expression,
                    "right": child_expression,
                },
                required=("op", "left", "right"),
            ),
            _object(
                {
                    "op": {"type": "string", "enum": ["and", "or"]},
                    "args": _array(child_predicate, minimum=2),
                },
                required=("op", "args"),
            ),
            _object(
                {"op": {"const": "not"}, "arg": child_predicate},
                required=("op", "arg"),
            ),
            _object(
                {
                    "op": {"const": "between"},
                    "value": child_expression,
                    "lower": child_expression,
                    "upper": child_expression,
                },
                required=("op", "value", "lower", "upper"),
            ),
            _object(
                {
                    "op": {"type": "string", "enum": ["in", "not_in"]},
                    "value": child_expression,
                    "values": _array(
                        deepcopy(SHARED_DEFS["literal_expression"]),
                        minimum=1,
                        maximum=50,
                    ),
                },
                required=("op", "value", "values"),
            ),
            _object(
                {
                    "op": {
                        "type": "string",
                        "enum": ["like", "not_like", "contains", "not_contains"],
                    },
                    "value": child_expression,
                    "pattern": child_expression,
                },
                required=("op", "value", "pattern"),
            ),
            _object(
                {
                    "op": {"type": "string", "enum": ["is_null", "is_not_null"]},
                    "value": child_expression,
                },
                required=("op", "value"),
            ),
        ]
    }


def _expand_finite_ast_refs(value: Any) -> Any:
    if isinstance(value, Mapping):
        ref = value.get("$ref")
        if isinstance(ref, str):
            replacements = {
                "#/$defs/column_expression": {"$ref": "#/$defs/column_expression"},
                "#/$defs/literal_expression": {"$ref": "#/$defs/literal_expression"},
                "#/$defs/expression": _depth_ref("expression", 1),
                "#/$defs/computed_expression": _depth_ref("computed_expression", 1),
                "#/$defs/predicate": _depth_ref("predicate", 1),
            }
            if ref in replacements:
                return replacements[ref]
        return {str(key): _expand_finite_ast_refs(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_finite_ast_refs(item) for item in value]
    return deepcopy(value)


def _finite_ast_defs() -> dict[str, Any]:
    definitions: dict[str, Any] = {
        "column_expression": deepcopy(SHARED_DEFS["column_expression"]),
        "literal_expression": deepcopy(SHARED_DEFS["literal_expression"]),
    }
    for depth in range(1, MAX_EXPRESSION_DEPTH + 1):
        definitions[f"expression_d{depth}"] = _finite_expression_schema(depth)
        definitions[f"predicate_d{depth}"] = _finite_predicate_schema(depth)
        if depth < MAX_EXPRESSION_DEPTH:
            definitions[f"computed_expression_d{depth}"] = (
                _finite_computed_expression_schema(depth)
            )
    return definitions


def _local_definition_refs(value: Any) -> set[str]:
    refs: set[str] = set()
    if isinstance(value, Mapping):
        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            refs.add(ref.removeprefix("#/$defs/"))
        for item in value.values():
            refs.update(_local_definition_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.update(_local_definition_refs(item))
    return refs


def _reachable_ast_defs(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only finite AST definitions reachable from one tool's root schema."""
    available = _finite_ast_defs()
    pending = list(_local_definition_refs(schema))
    reachable: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reachable:
            continue
        if name not in available:
            raise RuntimeError(f"finite AST schema references missing definition {name!r}")
        reachable.add(name)
        pending.extend(_local_definition_refs(available[name]) - reachable)
    return {
        name: definition
        for name, definition in available.items()
        if name in reachable
    }


ORDER_KEY = _object(
    {
        "column": COLUMN_NAME,
        "direction": {"type": "string", "enum": ["asc", "desc"]},
        "nulls": {"type": "string", "enum": ["first", "last"], "default": "last"},
    },
    required=("column", "direction"),
)
JOIN_EDGE = _object(
    {
        "left_column": COLUMN_NAME,
        "op": {"type": "string", "enum": list(COMPARISON_OPERATORS)},
        "right_column": COLUMN_NAME,
    },
    required=("left_column", "op", "right_column"),
)
PROJECT_OUTPUT = {
    "oneOf": [
        _object(
            {
                "expression": {"$ref": "#/$defs/column_expression"},
                "as": OUTPUT_IDENTIFIER,
            },
            required=("expression",),
        ),
        _object(
            {
                "expression": {
                    "oneOf": [
                        {"$ref": "#/$defs/literal_expression"},
                        {"$ref": "#/$defs/computed_expression"},
                    ]
                },
                "as": OUTPUT_IDENTIFIER,
            },
            required=("expression", "as"),
        ),
    ]
}
AGGREGATE_METRIC = {
    "oneOf": [
        _object(
            {
                "op": {"const": "count"},
                "column": {"const": "*"},
                "distinct": {"const": False, "default": False},
                "as": OUTPUT_IDENTIFIER,
            },
            required=("op", "column", "as"),
        ),
        _object(
            {
                "op": {"const": "count"},
                "column": {"type": "string", "minLength": 1, "not": {"const": "*"}},
                "distinct": {"type": "boolean", "default": False},
                "as": OUTPUT_IDENTIFIER,
            },
            required=("op", "column", "as"),
        ),
        _object(
            {
                "op": {"type": "string", "enum": ["sum", "avg", "min", "max"]},
                "column": {"type": "string", "minLength": 1, "not": {"const": "*"}},
                "distinct": {"type": "boolean", "default": False},
                "as": OUTPUT_IDENTIFIER,
            },
            required=("op", "column", "as"),
        ),
    ]
}


SEMANTIC_AGGREGATE_METRIC = {"oneOf": []}
SEMANTIC_COLUMN_OUTPUT = _object(
    {"column": COLUMN_NAME, "as": OUTPUT_IDENTIFIER},
    required=("column",),
)
SEMANTIC_METRIC_CONDITION = {
    "oneOf": [
        _object(
            {
                "column": COLUMN_NAME,
                "op": {"type": "string", "enum": list(COMPARISON_OPERATORS)},
                "value": LITERAL_VALUE,
            },
            required=("column", "op", "value"),
        ),
        _object(
            {
                "column": COLUMN_NAME,
                "op": {"type": "string", "enum": ["is_null", "is_not_null"]},
            },
            required=("column", "op"),
        ),
    ]
}
SCALAR_VALUE_REF = {
    "oneOf": [
        _object({"column": COLUMN_NAME}, required=("column",)),
        _object({"value": {"type": "number"}}, required=("value",)),
    ]
}
SCALAR_FORMULA = _object(
    {
        "op": {"type": "string", "enum": ["add", "subtract", "multiply", "divide"]},
        "left": SCALAR_VALUE_REF,
        "right": SCALAR_VALUE_REF,
        "multiplier": {"type": "number", "default": 1},
        "round_digits": {"type": "integer", "minimum": 0, "maximum": 10},
        "as": OUTPUT_IDENTIFIER,
    },
    required=("op", "left", "right", "as"),
)
for _semantic_metric in AGGREGATE_METRIC["oneOf"]:
    _variant = deepcopy(dict(_semantic_metric))
    _variant["properties"] = {
        **dict(_variant.get("properties", {})),
        "where": SEMANTIC_METRIC_CONDITION,
    }
    SEMANTIC_AGGREGATE_METRIC["oneOf"].append(_variant)


def _tool(
    description: str,
    parameters: Mapping[str, Any],
    *capabilities: str,
) -> ToolDefinition:
    return ToolDefinition(description, parameters, frozenset(capabilities))


TOOL_DEFINITIONS: dict[str, ToolDefinition] = {
    "describe_table": _tool(
        "Inspect exact schemas and row counts for current relations.",
        _object(
            {"tables": _array(TABLE_NAME, minimum=1, unique=True)},
            required=("tables",),
        ),
        "perception",
    ),
    "inspect_column": _tool(
        "Inspect a column's bounded values and completeness metadata.",
        _object(
            {
                "table": TABLE_NAME,
                "column": COLUMN_NAME,
                "top_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
            required=("table", "column"),
        ),
        "perception",
    ),
    "read_rows": _tool(
        "Observe bounded rows without creating a relation artifact.",
        _object(
            {
                "table": TABLE_NAME,
                "columns": _array(COLUMN_NAME, minimum=1, unique=True),
                "conditions": {"$ref": "#/$defs/predicate"},
                "order_by": _array(ORDER_KEY, minimum=1),
                "limit": {"type": "integer", "minimum": 1, "maximum": 20, "default": 20},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
            },
            required=("table",),
            all_of=(
                {
                    "if": {"required": ["offset"], "properties": {"offset": {"minimum": 1}}},
                    "then": {"required": ["order_by"]},
                },
            ),
        ),
        "perception",
    ),
    "execute_sql": _tool(
        "Execute one read-only SELECT or WITH statement and create a relation artifact.",
        _object({"sql": NONBLANK_STRING}, required=("sql",)),
        "direct-sql",
    ),
    "filter_rows": _tool(
        "Filter rows by a typed predicate; preserve all columns and duplicates.",
        _object(
            {
                "table": TABLE_NAME,
                "conditions": {"$ref": "#/$defs/predicate"},
            },
            required=("table", "conditions"),
        ),
        "atomic",
    ),
    "project": _tool(
        "Select or compute columns; do not filter, sort, or deduplicate rows.",
        _object(
            {
                "table": TABLE_NAME,
                "outputs": _array(PROJECT_OUTPUT, minimum=1),
            },
            required=("table", "outputs"),
        ),
        "atomic",
    ),
    "join": _tool(
        "Join exactly two relations without projecting the result.",
        _object(
            {
                "left": TABLE_NAME,
                "right": TABLE_NAME,
                "left_role": IDENTIFIER,
                "right_role": IDENTIFIER,
                "type": {
                    "type": "string",
                    "enum": ["inner", "left", "semi", "anti", "cross"],
                    "default": "inner",
                },
                "on": _array(JOIN_EDGE),
            },
            required=("left", "right", "on"),
            all_of=(
                {
                    "if": {"required": ["type"], "properties": {"type": {"const": "cross"}}},
                    "then": {"properties": {"on": {"maxItems": 0}}},
                    "else": {"properties": {"on": {"minItems": 1}}},
                },
            ),
        ),
        "atomic",
    ),
    "aggregate": _tool(
        "Group rows and compute metrics; filter and projection are separate operations.",
        _object(
            {
                "table": TABLE_NAME,
                "group_by": _array(COLUMN_NAME, unique=True),
                "metrics": _array(AGGREGATE_METRIC, minimum=1),
            },
            required=("table", "group_by", "metrics"),
        ),
        "atomic",
    ),
    "distinct": _tool(
        "Remove exact duplicates across every current column.",
        _object({"table": TABLE_NAME}, required=("table",)),
        "atomic",
    ),
    "set_operation": _tool(
        "Combine two positionally aligned relations with explicit set or bag semantics.",
        _object(
            {
                "left": TABLE_NAME,
                "right": TABLE_NAME,
                "op": {
                    "type": "string",
                    "enum": ["union", "union_all", "intersect", "except"],
                },
            },
            required=("left", "right", "op"),
        ),
        "atomic",
    ),
    "sort": _tool(
        "Establish relation ordering without limiting rows.",
        _object(
            {"table": TABLE_NAME, "keys": _array(ORDER_KEY, minimum=1)},
            required=("table", "keys"),
        ),
        "atomic",
    ),
    "limit": _tool(
        "Slice an already ordered relation without changing its columns.",
        _object(
            {
                "table": TABLE_NAME,
                "count": {"type": "integer", "minimum": 1},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
            },
            required=("table", "count"),
        ),
        "atomic",
    ),
    "add_rank": _tool(
        "Add one ranking column; do not filter rows or establish output ordering.",
        _object(
            {
                "table": TABLE_NAME,
                "partition_by": _array(COLUMN_NAME, unique=True, default=[]),
                "order_by": _array(ORDER_KEY, minimum=1),
                "method": {"type": "string", "enum": ["row_number", "rank", "dense_rank"]},
                "as": OUTPUT_IDENTIFIER,
            },
            required=("table", "order_by", "method", "as"),
        ),
        "atomic",
    ),
    "shape_rows": _tool(
        "Select or alias final columns and optionally remove exact duplicate rows.",
        _object(
            {
                "table": TABLE_NAME,
                "outputs": _array(SEMANTIC_COLUMN_OUTPUT, minimum=1),
                "distinct": {"type": "boolean", "default": False},
            },
            required=("table", "outputs"),
        ),
        "semantic-atomic",
        "projection",
    ),
    "group_aggregate": _tool(
        "Group one fixed input population and compute globally or locally conditioned metrics.",
        _object(
            {
                "table": TABLE_NAME,
                "group_by": _array(COLUMN_NAME, unique=True),
                "metrics": _array(SEMANTIC_AGGREGATE_METRIC, minimum=1),
            },
            required=("table", "group_by", "metrics"),
        ),
        "semantic-atomic",
        "aggregation",
    ),
    "scalar_compute": _tool(
        "Compute final scalar expressions from an existing exactly-one-row relation.",
        _object(
            {
                "table": TABLE_NAME,
                "outputs": _array(SCALAR_FORMULA, minimum=1),
            },
            required=("table", "outputs"),
        ),
        "semantic-atomic",
        "scalar",
    ),
    "rank_select": _tool(
        "Rank one fixed input, take top-k rows with explicit ties, and project answer columns.",
        _object(
            {
                "table": TABLE_NAME,
                "order_by": _array(ORDER_KEY, minimum=1),
                "top_k": {"type": "integer", "minimum": 1},
                "with_ties": {"type": "boolean", "default": False},
                "outputs": _array(SEMANTIC_COLUMN_OUTPUT, minimum=1),
            },
            required=("table", "order_by", "top_k", "outputs"),
        ),
        "semantic-atomic",
        "ranking",
    ),
    "commit_checkpoint": _tool(
        "Commit a semantic phase with new targets distinct from current and active-path goals; max 8.",
        _object(
            {
                "progress_summary": _array(
                    {"type": "string", "minLength": 1, "maxLength": 512},
                    minimum=1,
                    maximum=5,
                ),
                "remaining_uncertainties": _array(
                    {"type": "string", "minLength": 1, "maxLength": 512},
                    maximum=5,
                ),
                "next_targets": _array(
                    {"type": "string", "minLength": 1, "maxLength": 512},
                    minimum=1,
                    maximum=3,
                ),
            },
            required=("progress_summary", "remaining_uncertainties", "next_targets"),
        ),
        "checkpoint",
    ),
    "restore_checkpoint": _tool(
        "Restore one available checkpoint and begin a new recovery branch.",
        _object(
            {
                "checkpoint_id": NONEMPTY_STRING,
                "reason": {"type": "string", "minLength": 1, "maxLength": 1500},
                "next_targets": _array(
                    {"type": "string", "minLength": 1, "maxLength": 512},
                    minimum=1,
                    maximum=3,
                ),
            },
            required=("checkpoint_id", "reason", "next_targets"),
        ),
        "checkpoint",
        "restore",
    ),
    "answer": _tool(
        "Submit one active relation artifact as the exact final answer.",
        _object({"table": TABLE_NAME}, required=("table",)),
        "terminal",
    ),
}


PERCEPTION_TOOLS = ("describe_table", "inspect_column", "read_rows")
ATOMIC_TOOLS = (
    "filter_rows",
    "project",
    "join",
    "aggregate",
    "distinct",
    "set_operation",
    "sort",
    "limit",
    "add_rank",
)
SEMANTIC_ATOMIC_TOOLS = (
    "filter_rows",
    "shape_rows",
    "join",
    "group_aggregate",
    "scalar_compute",
    "rank_select",
    "set_operation",
)
CONTROL_TOOLS = ("commit_checkpoint", "restore_checkpoint", "answer")
MODE_TOOLS: dict[str, tuple[str, ...]] = {
    "direct": (*PERCEPTION_TOOLS, "execute_sql", *CONTROL_TOOLS),
    "atomic": (*PERCEPTION_TOOLS, *ATOMIC_TOOLS, *CONTROL_TOOLS),
    "hybrid": (*PERCEPTION_TOOLS, "execute_sql", *ATOMIC_TOOLS, *CONTROL_TOOLS),
}
MODE_TOOL_NAMES = MODE_TOOLS
TOOLS = MODE_TOOLS["hybrid"]
TOOL_SPECS = {name: definition.description for name, definition in TOOL_DEFINITIONS.items()}


def _assert_definition_integrity() -> None:
    expected = set(TOOL_DEFINITIONS)
    surfaced = set().union(*(set(names) for names in MODE_TOOLS.values())) | set(
        SEMANTIC_ATOMIC_TOOLS
    )
    if expected != surfaced:
        raise RuntimeError(
            f"tool definitions and mode surfaces drifted: missing={sorted(expected - surfaced)}, "
            f"unknown={sorted(surfaced - expected)}"
        )
    if len(ATOMIC_TOOLS) != 9:
        raise RuntimeError("checkpoint-relalg-v1 must expose exactly nine atomic operators")
    if set(MODE_TOOLS["direct"]) & set(ATOMIC_TOOLS):
        raise RuntimeError("direct mode leaked atomic tools")
    if "execute_sql" in MODE_TOOLS["atomic"]:
        raise RuntimeError("atomic mode leaked execute_sql")


_assert_definition_integrity()


def normalize_mode(mode: str) -> str:
    if mode not in MODE_TOOLS:
        raise ValueError(f"unknown checkpoint-relalg mode {mode!r}; expected one of {MODES}")
    return mode


def normalize_atomic_operator_profile(profile: str | None) -> str:
    value = (
        DEFAULT_ATOMIC_OPERATOR_PROFILE
        if profile is None
        else str(profile).strip().lower()
    )
    if value not in ATOMIC_OPERATOR_PROFILES:
        raise ValueError(
            f"unknown atomic operator profile {profile!r}; expected one of {ATOMIC_OPERATOR_PROFILES}"
        )
    return value


def tools_for_profile(
    mode: str,
    atomic_operator_profile: str | None = None,
) -> tuple[str, ...]:
    active_mode = normalize_mode(mode)
    profile = normalize_atomic_operator_profile(atomic_operator_profile)
    if profile == ATOMIC_OPERATOR_PROFILE_MICRO:
        return MODE_TOOLS[active_mode]
    if active_mode != "atomic":
        raise ValueError("semantic-v2 is currently isolated to atomic mode")
    return (*PERCEPTION_TOOLS, *SEMANTIC_ATOMIC_TOOLS, *CONTROL_TOOLS)


def _contains_ref(value: Any) -> bool:
    if isinstance(value, dict):
        return "$ref" in value or any(_contains_ref(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_ref(item) for item in value)
    return False


def parameter_schema(tool: str) -> dict[str, Any]:
    """Return the closed provider/runtime schema for one public tool."""
    try:
        definition = TOOL_DEFINITIONS[tool]
    except KeyError as exc:
        raise ValueError(f"unknown checkpoint-relalg tool {tool!r}") from exc
    schema = _expand_finite_ast_refs(deepcopy(dict(definition.parameters)))
    if _contains_ref(schema):
        schema["$defs"] = _reachable_ast_defs(schema)
    return schema


PARAMETER_SCHEMAS = {name: parameter_schema(name) for name in TOOL_DEFINITIONS}
MODEL_ARG_SCHEMA: dict[str, tuple[set[str], set[str]]] = {}
for _name, _schema in PARAMETER_SCHEMAS.items():
    _properties = set(_schema.get("properties", {}))
    _required = set(_schema.get("required", []))
    MODEL_ARG_SCHEMA[_name] = (_required, _properties - _required)


def provider_tool_definitions(
    mode: str,
    atomic_operator_profile: str = DEFAULT_ATOMIC_OPERATOR_PROFILE,
) -> list[dict[str, Any]]:
    """Generate the exact provider-native function surface for ``mode``."""
    active_mode = normalize_mode(mode)
    active_tools = tools_for_profile(active_mode, atomic_operator_profile)
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": TOOL_DEFINITIONS[name].description,
                "parameters": parameter_schema(name),
            },
        }
        for name in active_tools
    ]


get_tool_definitions = provider_tool_definitions
tools_for_mode = provider_tool_definitions


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def tool_schema_hash(
    mode: str | None = None,
    atomic_operator_profile: str = DEFAULT_ATOMIC_OPERATOR_PROFILE,
) -> str:
    """Hash one mode surface, or all isolated surfaces when ``mode`` is omitted."""
    payload: Any
    if mode is None:
        payload = {
            active_mode: provider_tool_definitions(active_mode, atomic_operator_profile)
            for active_mode in MODES
        }
    else:
        payload = provider_tool_definitions(normalize_mode(mode), atomic_operator_profile)
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _resolve_ref(ref: str, root: Mapping[str, Any]) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise RuntimeError(f"unsupported non-local schema reference {ref!r}")
    current: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping) or part not in current:
            raise RuntimeError(f"unresolvable schema reference {ref!r}")
        current = current[part]
    if not isinstance(current, Mapping):
        raise RuntimeError(f"schema reference {ref!r} does not resolve to an object")
    return current


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) != isinstance(right, bool):
        return False
    return left == right


def _json_type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (not isinstance(value, float) or math.isfinite(value))
        )
    if expected == "string":
        return isinstance(value, str)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    return False


def _validate_schema(
    value: Any,
    schema: Mapping[str, Any],
    *,
    root: Mapping[str, Any],
    path: str,
) -> None:
    if "$ref" in schema:
        _validate_schema(value, _resolve_ref(str(schema["$ref"]), root), root=root, path=path)
        return

    if "allOf" in schema:
        for item in schema["allOf"]:
            _validate_schema(value, item, root=root, path=path)
    if "anyOf" in schema:
        errors: list[ProtocolValidationError] = []
        for item in schema["anyOf"]:
            try:
                _validate_schema(value, item, root=root, path=path)
                break
            except ProtocolValidationError as exc:
                errors.append(exc)
        else:
            raise ProtocolValidationError(
                "does not match any allowed shape",
                code="invalid_shape",
                path=path,
            ) from errors[-1] if errors else None
    if "oneOf" in schema:
        matches = 0
        last_error: ProtocolValidationError | None = None
        for item in schema["oneOf"]:
            try:
                _validate_schema(value, item, root=root, path=path)
                matches += 1
            except ProtocolValidationError as exc:
                last_error = exc
        if matches != 1:
            message = "does not match exactly one allowed shape"
            raise ProtocolValidationError(
                message,
                code="invalid_shape",
                path=path,
            ) from last_error
    if "not" in schema:
        try:
            _validate_schema(value, schema["not"], root=root, path=path)
        except ProtocolValidationError:
            pass
        else:
            raise ProtocolValidationError("matches a forbidden shape", path=path)
    if "if" in schema:
        try:
            _validate_schema(value, schema["if"], root=root, path=path)
        except ProtocolValidationError:
            branch = schema.get("else")
        else:
            branch = schema.get("then")
        if branch is not None:
            _validate_schema(value, branch, root=root, path=path)

    if "const" in schema and not _json_equal(value, schema["const"]):
        raise ProtocolValidationError(
            f"must equal {schema['const']!r}",
            code="invalid_value",
            path=path,
        )
    if "enum" in schema and not any(_json_equal(value, item) for item in schema["enum"]):
        raise ProtocolValidationError(
            f"must be one of {schema['enum']!r}",
            code="invalid_value",
            path=path,
        )

    expected_type = schema.get("type")
    if expected_type is not None:
        expected_types = [expected_type] if isinstance(expected_type, str) else list(expected_type)
        if not any(_json_type_matches(value, item) for item in expected_types):
            raise ProtocolValidationError(
                f"must have JSON type {' or '.join(expected_types)}",
                code="invalid_type",
                path=path,
            )

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ProtocolValidationError("string is too short", path=path)
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            raise ProtocolValidationError("string is too long", path=path)
        if "pattern" in schema and re.search(str(schema["pattern"]), value) is None:
            raise ProtocolValidationError("string does not match the required pattern", path=path)

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ProtocolValidationError(f"must be >= {schema['minimum']}", path=path)
        if "maximum" in schema and value > schema["maximum"]:
            raise ProtocolValidationError(f"must be <= {schema['maximum']}", path=path)

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ProtocolValidationError("array has too few items", path=path)
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            raise ProtocolValidationError("array has too many items", path=path)
        if schema.get("uniqueItems"):
            encoded = [_canonical_json(item) for item in value]
            if len(encoded) != len(set(encoded)):
                raise ProtocolValidationError("array items must be unique", path=path)
        if "items" in schema:
            for index, item in enumerate(value):
                _validate_schema(item, schema["items"], root=root, path=f"{path}[{index}]")

    if isinstance(value, dict):
        if any(not isinstance(name, str) for name in value):
            raise ProtocolValidationError(
                "object field names must be strings",
                code="unexpected_field",
                path=path,
            )
        required = schema.get("required", [])
        for name in required:
            if name not in value:
                raise ProtocolValidationError(
                    f"missing required field {name!r}",
                    code="missing_required_field",
                    path=path,
                )
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            unexpected = sorted(set(value) - set(properties))
            if unexpected:
                raise ProtocolValidationError(
                    f"unexpected field(s): {', '.join(unexpected)}",
                    code="unexpected_field",
                    path=path,
                )
        for name, item in value.items():
            if name in properties:
                _validate_schema(item, properties[name], root=root, path=f"{path}.{name}")


def _ast_children(node: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    children: list[Mapping[str, Any]] = []
    for field in ("left", "right", "arg", "value", "lower", "upper", "pattern", "else"):
        child = node.get(field)
        if isinstance(child, Mapping):
            children.append(child)
    for child in node.get("args", []):
        if isinstance(child, Mapping):
            children.append(child)
    for child in node.get("values", []):
        if isinstance(child, Mapping):
            children.append(child)
    for branch in node.get("branches", []):
        if isinstance(branch, Mapping):
            when = branch.get("when")
            then = branch.get("then")
            if isinstance(when, Mapping):
                children.append(when)
            if isinstance(then, Mapping):
                children.append(then)
    return children


def _validate_ast_depth(node: Mapping[str, Any], *, depth: int = 1, path: str = "$") -> None:
    if depth > MAX_EXPRESSION_DEPTH:
        raise ProtocolValidationError(
            f"expression/predicate depth exceeds {MAX_EXPRESSION_DEPTH}",
            code="expression_depth_exceeded",
            path=path,
        )
    for index, child in enumerate(_ast_children(node)):
        _validate_ast_depth(child, depth=depth + 1, path=f"{path}.node[{index}]")


_ALLOWED_AST_OPERATORS = frozenset(
    {
        *BINARY_EXPRESSION_OPERATORS,
        *UNARY_EXPRESSION_OPERATORS,
        *VARIABLE_EXPRESSION_ARITY,
        *PREDICATE_OPERATORS,
        "cast",
        "case_when",
    }
)


def _validate_ast_operators(node: Mapping[str, Any], *, path: str = "$") -> None:
    operator = node.get("op")
    if isinstance(operator, str) and operator not in _ALLOWED_AST_OPERATORS:
        raise ProtocolValidationError(
            f"unsupported expression/predicate operator {operator!r}",
            code="unsupported_expression_operator",
            path=f"{path}.op",
        )
    for index, child in enumerate(_ast_children(node)):
        _validate_ast_operators(child, path=f"{path}.node[{index}]")


def _walk_ast_roots(value: Any, *, parent_key: str | None = None) -> list[Mapping[str, Any]]:
    roots: list[Mapping[str, Any]] = []
    if isinstance(value, dict):
        if parent_key in {"conditions", "expression", "where"}:
            roots.append(value)
            return roots
        for key, item in value.items():
            roots.extend(_walk_ast_roots(item, parent_key=key))
    elif isinstance(value, list):
        for item in value:
            roots.extend(_walk_ast_roots(item, parent_key=parent_key))
    return roots


def _validate_reserved_output_names(tool: str, arguments: Mapping[str, Any]) -> None:
    candidates: list[tuple[str, Any]] = []
    if tool in {"project", "shape_rows", "scalar_compute", "rank_select"} and isinstance(
        arguments.get("outputs"), list
    ):
        candidates.extend(
            (f"$.arguments.outputs[{index}].as", output.get("as"))
            for index, output in enumerate(arguments["outputs"])
            if isinstance(output, Mapping) and "as" in output
        )
    elif tool in {"aggregate", "group_aggregate"} and isinstance(arguments.get("metrics"), list):
        candidates.extend(
            (f"$.arguments.metrics[{index}].as", metric.get("as"))
            for index, metric in enumerate(arguments["metrics"])
            if isinstance(metric, Mapping) and "as" in metric
        )
    elif tool == "add_rank" and "as" in arguments:
        candidates.append(("$.arguments.as", arguments.get("as")))
    for path, value in candidates:
        if isinstance(value, str) and value.startswith("__"):
            raise ProtocolValidationError(
                "double-underscore output names are reserved for Harness metadata",
                code="reserved_output_column",
                path=path,
            )


def validate_arguments(
    tool: str,
    arguments: Mapping[str, Any],
    *,
    mode: str = "hybrid",
    atomic_operator_profile: str = DEFAULT_ATOMIC_OPERATOR_PROFILE,
) -> dict[str, Any]:
    """Validate one call against the same schema supplied to the provider."""
    active_mode = normalize_mode(mode)
    active_tools = tools_for_profile(active_mode, atomic_operator_profile)
    if tool not in active_tools:
        if tool in TOOL_DEFINITIONS:
            raise ProtocolValidationError(
                f"tool {tool!r} is unavailable in {active_mode!r} mode",
                code="tool_not_available_in_mode",
                path="$.tool",
            )
        raise ProtocolValidationError(
            f"unknown tool {tool!r}",
            code="unknown_tool",
            path="$.tool",
        )
    if not isinstance(arguments, Mapping):
        raise ProtocolValidationError(
            "tool arguments must be a JSON object",
            code="invalid_type",
            path="$.arguments",
        )
    normalized = deepcopy(dict(arguments))
    _validate_reserved_output_names(tool, normalized)
    schema = parameter_schema(tool)
    ast_roots = _walk_ast_roots(normalized)
    for index, root in enumerate(ast_roots):
        _validate_ast_depth(root, path=f"$.arguments.ast[{index}]")
        _validate_ast_operators(root, path=f"$.arguments.ast[{index}]")
    _validate_schema(normalized, schema, root=schema, path="$.arguments")
    return normalized


def validate_tool_call(
    mode: str,
    tool: str,
    arguments: Mapping[str, Any],
    *,
    atomic_operator_profile: str = DEFAULT_ATOMIC_OPERATOR_PROFILE,
) -> dict[str, Any]:
    return validate_arguments(
        tool,
        arguments,
        mode=mode,
        atomic_operator_profile=atomic_operator_profile,
    )


def validate_model_action(
    action: Mapping[str, Any],
    *,
    mode: str = "hybrid",
    atomic_operator_profile: str = DEFAULT_ATOMIC_OPERATOR_PROFILE,
) -> dict[str, Any]:
    """Validate the canonical single-action envelope used by replay/tests."""
    if not isinstance(action, Mapping):
        raise ProtocolValidationError("action must be an object", path="$")
    unexpected = sorted(set(action) - {"tool", "arguments"})
    if unexpected:
        raise ProtocolValidationError(
            f"unexpected action field(s): {', '.join(unexpected)}",
            code="unexpected_field",
            path="$",
        )
    if "tool" not in action or "arguments" not in action:
        raise ProtocolValidationError(
            "action requires exactly tool and arguments",
            code="missing_required_field",
            path="$",
        )
    tool = action["tool"]
    if not isinstance(tool, str):
        raise ProtocolValidationError("tool must be a string", path="$.tool")
    arguments = validate_arguments(
        tool,
        action["arguments"],
        mode=mode,
        atomic_operator_profile=atomic_operator_profile,
    )
    return {"tool": tool, "arguments": arguments}


def normalize_carrier(carrier: str | None) -> str:
    value = DEFAULT_CARRIER if carrier is None else str(carrier).strip().lower()
    if value not in CARRIERS:
        raise ValueError(f"unknown checkpoint-relalg carrier {carrier!r}")
    return value


def normalize_checkpoint_guidance_profile(profile: str | None) -> str:
    value = (
        DEFAULT_CHECKPOINT_GUIDANCE_PROFILE
        if profile is None
        else str(profile).strip().lower()
    )
    if value not in CHECKPOINT_GUIDANCE_PROFILES:
        raise ValueError(f"unknown checkpoint guidance profile {profile!r}")
    return value


def checkpoint_commit_eligibility_for_guidance_profile(profile: str | None) -> str:
    """Map one frozen teacher profile to its deterministic Harness policy."""

    active_profile = normalize_checkpoint_guidance_profile(profile)
    if active_profile in {
        CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V5,
        CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V6,
    }:
        return CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1
    return CHECKPOINT_COMMIT_ELIGIBILITY_NONE


def checkpoint_commit_eligibility_manifest(policy: str | None) -> dict[str, Any] | None:
    """Return the identity-bound, content-free parameters for one policy."""

    active_policy = normalize_checkpoint_commit_eligibility_policy(policy)
    if active_policy == CHECKPOINT_COMMIT_ELIGIBILITY_NONE:
        return None
    return {
        "policy": active_policy,
        "first_commit_min_milestone_producers": (
            FIRST_COMMIT_MIN_MILESTONE_PRODUCERS
        ),
        "later_commit_min_milestone_producers": (
            LATER_COMMIT_MIN_MILESTONE_PRODUCERS
        ),
        "milestone_producer_tools": sorted(CHECKPOINT_MILESTONE_PRODUCER_TOOLS),
        "requires_new_active_artifacts_at_least_quota": True,
    }


def assistant_carrier_protocol(carrier: str | None = None) -> str:
    active_carrier = normalize_carrier(carrier)
    if active_carrier == CARRIER_NATIVE_TOOL_CALLS:
        return NATIVE_ASSISTANT_CARRIER
    return TEXT_JSON_ASSISTANT_CARRIER


def carrier_experiment_arm(carrier: str | None = None) -> str:
    return "A" if normalize_carrier(carrier) == CARRIER_TEXT_JSON else "B"


_PROMPT_DIR = Path(__file__).with_name("prompts")


def _prompt_fragment(name: str) -> str:
    path = _PROMPT_DIR / f"{name}.txt"
    return path.read_text(encoding="utf-8").strip()


def get_system_prompt(
    mode: str,
    *,
    teacher: bool = False,
    carrier: str = DEFAULT_CARRIER,
    checkpoint_guidance_profile: str = DEFAULT_CHECKPOINT_GUIDANCE_PROFILE,
    atomic_operator_profile: str = DEFAULT_ATOMIC_OPERATOR_PROFILE,
) -> str:
    """Build Shared Core + one short mode clause + optional teacher-only guidance."""
    active_mode = normalize_mode(mode)
    active_carrier = normalize_carrier(carrier)
    active_operator_profile = normalize_atomic_operator_profile(atomic_operator_profile)
    tools_for_profile(active_mode, active_operator_profile)
    active_checkpoint_guidance = normalize_checkpoint_guidance_profile(
        checkpoint_guidance_profile
    )
    semantic_profile = active_operator_profile == ATOMIC_OPERATOR_PROFILE_SEMANTIC
    if (
        active_checkpoint_guidance
        in {
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE,
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V2,
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V3,
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V4,
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V5,
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V6,
        }
        and not (active_mode == "atomic" and semantic_profile)
    ):
        raise ValueError(
            "semantic-milestone-v1 requires atomic mode with semantic-v2 operators"
        )
    shared_core = _prompt_fragment("shared_core")
    fragments = [
        shared_core,
        _prompt_fragment("atomic_semantic" if semantic_profile else active_mode),
    ]
    if teacher:
        checkpoint_fragment = {
            CHECKPOINT_GUIDANCE_PROFILE_STANDARD: "teacher_checkpoint",
            CHECKPOINT_GUIDANCE_PROFILE_STRESS: "teacher_checkpoint_stress",
            CHECKPOINT_GUIDANCE_PROFILE_RESTORE_TRIGGER: (
                "teacher_checkpoint_restore_trigger"
            ),
            CHECKPOINT_GUIDANCE_PROFILE_RESTORE_TARGET: (
                "teacher_checkpoint_restore_target"
            ),
            CHECKPOINT_GUIDANCE_PROFILE_RESTORE_PROBE: (
                "teacher_checkpoint_restore_probe"
            ),
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE: (
                "teacher_checkpoint_semantic_milestone"
            ),
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V2: (
                "teacher_checkpoint_semantic_milestone_v2"
            ),
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V3: (
                "teacher_checkpoint_semantic_milestone_v3"
            ),
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V4: (
                "teacher_checkpoint_semantic_milestone_v4"
            ),
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V5: (
                "teacher_checkpoint_semantic_milestone_v5"
            ),
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V6: (
                "teacher_checkpoint_semantic_milestone_v6"
            ),
        }[active_checkpoint_guidance]
        fragments.append(_prompt_fragment(checkpoint_fragment))
        if semantic_profile:
            fragments.append(_prompt_fragment("teacher_atomic_semantic"))
    elif active_checkpoint_guidance != DEFAULT_CHECKPOINT_GUIDANCE_PROFILE:
        raise ValueError("non-default checkpoint guidance is teacher-only")
    if active_carrier == CARRIER_TEXT_JSON:
        native_clause = "Make exactly one native tool call per turn."
        if shared_core.count(native_clause) != 1:
            raise RuntimeError("shared prompt native carrier clause drifted")
        fragments[0] = shared_core.replace(
            native_clause,
            "Make exactly one action per turn using the TEXT-JSON CARRIER below.",
            1,
        )
        # Text JSON cannot receive provider ``tools``.  Render the exact same
        # name/description/parameters, without the native ``type=function``
        # transport wrapper that could be mistaken for the action envelope.
        compact_schemas = _canonical_json([
            deepcopy(item["function"])
            for item in provider_tool_definitions(active_mode, active_operator_profile)
        ])
        fragments.append(
            "TEXT-JSON CARRIER (diagnostic-only)\n"
            "Keep reasoning only in the provider reasoning_content field. Visible assistant "
            "content must be exactly one raw JSON object with exactly the keys tool and "
            "arguments: {\"tool\":\"tool_name\",\"arguments\":{}}. Do not return prose, "
            "Markdown fences, XML, an array, or multiple actions. A later user message whose "
            "JSON type is checkpoint_relalg_tool_result is Harness feedback, not a new task.\n"
            f"EXACT TOOL SCHEMAS FOR {active_mode.upper()} MODE\n{compact_schemas}"
        )
    if (
        teacher
        and active_checkpoint_guidance
        in {
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V2,
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V3,
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V4,
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V5,
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V6,
        }
    ):
        # Keep the actionable turn check after the large Text-JSON schema block.
        # Native transport also benefits from making it the last system clause.
        tail_name = {
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V2: (
                "teacher_checkpoint_semantic_milestone_v2_tail"
            ),
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V3: (
                "teacher_checkpoint_semantic_milestone_v3_tail"
            ),
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V4: (
                "teacher_checkpoint_semantic_milestone_v4_tail"
            ),
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V5: (
                "teacher_checkpoint_semantic_milestone_v5_tail"
            ),
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V6: (
                "teacher_checkpoint_semantic_milestone_v6_tail"
            ),
        }[active_checkpoint_guidance]
        fragments.append(_prompt_fragment(tail_name))
    return "\n\n".join(fragments)


build_system_prompt = get_system_prompt


def prompt_hash(
    mode: str,
    *,
    teacher: bool = False,
    carrier: str = DEFAULT_CARRIER,
    checkpoint_guidance_profile: str = DEFAULT_CHECKPOINT_GUIDANCE_PROFILE,
    atomic_operator_profile: str = DEFAULT_ATOMIC_OPERATOR_PROFILE,
) -> str:
    active_mode = normalize_mode(mode)
    active_carrier = normalize_carrier(carrier)
    active_operator_profile = normalize_atomic_operator_profile(atomic_operator_profile)
    active_checkpoint_guidance = normalize_checkpoint_guidance_profile(
        checkpoint_guidance_profile
    )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "scheme": SCHEME,
        "mode": active_mode,
        "audience": "teacher" if teacher else "student",
        "system_prompt": get_system_prompt(
            active_mode,
            teacher=teacher,
            carrier=active_carrier,
            checkpoint_guidance_profile=active_checkpoint_guidance,
            atomic_operator_profile=active_operator_profile,
        ),
    }
    if active_operator_profile != DEFAULT_ATOMIC_OPERATOR_PROFILE:
        payload["atomic_operator_profile"] = active_operator_profile
    if active_checkpoint_guidance != DEFAULT_CHECKPOINT_GUIDANCE_PROFILE:
        payload["checkpoint_guidance_profile"] = active_checkpoint_guidance
    # Preserve the frozen native prompt hashes.  Native is the implicit v1
    # carrier; the text diagnostic is explicitly namespaced in its hash.
    if active_carrier != CARRIER_NATIVE_TOOL_CALLS:
        payload["carrier_ablation_protocol_version"] = CARRIER_ABLATION_PROTOCOL_VERSION
        payload["carrier"] = active_carrier
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def carrier_protocol_hash(
    mode: str,
    carrier: str = DEFAULT_CARRIER,
    atomic_operator_profile: str = DEFAULT_ATOMIC_OPERATOR_PROFILE,
    checkpoint_commit_eligibility_policy: str = CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
) -> str:
    """Hash one carrier-specific protocol while preserving the native v1 identity."""

    active_mode = normalize_mode(mode)
    active_carrier = normalize_carrier(carrier)
    active_operator_profile = normalize_atomic_operator_profile(atomic_operator_profile)
    active_commit_eligibility = normalize_checkpoint_commit_eligibility_policy(
        checkpoint_commit_eligibility_policy
    )
    if active_commit_eligibility != CHECKPOINT_COMMIT_ELIGIBILITY_NONE and not (
        active_mode == "atomic"
        and active_operator_profile == ATOMIC_OPERATOR_PROFILE_SEMANTIC
    ):
        raise ValueError(
            "checkpoint commit eligibility is isolated to atomic semantic-v2"
        )
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "checkpoint_policy_version": CHECKPOINT_POLICY_VERSION,
        "checkpoint_goal_policy_version": CHECKPOINT_GOAL_POLICY_VERSION,
        "mode": active_mode,
        "student_prompt_sha256": prompt_hash(
            active_mode,
            teacher=False,
            carrier=active_carrier,
            atomic_operator_profile=active_operator_profile,
        ),
        "tool_schema_sha256": tool_schema_hash(active_mode, active_operator_profile),
    }
    if active_operator_profile != DEFAULT_ATOMIC_OPERATOR_PROFILE:
        payload["atomic_operator_profile"] = active_operator_profile
    eligibility_manifest = checkpoint_commit_eligibility_manifest(
        active_commit_eligibility
    )
    if eligibility_manifest is not None:
        payload["checkpoint_commit_eligibility"] = eligibility_manifest
    if active_carrier != CARRIER_NATIVE_TOOL_CALLS:
        payload.update({
            "carrier_policy_version": CARRIER_POLICY_VERSION,
            "carrier": active_carrier,
            "assistant_carrier": assistant_carrier_protocol(active_carrier),
        })
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def capability_manifest(
    mode: str,
    carrier: str = DEFAULT_CARRIER,
    atomic_operator_profile: str = DEFAULT_ATOMIC_OPERATOR_PROFILE,
    checkpoint_commit_eligibility_policy: str = CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
) -> dict[str, Any]:
    """Generate a frozen, serializable account of one isolated mode surface."""
    active_mode = normalize_mode(mode)
    active_carrier = normalize_carrier(carrier)
    active_operator_profile = normalize_atomic_operator_profile(atomic_operator_profile)
    active_commit_eligibility = normalize_checkpoint_commit_eligibility_policy(
        checkpoint_commit_eligibility_policy
    )
    if active_commit_eligibility != CHECKPOINT_COMMIT_ELIGIBILITY_NONE and not (
        active_mode == "atomic"
        and active_operator_profile == ATOMIC_OPERATOR_PROFILE_SEMANTIC
    ):
        raise ValueError(
            "checkpoint commit eligibility is isolated to atomic semantic-v2"
        )
    active_tools = tools_for_profile(active_mode, active_operator_profile)
    manifest = {
        "protocol_version": PROTOCOL_VERSION,
        "scheme": SCHEME,
        "tool_scheme": SCHEME,
        "mode": active_mode,
        "admission_status": ADMISSION_STATUS,
        "backend": BACKEND,
        "dialect": DIALECT,
        "environment_renderer_version": ENVIRONMENT_RENDERER_VERSION,
        "checkpoint_policy_version": CHECKPOINT_POLICY_VERSION,
        "checkpoint_goal_policy_version": CHECKPOINT_GOAL_POLICY_VERSION,
        "max_checkpoints": MAX_CHECKPOINTS,
        "executor_version": EXECUTOR_VERSION,
        "assistant_carrier": assistant_carrier_protocol(active_carrier),
        "single_tool_call_per_turn": True,
        "max_calls_per_turn": 1,
        "min_tool_calls_per_turn": 1,
        "max_tool_calls_per_turn": 1,
        "tools": list(active_tools),
        "tool_capabilities": {
            name: sorted(TOOL_DEFINITIONS[name].capabilities)
            for name in active_tools
        },
        "tool_schema_sha256": tool_schema_hash(active_mode, active_operator_profile),
        "student_prompt_sha256": prompt_hash(
            active_mode,
            carrier=active_carrier,
            atomic_operator_profile=active_operator_profile,
        ),
        "teacher_prompt_sha256": prompt_hash(
            active_mode,
            teacher=True,
            carrier=active_carrier,
            atomic_operator_profile=active_operator_profile,
        ),
        "max_expression_predicate_depth": MAX_EXPRESSION_DEPTH,
        "canonical_types": list(CANONICAL_TYPES),
        "expression_operators": [
            *BINARY_EXPRESSION_OPERATORS,
            *UNARY_EXPRESSION_OPERATORS,
            *VARIABLE_EXPRESSION_ARITY,
            "cast",
            "case_when",
        ],
        "predicate_operators": list(PREDICATE_OPERATORS),
    }
    if active_operator_profile != DEFAULT_ATOMIC_OPERATOR_PROFILE:
        manifest["atomic_operator_profile"] = active_operator_profile
        manifest["checkpoint_tools_model_visible"] = True
    eligibility_manifest = checkpoint_commit_eligibility_manifest(
        active_commit_eligibility
    )
    if eligibility_manifest is not None:
        manifest["checkpoint_commit_eligibility"] = eligibility_manifest
    # Keep the default native capability manifest byte-for-byte compatible.
    # The A/B record and run manifest still carry the explicit carrier name.
    if active_carrier != CARRIER_NATIVE_TOOL_CALLS:
        manifest["single_tool_call_per_turn"] = False
        manifest["min_tool_calls_per_turn"] = 0
        manifest["max_tool_calls_per_turn"] = 0
        manifest["single_action_per_turn"] = True
        manifest["min_actions_per_turn"] = 1
        manifest["max_actions_per_turn"] = 1
        manifest["carrier_ablation_protocol_version"] = CARRIER_ABLATION_PROTOCOL_VERSION
        manifest["carrier_policy_version"] = CARRIER_POLICY_VERSION
        manifest["experiment_arm"] = carrier_experiment_arm(active_carrier)
        manifest["carrier"] = active_carrier
    return manifest


get_capability_manifest = capability_manifest
CAPABILITY_MANIFESTS = {mode: capability_manifest(mode) for mode in MODES}


__all__ = [
    "ADMISSION_STATUS",
    "ATOMIC_TOOLS",
    "ATOMIC_OPERATOR_PROFILES",
    "ATOMIC_OPERATOR_PROFILE_MICRO",
    "ATOMIC_OPERATOR_PROFILE_SEMANTIC",
    "BACKEND",
    "BINARY_EXPRESSION_OPERATORS",
    "CANONICAL_TYPES",
    "CAPABILITY_MANIFESTS",
    "CARRIERS",
    "CARRIER_ABLATION_PROTOCOL_VERSION",
    "CARRIER_NATIVE_TOOL_CALLS",
    "CARRIER_POLICY_VERSION",
    "CARRIER_TEXT_JSON",
    "CHECKPOINT_GUIDANCE_PROFILES",
    "CHECKPOINT_GUIDANCE_PROFILE_RESTORE_TARGET",
    "CHECKPOINT_GUIDANCE_PROFILE_RESTORE_PROBE",
    "CHECKPOINT_GUIDANCE_PROFILE_RESTORE_TRIGGER",
    "CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE",
    "CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V2",
    "CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V3",
    "CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V4",
    "CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V5",
    "CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V6",
    "CHECKPOINT_GUIDANCE_PROFILE_STANDARD",
    "CHECKPOINT_GUIDANCE_PROFILE_STRESS",
    "checkpoint_commit_eligibility_for_guidance_profile",
    "checkpoint_commit_eligibility_manifest",
    "COMPARISON_OPERATORS",
    "CHECKPOINT_POLICY_VERSION",
    "CONTROL_TOOLS",
    "DEFAULT_CARRIER",
    "DEFAULT_CHECKPOINT_GUIDANCE_PROFILE",
    "DEFAULT_ATOMIC_OPERATOR_PROFILE",
    "MAX_AST_DEPTH",
    "MAX_EXPRESSION_DEPTH",
    "MODE_TOOLS",
    "MODE_TOOL_NAMES",
    "MODEL_ARG_SCHEMA",
    "MODES",
    "NATIVE_ASSISTANT_CARRIER",
    "TEXT_JSON_ASSISTANT_CARRIER",
    "DIALECT",
    "ENVIRONMENT_RENDERER_VERSION",
    "EXECUTOR_VERSION",
    "PARAMETER_SCHEMAS",
    "PERCEPTION_TOOLS",
    "SEMANTIC_ATOMIC_TOOLS",
    "PREDICATE_OPERATORS",
    "PROTOCOL_VERSION",
    "ProtocolValidationError",
    "SCHEME",
    "SHARED_DEFS",
    "TOOL_DEFINITIONS",
    "TOOL_SCHEME",
    "TOOL_SPECS",
    "TOOLS",
    "UNARY_EXPRESSION_OPERATORS",
    "VARIABLE_EXPRESSION_ARITY",
    "build_system_prompt",
    "assistant_carrier_protocol",
    "carrier_experiment_arm",
    "carrier_protocol_hash",
    "capability_manifest",
    "get_capability_manifest",
    "get_system_prompt",
    "get_tool_definitions",
    "normalize_mode",
    "normalize_carrier",
    "normalize_checkpoint_guidance_profile",
    "normalize_atomic_operator_profile",
    "parameter_schema",
    "prompt_hash",
    "provider_tool_definitions",
    "tool_schema_hash",
    "tools_for_mode",
    "tools_for_profile",
    "validate_arguments",
    "validate_model_action",
    "validate_tool_call",
]
