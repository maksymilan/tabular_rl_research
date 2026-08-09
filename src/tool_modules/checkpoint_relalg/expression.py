"""Typed, backend-independent expression validation for checkpoint-relalg-v1.

The atomic protocol deliberately does not accept SQL snippets.  This module is
the only path from the model-visible expression AST to SQL: column references
are resolved against an exact schema, operators come from a closed enum, and
all literal values are returned as DB-API parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
import re
from typing import Any, Mapping, Sequence

from .errors import CheckpointRelalgError


CANONICAL_TYPES = frozenset(
    {"NULL", "BOOLEAN", "INTEGER", "REAL", "TEXT", "DATE", "DATETIME", "BLOB"}
)
NUMERIC_TYPES = frozenset({"INTEGER", "REAL"})
MAX_AST_DEPTH = 5
MAX_VARIADIC_ARGS = 8

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BINARY_NUMERIC = frozenset({"add", "subtract", "multiply", "divide", "modulo"})
_UNARY_TEXT = frozenset({"lower", "upper", "trim", "length"})
_DATE_PARTS = frozenset({"extract_year", "extract_month", "extract_day"})
_FUNCTIONS = frozenset(
    {
        *_BINARY_NUMERIC,
        "abs",
        "round",
        *_UNARY_TEXT,
        "concat",
        "coalesce",
        "cast",
        *_DATE_PARTS,
        "date_diff_days",
        "case_when",
    }
)


class RelAlgError(CheckpointRelalgError):
    """Structured error surfaced by the relational operator layer."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__("argument_validation_error", code, message, dict(details or {}))
        self.path = path
        if path is not None:
            self.details.setdefault("path", path)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.path is not None:
            result["path"] = self.path
        if self.details:
            result["details"] = self.details
        return result


class RelAlgValidationError(RelAlgError):
    """The typed AST or an operator argument is invalid."""


class RelAlgExecutionError(RelAlgError):
    """A validated operation could not be executed atomically."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        CheckpointRelalgError.__init__(
            self, "execution_error", code, message, dict(details or {})
        )
        self.path = path
        if path is not None:
            self.details.setdefault("path", path)


class NonrecoverableExecutionError(RelAlgExecutionError):
    """Execution failed and the logical state was unexpectedly mutated."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        CheckpointRelalgError.__init__(
            self, "nonrecoverable_execution_error", code, message, dict(details or {})
        )
        self.path = path
        if path is not None:
            self.details.setdefault("path", path)


class RelAlgStateValidationError(RelAlgValidationError):
    """An exact model reference is absent from the active resident state."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        CheckpointRelalgError.__init__(
            self, "state_validation_error", code, message, dict(details or {})
        )
        self.path = path
        if path is not None:
            self.details.setdefault("path", path)


# Public four-field spelling used by the runner integration contract.  Internal
# compiler errors use the narrower RelAlgExecutionError convenience subclass.
RelalgExecutionError = CheckpointRelalgError


@dataclass(frozen=True)
class CompiledExpression:
    sql: str
    params: tuple[Any, ...]
    canonical_type: str
    # Literal identity is retained only for contextual DATE/DATETIME checks.
    literal_value: Any = _IDENTIFIER_RE
    column_name: str | None = None

    @property
    def is_literal(self) -> bool:
        return self.literal_value is not _IDENTIFIER_RE


def quote_identifier(identifier: str) -> str:
    """Quote one SQLite identifier without interpreting dots as qualifiers."""

    if not isinstance(identifier, str) or not identifier:
        raise RelAlgValidationError("invalid_identifier", "identifier must be non-empty")
    return '"' + identifier.replace('"', '""') + '"'


def require_simple_identifier(
    value: Any, *, path: str, code: str = "invalid_identifier"
) -> str:
    """Validate aliases and roles (logical joined columns may contain dots)."""

    if not isinstance(value, str) or _IDENTIFIER_RE.fullmatch(value) is None:
        raise RelAlgValidationError(
            code,
            "expected an identifier matching [A-Za-z_][A-Za-z0-9_]*",
            path=path,
        )
    return value


def normalize_canonical_type(value: Any, *, path: str = "type") -> str:
    candidate = getattr(value, "value", value)
    if not isinstance(candidate, str):
        raise RelAlgValidationError("invalid_type", "canonical type must be a string", path=path)
    candidate = candidate.upper()
    if candidate not in CANONICAL_TYPES:
        raise RelAlgValidationError(
            "invalid_type", f"unsupported canonical type {candidate!r}", path=path
        )
    return candidate


def column_type_map(columns: Sequence[Any]) -> dict[str, str]:
    """Build an exact-name schema map from Column dataclasses or mappings."""

    result: dict[str, str] = {}
    for index, column in enumerate(columns):
        if isinstance(column, Mapping):
            name = column.get("name")
            canonical_type = column.get("canonical_type", column.get("type"))
        else:
            name = getattr(column, "name", None)
            canonical_type = getattr(column, "canonical_type", getattr(column, "type", None))
        if not isinstance(name, str) or not name:
            raise RelAlgValidationError(
                "invalid_schema", "column name must be non-empty", path=f"columns[{index}].name"
            )
        if name in result:
            raise RelAlgValidationError(
                "invalid_schema", f"duplicate column name {name!r}", path=f"columns[{index}].name"
            )
        result[name] = normalize_canonical_type(
            canonical_type, path=f"columns[{index}].canonical_type"
        )
    return result


def literal_type(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RelAlgValidationError(
                "invalid_literal", "REAL literals must be finite JSON numbers"
            )
        return "REAL"
    if isinstance(value, str):
        return "TEXT"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "BLOB"
    raise RelAlgValidationError(
        "invalid_literal", f"unsupported literal value type {type(value).__name__}"
    )


def merge_types(left: str, right: str, *, path: str, equality: bool = False) -> str:
    """Return the common canonical type or raise ``type_mismatch``.

    NULL is an unknown value of the counterpart type. INTEGER/REAL is the only
    implicit promotion. BLOB and BOOLEAN are equality-compatible with their own
    type but never enter ordered/string/numeric operations.
    """

    left = normalize_canonical_type(left, path=path)
    right = normalize_canonical_type(right, path=path)
    if left == "NULL":
        return right
    if right == "NULL":
        return left
    if left == right:
        return left
    if left in NUMERIC_TYPES and right in NUMERIC_TYPES:
        return "REAL"
    raise RelAlgValidationError(
        "type_mismatch",
        f"incompatible canonical types {left} and {right}",
        path=path,
        details={"left_type": left, "right_type": right, "equality": equality},
    )


def validate_iso_literal(value: Any, target_type: str, *, path: str) -> None:
    """Validate a TEXT literal used in a DATE/DATETIME context."""

    if not isinstance(value, str):
        return
    try:
        if target_type == "DATE":
            # DATE is deliberately strict: datetime strings are not dates.
            if "T" in value or " " in value:
                raise ValueError
            date.fromisoformat(value)
        elif target_type == "DATETIME":
            datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RelAlgValidationError(
            "invalid_date_literal",
            f"{target_type} literal must use ISO-8601",
            path=path,
        ) from exc


def compatible_expressions(
    left: CompiledExpression,
    right: CompiledExpression,
    *,
    path: str,
    equality: bool = False,
) -> str:
    """Merge expression types, contextually admitting ISO date literals."""

    lt, rt = left.canonical_type, right.canonical_type
    if lt in {"DATE", "DATETIME"} and rt == "TEXT" and right.is_literal:
        validate_iso_literal(right.literal_value, lt, path=path)
        return lt
    if rt in {"DATE", "DATETIME"} and lt == "TEXT" and left.is_literal:
        validate_iso_literal(left.literal_value, rt, path=path)
        return rt
    return merge_types(lt, rt, path=path, equality=equality)


def _expect_object(node: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise RelAlgValidationError("invalid_expression", "expression must be an object", path=path)
    return node


def _exact_keys(
    node: Mapping[str, Any], required: set[str], optional: set[str] = set(), *, path: str
) -> None:
    keys = set(node)
    missing = required - keys
    extra = keys - required - optional
    if missing:
        raise RelAlgValidationError(
            "invalid_expression",
            f"missing required fields: {', '.join(sorted(missing))}",
            path=path,
        )
    if extra:
        raise RelAlgValidationError(
            "invalid_expression",
            f"unknown fields: {', '.join(sorted(extra))}",
            path=path,
        )


def _args(
    node: Mapping[str, Any], *, path: str, minimum: int, maximum: int
) -> list[Any]:
    value = node.get("args")
    if not isinstance(value, list) or not minimum <= len(value) <= maximum:
        qualifier = str(minimum) if minimum == maximum else f"{minimum}..{maximum}"
        raise RelAlgValidationError(
            "invalid_expression_arity", f"args must contain {qualifier} expressions", path=path
        )
    return value


class ExpressionCompiler:
    """Validate and compile the finite v1 expression AST."""

    def __init__(self, columns: Mapping[str, str] | Sequence[Any], *, table_alias: str = "src"):
        self.columns = (
            {name: normalize_canonical_type(value) for name, value in columns.items()}
            if isinstance(columns, Mapping)
            else column_type_map(columns)
        )
        self.table_alias = table_alias

    def compile(
        self, node: Any, *, path: str = "expression", depth: int = 1
    ) -> CompiledExpression:
        if depth > MAX_AST_DEPTH:
            raise RelAlgValidationError(
                "expression_depth_exceeded",
                f"combined expression/predicate AST depth exceeds {MAX_AST_DEPTH}",
                path=path,
            )
        obj = _expect_object(node, path=path)
        if "column" in obj:
            _exact_keys(obj, {"column"}, path=path)
            name = obj["column"]
            if not isinstance(name, str) or name not in self.columns:
                raise RelAlgStateValidationError(
                    "unknown_column",
                    f"unknown exact column {name!r}",
                    path=f"{path}.column",
                    details={"available_columns": list(self.columns)},
                )
            sql = f"{quote_identifier(self.table_alias)}.{quote_identifier(name)}"
            if self.columns[name] in {"TEXT", "DATE", "DATETIME"}:
                # Source DDL collations are backend metadata, not part of the
                # Atomic IR.  Freeze textual comparison semantics so an
                # operation has the same result before and after CTAS
                # materialization (which otherwise drops source collation).
                sql = f"({sql} COLLATE BINARY)"
            return CompiledExpression(sql, (), self.columns[name], column_name=name)
        if "value" in obj and "op" not in obj:
            _exact_keys(obj, {"value"}, path=path)
            value = obj["value"]
            return CompiledExpression("?", (value,), literal_type(value), literal_value=value)
        if "op" not in obj:
            raise RelAlgValidationError(
                "invalid_expression", "expected column, value, or op expression", path=path
            )
        op = obj["op"]
        if not isinstance(op, str) or op not in _FUNCTIONS:
            raise RelAlgValidationError(
                "unsupported_expression_operator",
                f"unsupported expression operator {op!r}",
                path=f"{path}.op",
            )
        if op == "case_when":
            return self._compile_case(obj, path=path, depth=depth)
        if op == "cast":
            _exact_keys(obj, {"op", "args", "to"}, path=path)
            args = _args(obj, path=f"{path}.args", minimum=1, maximum=1)
            child = self.compile(args[0], path=f"{path}.args[0]", depth=depth + 1)
            target = normalize_canonical_type(obj["to"], path=f"{path}.to")
            function = {
                "NULL": "_relalg_cast_null",
                "BOOLEAN": "_relalg_cast_boolean",
                "INTEGER": "_relalg_cast_integer",
                "REAL": "_relalg_cast_real",
                "TEXT": "_relalg_cast_text",
                "DATE": "_relalg_cast_date",
                "DATETIME": "_relalg_cast_datetime",
                "BLOB": "_relalg_cast_blob",
            }[target]
            return CompiledExpression(
                f"{function}({child.sql})", child.params, target
            )

        _exact_keys(obj, {"op", "args"}, path=path)
        if op in _BINARY_NUMERIC or op == "date_diff_days":
            raw_args = _args(obj, path=f"{path}.args", minimum=2, maximum=2)
        elif op in {"abs", *_UNARY_TEXT, *_DATE_PARTS}:
            raw_args = _args(obj, path=f"{path}.args", minimum=1, maximum=1)
        elif op == "round":
            raw_args = _args(obj, path=f"{path}.args", minimum=1, maximum=2)
        else:
            raw_args = _args(obj, path=f"{path}.args", minimum=2, maximum=MAX_VARIADIC_ARGS)
        args = [
            self.compile(arg, path=f"{path}.args[{index}]", depth=depth + 1)
            for index, arg in enumerate(raw_args)
        ]
        params = tuple(value for arg in args for value in arg.params)

        if op in _BINARY_NUMERIC:
            common = merge_types(args[0].canonical_type, args[1].canonical_type, path=path)
            if common not in NUMERIC_TYPES and common != "NULL":
                raise RelAlgValidationError(
                    "type_mismatch", f"{op} requires numeric operands", path=path
                )
            if op == "modulo" and common not in {"INTEGER", "NULL"}:
                raise RelAlgValidationError(
                    "type_mismatch", "modulo requires INTEGER operands", path=path
                )
            symbol = {"add": "+", "subtract": "-", "multiply": "*", "divide": "/", "modulo": "%"}[op]
            if op == "divide":
                # The harness UDF enforces REAL division and reports zero
                # denominators instead of inheriting SQLite's NULL-on-zero.
                sql = f"_relalg_divide({args[0].sql}, {args[1].sql})"
                output_type = "REAL"
            else:
                sql = f"({args[0].sql} {symbol} {args[1].sql})"
                output_type = common
            return CompiledExpression(sql, params, output_type)
        if op == "abs":
            if args[0].canonical_type not in NUMERIC_TYPES | {"NULL"}:
                raise RelAlgValidationError("type_mismatch", "abs requires a numeric operand", path=path)
            return CompiledExpression(f"abs({args[0].sql})", params, args[0].canonical_type)
        if op == "round":
            if args[0].canonical_type not in NUMERIC_TYPES | {"NULL"}:
                raise RelAlgValidationError("type_mismatch", "round requires a numeric operand", path=path)
            if len(args) == 2 and args[1].canonical_type not in {"INTEGER", "NULL"}:
                raise RelAlgValidationError(
                    "type_mismatch", "round precision must be INTEGER", path=f"{path}.args[1]"
                )
            return CompiledExpression(
                f"round({', '.join(arg.sql for arg in args)})", params, "REAL"
            )
        if op in {"lower", "upper", "trim", "length"}:
            if args[0].canonical_type not in {"TEXT", "NULL"}:
                raise RelAlgValidationError(
                    "type_mismatch", f"{op} requires a TEXT operand", path=path
                )
            output_type = "INTEGER" if op == "length" else "TEXT"
            return CompiledExpression(f"{op}({args[0].sql})", params, output_type)
        if op == "concat":
            if any(arg.canonical_type not in {"TEXT", "NULL"} for arg in args):
                raise RelAlgValidationError(
                    "type_mismatch", "concat requires TEXT operands", path=path
                )
            return CompiledExpression(
                "(" + " || ".join(arg.sql for arg in args) + ")", params, "TEXT"
            )
        if op == "coalesce":
            output_type = args[0].canonical_type
            for arg in args[1:]:
                output_type = merge_types(output_type, arg.canonical_type, path=path)
            return CompiledExpression(
                f"coalesce({', '.join(arg.sql for arg in args)})", params, output_type
            )
        if op in _DATE_PARTS:
            if args[0].canonical_type not in {"DATE", "DATETIME", "NULL"}:
                raise RelAlgValidationError(
                    "type_mismatch", f"{op} requires DATE or DATETIME", path=path
                )
            function = f"_relalg_{op}"
            return CompiledExpression(f"{function}({args[0].sql})", params, "INTEGER")
        if op == "date_diff_days":
            for index, arg in enumerate(args):
                if arg.canonical_type not in {"DATE", "DATETIME", "NULL"}:
                    raise RelAlgValidationError(
                        "type_mismatch",
                        "date_diff_days requires DATE or DATETIME operands",
                        path=f"{path}.args[{index}]",
                    )
            return CompiledExpression(
                f"_relalg_date_diff_days({args[0].sql}, {args[1].sql})", params, "REAL"
            )
        raise AssertionError(f"unhandled expression operator: {op}")

    def _compile_case(
        self, node: Mapping[str, Any], *, path: str, depth: int
    ) -> CompiledExpression:
        _exact_keys(node, {"op", "branches", "else"}, path=path)
        branches = node["branches"]
        if not isinstance(branches, list) or not branches:
            raise RelAlgValidationError(
                "invalid_expression_arity",
                "case_when branches must contain at least one item",
                path=f"{path}.branches",
            )
        # Local import avoids an expression/predicate import cycle.
        from .predicate import PredicateCompiler

        predicate_compiler = PredicateCompiler(self.columns, table_alias=self.table_alias)
        sql_parts: list[str] = ["CASE"]
        params: list[Any] = []
        output_type = "NULL"
        for index, raw_branch in enumerate(branches):
            branch_path = f"{path}.branches[{index}]"
            branch = _expect_object(raw_branch, path=branch_path)
            _exact_keys(branch, {"when", "then"}, path=branch_path)
            condition = predicate_compiler.compile(
                branch["when"], path=f"{branch_path}.when", depth=depth + 1
            )
            then = self.compile(branch["then"], path=f"{branch_path}.then", depth=depth + 1)
            output_type = merge_types(output_type, then.canonical_type, path=branch_path)
            sql_parts.append(f"WHEN {condition.sql} THEN {then.sql}")
            params.extend(condition.params)
            params.extend(then.params)
        otherwise = self.compile(node["else"], path=f"{path}.else", depth=depth + 1)
        output_type = merge_types(output_type, otherwise.canonical_type, path=f"{path}.else")
        sql_parts.append(f"ELSE {otherwise.sql} END")
        params.extend(otherwise.params)
        return CompiledExpression(" ".join(sql_parts), tuple(params), output_type)


def compile_expression(
    node: Any,
    columns: Mapping[str, str] | Sequence[Any],
    *,
    table_alias: str = "src",
) -> CompiledExpression:
    return ExpressionCompiler(columns, table_alias=table_alias).compile(node)


__all__ = [
    "CANONICAL_TYPES",
    "MAX_AST_DEPTH",
    "CompiledExpression",
    "ExpressionCompiler",
    "NonrecoverableExecutionError",
    "RelAlgError",
    "RelAlgExecutionError",
    "RelAlgStateValidationError",
    "RelAlgValidationError",
    "RelalgExecutionError",
    "column_type_map",
    "compatible_expressions",
    "compile_expression",
    "literal_type",
    "merge_types",
    "normalize_canonical_type",
    "quote_identifier",
    "require_simple_identifier",
]
