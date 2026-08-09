"""Typed SQL-three-valued predicates for checkpoint-relalg-v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .expression import (
    MAX_AST_DEPTH,
    CompiledExpression,
    ExpressionCompiler,
    RelAlgValidationError,
    column_type_map,
    compatible_expressions,
)


_COMPARISONS = frozenset({"=", "!=", ">", ">=", "<", "<="})
_LOGIC = frozenset({"and", "or"})
_MEMBERSHIP = frozenset({"in", "not_in"})
_PATTERNS = frozenset({"like", "not_like", "contains", "not_contains"})
_NULL_TESTS = frozenset({"is_null", "is_not_null"})
_ALL = _COMPARISONS | _LOGIC | _MEMBERSHIP | _PATTERNS | _NULL_TESTS | {"not", "between"}


@dataclass(frozen=True)
class CompiledPredicate:
    sql: str
    params: tuple[Any, ...]


def _object(node: Any, *, path: str) -> Mapping[str, Any]:
    if not isinstance(node, Mapping):
        raise RelAlgValidationError("invalid_predicate", "predicate must be an object", path=path)
    return node


def _keys(node: Mapping[str, Any], required: set[str], *, path: str) -> None:
    keys = set(node)
    missing = required - keys
    extra = keys - required
    if missing:
        raise RelAlgValidationError(
            "invalid_predicate",
            f"missing required fields: {', '.join(sorted(missing))}",
            path=path,
        )
    if extra:
        raise RelAlgValidationError(
            "invalid_predicate", f"unknown fields: {', '.join(sorted(extra))}", path=path
        )


class PredicateCompiler:
    """Compile only the frozen v1 predicate shapes.

    SQLite's WHERE/ON operators implement SQL three-valued logic: only TRUE
    rows survive filtering, comparisons with NULL are UNKNOWN, and NOT/AND/OR
    preserve UNKNOWN.  The validator prevents SQLite's implicit type coercions
    from becoming protocol semantics.
    """

    def __init__(self, columns: Mapping[str, str] | Sequence[Any], *, table_alias: str = "src"):
        self.columns = (
            dict(columns) if isinstance(columns, Mapping) else column_type_map(columns)
        )
        self.expressions = ExpressionCompiler(self.columns, table_alias=table_alias)

    def compile(
        self, node: Any, *, path: str = "conditions", depth: int = 1
    ) -> CompiledPredicate:
        if depth > MAX_AST_DEPTH:
            raise RelAlgValidationError(
                "expression_depth_exceeded",
                f"combined expression/predicate AST depth exceeds {MAX_AST_DEPTH}",
                path=path,
            )
        obj = _object(node, path=path)
        op = obj.get("op")
        if not isinstance(op, str) or op not in _ALL:
            raise RelAlgValidationError(
                "unsupported_predicate_operator",
                f"unsupported predicate operator {op!r}",
                path=f"{path}.op",
            )
        if op in _COMPARISONS:
            _keys(obj, {"op", "left", "right"}, path=path)
            left = self.expressions.compile(obj["left"], path=f"{path}.left", depth=depth + 1)
            right = self.expressions.compile(obj["right"], path=f"{path}.right", depth=depth + 1)
            common = compatible_expressions(
                left, right, path=path, equality=op in {"=", "!="}
            )
            if common in {"BOOLEAN", "BLOB"} and op not in {"=", "!="}:
                raise RelAlgValidationError(
                    "type_mismatch", f"{common} supports equality predicates only", path=path
                )
            sql_op = "<>" if op == "!=" else op
            return CompiledPredicate(
                f"({left.sql} {sql_op} {right.sql})", left.params + right.params
            )
        if op in _LOGIC:
            _keys(obj, {"op", "args"}, path=path)
            args = obj["args"]
            if not isinstance(args, list) or len(args) < 2:
                raise RelAlgValidationError(
                    "invalid_predicate_arity",
                    f"{op} requires at least 2 predicates",
                    path=f"{path}.args",
                )
            compiled = [
                self.compile(arg, path=f"{path}.args[{index}]", depth=depth + 1)
                for index, arg in enumerate(args)
            ]
            params = tuple(value for item in compiled for value in item.params)
            return CompiledPredicate(
                "(" + f" {op.upper()} ".join(item.sql for item in compiled) + ")", params
            )
        if op == "not":
            _keys(obj, {"op", "arg"}, path=path)
            arg = self.compile(obj["arg"], path=f"{path}.arg", depth=depth + 1)
            return CompiledPredicate(f"(NOT {arg.sql})", arg.params)
        if op == "between":
            _keys(obj, {"op", "value", "lower", "upper"}, path=path)
            value = self.expressions.compile(obj["value"], path=f"{path}.value", depth=depth + 1)
            lower = self.expressions.compile(obj["lower"], path=f"{path}.lower", depth=depth + 1)
            upper = self.expressions.compile(obj["upper"], path=f"{path}.upper", depth=depth + 1)
            common = compatible_expressions(value, lower, path=f"{path}.lower")
            compatible_expressions(
                CompiledExpression(value.sql, value.params, common, value.literal_value, value.column_name),
                upper,
                path=f"{path}.upper",
            )
            if common in {"BOOLEAN", "BLOB"}:
                raise RelAlgValidationError(
                    "type_mismatch", f"between does not support {common}", path=path
                )
            return CompiledPredicate(
                f"({value.sql} BETWEEN {lower.sql} AND {upper.sql})",
                value.params + lower.params + upper.params,
            )
        if op in _MEMBERSHIP:
            _keys(obj, {"op", "value", "values"}, path=path)
            value = self.expressions.compile(obj["value"], path=f"{path}.value", depth=depth + 1)
            raw_values = obj["values"]
            if not isinstance(raw_values, list) or not 1 <= len(raw_values) <= 50:
                raise RelAlgValidationError(
                    "invalid_predicate_arity",
                    "in/not_in values must contain 1..50 literal expressions",
                    path=f"{path}.values",
                )
            values: list[CompiledExpression] = []
            for index, raw in enumerate(raw_values):
                item_path = f"{path}.values[{index}]"
                if not isinstance(raw, Mapping) or set(raw) != {"value"}:
                    raise RelAlgValidationError(
                        "invalid_predicate",
                        "in/not_in accepts literal expressions only",
                        path=item_path,
                    )
                item = self.expressions.compile(raw, path=item_path, depth=depth + 1)
                compatible_expressions(value, item, path=item_path, equality=True)
                values.append(item)
            keyword = "NOT IN" if op == "not_in" else "IN"
            params = value.params + tuple(v for item in values for v in item.params)
            return CompiledPredicate(
                f"({value.sql} {keyword} ({', '.join(item.sql for item in values)}))", params
            )
        if op in _PATTERNS:
            _keys(obj, {"op", "value", "pattern"}, path=path)
            value = self.expressions.compile(obj["value"], path=f"{path}.value", depth=depth + 1)
            pattern = self.expressions.compile(
                obj["pattern"], path=f"{path}.pattern", depth=depth + 1
            )
            for item, item_path in ((value, f"{path}.value"), (pattern, f"{path}.pattern")):
                if item.canonical_type not in {"TEXT", "NULL"}:
                    raise RelAlgValidationError(
                        "type_mismatch", f"{op} requires TEXT expressions", path=item_path
                    )
            if op in {"like", "not_like"}:
                sql = f"({value.sql} {'NOT LIKE' if op == 'not_like' else 'LIKE'} {pattern.sql})"
            else:
                contains = f"(instr({value.sql}, {pattern.sql}) > 0)"
                sql = f"(NOT {contains})" if op == "not_contains" else contains
            return CompiledPredicate(sql, value.params + pattern.params)
        if op in _NULL_TESTS:
            _keys(obj, {"op", "value"}, path=path)
            value = self.expressions.compile(obj["value"], path=f"{path}.value", depth=depth + 1)
            return CompiledPredicate(
                f"({value.sql} IS {'NOT ' if op == 'is_not_null' else ''}NULL)", value.params
            )
        raise AssertionError(f"unhandled predicate operator: {op}")


def compile_predicate(
    node: Any,
    columns: Mapping[str, str] | Sequence[Any],
    *,
    table_alias: str = "src",
) -> CompiledPredicate:
    return PredicateCompiler(columns, table_alias=table_alias).compile(node)


__all__ = ["CompiledPredicate", "PredicateCompiler", "compile_predicate"]
