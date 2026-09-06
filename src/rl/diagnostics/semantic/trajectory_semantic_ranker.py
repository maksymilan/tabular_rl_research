#!/usr/bin/env python3
"""Deterministic trajectory-level semantic ranking for atomic version26 rollouts.

This module is intentionally an offline/reward-side diagnostic.  Gold SQL is compiled into a
small, path-independent semantic IR; the terminal evidence relation's recorded
``relation-derivation-v1`` lineage is compiled into the same IR.  No gold action sequence is ever
constructed and no model-authored reasoning is treated as evidence.

The first research use is trajectory-level ranking, not turn-level credit:

    quality = 0.10 * schema + 0.45 * semantic + 0.45 * answer
    reward  = 1                                      if exact result is correct
              -1 + failure_quality_scale * quality  otherwise

Correct trajectories deliberately receive the same optimization reward even when this diagnostic
semantic normalizer cannot recognize an equivalent implementation.  ``raw_quality`` remains
available to audit exactly those false-low correct trajectories before any RL run.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from sqlglot import exp, parse_one


SEMANTIC_CLASS_WEIGHTS: dict[str, float] = {
    "join": 0.20,
    "predicate": 0.20,
    "grain": 0.15,
    "aggregate_compute": 0.25,
    "set_distinct": 0.10,
    "rank_limit": 0.10,
}
QUALITY_WEIGHTS: dict[str, float] = {
    "schema": 0.10,
    "semantic": 0.45,
    "answer": 0.45,
}
SUPPORTED_AGGREGATES = frozenset({"avg", "sum", "count", "min", "max"})


class SemanticCompilationError(ValueError):
    """The semantic IR could not be compiled without guessing."""


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _identifier(value: Any) -> str:
    return str(value).strip().strip('"`[]').casefold()


def _literal(value: Any) -> list[Any]:
    if value is None:
        return ["literal", "null", None]
    if isinstance(value, bool):
        return ["literal", "bool", value]
    if isinstance(value, int):
        return ["literal", "number", value]
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return ["literal", "number", str(value)]
        return ["literal", "number", value]
    return ["literal", "text", str(value)]


def _sorted_operands(name: str, left: Any, right: Any) -> list[Any]:
    values = [left, right]
    if name in {"eq", "neq", "add", "mul"}:
        values.sort(key=_json_key)
    return [name, *values]


def _flatten_boolean(name: str, node: Any) -> list[Any]:
    values: list[Any] = []

    def visit(value: Any) -> None:
        if isinstance(value, list) and value and value[0] == name:
            for child in value[1:]:
                visit(child)
        else:
            values.append(value)

    visit(node)
    values.sort(key=_json_key)
    return [name, *values]


def _unit(kind: str, payload: Any) -> str:
    return _json_key({"kind": kind, "payload": payload})


def _unit_kind(unit: str) -> str:
    return str(json.loads(unit)["kind"])


def _counter_union_max(counters: Iterable[Counter[str]]) -> Counter[str]:
    result: Counter[str] = Counter()
    for counter in counters:
        result |= counter
    return result


def _set_jaccard(left: set[Any], right: set[Any]) -> float | None:
    if not left and not right:
        return None
    union = left | right
    return len(left & right) / len(union) if union else None


def _weighted_mean(items: Iterable[tuple[float | None, float]]) -> float | None:
    retained = [(value, weight) for value, weight in items if value is not None]
    if not retained:
        return None
    denominator = sum(weight for _, weight in retained)
    return sum(float(value) * weight for value, weight in retained) / denominator


def _row_key(row: Iterable[Any]) -> str:
    encoded = []
    for value in row:
        if isinstance(value, bytes):
            encoded.append({"type": "bytes", "value": value.hex()})
        else:
            encoded.append({"type": type(value).__name__, "value": value})
    return _json_key(encoded)


@dataclass(frozen=True)
class CompiledSemantics:
    units: Counter[str]
    tables: frozenset[str]
    columns: frozenset[str]
    eligible: bool
    reasons: tuple[str, ...] = ()

    def by_class(self) -> dict[str, Counter[str]]:
        result = {name: Counter() for name in SEMANTIC_CLASS_WEIGHTS}
        for unit, count in self.units.items():
            result[_unit_kind(unit)][unit] += count
        return result


@dataclass
class Artifact:
    handle: str
    columns: dict[str, Any]
    input_refs: tuple[str, ...]
    own_units: Counter[str]
    root_tables: set[str]
    used_columns: set[str]
    output_columns: tuple[str, ...]
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class SemanticOverlap:
    score: float | None
    tp_weight: float
    fp_weight: float
    fn_weight: float
    per_class: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class TrajectoryQuality:
    correct: bool
    semantic_eligible: bool
    schema_score: float | None
    semantic_score: float | None
    answer_score: float | None
    raw_quality: float
    training_reward: float
    evidence_handle: str | None
    artifact_selection: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GoldSemanticCompiler:
    """Compile one conservative flat SQLite SELECT into path-independent semantic units."""

    def __init__(self, table_columns: Mapping[str, Iterable[str]]):
        self.table_columns = {
            _identifier(table): {_identifier(column) for column in columns}
            for table, columns in table_columns.items()
        }
        self.aliases: dict[str, str] = {}
        self.projection_aliases: dict[str, exp.Expression] = {}
        self.reasons: list[str] = []
        self.tables: set[str] = set()
        self.columns: set[str] = set()

    def compile(self, sql: str) -> CompiledSemantics:
        try:
            tree = parse_one(sql, read="sqlite")
        except Exception as exc:
            return CompiledSemantics(
                units=Counter(),
                tables=frozenset(),
                columns=frozenset(),
                eligible=False,
                reasons=(f"parse_error:{type(exc).__name__}:{exc}",),
            )
        if not isinstance(tree, exp.Select):
            return CompiledSemantics(
                units=Counter(), tables=frozenset(), columns=frozenset(),
                eligible=False, reasons=(f"unsupported_root:{tree.key}",),
            )
        unsupported = []
        for node_type, label in (
            (exp.Subquery, "subquery"),
            (exp.Window, "window"),
        ):
            if any(True for _ in tree.find_all(node_type)):
                unsupported.append(label)
        if tree.args.get("with") is not None:
            unsupported.append("with")
        table_nodes = list(tree.find_all(exp.Table))
        physical = [_identifier(node.name) for node in table_nodes]
        if len(physical) != len(set(physical)):
            unsupported.append("repeated_physical_table")
        for node in table_nodes:
            table = _identifier(node.name)
            alias = _identifier(node.alias_or_name)
            self.aliases[alias] = table
            self.aliases[table] = table
            self.tables.add(table)
        for projection in tree.expressions:
            alias = projection.alias
            if alias:
                self.projection_aliases[_identifier(alias)] = projection.this
        if unsupported:
            return CompiledSemantics(
                units=Counter(), tables=frozenset(self.tables), columns=frozenset(),
                eligible=False, reasons=tuple(sorted(set(unsupported))),
            )

        units: Counter[str] = Counter()
        join_predicate_nodes: set[int] = set()
        for join in tree.args.get("joins") or []:
            join_type = self._join_type(join)
            on = join.args.get("on")
            if join_type == "cross":
                joined = join.this
                table = _identifier(joined.name) if isinstance(joined, exp.Table) else "unknown"
                units[_unit("join", ["cross", table])] += 1
                continue
            if on is None:
                self.reasons.append("join_without_on")
                continue
            atoms = self._and_atoms(on)
            for atom in atoms:
                if isinstance(atom, exp.EQ) and isinstance(atom.this, exp.Column) and isinstance(atom.expression, exp.Column):
                    left = self.expr(atom.this)
                    right = self.expr(atom.expression)
                    if join_type == "inner":
                        left, right = sorted([left, right], key=_json_key)
                    units[_unit("join", [join_type, left, right])] += 1
                    join_predicate_nodes.add(id(atom))
                else:
                    self.reasons.append("non_equality_join_condition")

        where = tree.args.get("where")
        if where is not None:
            for atom in self._and_atoms(where.this):
                if self._is_cross_table_equality(atom):
                    left = self.expr(atom.this)
                    right = self.expr(atom.expression)
                    left, right = sorted([left, right], key=_json_key)
                    units[_unit("join", ["inner", left, right])] += 1
                    continue
                units[_unit("predicate", ["where", self.expr(atom)])] += 1

        having = tree.args.get("having")
        if having is not None:
            for atom in self._and_atoms(having.this):
                units[_unit("predicate", ["having", self.expr(atom)])] += 1

        group = tree.args.get("group")
        group_expressions = list(group.expressions) if group is not None else []
        aggregates = list(tree.find_all(exp.AggFunc))
        if group_expressions:
            for expression in group_expressions:
                units[_unit("grain", self.expr(expression))] += 1
        elif aggregates:
            units[_unit("grain", ["global"])] += 1

        seen_aggregates: set[str] = set()
        for aggregate in aggregates:
            aggregate_unit = _unit("aggregate_compute", self.aggregate(aggregate))
            # Referencing the same aggregate from SELECT, HAVING, or ORDER BY does not perform a
            # second semantic computation.  Multiple genuinely different aggregate expressions
            # remain separate units.
            if aggregate_unit not in seen_aggregates:
                seen_aggregates.add(aggregate_unit)
                units[aggregate_unit] += 1

        for projection in tree.expressions:
            core = projection.this if isinstance(projection, exp.Alias) else projection
            if self._is_nontrivial_compute(core):
                units[_unit("aggregate_compute", ["compute", self.expr(core)])] += 1

        if tree.args.get("distinct") is not None:
            units[_unit("set_distinct", ["distinct"])] += 1

        order = tree.args.get("order")
        if order is not None:
            for index, ordered in enumerate(order.expressions):
                expression = ordered.this if isinstance(ordered, exp.Ordered) else ordered
                direction = "desc" if isinstance(ordered, exp.Ordered) and bool(ordered.args.get("desc")) else "asc"
                units[_unit("rank_limit", ["order", index, self.expr(expression), direction])] += 1
        limit = tree.args.get("limit")
        if limit is not None and limit.expression is not None:
            units[_unit("rank_limit", ["limit", self.expr(limit.expression)])] += 1
        offset = tree.args.get("offset")
        if offset is not None and offset.expression is not None:
            units[_unit("rank_limit", ["offset", self.expr(offset.expression)])] += 1

        eligible = not self.reasons
        return CompiledSemantics(
            units=units,
            tables=frozenset(self.tables),
            columns=frozenset(self.columns),
            eligible=eligible,
            reasons=tuple(sorted(set(self.reasons))),
        )

    @staticmethod
    def _join_type(join: exp.Join) -> str:
        side = str(join.args.get("side") or "").casefold()
        kind = str(join.args.get("kind") or "").casefold()
        if kind == "cross":
            return "cross"
        if side == "left":
            return "left"
        if side in {"right", "full"}:
            return side
        return "inner"

    @staticmethod
    def _and_atoms(node: exp.Expression) -> list[exp.Expression]:
        if isinstance(node, exp.And):
            return GoldSemanticCompiler._and_atoms(node.this) + GoldSemanticCompiler._and_atoms(node.expression)
        return [node]

    def _is_cross_table_equality(self, node: exp.Expression) -> bool:
        if not isinstance(node, exp.EQ) or not isinstance(node.this, exp.Column) or not isinstance(node.expression, exp.Column):
            return False
        left = self.column(node.this)
        right = self.column(node.expression)
        return left[1] != right[1]

    def column(self, node: exp.Column) -> list[Any]:
        column = _identifier(node.name)
        if node.table:
            owner = self.aliases.get(_identifier(node.table))
            if owner is None:
                self.reasons.append(f"unknown_column_owner:{node.table}")
                return ["column", "?", column]
        elif column in self.projection_aliases:
            return self.expr(self.projection_aliases[column])
        else:
            candidates = [
                table for table in self.tables
                if not self.table_columns.get(table) or column in self.table_columns[table]
            ]
            if len(candidates) != 1:
                self.reasons.append(f"ambiguous_column:{column}")
                return ["column", "?", column]
            owner = candidates[0]
        self.columns.add(f"{owner}.{column}")
        return ["column", owner, column]

    def aggregate(self, node: exp.AggFunc) -> list[Any]:
        operation = node.key.casefold()
        if operation not in SUPPORTED_AGGREGATES:
            self.reasons.append(f"unsupported_aggregate:{operation}")
        source = node.this
        distinct = isinstance(source, exp.Distinct)
        if distinct:
            expressions = list(source.expressions)
            source_expr = expressions[0] if len(expressions) == 1 else source
        else:
            source_expr = source
        if source_expr is None:
            rendered_source: Any = ["star"]
        elif isinstance(source_expr, exp.Star):
            rendered_source = ["star"]
        else:
            rendered_source = self.expr(source_expr)
        return ["aggregate", operation, distinct, rendered_source]

    def expr(self, node: exp.Expression | Any) -> Any:
        if node is None:
            return _literal(None)
        if not isinstance(node, exp.Expression):
            return _literal(node)
        if isinstance(node, exp.Alias):
            return self.expr(node.this)
        if isinstance(node, (exp.Paren, exp.Cast, exp.TryCast)):
            return self.expr(node.this)
        if isinstance(node, exp.Column):
            return self.column(node)
        if isinstance(node, exp.Star):
            return ["star"]
        if isinstance(node, exp.Null):
            return _literal(None)
        if isinstance(node, exp.Boolean):
            return _literal(bool(node.this))
        if isinstance(node, exp.Literal):
            if node.is_string:
                return _literal(node.this)
            text = str(node.this)
            try:
                return _literal(int(text))
            except ValueError:
                try:
                    return _literal(float(text))
                except ValueError:
                    return _literal(text)
        if isinstance(node, exp.AggFunc):
            return self.aggregate(node)
        binaries: tuple[tuple[type[exp.Expression], str], ...] = (
            (exp.EQ, "eq"), (exp.NEQ, "neq"), (exp.GT, "gt"), (exp.GTE, "gte"),
            (exp.LT, "lt"), (exp.LTE, "lte"), (exp.Add, "add"), (exp.Sub, "sub"),
            (exp.Mul, "mul"), (exp.Div, "div"),
        )
        for node_type, name in binaries:
            if isinstance(node, node_type):
                left = self.expr(node.this)
                right = self.expr(node.expression)
                if name == "gt":
                    return ["lt", right, left]
                if name == "gte":
                    return ["lte", right, left]
                return _sorted_operands(name, left, right)
        if isinstance(node, exp.And):
            return _flatten_boolean("and", ["and", self.expr(node.this), self.expr(node.expression)])
        if isinstance(node, exp.Or):
            return _flatten_boolean("or", ["or", self.expr(node.this), self.expr(node.expression)])
        if isinstance(node, exp.Not):
            return ["not", self.expr(node.this)]
        if isinstance(node, exp.Between):
            return ["between", self.expr(node.this), self.expr(node.args.get("low")), self.expr(node.args.get("high"))]
        if isinstance(node, exp.In):
            values = sorted((self.expr(item) for item in node.expressions), key=_json_key)
            if node.args.get("query") is not None:
                self.reasons.append("in_subquery")
            return ["in", self.expr(node.this), values]
        if isinstance(node, (exp.Like, exp.ILike)):
            return [node.key.casefold(), self.expr(node.this), self.expr(node.expression)]
        if isinstance(node, exp.Is):
            return ["is", self.expr(node.this), self.expr(node.expression)]
        if isinstance(node, exp.Neg):
            return ["neg", self.expr(node.this)]
        # Generic deterministic fallback covers common SQLite scalar functions (strftime,
        # coalesce, case/iif) while still failing closed on subqueries/windows above.
        arguments = []
        for key in sorted(node.args):
            value = node.args[key]
            if value is None or key in {"alias"}:
                continue
            if isinstance(value, list):
                rendered = [self.expr(item) if isinstance(item, exp.Expression) else item for item in value]
            elif isinstance(value, exp.Expression):
                rendered = self.expr(value)
            else:
                rendered = value
            arguments.append([key, rendered])
        return [node.key.casefold(), *arguments]

    @staticmethod
    def _is_nontrivial_compute(node: exp.Expression) -> bool:
        if isinstance(node, (exp.Column, exp.Literal, exp.Star, exp.AggFunc)):
            return False
        if isinstance(node, (exp.Cast, exp.TryCast, exp.Paren)):
            return GoldSemanticCompiler._is_nontrivial_compute(node.this)
        return any(True for _ in node.find_all(exp.Column, exp.AggFunc))


class TrajectorySemanticCompiler:
    """Compile recorded relation-derivation outputs into the same semantic IR."""

    def __init__(self, table_columns: Mapping[str, Iterable[str]]):
        self.table_columns = {
            _identifier(table): tuple(_identifier(column) for column in columns)
            for table, columns in table_columns.items()
        }
        self.artifacts: dict[str, Artifact] = {}
        self.step_artifacts: dict[str, Artifact] = {}
        self.reasons: list[str] = []

    def source(self, table: str) -> Artifact:
        name = _identifier(table)
        if name in self.artifacts:
            return self.artifacts[name]
        columns: dict[str, Any] = {}
        for column in self.table_columns.get(name, ()):
            expression = ["column", name, column]
            columns[column] = expression
            columns[f"{name}.{column}"] = expression
        artifact = Artifact(
            handle=name,
            columns=columns,
            input_refs=(),
            own_units=Counter(),
            root_tables={name},
            used_columns=set(),
            output_columns=tuple(self.table_columns.get(name, ())),
        )
        self.artifacts[name] = artifact
        return artifact

    def artifact(self, ref: Any) -> Artifact | None:
        if not isinstance(ref, str):
            return None
        return self.artifacts.get(ref) or self.artifacts.get(_identifier(ref)) or self.source(ref)

    @staticmethod
    def _output_columns(output: Mapping[str, Any]) -> tuple[str, ...]:
        values = output.get("columns") or []
        return tuple(str(value) for value in values if isinstance(value, str))

    def resolve(self, artifact: Artifact | None, column: Any, *, record_failure: bool = True) -> Any:
        if not isinstance(column, str):
            return _literal(column)
        name = _identifier(column)
        if artifact is None:
            if record_failure:
                self.reasons.append(f"column_without_input:{name}")
            return ["column", "?", name]
        exact = artifact.columns.get(name)
        if exact is not None:
            return exact
        suffix = name.rsplit(".", 1)[-1]
        matches = {
            _json_key(value): value
            for key, value in artifact.columns.items()
            if key.rsplit(".", 1)[-1] == suffix
        }
        if len(matches) == 1:
            return next(iter(matches.values()))
        if record_failure:
            self.reasons.append(f"ambiguous_trajectory_column:{artifact.handle}:{name}")
        return ["column", "?", name]

    def expression(self, text: Any, artifact: Artifact | None) -> Any:
        if not isinstance(text, str):
            return _literal(text)
        try:
            tree = parse_one(text, read="sqlite")
        except Exception:
            return self.resolve(artifact, text)

        def convert(node: exp.Expression) -> Any:
            if isinstance(node, exp.Alias):
                return convert(node.this)
            if isinstance(node, (exp.Paren, exp.Cast, exp.TryCast)):
                return convert(node.this)
            if isinstance(node, exp.Column):
                rendered = f"{node.table}.{node.name}" if node.table else node.name
                return self.resolve(artifact, rendered)
            if isinstance(node, exp.Star):
                return ["star"]
            if isinstance(node, exp.Literal):
                if node.is_string:
                    return _literal(node.this)
                try:
                    return _literal(int(str(node.this)))
                except ValueError:
                    try:
                        return _literal(float(str(node.this)))
                    except ValueError:
                        return _literal(str(node.this))
            if isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Add, exp.Sub, exp.Mul, exp.Div)):
                names = {
                    exp.EQ: "eq", exp.NEQ: "neq", exp.GT: "gt", exp.GTE: "gte",
                    exp.LT: "lt", exp.LTE: "lte", exp.Add: "add", exp.Sub: "sub",
                    exp.Mul: "mul", exp.Div: "div",
                }
                name = names[type(node)]
                left, right = convert(node.this), convert(node.expression)
                if name == "gt":
                    return ["lt", right, left]
                if name == "gte":
                    return ["lte", right, left]
                return _sorted_operands(name, left, right)
            if isinstance(node, exp.And):
                return _flatten_boolean("and", ["and", convert(node.this), convert(node.expression)])
            if isinstance(node, exp.Or):
                return _flatten_boolean("or", ["or", convert(node.this), convert(node.expression)])
            arguments = []
            for key in sorted(node.args):
                value = node.args[key]
                if value is None or key == "alias":
                    continue
                if isinstance(value, list):
                    rendered = [convert(item) if isinstance(item, exp.Expression) else item for item in value]
                elif isinstance(value, exp.Expression):
                    rendered = convert(value)
                else:
                    rendered = value
                arguments.append([key, rendered])
            return [node.key.casefold(), *arguments]

        return convert(tree)

    def condition(self, condition: Any, artifact: Artifact | None) -> Any:
        if isinstance(condition, list):
            values = [self.condition(item, artifact) for item in condition]
            return _flatten_boolean("and", ["and", *values])
        if not isinstance(condition, dict):
            return _literal(condition)
        if "and" in condition:
            values = [self.condition(item, artifact) for item in condition.get("and") or []]
            return _flatten_boolean("and", ["and", *values])
        if "or" in condition:
            values = [self.condition(item, artifact) for item in condition.get("or") or []]
            return _flatten_boolean("or", ["or", *values])
        if "not" in condition:
            return ["not", self.condition(condition["not"], artifact)]
        left = self.resolve(artifact, condition.get("column"))
        operation = str(condition.get("op") or "eq").casefold()
        if "column_value" in condition:
            right = self.resolve(artifact, condition["column_value"])
        elif "value" in condition:
            right = _literal(condition["value"])
        elif "values" in condition:
            values = sorted((_literal(value) for value in condition.get("values") or []), key=_json_key)
            return ["in", left, values]
        elif operation == "between":
            return ["between", left, _literal(condition.get("low")), _literal(condition.get("high"))]
        elif operation == "is_null":
            return ["is", left, _literal(None)]
        elif "value_ref" in condition:
            source_artifact = self.step_artifacts.get(str(condition["value_ref"]))
            if source_artifact and isinstance(condition.get("column"), str):
                right = self.resolve(source_artifact, condition["column"])
            elif source_artifact and len(source_artifact.output_columns) == 1:
                right = self.resolve(source_artifact, source_artifact.output_columns[0])
            else:
                self.reasons.append(f"unresolved_condition_value_ref:{condition['value_ref']}")
                right = ["value_ref", str(condition["value_ref"]), condition.get("column")]
        else:
            right = _literal(None)
        names = {"=": "eq", "!=": "neq", ">": "gt", ">=": "gte", "<": "lt", "<=": "lte", "contains": "like"}
        operation = names.get(operation, operation)
        if operation == "gt":
            return ["lt", right, left]
        if operation == "gte":
            return ["lte", right, left]
        return _sorted_operands(operation, left, right)

    def compile(self, steps: Iterable[Mapping[str, Any]]) -> tuple[CompiledSemantics, str | None, str]:
        terminal_evidence: str | None = None
        for step in steps:
            call = step.get("tool_call") or {}
            tool = call.get("tool")
            arguments = call.get("arguments") or {}
            output = step.get("tool_output") or {}
            step_id = str(step.get("step_id") or "")
            if tool == "answer_from_context":
                evidence = arguments.get("evidence")
                if isinstance(evidence, dict) and isinstance(evidence.get("table"), str):
                    terminal_evidence = evidence["table"]
                continue
            handle = output.get("table")
            derivation = output.get("derivation")
            if not isinstance(handle, str) or not isinstance(derivation, dict):
                continue
            artifact = self._build_artifact(handle, tool, arguments, output, derivation)
            self.artifacts[handle] = artifact
            self.step_artifacts[step_id] = artifact

        if terminal_evidence and terminal_evidence in self.artifacts:
            selected = terminal_evidence
            selection = "terminal_evidence"
        elif self.artifacts:
            # The caller scores every live artifact and chooses the best when there is no legal
            # terminal.  Return the newest here as a deterministic fallback for direct callers.
            derived = [key for key, value in self.artifacts.items() if value.input_refs]
            selected = derived[-1] if derived else None
            selection = "latest_derived_fallback"
        else:
            selected = None
            selection = "no_artifact"
        compiled = self.compiled_for(selected)
        return compiled, selected, selection

    def compiled_for(self, handle: str | None) -> CompiledSemantics:
        if handle is None or handle not in self.artifacts:
            return CompiledSemantics(
                units=Counter(), tables=frozenset(), columns=frozenset(),
                eligible=True, reasons=(),
            )
        visited: set[str] = set()
        units: Counter[str] = Counter()
        tables: set[str] = set()
        columns: set[str] = set()
        reasons: set[str] = set()

        def visit(ref: str) -> None:
            if ref in visited:
                return
            visited.add(ref)
            artifact = self.artifacts.get(ref)
            if artifact is None:
                source = self.artifact(ref)
                if source:
                    tables.update(source.root_tables)
                return
            for parent in artifact.input_refs:
                visit(parent)
            units.update(artifact.own_units)
            tables.update(artifact.root_tables)
            columns.update(artifact.used_columns)
            reasons.update(artifact.reasons)
            for expression in artifact.columns.values():
                columns.update(_columns_in_expr(expression))

        visit(handle)
        return CompiledSemantics(
            units=units,
            tables=frozenset(tables),
            columns=frozenset(columns),
            eligible=not reasons,
            reasons=tuple(sorted(reasons)),
        )

    def _build_artifact(
        self,
        handle: str,
        tool: str,
        arguments: Mapping[str, Any],
        output: Mapping[str, Any],
        derivation: Mapping[str, Any],
    ) -> Artifact:
        reason_start = len(self.reasons)
        inputs = list(derivation.get("inputs") or [])
        table_inputs = [item for item in inputs if item.get("kind") == "table" and isinstance(item.get("ref"), str)]
        input_refs_list = [str(item["ref"]) for item in table_inputs]
        for item in inputs:
            if item.get("kind") != "value" or not isinstance(item.get("ref"), str):
                continue
            value_artifact = self.step_artifacts.get(str(item["ref"]))
            if value_artifact is None:
                self.reasons.append(f"unresolved_value_ref:{item['ref']}")
            elif value_artifact.handle not in input_refs_list:
                input_refs_list.append(value_artifact.handle)
        input_refs = tuple(input_refs_list)
        input_artifacts = [self.artifact(ref) for ref in input_refs]
        table_artifacts = input_artifacts[:len(table_inputs)]
        input_bindings: dict[str, Artifact] = {}
        for item, artifact in zip(table_inputs, table_artifacts, strict=True):
            if artifact is None:
                continue
            for name in (item.get("namespace"), item.get("ref")):
                if isinstance(name, str):
                    input_bindings[_identifier(name)] = artifact
        roots = {table for artifact in input_artifacts if artifact for table in artifact.root_tables}
        used_columns: set[str] = set()
        own_units: Counter[str] = Counter()
        output_columns = self._output_columns(output)
        semantics = derivation.get("semantics") or {}
        primary = input_artifacts[0] if input_artifacts else None
        scalar_expression: Any | None = None

        if tool == "join_tables":
            accumulator = primary
            edges = list(semantics.get("edges") or [])
            for index, edge in enumerate(edges):
                right = input_artifacts[index + 1] if index + 1 < len(input_artifacts) else None
                join_type = str(edge.get("join_type") or "inner").casefold()
                on = list(edge.get("on") or [])
                if join_type == "cross":
                    right_name = sorted(right.root_tables)[0] if right and right.root_tables else "unknown"
                    own_units[_unit("join", ["cross", right_name])] += 1
                for pair in on:
                    left_expr = self.resolve(accumulator, pair.get("left"))
                    right_expr = self.resolve(right, pair.get("right"))
                    if join_type == "inner":
                        left_expr, right_expr = sorted([left_expr, right_expr], key=_json_key)
                    own_units[_unit("join", [join_type, left_expr, right_expr])] += 1
                    used_columns.update(_columns_in_expr(left_expr))
                    used_columns.update(_columns_in_expr(right_expr))
                accumulator = self._joined_view(f"{handle}#edge{index}", accumulator, right, edge)

        elif tool == "condition_filter":
            stage = "having" if primary and _artifact_has_aggregate(primary, self.artifacts) else "where"
            predicate = self.condition(semantics.get("predicate"), primary)
            own_units[_unit("predicate", [stage, predicate])] += 1
            used_columns.update(_columns_in_expr(predicate))

        elif tool == "group_aggregate":
            group_by = list(semantics.get("row_grain") or [])
            if group_by:
                for column in group_by:
                    expression = self.resolve(primary, column)
                    own_units[_unit("grain", expression)] += 1
                    used_columns.update(_columns_in_expr(expression))
            elif semantics.get("aggregations"):
                own_units[_unit("grain", ["global"])] += 1
            for aggregation in semantics.get("aggregations") or []:
                operation = str(aggregation.get("op") or "").casefold()
                distinct = operation == "count_distinct"
                operation = "count" if distinct else {"mean": "avg"}.get(operation, operation)
                source = aggregation.get("source", "*")
                source_expr = ["star"] if source == "*" else self.expression(source, primary)
                payload: list[Any] = ["aggregate", operation, distinct, source_expr]
                if aggregation.get("predicate") is not None:
                    payload.append(["where", self.condition(aggregation["predicate"], primary)])
                own_units[_unit("aggregate_compute", payload)] += 1
                used_columns.update(_columns_in_expr(payload))

        elif tool == "extreme_value_select":
            for index, item in enumerate(semantics.get("order_by") or []):
                text = str(item)
                match = re.match(r"^(.*?)(?:\s+(ASC|DESC))?$", text, re.I)
                expression = self.expression(match.group(1).strip(), primary) if match else self.expression(text, primary)
                direction = (match.group(2) or "ASC").casefold() if match else "asc"
                own_units[_unit("rank_limit", ["order", index, expression, direction])] += 1
                used_columns.update(_columns_in_expr(expression))
            if semantics.get("top_k") is not None:
                own_units[_unit("rank_limit", ["limit", _literal(semantics["top_k"])])] += 1

        elif tool == "project":
            if semantics.get("row_operation") == "deduplicate":
                own_units[_unit("set_distinct", ["distinct"])] += 1
            for item in semantics.get("column_lineage") or []:
                if item.get("kind") == "expression":
                    expression = self.expression(item.get("expression"), primary)
                    own_units[_unit("aggregate_compute", ["compute", expression])] += 1
                    used_columns.update(_columns_in_expr(expression))

        elif tool == "scalar_compute":
            operation = str(semantics.get("operation") or arguments.get("operation") or "").casefold()
            operands = []
            for operand in arguments.get("operands") or []:
                if isinstance(operand, dict) and "value" in operand:
                    operands.append(_literal(operand["value"]))
                elif isinstance(operand, dict) and isinstance(operand.get("value_ref"), str):
                    source_artifact = self.step_artifacts.get(operand["value_ref"])
                    if source_artifact and isinstance(operand.get("column"), str):
                        operands.append(self.resolve(source_artifact, operand["column"]))
                    elif source_artifact and len(source_artifact.output_columns) == 1:
                        operands.append(self.resolve(source_artifact, source_artifact.output_columns[0]))
                    else:
                        operands.append(["value_ref", operand["value_ref"], operand.get("column")])
                else:
                    operands.append(_literal(operand))
            expression = _compute_expression(operation, operands)
            scalar_expression = expression
            own_units[_unit("aggregate_compute", ["compute", expression])] += 1
            used_columns.update(_columns_in_expr(expression))

        elif tool == "set_op":
            own_units[_unit("set_distinct", ["set", str(semantics.get("operation") or "").casefold()])] += 1

        columns = self._derive_output_lineage(
            tool, output_columns, semantics, primary, input_artifacts, input_bindings
        )
        if tool == "scalar_compute" and scalar_expression is not None:
            result = str(semantics.get("result_column") or (output_columns[0] if output_columns else "value"))
            columns = {_identifier(result): scalar_expression}
        return Artifact(
            handle=handle,
            columns=columns,
            input_refs=input_refs,
            own_units=own_units,
            root_tables=roots,
            used_columns=used_columns,
            output_columns=output_columns,
            reasons=tuple(sorted(set(self.reasons[reason_start:]))),
        )

    def _joined_view(self, handle: str, left: Artifact | None, right: Artifact | None, edge: Mapping[str, Any]) -> Artifact:
        namespace = _identifier(edge.get("namespace") or edge.get("input") or "right")
        columns = dict(left.columns) if left else {}
        if right:
            for key, value in right.columns.items():
                suffix = key.rsplit(".", 1)[-1]
                columns[f"{namespace}.{suffix}"] = value
        roots = set(left.root_tables if left else set()) | set(right.root_tables if right else set())
        return Artifact(handle, columns, (), Counter(), roots, set(), tuple(columns))

    def _derive_output_lineage(
        self,
        tool: str,
        output_columns: tuple[str, ...],
        semantics: Mapping[str, Any],
        primary: Artifact | None,
        inputs: list[Artifact | None],
        input_bindings: Mapping[str, Artifact] | None = None,
    ) -> dict[str, Any]:
        columns: dict[str, Any] = {}
        if tool == "join_tables":
            for name in output_columns:
                prefix, _, suffix = name.partition(".")
                owner = _identifier(prefix)
                # Version26 public joins normally preserve a source/role namespace.  Prefer that
                # exact owner before trying a unique suffix across inputs; failed probes must not
                # make an otherwise unambiguous trajectory ineligible.
                source = (input_bindings or {}).get(owner)
                if source is None:
                    source = next(
                        (artifact for artifact in inputs if artifact and owner in artifact.root_tables),
                        None,
                    )
                if source is not None:
                    columns[_identifier(name)] = self.resolve(source, suffix)
                    continue
                matches = []
                for artifact in inputs:
                    if artifact is None:
                        continue
                    value = self.resolve(
                        artifact,
                        suffix if prefix else name,
                        record_failure=False,
                    )
                    if not (isinstance(value, list) and len(value) > 1 and value[0] == "column" and value[1] == "?"):
                        matches.append(value)
                unique = {_json_key(value): value for value in matches}
                if len(unique) == 1:
                    columns[_identifier(name)] = next(iter(unique.values()))
                else:
                    columns[_identifier(name)] = self.resolve(None, name)
            return columns
        if tool == "project":
            for item in semantics.get("column_lineage") or []:
                output = item.get("output")
                if not isinstance(output, str):
                    continue
                sources = list(item.get("sources") or [])
                if item.get("kind") == "column" and len(sources) == 1:
                    columns[_identifier(output)] = self.resolve(primary, sources[0])
                else:
                    columns[_identifier(output)] = self.expression(item.get("expression"), primary)
            return columns
        if tool == "group_aggregate":
            group_by = list(semantics.get("row_grain") or [])
            for output, source in zip(output_columns, group_by):
                columns[_identifier(output)] = self.resolve(primary, source)
            for aggregation in semantics.get("aggregations") or []:
                output = aggregation.get("output")
                if not isinstance(output, str):
                    continue
                operation = str(aggregation.get("op") or "").casefold()
                distinct = operation == "count_distinct"
                operation = "count" if distinct else {"mean": "avg"}.get(operation, operation)
                source = aggregation.get("source", "*")
                source_expr = ["star"] if source == "*" else self.expression(source, primary)
                columns[_identifier(output)] = ["aggregate", operation, distinct, source_expr]
            return columns
        if tool == "scalar_compute":
            result = str(semantics.get("result_column") or (output_columns[0] if output_columns else "value"))
            columns[_identifier(result)] = ["computed_scalar", _identifier(result)]
            return columns
        for name in output_columns:
            columns[_identifier(name)] = self.resolve(primary, name)
        return columns


def _artifact_has_aggregate(artifact: Artifact, artifacts: Mapping[str, Artifact]) -> bool:
    visited: set[str] = set()

    def visit(current: Artifact) -> bool:
        if current.handle in visited:
            return False
        visited.add(current.handle)
        if any(_unit_kind(unit) == "aggregate_compute" and json.loads(unit)["payload"][0] == "aggregate" for unit in current.own_units):
            return True
        return any(ref in artifacts and visit(artifacts[ref]) for ref in current.input_refs)

    return visit(artifact)


def _columns_in_expr(value: Any) -> set[str]:
    columns: set[str] = set()
    if isinstance(value, list):
        if len(value) == 3 and value[0] == "column" and value[1] != "?":
            columns.add(f"{value[1]}.{value[2]}")
        for item in value:
            columns.update(_columns_in_expr(item))
    elif isinstance(value, dict):
        for item in value.values():
            columns.update(_columns_in_expr(item))
    return columns


def _compute_expression(operation: str, operands: list[Any]) -> Any:
    if len(operands) < 2:
        return [operation, *operands]
    left, right = operands[0], operands[1]
    if operation == "add":
        return _sorted_operands("add", left, right)
    if operation == "subtract":
        return ["sub", left, right]
    if operation == "multiply":
        return _sorted_operands("mul", left, right)
    if operation == "divide":
        return ["div", left, right]
    if operation == "percent":
        return ["div", _sorted_operands("mul", left, _literal(100)), right]
    return [operation, *operands]


def semantic_overlap(gold: CompiledSemantics, candidate: CompiledSemantics) -> SemanticOverlap:
    if not gold.eligible or not candidate.eligible:
        return SemanticOverlap(None, 0.0, 0.0, 0.0, {})
    gold_by = gold.by_class()
    candidate_by = candidate.by_class()
    tp_weight = fp_weight = fn_weight = 0.0
    per_class: dict[str, dict[str, Any]] = {}
    any_gold = False
    for kind, class_weight in SEMANTIC_CLASS_WEIGHTS.items():
        gold_units = gold_by[kind]
        candidate_units = candidate_by[kind]
        gold_count = sum(gold_units.values())
        candidate_count = sum(candidate_units.values())
        if gold_count:
            any_gold = True
        matched = sum((gold_units & candidate_units).values())
        false_negative = max(0, gold_count - matched)
        false_positive = max(0, candidate_count - matched)
        denominator = max(1, gold_count)
        tp = class_weight * matched / denominator
        fn = class_weight * false_negative / denominator
        fp = class_weight * min(1.0, false_positive / denominator)
        tp_weight += tp
        fp_weight += fp
        fn_weight += fn
        per_class[kind] = {
            "gold": gold_count,
            "candidate": candidate_count,
            "matched": matched,
            "tp_weight": tp,
            "fp_weight": fp,
            "fn_weight": fn,
            "missing": sorted((gold_units - candidate_units).elements()),
            "extra": sorted((candidate_units - gold_units).elements()),
        }
    if not any_gold:
        return SemanticOverlap(None, 0.0, fp_weight, 0.0, per_class)
    denominator = tp_weight + fp_weight + fn_weight
    score = tp_weight / denominator if denominator else None
    return SemanticOverlap(score, tp_weight, fp_weight, fn_weight, per_class)


def load_sqlite_columns(db_path: str | Path) -> dict[str, tuple[str, ...]]:
    connection = sqlite3.connect(str(db_path))
    try:
        tables = [
            str(row[0]) for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        result = {}
        for table in tables:
            quoted = '"' + table.replace('"', '""') + '"'
            result[_identifier(table)] = tuple(str(row[1]) for row in connection.execute(f"PRAGMA table_info({quoted})"))
        return result
    finally:
        connection.close()


def answer_similarity(predicted_rows: Iterable[Iterable[Any]], gold_rows: Iterable[Iterable[Any]]) -> tuple[float, dict[str, Any]]:
    predicted = [tuple(row) for row in predicted_rows]
    gold = [tuple(row) for row in gold_rows]
    # Match the active BIRD EX contract: raw SQLite cell equality, row-order invariance, and
    # duplicate-row collapse.  In particular Python intentionally treats ``1`` and ``1.0`` as
    # equal here; the earlier type-tagged JSON key disagreed with the repository verifier.
    predicted_set = set(predicted)
    gold_set = set(gold)
    union = predicted_set | gold_set
    row_jaccard = len(predicted_set & gold_set) / len(union) if union else 1.0
    predicted_width = len(predicted[0]) if predicted else 0
    gold_width = len(gold[0]) if gold else 0
    width_match = float(predicted_width == gold_width)
    if not predicted and not gold:
        count_ratio = 1.0
    elif not predicted or not gold:
        count_ratio = 0.0
    else:
        count_ratio = min(len(predicted_set), len(gold_set)) / max(len(predicted_set), len(gold_set))
    score = 0.80 * row_jaccard + 0.10 * width_match + 0.10 * count_ratio
    return score, {
        "row_jaccard": row_jaccard,
        "width_match": width_match,
        "count_ratio": count_ratio,
        "predicted_rows": len(predicted_set),
        "gold_rows": len(gold_set),
        "predicted_width": predicted_width,
        "gold_width": gold_width,
    }


def trajectory_quality(
    *,
    correct: bool,
    gold: CompiledSemantics,
    candidate: CompiledSemantics,
    predicted_rows: Iterable[Iterable[Any]],
    gold_rows: Iterable[Iterable[Any]],
    evidence_handle: str | None,
    artifact_selection: str,
    failure_quality_scale: float = 0.4,
) -> TrajectoryQuality:
    overlap = semantic_overlap(gold, candidate)
    table_score = _set_jaccard(set(gold.tables), set(candidate.tables))
    column_score = _set_jaccard(set(gold.columns), set(candidate.columns))
    schema_score = _weighted_mean(((table_score, 0.4), (column_score, 0.6)))
    answer_score, answer_diagnostics = answer_similarity(predicted_rows, gold_rows)
    raw_quality = _weighted_mean(
        (
            (schema_score, QUALITY_WEIGHTS["schema"]),
            (overlap.score, QUALITY_WEIGHTS["semantic"]),
            (answer_score, QUALITY_WEIGHTS["answer"]),
        )
    )
    raw_quality = float(raw_quality or 0.0)
    training_reward = 1.0 if correct else -1.0 + failure_quality_scale * raw_quality
    return TrajectoryQuality(
        correct=bool(correct),
        semantic_eligible=bool(gold.eligible and candidate.eligible),
        schema_score=schema_score,
        semantic_score=overlap.score,
        answer_score=answer_score,
        raw_quality=raw_quality,
        training_reward=training_reward,
        evidence_handle=evidence_handle,
        artifact_selection=artifact_selection,
        diagnostics={
            "gold_reasons": list(gold.reasons),
            "candidate_reasons": list(candidate.reasons),
            "semantic_overlap": asdict(overlap),
            "answer": answer_diagnostics,
        },
    )
