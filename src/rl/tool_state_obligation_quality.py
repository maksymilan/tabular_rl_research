#!/usr/bin/env python3
"""Path-independent trajectory quality from deterministic tool state and Gold SQL.

This is the deliberately small v2 research signal.  It does not inspect answer rows,
correctness labels, model reasoning, tool errors, or a privileged Gold tool path.

Gold SQL and every successfully produced relation artifact are normalized into the same eight
sets of semantic obligations.  Each artifact is compared with macro Jaccard over the non-empty
categories.  A whole trajectory receives::

    Q_rank = C_terminal                  if terminal evidence exists
             0.5 * C_max                 otherwise

Relation ancestry is used only to reconstruct the selected artifact's deterministic semantics; it
is not an independent reward or credit-assignment signal.  Unsupported normalization fails closed
as NA.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from sqlglot import exp, parse_one

from src.rl.trajectory_semantic_ranker import (
    GoldSemanticCompiler,
    TrajectorySemanticCompiler,
    _identifier,
    _json_key,
    _unit_kind,
)


CATEGORIES = ("source", "join", "predicate", "grain", "value", "set", "rank", "output")
PRODUCER_TOOLS = frozenset({
    "condition_filter",
    "project",
    "join_tables",
    "group_aggregate",
    "scalar_compute",
    "extreme_value_select",
    "set_op",
})


@dataclass(frozen=True)
class ObligationGraph:
    categories: dict[str, frozenset[str]]
    eligible: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObligationOverlap:
    score: float | None
    per_category: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ToolStateQuality:
    score: float | None
    semantic_eligible: bool
    c_terminal: float | None
    c_max: float | None
    terminal_evidence: str | None
    max_handle: str | None
    scoring_mode: str
    reasons: tuple[str, ...] = ()
    terminal_overlap: dict[str, Any] = field(default_factory=dict)
    max_overlap: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _empty_categories() -> dict[str, set[str]]:
    return {category: set() for category in CATEGORIES}


def _payload(unit: str) -> Any:
    return json.loads(unit)["payload"]


def _atoms(expression: Any) -> list[Any]:
    if isinstance(expression, list) and expression and expression[0] == "and":
        result: list[Any] = []
        for child in expression[1:]:
            result.extend(_atoms(child))
        return result
    return [expression]


def _contains_semantic_value(expression: Any) -> bool:
    if not isinstance(expression, list) or not expression:
        return False
    if isinstance(expression[0], str) and expression[0] in {"column", "aggregate"}:
        return True
    return any(_contains_semantic_value(item) for item in expression[1:])


def _value_obligations(expression: Any) -> set[str]:
    """Collect final value semantics, excluding aliases and implementation intermediates."""
    result: set[str] = set()

    def visit(value: Any) -> None:
        if not isinstance(value, list) or not value:
            return
        if isinstance(value[0], str) and value[0] == "aggregate":
            result.add(_json_key(value))
        for child in value[1:]:
            visit(child)

    visit(expression)
    if (
        isinstance(expression, list)
        and expression
        and isinstance(expression[0], str)
        and expression[0] not in {"column", "literal", "star", "aggregate"}
        and _contains_semantic_value(expression)
    ):
        result.add(_json_key(["compute", expression]))
    return result


def _freeze(categories: Mapping[str, Iterable[str]]) -> dict[str, frozenset[str]]:
    return {name: frozenset(categories.get(name, ())) for name in CATEGORIES}


def compile_gold_obligations(
    sql: str,
    table_columns: Mapping[str, Iterable[str]],
) -> ObligationGraph:
    compiler = GoldSemanticCompiler(table_columns)
    compiled = compiler.compile(sql)
    if not compiled.eligible:
        return ObligationGraph(_freeze(_empty_categories()), False, compiled.reasons)
    try:
        tree = parse_one(sql, read="sqlite")
    except Exception as exc:  # pragma: no cover - the conservative compiler already catches this
        return ObligationGraph(
            _freeze(_empty_categories()), False, (f"parse_error:{type(exc).__name__}:{exc}",)
        )
    if not isinstance(tree, exp.Select):
        return ObligationGraph(
            _freeze(_empty_categories()), False, (f"unsupported_root:{tree.key}",)
        )

    categories = _empty_categories()
    categories["source"].update(_json_key(["source", table]) for table in compiled.tables)
    kind_map = {
        "join": "join",
        "predicate": "predicate",
        "grain": "grain",
        "set_distinct": "set",
        "rank_limit": "rank",
    }
    for unit in compiled.units:
        kind = _unit_kind(unit)
        if kind in kind_map:
            categories[kind_map[kind]].add(_json_key(_payload(unit)))

    for index, projection in enumerate(tree.expressions):
        core = projection.this if isinstance(projection, exp.Alias) else projection
        expression = compiler.expr(core)
        categories["output"].add(_json_key(["position", index, expression]))
        categories["value"].update(_value_obligations(expression))
    reasons = tuple(sorted(set(compiler.reasons)))
    return ObligationGraph(_freeze(categories), not reasons, reasons)


def compile_artifact_obligations(
    compiler: TrajectorySemanticCompiler,
    handle: str | None,
) -> ObligationGraph:
    compiled = compiler.compiled_for(handle)
    if not compiled.eligible:
        return ObligationGraph(_freeze(_empty_categories()), False, compiled.reasons)
    categories = _empty_categories()
    categories["source"].update(_json_key(["source", table]) for table in compiled.tables)
    kind_map = {
        "join": "join",
        "predicate": "predicate",
        "grain": "grain",
        "set_distinct": "set",
        "rank_limit": "rank",
    }
    for unit in compiled.units:
        kind = _unit_kind(unit)
        category = kind_map.get(kind)
        if category is None:
            continue
        payload = _payload(unit)
        if category == "predicate" and isinstance(payload, list) and len(payload) == 2:
            stage, expression = payload
            categories[category].update(_json_key([stage, atom]) for atom in _atoms(expression))
        else:
            categories[category].add(_json_key(payload))

    artifact = compiler.artifacts.get(handle or "")
    if artifact is None:
        return ObligationGraph(_freeze(categories), True, ())
    for index, name in enumerate(artifact.output_columns):
        expression = artifact.columns.get(_identifier(name))
        if expression is None:
            return ObligationGraph(
                _freeze(categories), False, (f"missing_output_lineage:{handle}:{name}",)
            )
        categories["output"].add(_json_key(["position", index, expression]))
        categories["value"].update(_value_obligations(expression))
    return ObligationGraph(_freeze(categories), True, ())


def obligation_overlap(gold: ObligationGraph, candidate: ObligationGraph) -> ObligationOverlap:
    if not gold.eligible or not candidate.eligible:
        return ObligationOverlap(None, {})
    scores: list[float] = []
    details: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        gold_values = set(gold.categories[category])
        candidate_values = set(candidate.categories[category])
        union = gold_values | candidate_values
        if not union:
            continue
        intersection = gold_values & candidate_values
        score = len(intersection) / len(union)
        scores.append(score)
        details[category] = {
            "score": score,
            "gold": len(gold_values),
            "candidate": len(candidate_values),
            "matched": len(intersection),
            "missing": sorted(gold_values - candidate_values),
            "extra": sorted(candidate_values - gold_values),
        }
    return ObligationOverlap(sum(scores) / len(scores) if scores else 1.0, details)


def _terminal_handle(steps: Iterable[Mapping[str, Any]]) -> str | None:
    terminal: str | None = None
    for step in steps:
        call = step.get("tool_call") or {}
        if call.get("tool") != "answer_from_context":
            continue
        evidence = (call.get("arguments") or {}).get("evidence")
        if isinstance(evidence, dict) and isinstance(evidence.get("table"), str):
            terminal = evidence["table"]
    return terminal


def score_tool_state_trajectory(
    *,
    gold_sql: str,
    table_columns: Mapping[str, Iterable[str]],
    steps: Iterable[Mapping[str, Any]],
) -> ToolStateQuality:
    steps = list(steps)
    gold = compile_gold_obligations(gold_sql, table_columns)
    if not gold.eligible:
        return ToolStateQuality(
            score=None,
            semantic_eligible=False,
            c_terminal=None,
            c_max=None,
            terminal_evidence=None,
            max_handle=None,
            scoring_mode="unscorable",
            reasons=tuple(f"gold:{reason}" for reason in gold.reasons),
        )

    compiler = TrajectorySemanticCompiler(table_columns)
    compiler.compile(steps)
    terminal = _terminal_handle(steps)
    producer_order: list[str] = []
    for step in steps:
        call = step.get("tool_call") or {}
        output = step.get("tool_output") or {}
        handle = output.get("table")
        if (
            call.get("tool") in PRODUCER_TOOLS
            and isinstance(handle, str)
            and isinstance(output.get("derivation"), dict)
            and handle in compiler.artifacts
        ):
            producer_order.append(handle)

    overlaps: dict[str, ObligationOverlap] = {}
    candidate_failures: list[str] = []
    for handle in producer_order:
        graph = compile_artifact_obligations(compiler, handle)
        if not graph.eligible:
            candidate_failures.extend(f"candidate:{handle}:{reason}" for reason in graph.reasons)
            continue
        overlaps[handle] = obligation_overlap(gold, graph)

    if producer_order and not overlaps:
        return ToolStateQuality(
            score=None,
            semantic_eligible=False,
            c_terminal=None,
            c_max=None,
            terminal_evidence=terminal,
            max_handle=None,
            scoring_mode="unscorable",
            reasons=tuple(sorted(set(candidate_failures))),
        )

    terminal_overlap = ObligationOverlap(0.0, {})
    if terminal is not None:
        if terminal in compiler.artifacts:
            terminal_graph = compile_artifact_obligations(compiler, terminal)
            if not terminal_graph.eligible:
                reasons = tuple(f"terminal:{reason}" for reason in terminal_graph.reasons)
                return ToolStateQuality(
                    score=None,
                    semantic_eligible=False,
                    c_terminal=None,
                    c_max=None,
                    terminal_evidence=terminal,
                    max_handle=None,
                    scoring_mode="unscorable",
                    reasons=reasons,
                )
            terminal_overlap = obligation_overlap(gold, terminal_graph)
        else:
            terminal_overlap = ObligationOverlap(0.0, {})

    max_handle: str | None = None
    c_max = 0.0
    for handle in producer_order:
        overlap = overlaps.get(handle)
        if overlap is None or overlap.score is None:
            continue
        if max_handle is None or overlap.score > c_max:
            max_handle = handle
            c_max = overlap.score

    c_terminal = float(terminal_overlap.score or 0.0)
    if terminal is not None:
        score = c_terminal
        scoring_mode = "terminal"
    else:
        score = 0.5 * c_max
        scoring_mode = "half_max_fallback"
    return ToolStateQuality(
        score=score,
        semantic_eligible=True,
        c_terminal=c_terminal,
        c_max=c_max,
        terminal_evidence=terminal,
        max_handle=max_handle,
        scoring_mode=scoring_mode,
        reasons=tuple(sorted(set(candidate_failures))),
        terminal_overlap=asdict(terminal_overlap),
        max_overlap=asdict(overlaps[max_handle]) if max_handle in overlaps else {},
    )
