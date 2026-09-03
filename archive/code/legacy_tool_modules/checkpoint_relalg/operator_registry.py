"""Single-source semantic registry for checkpoint-relalg-v1 operators.

Provider JSON schemas live in ``provider_tools``.  This registry intentionally
contains only schema-independent relational semantics so the runtime, renderer,
and audits can agree on arity and ordering without duplicating model schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping


OrderingEffect = Literal["preserve", "establish", "clear"]


@dataclass(frozen=True)
class OperatorSpec:
    name: str
    input_arity: Literal[1, 2]
    artifact_kind: str
    ordering_effect: OrderingEffect
    semantics: str
    preserves_duplicates: bool


_MICRO_SPECS = (
    OperatorSpec(
        "filter_rows",
        1,
        "filter",
        "preserve",
        "Select rows whose typed predicate is TRUE; preserve all columns and bag multiplicity.",
        True,
    ),
    OperatorSpec(
        "project",
        1,
        "project",
        "preserve",
        "Produce only declared typed expressions; do not filter, sort, or deduplicate.",
        True,
    ),
    OperatorSpec(
        "join",
        2,
        "join",
        "clear",
        "Binary theta join with AND-combined edges and SQL NULL comparison semantics.",
        True,
    ),
    OperatorSpec(
        "aggregate",
        1,
        "aggregate",
        "clear",
        "Group once and compute count/sum/avg/min/max metrics without embedded filtering.",
        False,
    ),
    OperatorSpec(
        "distinct",
        1,
        "distinct",
        "clear",
        "Eliminate duplicates over every visible input column.",
        False,
    ),
    OperatorSpec(
        "set_operation",
        2,
        "set",
        "clear",
        "Combine positionally aligned relations; only union_all preserves duplicates.",
        False,
    ),
    OperatorSpec(
        "sort",
        1,
        "sort",
        "establish",
        "Establish semantic ordering from non-empty model keys; add only hidden deterministic ties.",
        True,
    ),
    OperatorSpec(
        "limit",
        1,
        "limit",
        "preserve",
        "Slice an already ordered relation without changing columns or order.",
        True,
    ),
    OperatorSpec(
        "add_rank",
        1,
        "rank",
        "clear",
        "Add one window rank column without filtering rows or establishing output order.",
        True,
    ),
)

_SEMANTIC_SPECS = (
    OperatorSpec(
        "shape_rows",
        1,
        "shape",
        "clear",
        "Project declared expressions and optionally eliminate exact duplicate output rows.",
        False,
    ),
    OperatorSpec(
        "group_aggregate",
        1,
        "group_aggregate",
        "clear",
        "Filter one population, group once, and compute globally or locally conditioned metrics.",
        False,
    ),
    OperatorSpec(
        "scalar_compute",
        1,
        "scalar_compute",
        "preserve",
        "Compute grounded expressions from an exactly-one-row relation.",
        True,
    ),
    OperatorSpec(
        "rank_select",
        1,
        "rank_select",
        "establish",
        "Filter, rank, select top-k with explicit ties, and project declared outputs.",
        True,
    ),
)

OPERATOR_SPECS: Mapping[str, OperatorSpec] = MappingProxyType(
    {spec.name: spec for spec in _MICRO_SPECS}
)
SEMANTIC_OPERATOR_SPECS: Mapping[str, OperatorSpec] = MappingProxyType(
    {spec.name: spec for spec in _SEMANTIC_SPECS}
)


def get_operator_spec(name: str) -> OperatorSpec:
    try:
        return OPERATOR_SPECS.get(name) or SEMANTIC_OPERATOR_SPECS[name]
    except KeyError as exc:
        from .expression import RelAlgValidationError

        raise RelAlgValidationError(
            "unknown_operator", f"unknown atomic operator {name!r}"
        ) from exc


__all__ = [
    "OPERATOR_SPECS",
    "SEMANTIC_OPERATOR_SPECS",
    "OperatorSpec",
    "OrderingEffect",
    "get_operator_spec",
]
