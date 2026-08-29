#!/usr/bin/env python3
"""Private SQL-AST coverage profiles for task sampling.

The output of this module is deliberately non-executable.  It is a hidden sampling aid, not a
gold tool plan, action label, teacher hint, or trajectory.  Perception, checkpoint, restore, and
the model's actual operator sequence can only be measured after a causal model<->Harness episode.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.optimizer.scope import traverse_scope


PROFILE_VERSION = "sql-task-coverage-profile-v2"
DIFFICULTY_VERSION = "sql-task-difficulty-v2"

_MUTATING_TYPES = tuple(
    value for value in (
        getattr(exp, name, None)
        for name in ("Insert", "Update", "Delete", "Create", "Drop", "Alter", "Merge", "Command")
    ) if value is not None
)
_COMPARISON_TYPES = tuple(
    value for value in (
        getattr(exp, name, None)
        for name in ("EQ", "NEQ", "GT", "GTE", "LT", "LTE", "Like", "ILike", "In", "Between", "Is", "Exists")
    ) if value is not None
)
_ARITHMETIC_TYPES = tuple(
    value for value in (
        getattr(exp, name, None)
        for name in ("Add", "Sub", "Mul", "Div", "Mod", "Pow")
    ) if value is not None
)
_SET_TYPES = tuple(
    value for value in (getattr(exp, "Union", None), getattr(exp, "Intersect", None), getattr(exp, "Except", None))
    if value is not None
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value == 1:
        return "1"
    if value == 2:
        return "2"
    return "3+"


def _nearest_select(node: exp.Expression) -> exp.Select | None:
    current = node.parent
    while current is not None:
        if isinstance(current, exp.Select):
            return current
        current = current.parent
    return None


def _direct_nodes(select: exp.Select, node_types: type | tuple[type, ...]):
    for node in select.walk():
        if node is not select and isinstance(node, node_types) and _nearest_select(node) is select:
            yield node


def _predicate_depth(node: exp.Expression | None) -> int:
    if node is None:
        return 0
    if isinstance(node, (exp.And, exp.Or)):
        return 1 + max(_predicate_depth(node.this), _predicate_depth(node.expression))
    if isinstance(node, exp.Not):
        return 1 + _predicate_depth(node.this)
    return 0


def _inside_window(node: exp.Expression, owner: exp.Select) -> bool:
    current = node.parent
    while current is not None and current is not owner:
        if isinstance(current, exp.Window):
            return True
        current = current.parent
    return False


def _join_kind(join: exp.Join) -> str:
    side = str(join.args.get("side") or "").lower()
    kind = str(join.args.get("kind") or "").lower()
    method = str(join.args.get("method") or "").lower()
    return side or kind or method or "inner"


def _masked_template(tree: exp.Expression) -> str:
    def mask(node: exp.Expression) -> exp.Expression:
        if isinstance(node, exp.Literal):
            return exp.Literal.string("?")
        return node

    return tree.copy().transform(mask).sql(dialect="sqlite", normalize=True, pretty=False)


def _operator_skeleton(select: exp.Select) -> str:
    parts = ["FROM"]
    joins = list(select.args.get("joins") or [])
    if joins:
        parts.append(f"J{_bucket(len(joins))}")
    if select.args.get("where") is not None:
        parts.append("FW")
    has_agg = select.args.get("group") is not None or any(_direct_nodes(select, exp.AggFunc))
    if has_agg:
        parts.append("G")
    if select.args.get("having") is not None:
        parts.append("FH")
    if any(_direct_nodes(select, exp.Window)):
        parts.append("W")
    if select.args.get("distinct") is not None:
        parts.append("D")
    parts.append("P")
    if select.args.get("order") is not None:
        parts.append("O")
    if select.args.get("limit") is not None or select.args.get("offset") is not None:
        parts.append("L")
    return ">".join(parts)


@dataclass(frozen=True)
class SqlTaskCoverageProfile:
    version: str
    dialect: str
    difficulty_version: str
    difficulty: str
    difficulty_score: int
    direct_support: str
    atomic_support: str
    counts: dict[str, int]
    histograms: dict[str, dict[str, int]]
    scope_skeletons: tuple[str, ...]
    feature_keys: tuple[str, ...]
    canonical_sql_sha256: str
    literal_masked_template_sha256: str
    sampling_only: bool = True
    executable_actions: bool = False

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["scope_skeletons"] = list(self.scope_skeletons)
        out["feature_keys"] = list(self.feature_keys)
        return out


def profile_sql(sql: str, *, dialect: str = "sqlite") -> SqlTaskCoverageProfile:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL profile input must be a non-empty string")
    tree = sqlglot.parse_one(sql, read=dialect)
    if isinstance(tree, _MUTATING_TYPES) or any(isinstance(node, _MUTATING_TYPES) for node in tree.walk()):
        raise ValueError("only read-only SELECT/WITH queries are eligible")
    selects = list(tree.find_all(exp.Select))
    if not selects:
        raise ValueError("query contains no SELECT scope")

    counts: dict[str, int] = {
        "select_scopes": len(selects),
        "joins": 0,
        "join_edges": 0,
        "where_filters": 0,
        "having_filters": 0,
        "predicate_leaves": 0,
        "predicate_depth": 0,
        "aggregate_metrics": 0,
        "group_by_columns": 0,
        "output_columns": 0,
        "order_keys": 0,
        "set_operations": 0,
        "subqueries": 0,
        "correlated_subqueries": 0,
        "ctes": 0,
        "windows": 0,
        "case_expressions": 0,
        "scalar_expressions": 0,
        "distinct_scopes": 0,
        "limits": 0,
        "offsets": 0,
    }
    hist: dict[str, dict[str, int]] = {
        "join_type": {}, "predicate_op": {}, "aggregate_op": {}, "set_op": {},
        "subquery_location": {}, "expression_type": {},
    }

    def bump(group: str, key: str, amount: int = 1) -> None:
        hist[group][key] = hist[group].get(key, 0) + amount

    skeletons: list[str] = []
    for select in selects:
        skeletons.append(_operator_skeleton(select))
        joins = list(select.args.get("joins") or [])
        counts["joins"] += len(joins)
        for join in joins:
            bump("join_type", _join_kind(join))
            on = join.args.get("on")
            edges = sum(1 for node in (on.walk() if on is not None else ()) if isinstance(node, _COMPARISON_TYPES))
            counts["join_edges"] += max(1, edges) if on is not None else 0

        for clause_name, count_name in (("where", "where_filters"), ("having", "having_filters")):
            clause = select.args.get(clause_name)
            if clause is None:
                continue
            counts[count_name] += 1
            predicate = clause.this
            leaves = [node for node in predicate.walk() if isinstance(node, _COMPARISON_TYPES)]
            counts["predicate_leaves"] += len(leaves)
            counts["predicate_depth"] = max(counts["predicate_depth"], _predicate_depth(predicate))
            for leaf in leaves:
                bump("predicate_op", leaf.key.lower())

        aggregate_nodes = [
            node for node in _direct_nodes(select, exp.AggFunc)
            if not _inside_window(node, select)
        ]
        counts["aggregate_metrics"] += len(aggregate_nodes)
        for aggregate in aggregate_nodes:
            bump("aggregate_op", aggregate.key.lower())
        group = select.args.get("group")
        if group is not None:
            counts["group_by_columns"] += len(group.expressions)
        counts["output_columns"] += len(select.expressions)
        order = select.args.get("order")
        if order is not None:
            counts["order_keys"] += len(order.expressions)
        counts["limits"] += int(select.args.get("limit") is not None)
        counts["offsets"] += int(select.args.get("offset") is not None)
        counts["distinct_scopes"] += int(select.args.get("distinct") is not None)
        counts["windows"] += sum(1 for _ in _direct_nodes(select, exp.Window))
        counts["case_expressions"] += sum(1 for _ in _direct_nodes(select, exp.Case))
        scalar_nodes = list(_direct_nodes(select, _ARITHMETIC_TYPES))
        counts["scalar_expressions"] += len(scalar_nodes)
        for node in scalar_nodes:
            bump("expression_type", node.key.lower())

    set_nodes = [node for node in tree.walk() if isinstance(node, _SET_TYPES)]
    counts["set_operations"] = len(set_nodes)
    for node in set_nodes:
        key = node.key.lower()
        if isinstance(node, exp.Union) and node.args.get("distinct") is False:
            key = "union_all"
        bump("set_op", key)

    subqueries = list(tree.find_all(exp.Subquery))
    counts["subqueries"] = len(subqueries)
    for node in subqueries:
        parent = node.parent
        location = "other"
        while parent is not None:
            if isinstance(parent, exp.From):
                location = "from"
                break
            if isinstance(parent, exp.Where):
                location = "where"
                break
            if isinstance(parent, exp.Having):
                location = "having"
                break
            if isinstance(parent, exp.Select):
                location = "select"
                break
            parent = parent.parent
        bump("subquery_location", location)
    counts["ctes"] = sum(1 for _ in tree.find_all(exp.CTE))
    counts["correlated_subqueries"] = sum(
        1 for scope in traverse_scope(tree) if scope.is_correlated_subquery
    )

    presence: list[str] = ["project"]
    checks = {
        "filter": counts["where_filters"] + counts["having_filters"],
        "join": counts["joins"],
        "aggregate": counts["aggregate_metrics"] + counts["group_by_columns"],
        "rank": counts["order_keys"] + counts["limits"] + counts["offsets"],
        "set": counts["set_operations"],
        "subquery": counts["subqueries"],
        "cte": counts["ctes"],
        "window": counts["windows"],
        "distinct": counts["distinct_scopes"],
        "scalar": counts["scalar_expressions"],
        "case": counts["case_expressions"],
    }
    presence.extend(key for key, value in checks.items() if value)
    presence = sorted(set(presence))

    features = {f"primitive:{name}" for name in presence}
    for name, value in counts.items():
        features.add(f"count:{name}:{_bucket(value)}")
    for group, values in hist.items():
        features.update(f"{group}:{name}" for name in values)
    for size, prefix in ((2, "pair"), (3, "triple")):
        features.update(f"{prefix}:{'+'.join(combo)}" for combo in itertools.combinations(presence, size))
    features.update(f"skeleton:{value}" for value in skeletons)

    score = 0
    score += min(3, counts["joins"])
    score += min(4, max(0, counts["select_scopes"] - 1) * 2)
    score += int(counts["aggregate_metrics"] > 0)
    score += int(counts["group_by_columns"] > 0)
    score += counts["having_filters"] * 2
    score += min(4, counts["set_operations"] * 3)
    score += min(3, counts["windows"] * 3)
    score += int(counts["predicate_leaves"] >= 3)
    score += int(counts["predicate_depth"] >= 2)
    score += int(counts["output_columns"] >= 3)
    score += int(counts["case_expressions"] > 0)
    score += int(counts["ctes"] > 0)
    score += min(4, counts["correlated_subqueries"] * 2)
    score += int(counts["scalar_expressions"] > 0)
    score += int(counts["order_keys"] >= 2)

    forced_hard = (
        counts["set_operations"] >= 1
        or counts["windows"] >= 1
        or counts["select_scopes"] >= 3
        or counts["joins"] >= 3
        or counts["correlated_subqueries"] >= 2
        or (counts["ctes"] and counts["subqueries"])
    )
    if forced_hard or score >= 7:
        difficulty = "hard"
    elif score >= 3 or counts["select_scopes"] > 1 or counts["joins"] > 1:
        difficulty = "medium"
    else:
        difficulty = "easy"

    join_types = set(hist["join_type"])
    atomic_risks: list[str] = []
    if counts["windows"]:
        atomic_risks.append("window")
    if "full" in join_types or "right" in join_types:
        atomic_risks.append("join_side")
    if counts["subqueries"] or counts["correlated_subqueries"]:
        atomic_risks.append("subquery_rewrite")
    with_clause = tree.args.get("with_")
    if with_clause is not None and with_clause.args.get("recursive"):
        atomic_risks.append("recursive_cte")
    atomic_support = "direct" if not atomic_risks else "rewrite-risk:" + ",".join(sorted(set(atomic_risks)))

    canonical = tree.sql(dialect="sqlite", normalize=True, pretty=False)
    template = _masked_template(tree)
    return SqlTaskCoverageProfile(
        version=PROFILE_VERSION,
        dialect=dialect,
        difficulty_version=DIFFICULTY_VERSION,
        difficulty=difficulty,
        difficulty_score=score,
        direct_support="read-only-sql",
        atomic_support=atomic_support,
        counts=counts,
        histograms=hist,
        scope_skeletons=tuple(skeletons),
        feature_keys=tuple(sorted(features)),
        canonical_sql_sha256=_sha(canonical),
        literal_masked_template_sha256=_sha(template),
    )


def profile_identity(profile: SqlTaskCoverageProfile) -> str:
    return _sha(json.dumps(profile.to_json(), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
