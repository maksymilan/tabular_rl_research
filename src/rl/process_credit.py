#!/usr/bin/env python3
"""Harness-grounded step credit for table-tool RL trajectories.

The module implements the normalized reward proposed in the project notes:

    r_t = C*c_t^+ + (1-C)*eta*DeltaPhi_t + lambda_A*A_t - P*c_t^-

It deliberately separates replay-derived feature extraction from the pure reward algebra.  Online
RL can feed the same feature rows directly; the offline adapter replays verified teacher episodes
through the canonical harness so provenance, row counts, and semantic state changes are not inferred
from model prose.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from sqlglot import exp, parse_one

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from executor import Harness  # noqa: E402
from observation_binding import action_literal_slots, same_scalar  # noqa: E402
from provenance import (  # noqa: E402
    PERCEPTION_TOOLS,
    backward_slice,
    build_grounding_references,
)
from target_support import (  # noqa: E402
    TargetSupport,
    build_target_support,
    canonical_table,
    matching_target_columns,
    observed_target_row_units,
    table_target_row_units,
)
from task_support import task_text_supports_literal  # noqa: E402
from trajectory_replay import remap_replay_handles  # noqa: E402
from rollout import (  # noqa: E402
    execute_tool,
    new_ctx,
    overview,
    score,
    state_digest,
)


SELECTIVE_TOOLS = frozenset({"condition_filter", "extreme_value_select"})
TABLE_PRODUCING_TOOLS = frozenset(
    {
        "condition_filter",
        "project",
        "scalar_compute",
        "join_tables",
        "group_aggregate",
        "extreme_value_select",
        "set_op",
    }
)
LINEAGE_PRESERVING_TOOLS = frozenset(
    {"condition_filter", "extreme_value_select", "project", "derive_column", "window"}
)
ROWSET_TOOLS = frozenset(
    {
        "condition_filter",
        "extreme_value_select",
        "project",
        "derive_column",
        "window",
        "group_aggregate",
        "scalar_compute",
        "join_tables",
        "set_op",
        "read_subtable",
    }
)
MAX_GROUNDING_ROWS = 10_000


@dataclass(frozen=True)
class ProcessRewardConfig:
    """Configurable weights; defaults are audit values, not tuned research conclusions."""

    w_terminal_correct: float = 0.0
    w_back_slice: float = 1.0
    w_new_evidence: float = 1.0
    w_search_reduction: float = 1.0
    w_feedback_response: float = 1.0
    w_target_potential: float = 1.0
    omega_target_table: float = 0.25
    omega_target_column: float = 0.25
    omega_target_row: float = 0.50
    eta_failure_progress: float = 0.05
    lambda_answer_format: float = 0.02
    lambda_terminal_failure: float = 0.30
    lambda_tool_error: float = 0.08
    lambda_adjacent_repeat: float = 0.0
    lambda_repeat_without_feedback: float = 0.06
    lambda_legal_no_state_change: float = 0.03
    lambda_ignored_feedback: float = 0.05
    lambda_unsupported_guess: float = 0.08
    lambda_empty_result: float = 0.10
    penalty_cap: float = 0.80

    def validate(self) -> None:
        values = asdict(self)
        if any(value < 0 for key, value in values.items() if key != "penalty_cap"):
            raise ValueError("all reward and penalty weights must be non-negative")
        if not 0 < self.penalty_cap < 1:
            raise ValueError("penalty_cap must satisfy 0 < P_max < 1")
        omega = self.omega_target_table + self.omega_target_column + self.omega_target_row
        if not math.isclose(omega, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("target-potential omega weights must sum to 1")
        auxiliary_positive = self.eta_failure_progress + self.lambda_answer_format
        if self.lambda_terminal_failure > 0:
            if not (
                0
                < auxiliary_positive
                < self.lambda_terminal_failure
                <= self.penalty_cap
                < 1
            ):
                raise ValueError(
                    "reward bounds require 0 < eta+lambda_A < lambda_fail <= P_max < 1"
                )
        elif auxiliary_positive != 0:
            raise ValueError(
                "eta_failure_progress and lambda_answer_format must both be zero "
                "when terminal-failure shaping is disabled"
            )


@dataclass
class StepFeature:
    action_index: int
    step_id: str
    tool: str | None
    legal_success: bool
    is_terminal: bool = False
    error_type: str | None = None
    action_signature: str | None = None
    state_changed: bool = False
    empty_result: bool = False
    repeated_call: bool = False
    adjacent_repeat: bool = False
    feedback_error_before: bool = False
    feedback_empty_before: bool = False
    action_changed_after_empty: bool = False
    action_changed_after_feedback: bool = False
    back_slice: float = 0.0
    attempted_back_slice: float = 0.0
    new_used_evidence: float = 0.0
    search_reduction: float = 0.0
    empty_result_penalty: float = 0.0
    verified_negative_evidence: bool = False
    feedback_response: float = 0.0
    target_table_delta: float = 0.0
    target_column_delta: float = 0.0
    target_row_delta: float = 0.0
    target_potential_delta: float = 0.0
    answer_format: float = 0.0
    unsupported_guess: float = 0.0
    terminal_failure: float = 0.0
    tool_error: float = 0.0
    repeat_without_feedback: float = 0.0
    legal_no_state_change: float = 0.0
    ignored_feedback: float = 0.0
    failure_responsibility: float = 0.0
    n_root: int | None = None
    n_in: int | None = None
    n_out: int | None = None
    references: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class StepReward:
    action_index: int
    step_id: str
    tool: str | None
    g_positive: float
    p_outcome: float
    p_local: float
    p_raw: float
    c_positive: float
    c_negative: float
    reward: float
    features: dict[str, Any]


@dataclass
class EpisodeReward:
    trajectory_id: str
    correct: bool
    grounding_method: str
    grounding_handle: str | None
    back_slice_step_ids: list[str]
    used_evidence_units: list[str]
    positive_mass: float
    raw_penalty_mass: float
    capped_penalty: float
    total_reward: float
    fallback_terminal_credit: bool
    process_update: bool
    steps: list[StepReward]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "correct": self.correct,
            "grounding_method": self.grounding_method,
            "grounding_handle": self.grounding_handle,
            "back_slice_step_ids": self.back_slice_step_ids,
            "used_evidence_units": self.used_evidence_units,
            "positive_mass": self.positive_mass,
            "raw_penalty_mass": self.raw_penalty_mass,
            "capped_penalty": self.capped_penalty,
            "total_reward": self.total_reward,
            "fallback_terminal_credit": self.fallback_terminal_credit,
            "process_update": self.process_update,
            "steps": [asdict(step) for step in self.steps],
            "diagnostics": self.diagnostics,
        }


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _flatten_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [scalar for item in value.values() for scalar in _flatten_values(item)]
    if isinstance(value, (list, tuple)):
        return [scalar for item in value for scalar in _flatten_values(item)]
    return [value]


def _audit_evidence_rows(rows: list[list[Any]], answer: Any, limit: int = 10) -> list[list[Any]]:
    wanted = _flatten_values(answer)
    matches = [
        row for row in rows
        if any(cell == value for cell in _flatten_values(row) for value in wanted)
    ]
    selected: list[list[Any]] = []
    seen: set[str] = set()
    for row in list(rows[:3]) + matches:
        key = canonical_json(row)
        if key not in seen:
            seen.add(key)
            selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def action_signature(tool: str, arguments: dict[str, Any]) -> str:
    payload = {"tool": tool, "arguments": arguments}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:16]


def _evidence_table(arguments: dict[str, Any]) -> str | None:
    evidence = arguments.get("evidence")
    if isinstance(evidence, str):
        return evidence
    if isinstance(evidence, dict) and isinstance(evidence.get("table"), str):
        return evidence["table"]
    return None


def _semantic_state(value: Any) -> Any:
    """Remove allocation/provenance identities before comparing environment meaning."""
    volatile = {
        "from_step",
        "created_by",
        "step_id",
        "produced_by",
        "evidence_step_id",
    }
    if isinstance(value, list):
        return [_semantic_state(item) for item in value]
    if not isinstance(value, dict):
        return value
    normalized = {
        key: _semantic_state(item)
        for key, item in value.items()
        if key not in volatile
    }
    # Derived handles are allocation identities. Compare their semantic artifact payloads as a
    # sorted multiset so repeating the same operation cannot manufacture a state change.
    tables = normalized.get("tables")
    if isinstance(tables, dict):
        sources = {}
        derived: dict[str, Any] = {}
        for name, table in tables.items():
            if isinstance(table, dict) and table.get("kind") == "source":
                sources[name] = table
            else:
                derived[canonical_json(table)] = table
        normalized["tables"] = {
            "sources": sources,
            "derived": [derived[key] for key in sorted(derived)],
        }
    return normalized


def semantic_state_digest(state: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(_semantic_state(state)).encode("utf-8")).hexdigest()


def _condition_columns(condition: Any) -> set[str]:
    if isinstance(condition, list):
        return {column for item in condition for column in _condition_columns(item)}
    if not isinstance(condition, dict):
        return set()
    columns: set[str] = set()
    for key in ("and", "or"):
        for item in condition.get(key, []) or []:
            columns.update(_condition_columns(item))
    if "not" in condition:
        columns.update(_condition_columns(condition["not"]))
    for key in ("column", "column_value"):
        value = condition.get(key)
        if isinstance(value, str):
            columns.add(value)
    return columns


def _expression_columns(expression: str) -> set[str]:
    """Extract logical columns from a SQLite expression, including quoted dotted names."""
    try:
        tree = parse_one(expression, read="sqlite")
    except Exception:
        return {expression}
    columns = set()
    for column in tree.find_all(exp.Column):
        if column.table:
            columns.add(f"{column.table}.{column.name}")
        else:
            columns.add(column.name)
    return columns


def _expression_predicate_literals(expression: str) -> list[tuple[str | None, Any]]:
    """Extract domain constants from expression predicates, excluding arithmetic/CASE encodings."""
    try:
        tree = parse_one(expression, read="sqlite")
    except Exception:
        return []
    targets: list[tuple[str | None, Any]] = []
    predicates = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.ILike)
    for predicate in tree.find_all(*predicates):
        predicate_columns = _expression_columns(predicate.sql(dialect="sqlite"))
        column = next(iter(predicate_columns)) if len(predicate_columns) == 1 else None
        for literal in predicate.find_all(exp.Literal):
            value: Any = literal.this
            if not literal.is_string:
                try:
                    value = int(value)
                except (TypeError, ValueError):
                    try:
                        value = float(value)
                    except (TypeError, ValueError):
                        pass
            targets.append((column, value))
    return targets


def _tool_columns(tool: str, arguments: dict[str, Any], output: dict[str, Any]) -> set[str]:
    columns: set[str] = set()
    if tool == "describe_table":
        for described in output.get("tables", []) or []:
            for column in described.get("columns", []) or []:
                if isinstance(column, dict) and isinstance(column.get("name"), str):
                    columns.add(column["name"])
    elif tool == "inspect_column":
        if isinstance(arguments.get("column"), str):
            columns.add(arguments["column"])
    elif tool == "read_subtable":
        requested = arguments.get("columns")
        if isinstance(requested, list):
            columns.update(item for item in requested if isinstance(item, str))
        elif isinstance(output.get("columns"), list):
            columns.update(item for item in output["columns"] if isinstance(item, str))
    elif tool == "condition_filter":
        columns.update(_condition_columns(arguments.get("conditions")))
        columns.update(arguments.get("return_columns") or [])
    elif tool == "project":
        derivation = output.get("derivation") or {}
        for item in (derivation.get("semantics") or {}).get("column_lineage", []) or []:
            columns.update(source for source in item.get("sources", []) if isinstance(source, str))
    elif tool == "join_tables":
        for join in arguments.get("joins") or []:
            for edge in join.get("on", []) or []:
                columns.update(
                    value for value in (edge.get("left"), edge.get("right"))
                    if isinstance(value, str)
                )
    elif tool == "group_aggregate":
        columns.update(arguments.get("group_by") or [])
        columns.update(arguments.get("passthrough") or [])
        for aggregation in arguments.get("aggregations") or []:
            column = aggregation.get("column")
            if isinstance(column, str) and column != "*":
                columns.update(_expression_columns(column))
            columns.update(_condition_columns(aggregation.get("where")))
    elif tool == "extreme_value_select":
        for item in arguments.get("order_by") or []:
            if isinstance(item, str):
                columns.add(re.sub(r"\s+(?:ASC|DESC)\s*$", "", item, flags=re.IGNORECASE))
        columns.update(arguments.get("return_columns") or [])
    elif tool == "scalar_compute":
        for operand in arguments.get("operands") or []:
            if isinstance(operand, dict) and isinstance(operand.get("column"), str):
                columns.add(operand["column"])
    return {column for column in columns if isinstance(column, str) and column != "*"}


def _action_literal_targets(tool: str, arguments: dict[str, Any]) -> list[tuple[str | None, Any]]:
    targets = [
        (slot.column, slot.value)
        for slot in action_literal_slots(tool, arguments)
    ]
    if tool == "group_aggregate":
        targets.extend(
            pair
            for aggregation in arguments.get("aggregations") or []
            if isinstance(aggregation, dict)
            and isinstance(aggregation.get("column"), str)
            for pair in _expression_predicate_literals(aggregation["column"])
        )
    elif tool == "scalar_compute":
        targets.extend(
            (operand.get("column"), operand["value"])
            for operand in arguments.get("operands") or []
            if isinstance(operand, dict) and "value" in operand
        )
    return targets


def _visible_output_values(output: dict[str, Any]) -> list[Any]:
    """Return factual scalar cells actually rendered in one model-visible tool output."""
    values: list[Any] = []

    def visit(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif not isinstance(value, dict):
            values.append(value)

    for key in ("rows", "frequent_values", "result_sample"):
        if key in output:
            visit(output[key])
    return values


def _grounding_reference_redundant_with_task(
    reference: dict[str, Any],
    task_text: str,
) -> bool:
    """Whether a row/domain edge only repeats literals already supplied by the task."""
    if (
        reference.get("type") != "grounding"
        or reference.get("role") not in {"domain_observation", "row_observation"}
    ):
        return False
    target = reference.get("target") or {}
    values = _flatten_values(target.get("values"))
    if not values:
        return False
    columns = {
        column
        for column in (
            target.get("column"),
            *(
                match.get("target_column")
                for match in target.get("column_matches") or []
                if isinstance(match, dict)
            ),
        )
        if isinstance(column, str)
    }
    candidates: tuple[str | None, ...] = tuple(sorted(columns)) or (None,)
    return all(
        any(
            task_text_supports_literal(value, task_text, column=column)
            for column in candidates
        )
        for value in values
    )


def _source_roots_from_references(
    references: list[dict[str, Any]],
    roots_by_step: dict[str, set[str]],
) -> set[str]:
    roots: set[str] = set()
    for reference in references:
        if reference.get("type", "data") != "data":
            continue
        source = reference.get("source")
        if isinstance(source, str):
            roots.add(canonical_table(source))
        parent = reference.get("step")
        if isinstance(parent, str):
            roots.update(roots_by_step.get(parent, set()))
    return roots


def _fraction_delta(before: set[Any], after: set[Any], target: set[Any] | frozenset[Any]) -> float:
    if not target:
        return 0.0
    return len((after - before) & set(target)) / len(target)


def _table_ref(arguments: dict[str, Any], ctx: dict[str, Any]) -> str | None:
    ref = arguments.get("table")
    if not isinstance(ref, str):
        return None
    history = ctx.get("history", {}).get(ref)
    if isinstance(history, dict):
        output = history.get("output") or {}
        return output.get("table") or ref
    return ref


def _row_count(harness: Harness, table: str | None) -> int | None:
    if not table:
        return None
    try:
        return int(harness.conn.execute(f"SELECT COUNT(*) FROM {harness._src(table)}").fetchone()[0])
    except Exception:
        return None


def _is_empty_result(tool: str | None, output: dict[str, Any] | None) -> bool:
    if tool == "inspect_column" and isinstance(output, dict):
        return output.get("distinct_count") == 0
    if tool not in ROWSET_TOOLS or not isinstance(output, dict):
        return False
    row_count = output.get("row_count")
    if isinstance(row_count, int):
        return row_count == 0
    rows = output.get("rows")
    return isinstance(rows, list) and not rows


def normalized_search_reduction(n_root: int, n_in: int, n_out: int) -> float:
    """Fixed-root log reduction; invalid, expanding, or empty outputs receive zero."""
    if n_root <= 0 or n_in < 0 or n_out <= 0 or n_out > n_in:
        return 0.0
    return (math.log(n_in + 1) - math.log(n_out + 1)) / math.log(n_root + 1)


def _automatic_final_table(
    table_history: list[tuple[str, str, str]],
) -> tuple[str, str | None, str | None, str | None]:
    """Use the nearest prior table-bearing action as the final dependency seed.

    The model's optional answer evidence is deliberately irrelevant here. The harness owns the
    execution history, so it can identify the most recent table that was produced or observed.
    """
    if not table_history:
        return "no_prior_table", None, None, None
    step_id, handle, role = table_history[-1]
    return "automatic_last_table", handle, step_id, role


def _history_value_support(
    handle: str,
    history: dict[str, dict[str, Any]],
    values: list[Any],
) -> tuple[set[int], str | None]:
    """Return answer-value indices visibly supported by this handle's produced/read rows."""
    supported: set[int] = set()
    latest_observation: str | None = None
    for step_id, record in history.items():
        output = record.get("output") or {}
        is_producer = output.get("table") == handle
        is_read = record.get("tool") == "read_subtable" and (record.get("arguments") or {}).get("table") == handle
        if not (is_producer or is_read):
            continue
        rows = output.get("rows")
        if not isinstance(rows, list):
            continue
        if is_read:
            latest_observation = step_id
        cells = _flatten_values(rows)
        for index, value in enumerate(values):
            if any(same_scalar(value, cell) for cell in cells):
                supported.add(index)
    return supported, latest_observation


def _lineage_for_output(
    tool: str,
    input_table: str | None,
    output_table: str | None,
    lineages: dict[str, tuple[str, int]],
) -> None:
    if output_table and input_table and tool in LINEAGE_PRESERVING_TOOLS and input_table in lineages:
        lineages[output_table] = lineages[input_table]


def replay_step_features(
    trajectory: dict[str, Any],
    *,
    denotation_comparison: str = "bird-set",
) -> tuple[list[StepFeature], dict[str, Any]]:
    """Replay one normalized trajectory and return harness-authored feature rows."""
    if denotation_comparison != "bird-set":
        raise ValueError("active process reward requires denotation_comparison='bird-set'")
    source = trajectory.get("source") or {}
    db_path = source.get("db_path")
    gold_sql = source.get("gold_sql")
    if not db_path or not gold_sql:
        raise ValueError("trajectory source must contain db_path and gold_sql")

    harness = Harness(str(db_path))
    try:
        comparison = denotation_comparison
        target_support = build_target_support(harness, gold_sql)
        recorded_catalog = (trajectory.get("initial_state") or {}).get("dataset_overview")
        catalog = (
            deepcopy(recorded_catalog)
            if isinstance(recorded_catalog, dict)
            else overview(harness)
        )
        catalog_source = "recorded_harness_initial_state" if recorded_catalog else "fresh_database"
        ctx = new_ctx(catalog)
        created: set[str] = set()
        table_history: list[tuple[str, str, str]] = []
        lineages: dict[str, tuple[str, int]] = {}
        roots_by_step: dict[str, set[str]] = {}
        roots_by_table: dict[str, set[str]] = {}
        discovered_tables: set[str] = set()
        discovered_columns: set[str] = set()
        discovered_rows: set[tuple[Any, ...] | str] = set()
        target_row_match_incomplete_steps: list[str] = []
        for table in catalog.get("tables", []):
            name = table.get("table_name")
            count = table.get("row_count", table.get("num_rows"))
            if isinstance(name, str) and isinstance(count, int):
                lineages[name] = (name, count)
                roots_by_table[name] = {canonical_table(name)}

        legal_steps = {int(step["step_id"].rsplit("_", 1)[1]): step for step in trajectory.get("steps") or []}
        error_events = {
            int(event["action_index"]): event
            for event in (trajectory.get("rollout_generation") or {}).get("error_events") or []
        }
        action_indices = sorted(set(legal_steps) | set(error_events))
        if not action_indices:
            raise ValueError("trajectory has no actions")

        features: list[StepFeature] = []
        provenance_steps: list[dict[str, Any]] = []
        seen_signatures: set[str] = set()
        previous_action_signature: str | None = None
        seen_observation_signatures: set[str] = set()
        final_arguments: dict[str, Any] | None = None
        final_step_id: str | None = None
        replay_correct = False
        replay_handle_map: dict[str, str] = {}
        unsupported_action_literals: list[dict[str, Any]] = []
        unsupported_action_columns: list[dict[str, Any]] = []
        visible_observation_values: list[Any] = []
        task_text = "\n".join(filter(None, [
            trajectory.get("question"),
            source.get("external_knowledge"),
        ]))

        for action_index in action_indices:
            if action_index in error_events:
                event = error_events[action_index]
                attempted_signature = (
                    action_signature(
                        event["attempted_tool"], event.get("attempted_arguments") or {}
                    )
                    if event.get("attempted_tool")
                    else None
                )
                features.append(
                    StepFeature(
                        action_index=action_index,
                        step_id=str(event.get("step_id") or f"step_{action_index}"),
                        tool=event.get("attempted_tool"),
                        legal_success=False,
                        error_type=event.get("error_type"),
                        action_signature=attempted_signature,
                        adjacent_repeat=bool(
                            attempted_signature
                            and attempted_signature == previous_action_signature
                        ),
                        state_changed=event.get("state_before_hash") != event.get("state_after_hash"),
                        tool_error=1.0,
                    )
                )
                previous_action_signature = attempted_signature
                continue

            step = legal_steps[action_index]
            step_id = step["step_id"]
            call = step["tool_call"]
            tool = call["tool"]
            authored_arguments = call.get("arguments") or {}
            arguments = remap_replay_handles(authored_arguments, replay_handle_map)
            before = ctx["environment"].snapshot()
            signature = action_signature(tool, authored_arguments)
            repeated = signature in seen_signatures
            seen_signatures.add(signature)
            adjacent_repeat = signature == previous_action_signature
            previous_action_signature = signature
            input_table = _table_ref(arguments, ctx)
            n_in = _row_count(harness, input_table)
            references: list[dict[str, Any]] = []
            output: dict[str, Any] = {}
            output_table: str | None = None
            tables_before = set(discovered_tables)
            columns_before = set(discovered_columns)
            rows_before = set(discovered_rows)
            verified_negative_evidence = False
            step_unsupported_literals: list[dict[str, Any]] = []
            step_unsupported_columns: list[dict[str, Any]] = []
            known_column_suffixes: set[str] = set()
            for resident in (before.get("tables") or {}).values():
                if not isinstance(resident, dict):
                    continue
                resident_columns = resident.get("columns")
                if not isinstance(resident_columns, list):
                    resident_columns = [
                        item.get("name")
                        for item in ((resident.get("schema") or {}).get("columns") or [])
                        if isinstance(item, dict)
                    ]
                known_column_suffixes.update(
                    str(item).rsplit(".", 1)[-1].casefold()
                    for item in resident_columns
                    if isinstance(item, str)
                )

            if tool == "answer_from_context":
                replay_correct, _, _ = score(
                    harness,
                    gold_sql,
                    arguments,
                    created,
                    denotation_comparison=comparison,
                )
                final_arguments = arguments
                final_step_id = step_id
                after = before
            else:
                output, output_table = execute_tool(harness, tool, arguments, ctx, step_id)
                references = list((ctx["history"].get(step_id) or {}).get("references") or [])
                references = [
                    reference
                    for reference in references
                    if not _grounding_reference_redundant_with_task(reference, task_text)
                ]
                for column, literal in _action_literal_targets(tool, arguments):
                    if task_text_supports_literal(literal, task_text, column=column):
                        continue
                    if any(
                        same_scalar(literal, value)
                        for value in visible_observation_values
                    ):
                        continue
                    item = {"step_id": step_id, "column": column, "value": literal}
                    unsupported_action_literals.append(item)
                    step_unsupported_literals.append(item)
                after = ctx["environment"].snapshot()
                visible_observation_values.extend(_visible_output_values(output))
                if output_table:
                    recorded_output = step.get("tool_output") or {}
                    recorded_handle = recorded_output.get("table")
                    if isinstance(recorded_handle, str):
                        replay_handle_map[recorded_handle] = output_table
                    created.add(output_table)
                    table_history.append((step_id, output_table, "produced_table"))
                    _lineage_for_output(tool, input_table, output_table, lineages)
                elif input_table and tool in {"aggregate", "inspect_column", "read_subtable"}:
                    table_history.append((step_id, input_table, tool))

            observation_changed = False
            if tool in PERCEPTION_TOOLS and output:
                observation_signature = canonical_json(
                    {"tool": tool, "arguments": arguments, "output": output}
                )
                observation_changed = observation_signature not in seen_observation_signatures
                seen_observation_signatures.add(observation_signature)

            step_roots = _source_roots_from_references(references, roots_by_step)
            if input_table:
                step_roots.update(
                    roots_by_table.get(input_table, {canonical_table(input_table)})
                )
            if tool == "describe_table":
                step_roots.update(
                    canonical_table(item)
                    for item in arguments.get("tables") or []
                    if isinstance(item, str)
                )
            if tool == "join_tables":
                for joined in arguments.get("joins") or []:
                    table = joined.get("table") if isinstance(joined, dict) else None
                    if isinstance(table, str):
                        step_roots.update(
                            roots_by_table.get(table, {canonical_table(table)})
                        )
            roots_by_step[step_id] = step_roots
            if output_table:
                roots_by_table[output_table] = set(step_roots)

            discovered_tables.update(step_roots & set(target_support.tables))
            column_output = dict(output)
            history_record = ctx["history"].get(step_id) or {}
            if tool == "read_subtable" and history_record.get("observed_columns"):
                column_output["columns"] = history_record["observed_columns"]
            observed_columns = _tool_columns(tool, arguments, column_output)
            if tool != "describe_table":
                for column in observed_columns:
                    suffix = column.rsplit(".", 1)[-1].casefold()
                    if suffix not in known_column_suffixes:
                        item = {"step_id": step_id, "column": column}
                        unsupported_action_columns.append(item)
                        step_unsupported_columns.append(item)
            if tool == "describe_table":
                for described in output.get("tables", []) or []:
                    table = described.get("table_name")
                    if not isinstance(table, str):
                        continue
                    for item in described.get("columns", []) or []:
                        column = item.get("name") if isinstance(item, dict) else None
                        if isinstance(column, str):
                            discovered_columns.update(
                                matching_target_columns(
                                    column,
                                    {canonical_table(table)},
                                    target_support,
                                )
                            )
            else:
                for column in observed_columns:
                    discovered_columns.update(
                        matching_target_columns(column, step_roots, target_support)
                    )
            if output_table and tool in TABLE_PRODUCING_TOOLS:
                row_units, row_audit = table_target_row_units(
                    harness,
                    output_table,
                    target_support,
                )
                discovered_rows.update(row_units)
                verified_negative_evidence = bool(
                    row_units and all(isinstance(item, str) for item in row_units)
                )
                if not row_audit.get("rows_complete", True):
                    target_row_match_incomplete_steps.append(step_id)
            elif tool == "read_subtable":
                observed_rows = output.get("rows")
                read_columns = history_record.get("observed_columns")
                if isinstance(observed_rows, list) and isinstance(read_columns, list):
                    discovered_rows.update(
                        observed_target_row_units(
                            [str(column) for column in read_columns],
                            observed_rows,
                            target_support,
                        )
                    )

            n_out = output.get("row_count") if isinstance(output.get("row_count"), int) else None
            root = lineages.get(input_table or "")
            feature = StepFeature(
                action_index=action_index,
                step_id=step_id,
                tool=tool,
                legal_success=True,
                is_terminal=tool == "answer_from_context",
                action_signature=signature,
                state_changed=(
                    semantic_state_digest(before) != semantic_state_digest(after)
                    or observation_changed
                ),
                empty_result=_is_empty_result(tool, output),
                repeated_call=repeated,
                adjacent_repeat=adjacent_repeat,
                verified_negative_evidence=verified_negative_evidence,
                target_table_delta=_fraction_delta(
                    tables_before, discovered_tables, target_support.tables
                ),
                target_column_delta=_fraction_delta(
                    columns_before, discovered_columns, target_support.columns
                ),
                target_row_delta=_fraction_delta(
                    rows_before, discovered_rows, target_support.row_units
                ),
                answer_format=float(tool == "answer_from_context"),
                unsupported_guess=float(
                    bool(step_unsupported_literals or step_unsupported_columns)
                ),
                n_root=root[1] if root else None,
                n_in=n_in,
                n_out=n_out,
                references=references,
            )
            features.append(feature)
            if tool != "answer_from_context":
                provenance_steps.append({"step_id": step_id, "references": references})

        if final_arguments is None or final_step_id is None:
            replay_correct = False
            grounding_method, grounding_handle = "no_final_answer", None
            grounding_step_id = grounding_step_role = None
            final_refs: list[dict[str, Any]] = []
            grounding_handles: list[str] = []
            final_values: list[Any] = []
            required_value_indices: set[int] = set()
            supported_value_indices: set[int] = set()
        else:
            grounding_handle = _evidence_table(final_arguments)
            if grounding_handle:
                grounding_method = "terminal_evidence"
                grounding_step_id = grounding_step_role = None
            else:
                # Historical explicit-answer trajectories predate the mandatory evidence table.
                grounding_method, grounding_handle, grounding_step_id, grounding_step_role = (
                    _automatic_final_table(table_history)
                )
            grounding_handles = [grounding_handle] if grounding_handle else []
            final_refs = build_grounding_references(
                "answer_from_context", final_arguments, ctx["history"]
            )
            producer = ctx["handle_to_step"].get(grounding_handle) if grounding_handle else None
            if producer:
                final_refs.append(
                    {
                        "type": "data",
                        "step": producer,
                        "role": (
                            "terminal_evidence"
                            if grounding_method == "terminal_evidence"
                            else "automatic_final_table"
                        ),
                        "target": {"handle": grounding_handle},
                    }
                )
            if grounding_step_id and grounding_step_id != producer:
                final_refs.append(
                    {
                        "type": "grounding",
                        "step": grounding_step_id,
                        "role": "automatic_final_observation",
                        "target": {
                            "handle": grounding_handle,
                            "observation_tool": grounding_step_role,
                        },
                    }
                )
            final_values = _flatten_values(final_arguments.get("answer"))
            required_value_indices = {
                index for index, value in enumerate(final_values) if value not in (None, "")
            }
            supported_value_indices: set[int] = set()
            if grounding_method == "terminal_evidence" and replay_correct:
                # The exact cited relation was execution-scored under the named denotation metric;
                # there is no model-authored answer payload left to ground cell by cell.
                supported_value_indices.update(required_value_indices)
            elif grounding_handle:
                supported, observed_at = _history_value_support(
                    grounding_handle, ctx["history"], final_values
                )
                supported_value_indices.update(supported)
                if observed_at and observed_at not in {producer, grounding_step_id}:
                    final_refs.append({
                        "type": "grounding",
                        "step": observed_at,
                        "role": "automatic_final_observation",
                        "target": {"handle": grounding_handle, "observation_tool": "read_subtable"},
                    })

            if grounding_method != "terminal_evidence":
                # Replay-only compatibility for old explicit multi-handle answer payloads.
                latest_by_handle: dict[str, tuple[str, str]] = {}
                for history_step_id, handle, role in table_history:
                    latest_by_handle[handle] = (history_step_id, role)
                for candidate_handle in reversed(list(latest_by_handle)):
                    missing_strings = {
                        index
                        for index in required_value_indices - supported_value_indices
                        if isinstance(final_values[index], str)
                    }
                    if not missing_strings or candidate_handle in grounding_handles:
                        continue
                    supported, observed_at = _history_value_support(
                        candidate_handle, ctx["history"], final_values
                    )
                    useful = supported & missing_strings
                    if not useful:
                        continue
                    grounding_handles.append(candidate_handle)
                    supported_value_indices.update(useful)
                    candidate_producer = ctx["handle_to_step"].get(candidate_handle)
                    if candidate_producer:
                        final_refs.append({
                            "type": "data",
                            "step": candidate_producer,
                            "role": "automatic_additional_final_table",
                            "target": {"handle": candidate_handle},
                        })
                    observation_step = observed_at or latest_by_handle[candidate_handle][0]
                    if observation_step and observation_step != candidate_producer:
                        final_refs.append({
                            "type": "grounding",
                            "step": observation_step,
                            "role": "automatic_additional_final_observation",
                            "target": {
                                "handle": candidate_handle,
                                "observation_tool": "read_subtable",
                            },
                        })
                if len(grounding_handles) > 1:
                    grounding_method = "automatic_multi_table"
            deduped_refs: list[dict[str, Any]] = []
            seen_refs: set[str] = set()
            for ref in final_refs:
                key = canonical_json(ref)
                if key not in seen_refs:
                    seen_refs.add(key)
                    deduped_refs.append(ref)
            final_refs = deduped_refs
            provenance_steps.append({"step_id": final_step_id, "references": final_refs})

        # B is the answer's data/value dependency chain.  Row/domain observations carry factual
        # values and therefore remain causal data dependencies; schema observations are control
        # evidence and belong to E only.  Keeping schema edges out of B prevents describe_table
        # from receiving duplicate B+E credit for the same observation.
        causal_steps = []
        for node in provenance_steps:
            causal_steps.append({
                **node,
                "references": [
                    ref
                    for ref in node.get("references") or []
                    if not (
                        ref.get("type") == "grounding"
                        and ref.get("role") == "schema_observation"
                    )
                ],
            })
        attempted_slice_ids = backward_slice(
            {"steps": causal_steps},
            reference_type=("data", "value", "grounding"),
        )
        slice_ids = attempted_slice_ids if replay_correct else set()
        evidence_producer = ctx["handle_to_step"].get(grounding_handle) if grounding_handle else None
        grounding_table_audit: dict[str, Any] | None = None
        if grounding_handle:
            try:
                audit_columns = list(harness._cols(grounding_handle))  # noqa: SLF001
                audit_probe_rows = [
                    list(row)
                    for row in harness.conn.execute(
                        f"SELECT * FROM {harness._src(grounding_handle)} LIMIT 100"  # noqa: SLF001
                    ).fetchall()
                ]
                final_answer = final_arguments.get("answer") if final_arguments else None
                grounding_table_audit = {
                    "handle": grounding_handle,
                    "row_count": _row_count(harness, grounding_handle),
                    "columns": audit_columns,
                    "rows_sample": _audit_evidence_rows(audit_probe_rows, final_answer),
                    "final_answer": final_answer,
                }
            except Exception as exc:  # audit metadata must never change reward semantics
                grounding_table_audit = {
                    "handle": grounding_handle,
                    "audit_error": f"{type(exc).__name__}: {exc}",
                }
        used_units: set[str] = set()
        evidence_steps: set[str] = set()
        by_step = {feature.step_id: feature for feature in features}
        if replay_correct and evidence_producer:
            evidence_steps.add(evidence_producer)
            used_units.add(f"table:{grounding_handle}")
        if replay_correct:
            for handle in grounding_handles:
                producer_step = ctx["handle_to_step"].get(handle)
                if producer_step:
                    evidence_steps.add(producer_step)
                    used_units.add(f"table:{handle}")
        for node in provenance_steps:
            if node.get("step_id") != final_step_id and node.get("step_id") not in slice_ids:
                continue
            for ref in node.get("references") or []:
                ref_step = ref.get("step")
                if ref.get("type") != "grounding" or ref_step not in by_step:
                    continue
                evidence_steps.add(ref_step)
                used_units.add(f"{ref.get('role', 'grounding')}:{ref_step}")

        for step_id in attempted_slice_ids:
            if step_id in by_step:
                by_step[step_id].attempted_back_slice = 1.0
        for step_id in slice_ids:
            if step_id in by_step:
                by_step[step_id].back_slice = 1.0
        for step_id in evidence_steps:
            if step_id in by_step:
                by_step[step_id].new_used_evidence = 1.0

        previous: StepFeature | None = None
        for feature in features:
            if previous is not None:
                feature.feedback_error_before = bool(previous.error_type)
                feature.feedback_empty_before = previous.empty_result
                if (
                    (previous.error_type or previous.empty_result)
                    and previous.action_signature
                    and feature.action_signature
                ):
                    feature.action_changed_after_feedback = (
                        previous.action_signature != feature.action_signature
                    )
                    feature.action_changed_after_empty = bool(
                        previous.empty_result and feature.action_changed_after_feedback
                    )
            if (
                feature.tool in SELECTIVE_TOOLS
                and feature.back_slice
                and feature.n_root is not None
                and feature.n_root > 0
                and feature.n_in is not None
                and feature.n_out is not None
                and 0 < feature.n_out <= feature.n_in
            ):
                feature.search_reduction = normalized_search_reduction(
                    feature.n_root, feature.n_in, feature.n_out
                )
            if (
                feature.tool in SELECTIVE_TOOLS
                and feature.n_root is not None
                and feature.n_in is not None
                and feature.n_out == 0
                and feature.n_root >= feature.n_in > 0
                and not feature.verified_negative_evidence
            ):
                feature.empty_result_penalty = 1.0
            feedback_before = feature.feedback_error_before or feature.feedback_empty_before
            verifiable_progress = bool(
                feature.target_table_delta
                + feature.target_column_delta
                + feature.target_row_delta
                > 0
                or (
                    replay_correct
                    and feature.back_slice
                    + feature.new_used_evidence
                    + feature.search_reduction
                    > 0
                )
            )
            valid_response = bool(
                feature.legal_success
                and (
                    replay_correct and feature.is_terminal
                    or (
                        feature.action_changed_after_feedback
                        and feature.state_changed
                        and verifiable_progress
                    )
                )
            )
            feature.feedback_response = float(feedback_before and valid_response)
            feature.repeat_without_feedback = float(
                feature.repeated_call
                and not feature.feedback_error_before
                and not feature.feedback_empty_before
            )
            feature.legal_no_state_change = float(
                feature.legal_success and not feature.is_terminal and not feature.state_changed
            )
            feature.ignored_feedback = float(
                feedback_before and not valid_response
            )
            previous = feature

        return features, {
            "denotation_comparison": comparison,
            "catalog_source": catalog_source,
            "replay_correct": replay_correct,
            "replay_handle_map": replay_handle_map,
            "grounding_method": grounding_method,
            "grounding_handle": grounding_handle,
            "grounding_handles": grounding_handles,
            "back_slice_step_ids": sorted(slice_ids, key=lambda value: int(value.rsplit("_", 1)[1])),
            "attempted_back_slice_step_ids": sorted(
                attempted_slice_ids, key=lambda value: int(value.rsplit("_", 1)[1])
            ),
            "used_evidence_units": sorted(used_units),
            "structured_grounding_available": bool(slice_ids),
            "grounding_step_id": grounding_step_id,
            "grounding_step_role": grounding_step_role,
            "grounding_table_audit": grounding_table_audit,
            "final_references": final_refs,
            "unsupported_final_values": [
                final_values[index]
                for index in sorted(required_value_indices - supported_value_indices)
            ],
            "final_value_grounding_complete": (
                not required_value_indices or required_value_indices <= supported_value_indices
            ),
            "unsupported_action_literals": unsupported_action_literals,
            "unsupported_action_columns": unsupported_action_columns,
            "action_literal_grounding_complete": not unsupported_action_literals,
            "action_grounding_clean": (
                not unsupported_action_literals and not unsupported_action_columns
            ),
            "deterministic_grounding_complete": (
                (not required_value_indices or required_value_indices <= supported_value_indices)
                and target_support.sql_parse_complete
                and target_support.rows_complete
                and (not replay_correct or bool(slice_ids))
            ),
            "target_support": {
                "tables": sorted(target_support.tables),
                "columns": sorted(target_support.columns),
                "row_units": len(target_support.row_units),
                "sql_parse_complete": target_support.sql_parse_complete,
                "rows_complete": target_support.rows_complete,
                "row_match_incomplete_steps": target_row_match_incomplete_steps,
                **target_support.diagnostics,
            },
            "discovered_target_support": {
                "tables": sorted(discovered_tables),
                "columns": sorted(discovered_columns),
                "row_units": len(discovered_rows),
            },
            "max_grounding_rows": MAX_GROUNDING_ROWS,
        }
    finally:
        harness.conn.close()


def allocate_process_rewards(
    trajectory_id: str,
    features: list[StepFeature],
    *,
    correct: bool,
    config: ProcessRewardConfig | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> EpisodeReward:
    config = config or ProcessRewardConfig()
    config.validate()
    if not features:
        raise ValueError("cannot allocate reward over an empty trajectory")

    for feature in features:
        feature.tool_error = max(feature.tool_error, float(bool(feature.error_type)))
        feature.terminal_failure = float(not correct and feature is features[-1])
        feature.failure_responsibility = feature.terminal_failure
        feature.target_potential_delta = (
            config.omega_target_table * feature.target_table_delta
            + config.omega_target_column * feature.target_column_delta
            + config.omega_target_row * feature.target_row_delta
        )

    positive = [
        config.w_terminal_correct * float(feature.is_terminal)
        + config.w_back_slice * feature.back_slice
        + config.w_new_evidence * feature.new_used_evidence
        + config.w_search_reduction * feature.search_reduction
        + config.w_feedback_response * feature.feedback_response
        + config.w_target_potential * feature.target_potential_delta
        for feature in features
    ]
    outcome_penalties = [
        config.lambda_terminal_failure * feature.terminal_failure
        for feature in features
    ]
    local_penalties = [
        config.lambda_tool_error * feature.tool_error
        + config.lambda_adjacent_repeat * float(feature.adjacent_repeat)
        + config.lambda_repeat_without_feedback * feature.repeat_without_feedback
        + config.lambda_legal_no_state_change * feature.legal_no_state_change
        + config.lambda_ignored_feedback * feature.ignored_feedback
        + config.lambda_unsupported_guess * feature.unsupported_guess
        + config.lambda_empty_result * feature.empty_result_penalty
        for feature in features
    ]
    penalties = [
        outcome + local
        for outcome, local in zip(outcome_penalties, local_penalties, strict=True)
    ]
    positive_mass = sum(positive)
    raw_penalty_mass = sum(penalties)
    capped_penalty = min(config.penalty_cap, raw_penalty_mass)
    process_update = bool(not correct or positive_mass > 0)
    c_positive = [0.0] * len(features)
    if positive_mass > 0:
        c_positive = [value / positive_mass for value in positive]
    c_negative = (
        [value / raw_penalty_mass for value in penalties]
        if raw_penalty_mass > 0
        else [0.0] * len(features)
    )
    rewards = [
        (
            float(correct) * plus
            + float(not correct) * config.eta_failure_progress * feature.target_potential_delta
            + config.lambda_answer_format * feature.answer_format
            - capped_penalty * minus
        )
        for feature, plus, minus in zip(features, c_positive, c_negative, strict=True)
    ]
    expected = (
        float(correct and positive_mass > 0)
        + float(not correct)
        * config.eta_failure_progress
        * sum(feature.target_potential_delta for feature in features)
        + config.lambda_answer_format * sum(feature.answer_format for feature in features)
        - capped_penalty
    )
    if not math.isclose(sum(rewards), expected, rel_tol=0.0, abs_tol=1e-9):
        raise AssertionError(f"reward conservation failed: {sum(rewards)} != {expected}")
    if correct and process_update and sum(rewards) <= 0:
        raise AssertionError("a correct trajectory must retain positive total reward")
    if not correct and sum(rewards) > 0:
        raise AssertionError("a failed trajectory cannot receive positive total reward")

    step_rewards = []
    for feature, g_value, p_outcome, p_local, p_value, plus, minus, reward in zip(
        features,
        positive,
        outcome_penalties,
        local_penalties,
        penalties,
        c_positive,
        c_negative,
        rewards,
        strict=True,
    ):
        step_rewards.append(
            StepReward(
                action_index=feature.action_index,
                step_id=feature.step_id,
                tool=feature.tool,
                g_positive=round(g_value, 10),
                p_outcome=round(p_outcome, 10),
                p_local=round(p_local, 10),
                p_raw=round(p_value, 10),
                c_positive=round(plus, 10),
                c_negative=round(minus, 10),
                reward=round(reward, 10),
                features={
                    key: value
                    for key, value in asdict(feature).items()
                    if key not in {"references", "action_signature"}
                },
            )
        )

    diagnostics = dict(diagnostics or {})
    diagnostics["failure_outcome_allocation"] = "terminal_boundary"
    diagnostics["fallback_positive_allocation"] = "disabled_exclude_correct_G0"
    diagnostics["outcome_penalty_mass"] = round(sum(outcome_penalties), 10)
    diagnostics["local_penalty_mass"] = round(sum(local_penalties), 10)
    diagnostics["answer_format_mass"] = round(
        config.lambda_answer_format * sum(feature.answer_format for feature in features),
        10,
    )
    diagnostics["failure_progress_mass"] = round(
        float(not correct)
        * config.eta_failure_progress
        * sum(feature.target_potential_delta for feature in features),
        10,
    )
    diagnostics["process_update"] = process_update
    diagnostics["expected_total_reward"] = round(expected, 10)
    return EpisodeReward(
        trajectory_id=trajectory_id,
        correct=correct,
        grounding_method=str(diagnostics.get("grounding_method", "unknown")),
        grounding_handle=diagnostics.get("grounding_handle"),
        back_slice_step_ids=list(diagnostics.get("back_slice_step_ids") or []),
        used_evidence_units=list(diagnostics.get("used_evidence_units") or []),
        positive_mass=round(positive_mass, 10),
        raw_penalty_mass=round(raw_penalty_mass, 10),
        capped_penalty=round(capped_penalty, 10),
        total_reward=round(sum(rewards), 10),
        fallback_terminal_credit=False,
        process_update=process_update,
        steps=step_rewards,
        diagnostics=diagnostics,
    )


def score_verified_trajectory(
    trajectory: dict[str, Any],
    config: ProcessRewardConfig | None = None,
    *,
    denotation_comparison: str = "bird-set",
) -> EpisodeReward:
    features, diagnostics = replay_step_features(
        trajectory,
        denotation_comparison=denotation_comparison,
    )
    correct = bool(diagnostics["replay_correct"] and trajectory.get("label_status") == "verified")
    return allocate_process_rewards(
        str(trajectory.get("trajectory_id", "unknown")),
        features,
        correct=correct,
        config=config,
        diagnostics=diagnostics,
    )


def score_rollout_trajectory(
    trajectory: dict[str, Any],
    config: ProcessRewardConfig | None = None,
    *,
    denotation_comparison: str = "bird-set",
) -> EpisodeReward:
    """Score one freshly sampled semantic rollout without requiring an SFT verification label."""
    features, diagnostics = replay_step_features(
        trajectory,
        denotation_comparison=denotation_comparison,
    )
    return allocate_process_rewards(
        str(trajectory.get("trajectory_id", "unknown")),
        features,
        correct=bool(diagnostics["replay_correct"]),
        config=config,
        diagnostics=diagnostics,
    )
