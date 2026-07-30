#!/usr/bin/env python3
"""Harness-grounded step credit for table-tool RL trajectories.

The module implements the normalized reward proposed in the project notes:

    r_t = C * c_t^+ - P * c_t^-

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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from executor import Harness  # noqa: E402
from provenance import (  # noqa: E402
    PERCEPTION_TOOLS,
    backward_slice,
    build_grounding_references,
    condition_literal_targets,
)
from rollout import (  # noqa: E402
    execute_tool,
    new_ctx,
    overview,
    score,
    state_digest,
)


SELECTIVE_TOOLS = frozenset({"condition_filter", "extreme_value_select"})
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
        "join",
        "join_tables",
        "set_op",
        "read_subtable",
    }
)
MAX_GROUNDING_ROWS = 10_000


@dataclass(frozen=True)
class ProcessRewardConfig:
    """Configurable weights; defaults are audit values, not tuned research conclusions."""

    w_back_slice: float = 1.0
    w_new_evidence: float = 1.0
    w_search_reduction: float = 1.0
    w_feedback_response: float = 1.0
    lambda_terminal_failure: float = 0.30
    lambda_tool_error: float = 0.08
    lambda_repeat_without_feedback: float = 0.06
    lambda_legal_no_state_change: float = 0.03
    lambda_ignored_feedback: float = 0.05
    failure_chain_weight: float = 1.0
    failure_state_change_bonus: float = 0.25
    failure_terminal_bonus: float = 0.50
    fallback_legal_weight: float = 0.25
    fallback_state_change_weight: float = 1.0
    fallback_terminal_weight: float = 0.50
    penalty_cap: float = 0.80

    def validate(self) -> None:
        values = asdict(self)
        if any(value < 0 for key, value in values.items() if key != "penalty_cap"):
            raise ValueError("all reward and penalty weights must be non-negative")
        if not 0 < self.penalty_cap < 1:
            raise ValueError("penalty_cap must satisfy 0 < P_max < 1")


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
    feedback_error_before: bool = False
    feedback_empty_before: bool = False
    action_changed_after_empty: bool = False
    back_slice: float = 0.0
    attempted_back_slice: float = 0.0
    new_used_evidence: float = 0.0
    search_reduction: float = 0.0
    feedback_response: float = 0.0
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


def _remap_replay_handles(value: Any, handle_map: dict[str, str]) -> Any:
    """Translate recorded online handles to handles allocated during legal-only replay.

    Recoverable execution failures are retained as audit events but omitted from the legal SFT
    step list. Some executor failures consume an internal handle number before raising, so a later
    online handle can be ``project_006`` while legal-only replay allocates ``project_005``. Keep
    model-authored calls unchanged in the stored trajectory, but translate exact table references
    and logical ``handle.column`` namespaces at replay time.
    """
    if isinstance(value, list):
        return [_remap_replay_handles(item, handle_map) for item in value]
    if isinstance(value, dict):
        return {key: _remap_replay_handles(item, handle_map) for key, item in value.items()}
    if not isinstance(value, str) or not handle_map:
        return value
    if value in handle_map:
        return handle_map[value]
    rendered = value
    for recorded, replayed in sorted(handle_map.items(), key=lambda item: -len(item[0])):
        rendered = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(recorded)}(?=\.)",
            replayed,
            rendered,
        )
    return rendered


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


def _same_grounded_value(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    try:
        return left == right
    except Exception:
        return False


def _task_text_supports_literal(value: Any, text: str) -> bool:
    """Conservative check for constants supplied directly by the question/evidence."""
    if value is None:
        return True
    needle = str(value).strip().strip("%_").casefold()
    if not needle:
        return True
    haystack = " ".join(str(text).casefold().split())
    # Numeric boundaries prevent id 7 from matching 2017.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        import re
        return bool(re.search(rf"(?<!\d){re.escape(needle)}(?!\d)", haystack))
    return needle in haystack


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
            if any(_same_grounded_value(value, cell) for cell in cells):
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


def replay_step_features(trajectory: dict[str, Any]) -> tuple[list[StepFeature], dict[str, Any]]:
    """Replay one normalized trajectory and return harness-authored feature rows."""
    source = trajectory.get("source") or {}
    db_path = source.get("db_path")
    gold_sql = source.get("gold_sql")
    if not db_path or not gold_sql:
        raise ValueError("trajectory source must contain db_path and gold_sql")

    harness = Harness(str(db_path))
    try:
        catalog = overview(harness)
        ctx = new_ctx(catalog)
        created: set[str] = set()
        table_history: list[tuple[str, str, str]] = []
        lineages: dict[str, tuple[str, int]] = {}
        for table in catalog.get("tables", []):
            name = table.get("table_name")
            count = table.get("row_count", table.get("num_rows"))
            if isinstance(name, str) and isinstance(count, int):
                lineages[name] = (name, count)

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
        final_arguments: dict[str, Any] | None = None
        final_step_id: str | None = None
        replay_correct = False
        replay_handle_map: dict[str, str] = {}
        unsupported_action_literals: list[dict[str, Any]] = []
        task_text = "\n".join(filter(None, [
            trajectory.get("question"),
            source.get("external_knowledge"),
        ]))

        for action_index in action_indices:
            if action_index in error_events:
                event = error_events[action_index]
                features.append(
                    StepFeature(
                        action_index=action_index,
                        step_id=str(event.get("step_id") or f"step_{action_index}"),
                        tool=event.get("attempted_tool"),
                        legal_success=False,
                        error_type=event.get("error_type"),
                        action_signature=(
                            action_signature(
                                event["attempted_tool"], event.get("attempted_arguments") or {}
                            )
                            if event.get("attempted_tool")
                            else None
                        ),
                        state_changed=event.get("state_before_hash") != event.get("state_after_hash"),
                        tool_error=1.0,
                    )
                )
                continue

            step = legal_steps[action_index]
            step_id = step["step_id"]
            call = step["tool_call"]
            tool = call["tool"]
            authored_arguments = call.get("arguments") or {}
            arguments = _remap_replay_handles(authored_arguments, replay_handle_map)
            before = ctx["environment"].snapshot()
            signature = action_signature(tool, authored_arguments)
            repeated = signature in seen_signatures
            seen_signatures.add(signature)
            input_table = _table_ref(arguments, ctx)
            n_in = _row_count(harness, input_table)
            references: list[dict[str, Any]] = []
            output: dict[str, Any] = {}
            output_table: str | None = None

            if tool == "answer_from_context":
                replay_correct, _, _ = score(harness, gold_sql, arguments, created)
                final_arguments = arguments
                final_step_id = step_id
                after = before
            else:
                output, output_table = execute_tool(harness, tool, arguments, ctx, step_id)
                references = list((ctx["history"].get(step_id) or {}).get("references") or [])
                if tool == "condition_filter":
                    observed_values = [
                        value
                        for ref in references
                        if ref.get("type") == "grounding"
                        and ref.get("role") in {"domain_observation", "row_observation"}
                        for value in (ref.get("target") or {}).get("values", [])
                    ]
                    for column, literal in condition_literal_targets(arguments.get("conditions")):
                        if _task_text_supports_literal(literal, task_text):
                            continue
                        if any(_same_grounded_value(literal, value) for value in observed_values):
                            continue
                        unsupported_action_literals.append({
                            "step_id": step_id,
                            "column": column,
                            "value": literal,
                        })
                after = ctx["environment"].snapshot()
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

            n_out = output.get("row_count") if isinstance(output.get("row_count"), int) else None
            root = lineages.get(input_table or "")
            feature = StepFeature(
                action_index=action_index,
                step_id=step_id,
                tool=tool,
                legal_success=True,
                is_terminal=tool == "answer_from_context",
                action_signature=signature,
                state_changed=state_digest(before) != state_digest(after),
                empty_result=_is_empty_result(tool, output),
                repeated_call=repeated,
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
        else:
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
                        "role": "automatic_final_table",
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
            if grounding_handle:
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

            # A row-valued answer may combine values from several separately read handles. Search
            # backward only for still-uncovered string values; numeric equality is too collision-
            # prone to establish an additional dependency without an explicit data edge.
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
                        "target": {"handle": candidate_handle, "observation_tool": "read_subtable"},
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

        if final_arguments is None:
            grounding_handles = []
            final_values = []
            required_value_indices = set()
            supported_value_indices = set()

        attempted_slice_ids = backward_slice(
            {"steps": provenance_steps},
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
                if ref.get("type") != "grounding" or ref_step not in slice_ids:
                    continue
                evidence_steps.add(ref_step)
                used_units.add(f"{ref.get('role', 'grounding')}:{ref_step}")

        by_step = {feature.step_id: feature for feature in features}
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
                if previous.empty_result and previous.action_signature and feature.action_signature:
                    feature.action_changed_after_empty = previous.action_signature != feature.action_signature
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
            if feature.legal_success:
                feature.feedback_response = min(
                    1.0,
                    float(feature.feedback_error_before)
                    + float(
                        feature.feedback_empty_before
                        and feature.action_changed_after_empty
                        and feature.state_changed
                    ),
                )
            feature.repeat_without_feedback = float(
                feature.repeated_call
                and not feature.feedback_error_before
                and not feature.feedback_empty_before
            )
            feature.legal_no_state_change = float(
                feature.legal_success and not feature.is_terminal and not feature.state_changed
            )
            feature.ignored_feedback = float(
                (feature.feedback_error_before or feature.feedback_empty_before)
                and not feature.is_terminal
                and (
                    (previous is not None and previous.action_signature == feature.action_signature)
                    or not feature.state_changed
                )
            )
            previous = feature

        return features, {
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
            "action_literal_grounding_complete": not unsupported_action_literals,
            "deterministic_grounding_complete": (
                (not required_value_indices or required_value_indices <= supported_value_indices)
                and not unsupported_action_literals
            ),
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
        feature.failure_responsibility = 0.0

    positive = [
        config.w_back_slice * feature.back_slice
        + config.w_new_evidence * feature.new_used_evidence
        + config.w_search_reduction * feature.search_reduction
        + config.w_feedback_response * feature.feedback_response
        for feature in features
    ]
    local_penalties = [
        config.lambda_tool_error * feature.tool_error
        + config.lambda_repeat_without_feedback * feature.repeat_without_feedback
        + config.lambda_legal_no_state_change * feature.legal_no_state_change
        + config.lambda_ignored_feedback * feature.ignored_feedback
        for feature in features
    ]
    outcome_weights = [0.0] * len(features)
    if not correct:
        has_attempted_slice = any(feature.attempted_back_slice for feature in features)
        has_legal_action = any(feature.legal_success for feature in features)
        for index, feature in enumerate(features):
            if has_attempted_slice:
                eligible = bool(feature.attempted_back_slice or feature.is_terminal)
            elif has_legal_action:
                eligible = feature.legal_success
            else:
                eligible = True
            if not eligible:
                continue
            weight = config.failure_chain_weight
            if feature.state_changed:
                weight += config.failure_state_change_bonus
            if feature.is_terminal or index == len(features) - 1:
                weight += config.failure_terminal_bonus
            outcome_weights[index] = weight
            feature.failure_responsibility = weight
    outcome_weight_mass = sum(outcome_weights)
    if not correct and outcome_weight_mass <= 0:
        # A zeroed legacy/custom configuration must still conserve the requested failure budget.
        outcome_weights[-1] = 1.0
        features[-1].failure_responsibility = 1.0
        outcome_weight_mass = 1.0
    outcome_penalties = [0.0] * len(features)
    if not correct and outcome_weight_mass > 0:
        outcome_penalties = [
            config.lambda_terminal_failure * weight / outcome_weight_mass
            for weight in outcome_weights
        ]
    penalties = [
        outcome + local
        for outcome, local in zip(outcome_penalties, local_penalties, strict=True)
    ]
    positive_mass = sum(positive)
    raw_penalty_mass = sum(penalties)
    capped_penalty = min(config.penalty_cap, raw_penalty_mass)
    fallback = positive_mass <= 0
    c_positive = [0.0] * len(features)
    if positive_mass > 0:
        c_positive = [value / positive_mass for value in positive]
    else:
        fallback_weights = [
            config.fallback_legal_weight * float(feature.legal_success)
            + config.fallback_state_change_weight * float(feature.state_changed)
            + config.fallback_terminal_weight * float(feature.is_terminal)
            for feature in features
        ]
        fallback_mass = sum(fallback_weights)
        if fallback_mass > 0:
            c_positive = [value / fallback_mass for value in fallback_weights]
        else:
            c_positive[-1] = 1.0
    c_negative = (
        [value / raw_penalty_mass for value in penalties]
        if raw_penalty_mass > 0
        else [0.0] * len(features)
    )
    rewards = [
        float(correct) * plus - capped_penalty * minus
        for plus, minus in zip(c_positive, c_negative, strict=True)
    ]
    expected = float(correct) - capped_penalty
    if not math.isclose(sum(rewards), expected, rel_tol=0.0, abs_tol=1e-9):
        raise AssertionError(f"reward conservation failed: {sum(rewards)} != {expected}")
    if correct and sum(rewards) <= 0:
        raise AssertionError("a correct trajectory must retain positive total reward")

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
    diagnostics["failure_outcome_allocation"] = "attempted_dependency_chain"
    diagnostics["fallback_positive_allocation"] = "environment_weighted"
    diagnostics["outcome_penalty_mass"] = round(sum(outcome_penalties), 10)
    diagnostics["local_penalty_mass"] = round(sum(local_penalties), 10)
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
        fallback_terminal_credit=fallback,
        steps=step_rewards,
        diagnostics=diagnostics,
    )


def score_verified_trajectory(
    trajectory: dict[str, Any], config: ProcessRewardConfig | None = None
) -> EpisodeReward:
    features, diagnostics = replay_step_features(trajectory)
    correct = bool(diagnostics["replay_correct"] and trajectory.get("label_status") == "verified")
    return allocate_process_rewards(
        str(trajectory.get("trajectory_id", "unknown")),
        features,
        correct=correct,
        config=config,
        diagnostics=diagnostics,
    )
