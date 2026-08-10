"""Compact, no-leak model rendering for checkpoint-relalg state."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any

from .checkpoint_store import CheckpointNode, CheckpointStore
from .environment_state import EnvironmentState, Observation, StepRecord
from .relation_artifact import ForeignKey, RelationArtifact, SourceRelation


_FORBIDDEN_KEY = re.compile(r"[^a-z0-9]+")


def _is_forbidden_key(key: Any) -> bool:
    normalized = _FORBIDDEN_KEY.sub("_", str(key).lower()).strip("_")
    compact = normalized.replace("_", "")
    return (
        normalized.startswith("gold")
        or compact.startswith("gold")
        or "evaluator" in normalized
        or normalized in {
            "evaluation",
            "evaluation_result",
            "expected_answer",
            "expected_rows",
            "reference_answer",
            "reference_sql",
        }
    )


def _safe_value(value: Any) -> Any:
    """Drop evaluation-only keys without editing legitimate textual values."""

    if hasattr(value, "to_payload") and callable(value.to_payload):
        return _safe_value(value.to_payload())
    if isinstance(value, Mapping):
        return {
            str(key): _safe_value(item)
            for key, item in value.items()
            if not _is_forbidden_key(key)
        }
    if isinstance(value, (tuple, list)):
        return [_safe_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_safe_value(item) for item in value), key=str)
    if isinstance(value, bytes):
        return f"<BLOB {len(value)} bytes>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _compact(value: Any) -> str:
    safe = _safe_value(value)
    if isinstance(safe, str):
        return safe
    return json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _contract(value: Any) -> str:
    safe = _safe_value(value)
    if safe in (None, "", {}, []):
        return "(none)"
    if isinstance(safe, str):
        return safe
    if isinstance(safe, Mapping) and len(safe) == 1:
        only_value = next(iter(safe.values()))
        if isinstance(only_value, str):
            return only_value
    return _compact(safe)


def _render_source(source: SourceRelation) -> list[str]:
    lines = [f"{source.name}[{source.row_count}]:"]
    primary = set(source.primary_key)
    simple_foreign: dict[str, ForeignKey] = {
        key.columns[0]: key
        for key in source.foreign_keys
        if len(key.columns) == 1 and len(key.ref_columns) == 1
    }
    for column in source.columns:
        annotations: list[str] = []
        if column.name in primary:
            annotations.append("PK")
        foreign = simple_foreign.get(column.name)
        if foreign is not None:
            annotations.append(f"FK->{foreign.ref_table}.{foreign.ref_columns[0]}")
        suffix = f" {' '.join(annotations)}" if annotations else ""
        lines.append(f"  {column.name} {column.canonical_type}{suffix}")
    for foreign in source.foreign_keys:
        if len(foreign.columns) > 1:
            lines.append(
                "  FK("
                + ",".join(foreign.columns)
                + f")->{foreign.ref_table}("
                + ",".join(foreign.ref_columns)
                + ")"
            )
    return lines


def _render_artifact(artifact: RelationArtifact) -> list[str]:
    columns = ", ".join(
        f"{column.name} {column.canonical_type}" for column in artifact.columns
    ) or "(zero columns)"
    if artifact.ordered_by:
        ordering = ", ".join(
            f"{key.column} {key.direction} nulls {key.nulls}" for key in artifact.ordered_by
        )
    else:
        ordering = "none"
    lines = [
        f"{artifact.table}[{artifact.row_count}] kind={artifact.kind}",
        f"  columns: {columns}",
        f"  order: {ordering}",
        f"  derived: {_compact(artifact.derivation) if artifact.derivation else 'source-independent'}",
    ]
    # ``None`` is both the dataclass default and SQL NULL.  Shape determines
    # whether a scalar exists, so a grounded 1x1 NULL must still be rendered.
    if artifact.row_count == 1 and len(artifact.columns) == 1:
        lines.append(f"  scalar: {_compact(artifact.scalar_cell)}")
    return lines


def _render_observation(observation: Observation) -> str:
    return (
        f"{observation.observation_id} {observation.kind}({observation.table}): "
        f"{_compact(observation.payload)}"
    )


def _render_step(step: StepRecord) -> str:
    handles = ", ".join(step.produced_artifact_ids)
    return f"{step.step_id} {step.tool} -> {handles}"


def _render_history(checkpoints: CheckpointStore) -> list[str]:
    lines = [
        "Working memory only; not database evidence. Model-authored values must be",
        "rechecked against CURRENT ENVIRONMENT STATE before use as evidence.",
        "Legend: + progress  ? uncertainty  ! restore reason  -> next target",
    ]
    for node in sorted(checkpoints.nodes.values(), key=lambda item: item.sequence):
        parent = node.parent_id if node.parent_id is not None else "-"
        transition = ""
        if node.created_by == "restore_checkpoint":
            transition = ", restored"
        elif node.created_by == "bootstrap_checkpoint":
            transition = ", initial-targets"
        lines.append(f"{node.checkpoint_id}({parent}, {node.status}{transition})")
        if node.phase_targets:
            lines.append("  phase: " + " | ".join(node.phase_targets))
        elif node.checkpoint_id != "root":
            lines.append("  phase: bootstrap exploration")
        for item in node.progress_summary:
            lines.append(f"  + {item}")
        for item in node.remaining_uncertainties:
            lines.append(f"  ? {item}")
        if node.restore_reason:
            lines.append(f"  ! {node.restore_reason}")
        for item in node.next_targets:
            lines.append(f"  -> {item}")
    return lines


def _render_root_history(state: EnvironmentState) -> list[str]:
    """Render a truthful synthetic root when no checkpoint store exists yet."""

    lines = [
        "Working memory only; not database evidence. Model-authored values must be",
        "rechecked against CURRENT ENVIRONMENT STATE before use as evidence.",
        "Legend: + progress  ? uncertainty  ! restore reason  -> next target",
        "root(-, active_path)",
    ]
    for target in state.current_targets:
        lines.append(f"  -> {target}")
    return lines


class EnvironmentRenderer:
    """Render the six protocol sections in their fixed authority order."""

    def render(
        self,
        question: Any,
        external_knowledge: Any,
        state: EnvironmentState,
        checkpoints: CheckpointStore | None = None,
        *,
        checkpoint_store: CheckpointStore | None = None,
    ) -> str:
        if checkpoints is not None and checkpoint_store is not None:
            raise ValueError("provide checkpoints or checkpoint_store, not both")
        store = checkpoints if checkpoints is not None else checkpoint_store
        if store is not None and store.state is not state:
            raise ValueError("checkpoint store and renderer state must be identical")
        if store is None and state.checkpoint_id != "root":
            raise ValueError("a checkpoint store is required after the bootstrap phase")

        targets = (
            "\n".join(f"- {target}" for target in state.current_targets)
            if state.current_targets
            else "bootstrap exploration (no local target)"
        )
        environment = self.render_environment(state)
        sections = [
            ("QUESTION", _contract(question)),
            ("EXTERNAL KNOWLEDGE", _contract(external_knowledge)),
            ("CURRENT PHASE TARGETS", targets),
            (
                "CHECKPOINT HISTORY",
                "\n".join(_render_history(store) if store is not None else _render_root_history(state)),
            ),
            ("CURRENT ENVIRONMENT STATE", environment),
        ]
        if state.last_error is not None:
            sections.append(("LAST ERROR", _compact(state.last_error)))
        return "\n\n".join(f"{heading}\n\n{body}" for heading, body in sections)

    def render_environment(self, state: EnvironmentState) -> str:
        lines = ["Authoritative database facts for the current active branch."]

        # The opening catalog is always visible so the model can choose which
        # schema to inspect.  It exposes table-level topology, not undiscovered
        # column names; exact FK columns appear only in a discovered schema.
        lines.append("CATALOG")
        if state.sources:
            for table in sorted(state.sources):
                source = state.sources[table]
                referenced_tables = sorted({key.ref_table for key in source.foreign_keys})
                relation_links = (
                    " " + " ".join(f"FK->{target}" for target in referenced_tables)
                    if referenced_tables
                    else ""
                )
                lines.append(f"{source.name}[{source.row_count}]{relation_links}")
        else:
            lines.append("(empty catalog)")

        lines.append("SOURCES")
        if state.discovered_schema_ids:
            for table in sorted(state.discovered_schema_ids):
                lines.extend(_render_source(state.sources[table]))
        else:
            lines.append("(none discovered)")

        lines.append("ARTIFACTS")
        if state.active_artifact_ids:
            for table in sorted(state.active_artifact_ids):
                lines.extend(_render_artifact(state.artifacts[table]))
        else:
            lines.append("(none)")

        lines.append("OBSERVATIONS")
        if state.active_observation_ids:
            for observation_id in sorted(state.active_observation_ids):
                lines.append(_render_observation(state.observations[observation_id]))
        else:
            lines.append("(none)")

        lines.append("USABLE PRODUCING STEPS")
        if state.usable_step_ids:
            for step_id in sorted(state.usable_step_ids):
                lines.append(_render_step(state.steps[step_id]))
        else:
            lines.append("(none)")
        return "\n".join(lines)


def render_context(
    question: Any,
    external_knowledge: Any,
    state: EnvironmentState,
    checkpoints: CheckpointStore,
) -> str:
    return EnvironmentRenderer().render(question, external_knowledge, state, checkpoints)


def render_environment(state: EnvironmentState) -> str:
    return EnvironmentRenderer().render_environment(state)


__all__ = ["EnvironmentRenderer", "render_context", "render_environment"]
