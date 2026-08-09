"""Resident logical state for checkpoint-relalg episodes.

Facts are append-only in the backing stores.  Branching only changes explicit
membership sets, which makes a checkpoint snapshot small and makes abandoned
facts impossible to resolve through :meth:`EnvironmentState.get_relation`.
"""

from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Iterable, Mapping

from .relation_artifact import (
    Relation,
    RelationArtifact,
    SourceRelation,
    _freeze_value,
    _thaw_value,
    sqlite_identifier_key,
)


class StateError(ValueError):
    """Structured validation error safe to return through the tool protocol."""

    def __init__(
        self,
        code: str,
        message: str | None = None,
        details: Mapping[str, Any] | None = None,
        *,
        error_type: str = "state_validation_error",
    ) -> None:
        self.type = error_type
        self.code = code
        self.message = message or code.replace("_", " ")
        self.details = deepcopy(dict(details or {}))
        super().__init__(self.message)

    def to_payload(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "code": self.code,
            "message": self.message,
            "details": deepcopy(self.details),
        }


@dataclass(frozen=True, slots=True)
class Observation:
    observation_id: str
    kind: str
    table: str
    payload: Any

    def __post_init__(self) -> None:
        for field_name in ("observation_id", "kind", "table"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        object.__setattr__(self, "payload", _freeze_value(self.payload))

    def to_payload(self) -> dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "kind": self.kind,
            "table": self.table,
            "payload": _thaw_value(self.payload),
        }


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_id: str
    tool: str
    status: str
    output: Any
    error: Any = None
    phase_id: str = ""
    checkpoint_id: str = "root"
    produced_artifact_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("step_id", "tool", "status"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if not isinstance(self.phase_id, str) or not isinstance(self.checkpoint_id, str):
            raise ValueError("phase_id and checkpoint_id must be strings")
        produced = tuple(self.produced_artifact_ids)
        if any(not isinstance(item, str) or not item for item in produced):
            raise ValueError("produced_artifact_ids must contain non-empty strings")
        if len(set(produced)) != len(produced):
            raise ValueError("produced_artifact_ids must be unique")
        object.__setattr__(self, "produced_artifact_ids", produced)
        object.__setattr__(self, "output", _freeze_value(self.output))
        object.__setattr__(self, "error", _freeze_value(self.error))

    @property
    def succeeded(self) -> bool:
        return self.status == "success"

    def to_payload(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool": self.tool,
            "status": self.status,
            "output": _thaw_value(self.output),
            "error": _thaw_value(self.error),
            "phase_id": self.phase_id,
            "checkpoint_id": self.checkpoint_id,
            "produced_artifact_ids": list(self.produced_artifact_ids),
        }

    def logical_payload(self) -> dict[str, Any]:
        """Return the model-usable producer identity, never a tool transcript."""

        return {
            "step_id": self.step_id,
            "tool": self.tool,
            "phase_id": self.phase_id,
            "checkpoint_id": self.checkpoint_id,
            "produced_artifact_ids": list(self.produced_artifact_ids),
        }


def _canonicalize(value: Any) -> Any:
    if hasattr(value, "to_payload") and callable(value.to_payload):
        return _canonicalize(value.to_payload())
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (set, frozenset)):
        canonical = [_canonicalize(item) for item in value]
        return sorted(canonical, key=lambda item: json.dumps(item, sort_keys=True, default=str))
    if isinstance(value, bytes):
        return {"$blob_hex": value.hex()}
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported logical-state value {type(value).__name__}")


_NUMERIC_SUFFIX = re.compile(r"_(\d+)$")


class EnvironmentState:
    """Append-only episode facts plus branch-local active membership.

    ``sources`` is the immutable catalog.  A source becomes model-visible only
    after :meth:`discover_schema`; source names remain legal inputs to schema
    perception tools before discovery.  Derived relations are legal only while
    their handles occur in ``active_artifact_ids``.
    """

    def __init__(
        self,
        sources: Mapping[str, SourceRelation] | Iterable[SourceRelation] | None = None,
        *,
        source_relations: Mapping[str, SourceRelation] | Iterable[SourceRelation] | None = None,
        current_targets: Iterable[str] = (),
        phase_id: str = "phase_000",
        checkpoint_id: str = "root",
    ) -> None:
        if sources is not None and source_relations is not None:
            raise ValueError("provide sources or source_relations, not both")
        initial_sources = source_relations if source_relations is not None else sources

        self.sources: dict[str, SourceRelation] = {}
        self.artifacts: dict[str, RelationArtifact] = {}
        self.observations: dict[str, Observation] = {}
        self.steps: dict[str, StepRecord] = {}

        self.active_artifact_ids: set[str] = set()
        self.active_observation_ids: set[str] = set()
        self.usable_step_ids: set[str] = set()
        self.discovered_schema_ids: set[str] = set()

        self.current_targets: tuple[str, ...] = tuple(current_targets)
        self.phase_id = phase_id
        self.checkpoint_id = checkpoint_id
        self.last_error: Any = None

        # Counters are episode-audit mechanics.  They intentionally survive
        # restore and intentionally do not participate in logical_hash().
        self._step_counter = 0
        self._artifact_counter = 0
        self._observation_counter = 0

        if initial_sources:
            iterable = initial_sources.values() if isinstance(initial_sources, Mapping) else initial_sources
            for source in iterable:
                self.add_source(source)

    @property
    def source_relations(self) -> dict[str, SourceRelation]:
        return self.sources

    @property
    def active_artifacts(self) -> tuple[RelationArtifact, ...]:
        return tuple(self.artifacts[item] for item in sorted(self.active_artifact_ids))

    @property
    def active_observations(self) -> tuple[Observation, ...]:
        return tuple(self.observations[item] for item in sorted(self.active_observation_ids))

    def add_source(self, source: SourceRelation, *, discovered: bool = False) -> SourceRelation:
        if not isinstance(source, SourceRelation):
            raise TypeError("source must be a SourceRelation")
        occupied = {
            sqlite_identifier_key(name) for name in (*self.sources, *self.artifacts)
        }
        if sqlite_identifier_key(source.name) in occupied:
            raise StateError(
                "duplicate_table",
                f"relation {source.name!r} is already registered",
                {"table": source.name},
            )
        self.sources[source.name] = source
        if discovered:
            self.discovered_schema_ids.add(source.name)
        return source

    register_source = add_source

    def discover_schema(self, table: str) -> SourceRelation:
        source = self.sources.get(table)
        if source is None:
            raise StateError(
                "unknown_table",
                f"unknown source table {table!r}",
                {"requested": table, "available_tables": sorted(self.sources)},
            )
        self.discovered_schema_ids.add(table)
        return source

    mark_schema_discovered = discover_schema

    def get_relation(self, table: str) -> Relation:
        """Resolve an exact source name or active derived handle."""

        if table in self.sources:
            return self.sources[table]
        artifact = self.artifacts.get(table)
        if artifact is not None:
            if table not in self.active_artifact_ids:
                raise StateError(
                    "inactive_handle",
                    f"artifact {table!r} belongs to an inactive branch",
                    {
                        "requested": table,
                        "available_handles": sorted(self.active_artifact_ids),
                    },
                )
            return artifact
        raise StateError(
            "unknown_table",
            f"unknown relation {table!r}",
            {
                "requested": table,
                "available_sources": sorted(self.sources),
                "available_handles": sorted(self.active_artifact_ids),
            },
        )

    resolve_relation = get_relation

    def add_artifact(self, artifact: RelationArtifact) -> RelationArtifact:
        if not isinstance(artifact, RelationArtifact):
            raise TypeError("artifact must be a RelationArtifact")
        occupied = {
            sqlite_identifier_key(name) for name in (*self.sources, *self.artifacts)
        }
        if sqlite_identifier_key(artifact.table) in occupied:
            raise StateError(
                "duplicate_table",
                f"relation {artifact.table!r} is already registered",
                {"table": artifact.table},
            )
        self.artifacts[artifact.table] = artifact
        self.active_artifact_ids.add(artifact.table)
        self._observe_counter(artifact.table, "artifact")
        return artifact

    register_artifact = add_artifact

    def add_observation(self, observation: Observation) -> Observation:
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        if observation.observation_id in self.observations:
            raise StateError(
                "duplicate_observation",
                f"observation {observation.observation_id!r} already exists",
                {"observation_id": observation.observation_id},
            )
        # Observations about an abandoned artifact cannot become active again by
        # accident; the caller must restore the checkpoint containing them.
        self.get_relation(observation.table)
        self.observations[observation.observation_id] = observation
        self.active_observation_ids.add(observation.observation_id)
        self._observe_counter(observation.observation_id, "observation")
        return observation

    register_observation = add_observation

    def add_step(self, step: StepRecord) -> StepRecord:
        if not isinstance(step, StepRecord):
            raise TypeError("step must be a StepRecord")
        if step.step_id in self.steps:
            raise StateError(
                "duplicate_step",
                f"step {step.step_id!r} already exists",
                {"step_id": step.step_id},
            )
        if step.produced_artifact_ids and not step.succeeded:
            raise StateError(
                "invalid_step",
                "a failed step cannot declare produced artifacts",
                {"step_id": step.step_id},
            )
        missing_artifacts = sorted(set(step.produced_artifact_ids).difference(self.artifacts))
        inactive_artifacts = sorted(
            set(step.produced_artifact_ids).difference(self.active_artifact_ids)
        )
        if missing_artifacts or inactive_artifacts:
            raise StateError(
                "invalid_step",
                "a producing step must reference current active artifacts",
                {
                    "step_id": step.step_id,
                    "missing_artifacts": missing_artifacts,
                    "inactive_artifacts": inactive_artifacts,
                },
            )
        self.steps[step.step_id] = step
        self._observe_counter(step.step_id, "step")
        if step.succeeded:
            if step.produced_artifact_ids:
                self.usable_step_ids.add(step.step_id)
            self.last_error = None
        else:
            # Failed calls stay in the append-only audit log but are neither
            # usable nor part of the logical environment hash.
            self.usable_step_ids.discard(step.step_id)
            self.last_error = _thaw_value(step.error if step.error is not None else step.output)
        return step

    register_step = add_step

    def record_step(
        self,
        tool: str,
        status: str,
        output: Any,
        error: Any = None,
        *,
        step_id: str | None = None,
    ) -> StepRecord:
        record = StepRecord(
            step_id=step_id or self.allocate_step_id(),
            tool=tool,
            status=status,
            output=output,
            error=error,
            phase_id=self.phase_id,
            checkpoint_id=self.checkpoint_id,
        )
        return self.add_step(record)

    def set_last_error(self, error: Any) -> None:
        self.last_error = deepcopy(error)

    def clear_last_error(self) -> None:
        self.last_error = None

    def allocate_step_id(self) -> str:
        self._step_counter += 1
        while f"step_{self._step_counter:03d}" in self.steps:
            self._step_counter += 1
        return f"step_{self._step_counter:03d}"

    next_step_id = allocate_step_id

    def allocate_observation_id(self) -> str:
        self._observation_counter += 1
        while f"observation_{self._observation_counter:03d}" in self.observations:
            self._observation_counter += 1
        return f"observation_{self._observation_counter:03d}"

    next_observation_id = allocate_observation_id

    def allocate_artifact_handle(self, kind: str = "relation") -> str:
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("artifact kind must be a non-empty string")
        prefix = re.sub(r"[^A-Za-z0-9_]", "_", kind.strip()).strip("_").lower() or "relation"
        self._artifact_counter += 1
        candidate = f"{prefix}_{self._artifact_counter:03d}"
        occupied = {
            sqlite_identifier_key(name) for name in (*self.sources, *self.artifacts)
        }
        while sqlite_identifier_key(candidate) in occupied:
            self._artifact_counter += 1
            candidate = f"{prefix}_{self._artifact_counter:03d}"
        return candidate

    next_artifact_handle = allocate_artifact_handle

    def _observe_counter(self, identifier: str, counter: str) -> None:
        match = _NUMERIC_SUFFIX.search(identifier)
        if not match:
            return
        value = int(match.group(1))
        attr = f"_{counter}_counter"
        setattr(self, attr, max(getattr(self, attr), value))

    def restore_membership(
        self,
        *,
        discovered_schema_ids: Iterable[str],
        active_artifact_ids: Iterable[str],
        active_observation_ids: Iterable[str],
        usable_step_ids: Iterable[str],
    ) -> None:
        discovered = set(discovered_schema_ids)
        artifacts = set(active_artifact_ids)
        observations = set(active_observation_ids)
        steps = set(usable_step_ids)
        missing_sources = discovered.difference(self.sources)
        missing_artifacts = artifacts.difference(self.artifacts)
        missing_observations = observations.difference(self.observations)
        missing_steps = steps.difference(self.steps)
        if missing_sources or missing_artifacts or missing_observations or missing_steps:
            raise StateError(
                "invalid_checkpoint",
                "checkpoint snapshot references unknown immutable records",
                {
                    "missing_sources": sorted(missing_sources),
                    "missing_artifacts": sorted(missing_artifacts),
                    "missing_observations": sorted(missing_observations),
                    "missing_steps": sorted(missing_steps),
                },
            )
        failed_steps = sorted(step_id for step_id in steps if not self.steps[step_id].succeeded)
        if failed_steps:
            raise StateError(
                "invalid_checkpoint",
                "checkpoint marks failed steps as usable",
                {"failed_steps": failed_steps},
            )
        non_producers = sorted(
            step_id for step_id in steps if not self.steps[step_id].produced_artifact_ids
        )
        inactive_producers = sorted(
            step_id
            for step_id in steps
            if not set(self.steps[step_id].produced_artifact_ids).issubset(artifacts)
        )
        if non_producers or inactive_producers:
            raise StateError(
                "invalid_checkpoint",
                "checkpoint usable steps must produce active artifacts",
                {
                    "non_producing_steps": non_producers,
                    "inactive_producing_steps": inactive_producers,
                },
            )
        self.discovered_schema_ids = discovered
        self.active_artifact_ids = artifacts
        self.active_observation_ids = observations
        self.usable_step_ids = steps
        self.last_error = None

    def advance_phase(
        self,
        *,
        checkpoint_id: str,
        current_targets: Iterable[str],
        phase_id: str,
    ) -> None:
        self.checkpoint_id = checkpoint_id
        self.current_targets = tuple(current_targets)
        self.phase_id = phase_id
        self.last_error = None

    def logical_payload(self) -> dict[str, Any]:
        """Return only branch-active database facts and successful producers.

        Counters, phase/checkpoint control metadata, ``last_error``, and failed
        audit steps are excluded.  Consequently a rejected call does not change
        this payload or its hash.
        """

        payload = {
            "discovered_sources": [
                self.sources[name].to_payload() for name in sorted(self.discovered_schema_ids)
            ],
            "active_artifacts": [
                self.artifacts[table].to_payload() for table in sorted(self.active_artifact_ids)
            ],
            "active_observations": [
                self.observations[item].to_payload()
                for item in sorted(self.active_observation_ids)
            ],
            "usable_steps": [
                self.steps[item].logical_payload() for item in sorted(self.usable_step_ids)
            ],
        }
        return _canonicalize(payload)

    def logical_hash(self) -> str:
        encoded = json.dumps(
            self.logical_payload(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def environment_state_hash(self) -> str:
        return self.logical_hash()

    state_hash = logical_hash


__all__ = [
    "EnvironmentState",
    "Observation",
    "StateError",
    "StepRecord",
]
