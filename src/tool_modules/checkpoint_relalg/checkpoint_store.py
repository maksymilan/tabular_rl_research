"""Structural-sharing checkpoint tree for checkpoint-relalg-v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .environment_state import EnvironmentState, StateError


@dataclass(frozen=True, slots=True)
class CheckpointSnapshot:
    """A logical snapshot: membership ids plus a verification hash.

    Immutable source/artifact/observation/step records remain in
    :class:`EnvironmentState`; snapshots never duplicate table data.
    """

    discovered_schema_ids: frozenset[str]
    active_artifact_ids: frozenset[str]
    active_observation_ids: frozenset[str]
    usable_step_ids: frozenset[str]
    environment_state_hash: str

    @property
    def visible_source_schema_ids(self) -> frozenset[str]:
        return self.discovered_schema_ids

    @property
    def visible_observation_ids(self) -> frozenset[str]:
        return self.active_observation_ids

    def to_payload(self) -> dict[str, Any]:
        return {
            "discovered_schema_ids": sorted(self.discovered_schema_ids),
            "active_artifact_ids": sorted(self.active_artifact_ids),
            "active_observation_ids": sorted(self.active_observation_ids),
            "usable_step_ids": sorted(self.usable_step_ids),
            "environment_state_hash": self.environment_state_hash,
        }


@dataclass(slots=True)
class CheckpointNode:
    checkpoint_id: str
    parent_id: str | None
    status: str
    phase_targets: tuple[str, ...]
    progress_summary: tuple[str, ...]
    remaining_uncertainties: tuple[str, ...]
    next_targets: tuple[str, ...]
    snapshot: CheckpointSnapshot
    created_by: str = "commit_checkpoint"
    restore_reason: str | None = None
    restored_from: str | None = None
    abandoned_checkpoints: tuple[str, ...] = ()
    sequence: int = 0

    @property
    def parent_checkpoint_id(self) -> str | None:
        return self.parent_id

    def to_payload(self, *, include_snapshot: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "checkpoint_id": self.checkpoint_id,
            "parent_checkpoint_id": self.parent_id,
            "status": self.status,
            "phase_targets": list(self.phase_targets),
            "progress_summary": list(self.progress_summary),
            "remaining_uncertainties": list(self.remaining_uncertainties),
            "next_targets": list(self.next_targets),
            "created_by": self.created_by,
            "restore_reason": self.restore_reason,
            "restored_from": self.restored_from,
            "abandoned_checkpoints": list(self.abandoned_checkpoints),
            "sequence": self.sequence,
        }
        if include_snapshot:
            payload["snapshot"] = self.snapshot.to_payload()
        return payload


def _bounded_strings(
    value: Iterable[str],
    *,
    field_name: str,
    minimum: int,
    maximum: int,
    max_chars: int = 512,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise StateError(
            "invalid_checkpoint",
            f"{field_name} must be an array of strings",
            {"field": field_name},
            error_type="argument_validation_error",
        )
    try:
        result = tuple(value)
    except TypeError as exc:
        raise StateError(
            "invalid_checkpoint",
            f"{field_name} must be an array of strings",
            {"field": field_name},
            error_type="argument_validation_error",
        ) from exc
    if not minimum <= len(result) <= maximum:
        raise StateError(
            "invalid_checkpoint",
            f"{field_name} must contain {minimum} to {maximum} items",
            {"field": field_name, "item_count": len(result)},
            error_type="argument_validation_error",
        )
    for index, item in enumerate(result):
        if not isinstance(item, str) or not item.strip():
            raise StateError(
                "invalid_checkpoint",
                f"{field_name}[{index}] must be a non-empty string",
                {"field": field_name, "index": index},
                error_type="argument_validation_error",
            )
        if len(item) > max_chars:
            raise StateError(
                "invalid_checkpoint",
                f"{field_name}[{index}] exceeds {max_chars} characters",
                {"field": field_name, "index": index, "max_chars": max_chars},
                error_type="argument_validation_error",
            )
    return result


class CheckpointStore:
    """Own the checkpoint tree and apply exact branch membership restores."""

    def __init__(
        self,
        state: EnvironmentState,
        *,
        max_checkpoints: int = 32,
        max_restores: int = 8,
    ) -> None:
        if not isinstance(state, EnvironmentState):
            raise TypeError("state must be an EnvironmentState")
        if isinstance(max_checkpoints, bool) or not isinstance(max_checkpoints, int) or max_checkpoints < 0:
            raise ValueError("max_checkpoints must be a non-negative integer")
        if isinstance(max_restores, bool) or not isinstance(max_restores, int) or max_restores < 0:
            raise ValueError("max_restores must be a non-negative integer")
        self.state = state
        self.max_checkpoints = max_checkpoints
        self.max_restores = max_restores
        self._checkpoint_counter = 0
        self._phase_counter = self._phase_number(state.phase_id)
        self.restore_count = 0

        root_snapshot = self.capture_snapshot()
        self.nodes: dict[str, CheckpointNode] = {
            "root": CheckpointNode(
                checkpoint_id="root",
                parent_id=None,
                status="active_path",
                phase_targets=(),
                progress_summary=(),
                remaining_uncertainties=(),
                next_targets=tuple(state.current_targets),
                snapshot=root_snapshot,
                created_by="synthetic_root",
                sequence=0,
            )
        }
        # A store always starts from the synthetic root, irrespective of stale
        # caller control labels.
        self.state.checkpoint_id = "root"

    @staticmethod
    def _phase_number(phase_id: str) -> int:
        try:
            return int(phase_id.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            return 0

    @property
    def active_checkpoint_id(self) -> str:
        return self.state.checkpoint_id

    @property
    def checkpoint_count(self) -> int:
        return self._checkpoint_counter

    @property
    def snapshots(self) -> dict[str, CheckpointSnapshot]:
        return {checkpoint_id: node.snapshot for checkpoint_id, node in self.nodes.items()}

    @property
    def active_checkpoint_path(self) -> tuple[str, ...]:
        path: list[str] = []
        checkpoint_id: str | None = self.active_checkpoint_id
        seen: set[str] = set()
        while checkpoint_id is not None:
            if checkpoint_id in seen or checkpoint_id not in self.nodes:
                raise StateError(
                    "invalid_checkpoint",
                    "checkpoint parent graph is corrupt",
                    {"checkpoint_id": checkpoint_id},
                )
            seen.add(checkpoint_id)
            path.append(checkpoint_id)
            checkpoint_id = self.nodes[checkpoint_id].parent_id
        return tuple(reversed(path))

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointNode:
        node = self.nodes.get(checkpoint_id)
        if node is None:
            raise StateError(
                "invalid_checkpoint",
                f"unknown checkpoint {checkpoint_id!r}",
                {"requested": checkpoint_id, "available_checkpoints": list(self.nodes)},
            )
        return node

    get = get_checkpoint

    def capture_snapshot(self) -> CheckpointSnapshot:
        return CheckpointSnapshot(
            discovered_schema_ids=frozenset(self.state.discovered_schema_ids),
            active_artifact_ids=frozenset(self.state.active_artifact_ids),
            active_observation_ids=frozenset(self.state.active_observation_ids),
            usable_step_ids=frozenset(self.state.usable_step_ids),
            environment_state_hash=self.state.logical_hash(),
        )

    snapshot = capture_snapshot

    def commit(
        self,
        progress_summary: Iterable[str],
        remaining_uncertainties: Iterable[str],
        next_targets: Iterable[str],
    ) -> CheckpointNode:
        progress = _bounded_strings(
            progress_summary,
            field_name="progress_summary",
            minimum=1,
            maximum=5,
        )
        uncertainties = _bounded_strings(
            remaining_uncertainties,
            field_name="remaining_uncertainties",
            minimum=0,
            maximum=5,
        )
        targets = _bounded_strings(
            next_targets,
            field_name="next_targets",
            minimum=1,
            maximum=3,
        )
        self._ensure_checkpoint_budget()
        parent_id = self.active_checkpoint_id
        if parent_id not in self.nodes:
            raise StateError(
                "invalid_checkpoint",
                "active checkpoint does not exist",
                {"checkpoint_id": parent_id},
            )
        snapshot = self.capture_snapshot()
        checkpoint_id = self._allocate_checkpoint_id()
        node = CheckpointNode(
            checkpoint_id=checkpoint_id,
            parent_id=parent_id,
            status="active_path",
            phase_targets=tuple(self.state.current_targets),
            progress_summary=progress,
            remaining_uncertainties=uncertainties,
            next_targets=targets,
            snapshot=snapshot,
            created_by="commit_checkpoint",
            sequence=self._checkpoint_counter,
        )
        self.nodes[checkpoint_id] = node
        self._start_phase(checkpoint_id, targets)
        self._recompute_statuses()
        return node

    commit_checkpoint = commit

    def restore(
        self,
        checkpoint_id: str,
        reason: str,
        next_targets: Iterable[str],
    ) -> CheckpointNode:
        if not isinstance(checkpoint_id, str) or checkpoint_id not in self.nodes:
            raise StateError(
                "invalid_checkpoint",
                f"unknown checkpoint {checkpoint_id!r}",
                {"requested": checkpoint_id, "available_checkpoints": list(self.nodes)},
            )
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1500:
            raise StateError(
                "invalid_checkpoint",
                "reason must be non-empty and at most 1500 characters",
                {"field": "reason", "max_chars": 1500},
                error_type="argument_validation_error",
            )
        targets = _bounded_strings(
            next_targets,
            field_name="next_targets",
            minimum=1,
            maximum=3,
        )
        self._ensure_checkpoint_budget()
        if self.restore_count >= self.max_restores:
            raise StateError(
                "restore_limit_reached",
                "restore budget exhausted",
                {"max_restores": self.max_restores},
                error_type="resource_limit_error",
            )

        target = self.nodes[checkpoint_id]
        old_active = {
            item for item, node in self.nodes.items() if node.status == "active_path" and item != "root"
        }
        prior_membership = {
            "discovered_schema_ids": set(self.state.discovered_schema_ids),
            "active_artifact_ids": set(self.state.active_artifact_ids),
            "active_observation_ids": set(self.state.active_observation_ids),
            "usable_step_ids": set(self.state.usable_step_ids),
        }
        self._apply_snapshot(target.snapshot, rollback_membership=prior_membership)

        new_id = self._allocate_checkpoint_id()
        # Statuses are determined after the recovery node has been linked.
        node = CheckpointNode(
            checkpoint_id=new_id,
            parent_id=checkpoint_id,
            status="active_path",
            phase_targets=tuple(self.state.current_targets),
            progress_summary=(),
            remaining_uncertainties=(),
            next_targets=targets,
            snapshot=target.snapshot,
            created_by="restore_checkpoint",
            restore_reason=reason,
            restored_from=checkpoint_id,
            sequence=self._checkpoint_counter,
        )
        self.nodes[new_id] = node
        self.restore_count += 1
        self._start_phase(new_id, targets)
        self._recompute_statuses()
        newly_abandoned = tuple(
            item
            for item in sorted(old_active, key=lambda item: self.nodes[item].sequence)
            if self.nodes[item].status == "abandoned"
        )
        node.abandoned_checkpoints = newly_abandoned
        return node

    restore_checkpoint = restore

    def _apply_snapshot(
        self,
        snapshot: CheckpointSnapshot,
        *,
        rollback_membership: dict[str, set[str]],
    ) -> None:
        self.state.restore_membership(
            discovered_schema_ids=snapshot.discovered_schema_ids,
            active_artifact_ids=snapshot.active_artifact_ids,
            active_observation_ids=snapshot.active_observation_ids,
            usable_step_ids=snapshot.usable_step_ids,
        )
        actual_hash = self.state.logical_hash()
        if actual_hash != snapshot.environment_state_hash:
            self.state.restore_membership(**rollback_membership)
            raise StateError(
                "invalid_checkpoint",
                "checkpoint snapshot hash does not match immutable records",
                {
                    "expected_hash": snapshot.environment_state_hash,
                    "actual_hash": actual_hash,
                },
            )

    def _ensure_checkpoint_budget(self) -> None:
        if self._checkpoint_counter >= self.max_checkpoints:
            raise StateError(
                "checkpoint_limit_reached",
                "checkpoint budget exhausted",
                {"max_checkpoints": self.max_checkpoints},
                error_type="resource_limit_error",
            )

    def _allocate_checkpoint_id(self) -> str:
        self._checkpoint_counter += 1
        checkpoint_id = f"checkpoint_{self._checkpoint_counter:03d}"
        while checkpoint_id in self.nodes:
            self._checkpoint_counter += 1
            checkpoint_id = f"checkpoint_{self._checkpoint_counter:03d}"
        return checkpoint_id

    def _start_phase(self, checkpoint_id: str, targets: tuple[str, ...]) -> None:
        self._phase_counter += 1
        self.state.advance_phase(
            checkpoint_id=checkpoint_id,
            current_targets=targets,
            phase_id=f"phase_{self._phase_counter:03d}",
        )

    def _recompute_statuses(self) -> None:
        active = set(self.active_checkpoint_path)
        for checkpoint_id, node in self.nodes.items():
            node.status = "active_path" if checkpoint_id in active else "abandoned"

    def export_history(self, *, include_snapshots: bool = True) -> list[dict[str, Any]]:
        return [
            node.to_payload(include_snapshot=include_snapshots)
            for node in sorted(self.nodes.values(), key=lambda item: item.sequence)
        ]


__all__ = ["CheckpointNode", "CheckpointSnapshot", "CheckpointStore"]
