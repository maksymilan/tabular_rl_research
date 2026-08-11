"""Structural-sharing checkpoint tree for checkpoint-relalg-v1."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable
import unicodedata

from .environment_state import EnvironmentState, StateError


MAX_CHECKPOINTS = 8
CHECKPOINT_GOAL_POLICY_VERSION = "active-path-distinct-normalized-goals-v1"
CHECKPOINT_COMMIT_ELIGIBILITY_NONE = "none"
CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1 = (
    "checkpoint-relalg-ordinal-milestone-progress-v1"
)
CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V1 = (
    "checkpoint-relalg-initial-target-ordinal-milestone-v1"
)
CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V2 = (
    "checkpoint-relalg-ordered-target-transition-v2"
)
CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V3 = (
    "checkpoint-relalg-perception-bootstrap-ordered-target-v3"
)
CHECKPOINT_COMMIT_ELIGIBILITY_POLICIES = (
    CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
    CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1,
    CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V1,
    CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V2,
    CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V3,
)
CHECKPOINT_BOOTSTRAP_PERCEPTION_TOOLS = frozenset(
    {"describe_table", "inspect_column", "read_rows"}
)
CHECKPOINT_MILESTONE_PRODUCER_TOOLS = frozenset(
    {"filter_rows", "join", "group_aggregate", "set_operation"}
)
FIRST_COMMIT_MIN_MILESTONE_PRODUCERS = 2
LATER_COMMIT_MIN_MILESTONE_PRODUCERS = 3
BOOTSTRAP_TARGET_MIN_TARGETS = 2
_GOAL_TRAILING_PUNCTUATION = " .!?;:\u3002\uff01\uff1f\uff1b\uff1a"


def normalize_checkpoint_commit_eligibility_policy(policy: str | None) -> str:
    value = CHECKPOINT_COMMIT_ELIGIBILITY_NONE if policy is None else str(policy).strip()
    if value not in CHECKPOINT_COMMIT_ELIGIBILITY_POLICIES:
        raise ValueError(
            "checkpoint_commit_eligibility_policy must be one of "
            f"{CHECKPOINT_COMMIT_ELIGIBILITY_POLICIES}"
        )
    return value


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


def _canonical_goal_set(targets: Iterable[str]) -> frozenset[str]:
    """Canonicalize model-authored phase goals for deterministic novelty checks.

    This deliberately recognizes only obvious textual equivalence.  The Harness
    never asks another model to judge goal semantics and never treats a
    paraphrase detector as factual authority.
    """

    normalized: list[str] = []
    for target in targets:
        text = unicodedata.normalize("NFKC", target)
        text = re.sub(r"\s+", " ", text).strip().casefold()
        text = text.rstrip(_GOAL_TRAILING_PUNCTUATION)
        if not text:
            raise StateError(
                "invalid_checkpoint_goal",
                "next_targets must contain a substantive phase goal",
                {"field": "next_targets"},
                error_type="argument_validation_error",
            )
        normalized.append(text)
    canonical = frozenset(normalized)
    if len(canonical) != len(normalized):
        raise StateError(
            "checkpoint_goal_not_distinct",
            "next_targets contains duplicate goals after normalization",
            {"field": "next_targets", "conflict_scope": "same_checkpoint"},
            error_type="argument_validation_error",
        )
    return canonical


class CheckpointStore:
    """Own the checkpoint tree and apply exact branch membership restores."""

    def __init__(
        self,
        state: EnvironmentState,
        *,
        max_checkpoints: int = MAX_CHECKPOINTS,
        max_restores: int = 8,
        checkpoint_commit_eligibility_policy: str = CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
    ) -> None:
        if not isinstance(state, EnvironmentState):
            raise TypeError("state must be an EnvironmentState")
        if (
            isinstance(max_checkpoints, bool)
            or not isinstance(max_checkpoints, int)
            or not 0 <= max_checkpoints <= MAX_CHECKPOINTS
        ):
            raise ValueError(
                f"max_checkpoints must be an integer from 0 to {MAX_CHECKPOINTS}"
            )
        if isinstance(max_restores, bool) or not isinstance(max_restores, int) or max_restores < 0:
            raise ValueError("max_restores must be a non-negative integer")
        self.state = state
        self.max_checkpoints = max_checkpoints
        self.max_restores = max_restores
        self.checkpoint_commit_eligibility_policy = (
            normalize_checkpoint_commit_eligibility_policy(
                checkpoint_commit_eligibility_policy
            )
        )
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
        self._ensure_distinct_checkpoint_goal(targets)
        created_by = self._ensure_checkpoint_phase_progress(targets)
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
            created_by=created_by,
            sequence=self._checkpoint_counter,
        )
        self.nodes[checkpoint_id] = node
        self._start_phase(checkpoint_id, targets)
        self._recompute_statuses()
        return node

    commit_checkpoint = commit

    def _ensure_distinct_checkpoint_goal(self, targets: tuple[str, ...]) -> None:
        proposed = _canonical_goal_set(targets)
        conflicts: list[str] = []
        current_targets = tuple(self.state.current_targets)
        if current_targets and proposed == _canonical_goal_set(current_targets):
            conflicts.append("current_phase")
        for checkpoint_id in self.active_checkpoint_path:
            node = self.nodes[checkpoint_id]
            if node.next_targets and proposed == _canonical_goal_set(node.next_targets):
                conflicts.append(checkpoint_id)
        if conflicts:
            raise StateError(
                "checkpoint_goal_not_distinct",
                (
                    "next_targets must define a new phase goal distinct from the "
                    "current phase and every checkpoint on the active path"
                ),
                {
                    "field": "next_targets",
                    "conflicts": sorted(set(conflicts)),
                },
                error_type="argument_validation_error",
            )

    def _ensure_checkpoint_phase_progress(self, targets: tuple[str, ...]) -> str:
        if (
            self.checkpoint_commit_eligibility_policy
            == CHECKPOINT_COMMIT_ELIGIBILITY_NONE
        ):
            return "commit_checkpoint"

        if (
            self.checkpoint_commit_eligibility_policy
            in {
                CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V1,
                CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V2,
                CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V3,
            }
            and self._is_bootstrap_eligible_phase()
        ):
            if len(targets) < BOOTSTRAP_TARGET_MIN_TARGETS:
                raise StateError(
                    "checkpoint_bootstrap_targets_insufficient",
                    "an initial target checkpoint must partition at least two targets",
                    {
                        "policy": self.checkpoint_commit_eligibility_policy,
                        "minimum_target_count": BOOTSTRAP_TARGET_MIN_TARGETS,
                        "target_count": len(targets),
                    },
                    error_type="state_validation_error",
                )
            return "bootstrap_checkpoint"

        if self.checkpoint_commit_eligibility_policy not in {
            CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1,
            CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V1,
            CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V2,
            CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V3,
        }:
            raise AssertionError(
                "unhandled checkpoint commit eligibility policy "
                f"{self.checkpoint_commit_eligibility_policy!r}"
            )

        progress = self.phase_progress()
        quota = progress["quota"]
        producer_count = progress["successful_milestone_producer_count"]
        new_artifact_count = progress["new_active_artifact_count"]
        if producer_count >= quota and new_artifact_count >= quota:
            self._ensure_ordered_target_progression(targets)
            return "commit_checkpoint"
        raise StateError(
            "checkpoint_phase_progress_insufficient",
            "the current phase has not reached the checkpoint progress quota",
            {
                "policy": self.checkpoint_commit_eligibility_policy,
                "quota": quota,
                "successful_commit_ordinal": progress["successful_commit_ordinal"],
                "successful_milestone_producer_count": producer_count,
                "new_active_artifact_count": new_artifact_count,
                "milestone_tools": sorted(CHECKPOINT_MILESTONE_PRODUCER_TOOLS),
                "observed_milestone_tools": progress["observed_milestone_tools"],
            },
            error_type="state_validation_error",
        )

    def phase_progress(self) -> dict[str, Any]:
        """Return deterministic, content-free progress for the active phase."""

        successful_commit_count = sum(
            node.created_by == "commit_checkpoint" for node in self.nodes.values()
        )
        quota = (
            FIRST_COMMIT_MIN_MILESTONE_PRODUCERS
            if successful_commit_count == 0
            else LATER_COMMIT_MIN_MILESTONE_PRODUCERS
        )
        phase_start = self.get_checkpoint(self.active_checkpoint_id).snapshot
        new_active_artifacts = set(self.state.active_artifact_ids).difference(
            phase_start.active_artifact_ids
        )
        qualifying_steps = []
        for step in self.state.steps.values():
            if (
                step.phase_id != self.state.phase_id
                or step.checkpoint_id != self.active_checkpoint_id
                or not step.succeeded
                or step.step_id not in self.state.usable_step_ids
                or step.tool not in CHECKPOINT_MILESTONE_PRODUCER_TOOLS
                or not set(step.produced_artifact_ids).intersection(new_active_artifacts)
            ):
                continue
            qualifying_steps.append(step)
        producer_count = len(qualifying_steps)
        new_artifact_count = len(new_active_artifacts)
        return {
            "quota": quota,
            "successful_commit_ordinal": successful_commit_count + 1,
            "successful_milestone_producer_count": producer_count,
            "new_active_artifact_count": new_artifact_count,
            "observed_milestone_tools": sorted(
                {step.tool for step in qualifying_steps}
            ),
            "eligible": producer_count >= quota and new_artifact_count >= quota,
        }

    def ordered_target_status(self) -> dict[str, Any] | None:
        """Expose model-authored target control state for initial-target-v2."""

        if (
            self.checkpoint_commit_eligibility_policy
            not in {
                CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V2,
                CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V3,
            }
            or not self._has_active_bootstrap_checkpoint()
            or not self.state.current_targets
        ):
            return None
        progress = self.phase_progress()
        targets = tuple(self.state.current_targets)
        remaining = targets[1:]
        transition_required = bool(remaining) and bool(progress["eligible"])
        return {
            "active_target": targets[0],
            "remaining_targets": list(remaining),
            "milestone_quota": progress["quota"],
            "successful_milestone_producer_count": progress[
                "successful_milestone_producer_count"
            ],
            "new_active_artifact_count": progress["new_active_artifact_count"],
            "checkpoint_eligible": bool(progress["eligible"]),
            "target_transition_required": transition_required,
        }

    def ensure_ordered_target_action_allowed(self, tool: str) -> None:
        """Require a checkpoint before crossing an eligible target boundary."""

        status = self.ordered_target_status()
        if not status or not status["target_transition_required"]:
            return
        if tool not in {*CHECKPOINT_MILESTONE_PRODUCER_TOOLS, "answer"}:
            return
        raise StateError(
            "checkpoint_target_transition_required",
            (
                "the active target reached its milestone quota; commit a checkpoint "
                "that advances to remaining targets before another milestone producer "
                "or answer"
            ),
            {
                "policy": self.checkpoint_commit_eligibility_policy,
                "blocked_tool": tool,
                "milestone_quota": status["milestone_quota"],
                "successful_milestone_producer_count": status[
                    "successful_milestone_producer_count"
                ],
                "remaining_target_count": len(status["remaining_targets"]),
            },
            error_type="state_validation_error",
        )

    def _ensure_ordered_target_progression(self, targets: tuple[str, ...]) -> None:
        if (
            self.checkpoint_commit_eligibility_policy
            not in {
                CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V2,
                CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V3,
            }
            or not self._has_active_bootstrap_checkpoint()
            or len(self.state.current_targets) <= 1
        ):
            return
        active = next(iter(_canonical_goal_set((self.state.current_targets[0],))))
        remaining = _canonical_goal_set(self.state.current_targets[1:])
        proposed = _canonical_goal_set(targets)
        if active in proposed:
            raise StateError(
                "checkpoint_active_target_not_completed",
                "next_targets must remove the completed active target",
                {"field": "next_targets"},
                error_type="argument_validation_error",
            )
        if proposed.isdisjoint(remaining):
            raise StateError(
                "checkpoint_remaining_target_lost",
                (
                    "next_targets must retain at least one exact remaining target "
                    "from the ordered target plan"
                ),
                {"field": "next_targets"},
                error_type="argument_validation_error",
            )

    def _has_active_bootstrap_checkpoint(self) -> bool:
        return any(
            self.nodes[checkpoint_id].created_by == "bootstrap_checkpoint"
            for checkpoint_id in self.active_checkpoint_path
        )

    def _is_bootstrap_eligible_phase(self) -> bool:
        base_eligible = (
            self.checkpoint_count == 0
            and self.active_checkpoint_id == "root"
            and self.state.phase_id == "phase_000"
            and not self.state.current_targets
            and not self.state.active_artifact_ids
            and not self.state.usable_step_ids
        )
        if not base_eligible:
            return False
        if (
            self.checkpoint_commit_eligibility_policy
            != CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V3
        ):
            return (
                not self.state.steps
                and not self.state.discovered_schema_ids
                and not self.state.active_observation_ids
            )
        return all(
            step.tool in CHECKPOINT_BOOTSTRAP_PERCEPTION_TOOLS
            for step in self.state.steps.values()
        )

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


__all__ = [
    "BOOTSTRAP_TARGET_MIN_TARGETS",
    "CHECKPOINT_BOOTSTRAP_PERCEPTION_TOOLS",
    "CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V1",
    "CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V2",
    "CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V3",
    "CHECKPOINT_COMMIT_ELIGIBILITY_NONE",
    "CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1",
    "CHECKPOINT_COMMIT_ELIGIBILITY_POLICIES",
    "CHECKPOINT_GOAL_POLICY_VERSION",
    "CHECKPOINT_MILESTONE_PRODUCER_TOOLS",
    "FIRST_COMMIT_MIN_MILESTONE_PRODUCERS",
    "LATER_COMMIT_MIN_MILESTONE_PRODUCERS",
    "MAX_CHECKPOINTS",
    "CheckpointNode",
    "CheckpointSnapshot",
    "CheckpointStore",
    "normalize_checkpoint_commit_eligibility_policy",
]
