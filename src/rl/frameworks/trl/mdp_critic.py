"""Harness-conditioned SMDP transition extraction and critic feasibility audit.

This module consumes the flat Atomic v26 rollout log written by the current
trainer. It deliberately keeps only what the policy could observe and what the
Harness scored: exact model-visible state, typed action, terminal reward, and
episode outcome. Reference fields such as ``gold_sql`` are rejected rather than
silently propagated.

The output is an audit artifact, not a new actor trainer. The audit answers the
first question required by an IQL design: do repeated states contain action
variation and outcome variation, or is the proposed critic merely a
trajectory-success classifier?
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from rl.runtime.terminal_reward import terminal_result_reward


SCHEMA_VERSION = "atomic-v26-harness-smdp-transition-v1"
AUDIT_SCHEMA_VERSION = "atomic-v26-harness-smdp-iql-feasibility-audit-v1"
FORBIDDEN_KEYS = frozenset({
    "gold_sql",
    "gold_sample",
    "reference_answer",
    "reference_sql",
})
SAMPLE_INDEX_RE = re.compile(r"_sample_(\d+)$")


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _assert_no_forbidden_keys(value: Any, *, location: str) -> None:
    if isinstance(value, dict):
        found = FORBIDDEN_KEYS.intersection(value)
        if found:
            raise ValueError(
                f"{location} contains forbidden reference fields: {sorted(found)}"
            )
        for key, child in value.items():
            _assert_no_forbidden_keys(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_forbidden_keys(child, location=f"{location}[{index}]")


def _sample_index(trajectory_id: str) -> int:
    match = SAMPLE_INDEX_RE.search(trajectory_id)
    if match is None:
        raise ValueError(f"trajectory_id has no _sample_<index> suffix: {trajectory_id}")
    return int(match.group(1))


def _action_from_turn(turn: dict[str, Any], *, location: str) -> dict[str, Any]:
    parsed = turn.get("parsed")
    if not isinstance(parsed, dict):
        raise ValueError(f"{location} has no parsed action")
    tool = parsed.get("tool")
    arguments = parsed.get("arguments")
    if not isinstance(tool, str) or not tool:
        raise ValueError(f"{location} has no typed action tool")
    if not isinstance(arguments, dict):
        raise ValueError(f"{location} has no typed action arguments")
    action = {"tool": tool, "arguments": arguments}
    _assert_no_forbidden_keys(action, location=f"{location}.action")
    return action


def _model_visible_state(turn: dict[str, Any], *, location: str) -> Any:
    state = turn.get("model_input")
    if not isinstance(state, list) or not state:
        raise ValueError(f"{location} has no non-empty model_input")
    _assert_no_forbidden_keys(state, location=f"{location}.model_input")
    return state


def _terminal_reward(row: dict[str, Any]) -> tuple[float, str]:
    result = row.get("result_reward")
    if isinstance(result, dict):
        profile = result.get("profile")
        value = result.get("value")
        if profile != "four-level":
            raise ValueError(f"unsupported/non-registered result reward profile: {profile!r}")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value), "logged_four_level"
    # Old v26 logs may not have a scalar result_reward value. Recompute only
    # from Harness-owned fields, never from model-authored reasoning.
    errors = bool(row.get("errors") or row.get("error_events"))
    reward = terminal_result_reward(
        row.get("correct", False),
        profile="four-level",
        has_errors=errors,
    )
    return float(reward), "recomputed_four_level_from_harness_flags"


@dataclass(frozen=True)
class SMDPTransition:
    schema_version: str
    task_key: str
    example_index: int
    db_id: str
    trajectory_id: str
    episode_key: str
    sample_index: int
    turn_index: int
    policy_global_step: int | None
    policy_micro_step: int | None
    state: Any
    state_hash: str
    action: dict[str, Any]
    action_hash: str
    next_state: Any | None
    next_state_hash: str | None
    reward: float
    trajectory_terminal_reward: float
    done: bool
    gamma_duration: int
    trajectory_correct: bool
    trajectory_legal: bool
    trajectory_has_harness_error: bool
    result_reward_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_transitions(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[SMDPTransition], list[dict[str, Any]]]:
    """Build one semantic-step transition per valid rollout turn.

    A malformed turn excludes its whole trajectory and emits an issue. This
    avoids training on a partial episode whose terminal return is ambiguous.
    """
    transitions: list[SMDPTransition] = []
    issues: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        trajectory_id = row.get("trajectory_id")
        turns = row.get("turns")
        try:
            if not isinstance(trajectory_id, str) or not trajectory_id:
                raise ValueError("missing trajectory_id")
            if not isinstance(turns, list) or not turns:
                raise ValueError("missing non-empty turns")
            example_index = row.get("example_index")
            if type(example_index) is not int:
                raise ValueError("example_index must be an integer")
            sample_index = _sample_index(trajectory_id)
            reward, reward_source = _terminal_reward(row)
            db_id = row.get("db_id")
            if not isinstance(db_id, str) or not db_id:
                raise ValueError("missing db_id")
            task_key = f"example_index:{example_index}"
            global_step = row.get("policy_global_step")
            micro_step = row.get("policy_micro_step")
            if global_step is not None and type(global_step) is not int:
                raise ValueError("policy_global_step must be an integer when present")
            if micro_step is not None and type(micro_step) is not int:
                raise ValueError("policy_micro_step must be an integer when present")
            if global_step is None and micro_step is None:
                episode_key = trajectory_id
            else:
                episode_key = (
                    f"{trajectory_id}@global_step={global_step}"
                    f"@micro_step={micro_step}"
                )
            has_error = bool(row.get("errors") or row.get("error_events"))
            actions = [
                _action_from_turn(turn, location=f"row[{row_index}].turn[{index}]")
                for index, turn in enumerate(turns)
            ]
            states = [
                _model_visible_state(turn, location=f"row[{row_index}].turn[{index}]")
                for index, turn in enumerate(turns)
            ]
        except (TypeError, ValueError, KeyError) as exc:
            issues.append({
                "row_index": row_index,
                "trajectory_id": trajectory_id,
                "reason": str(exc),
            })
            continue

        # The remote diagnostic hosts include Python versions before the
        # ``zip(strict=...)`` keyword. Length equality is guaranteed by the two
        # comprehensions above, so the explicit compatibility form is safe.
        for turn_index, (state, action) in enumerate(zip(states, actions)):
            done = turn_index == len(turns) - 1
            next_state = None if done else states[turn_index + 1]
            transitions.append(SMDPTransition(
                schema_version=SCHEMA_VERSION,
                task_key=task_key,
                example_index=example_index,
                db_id=db_id,
                trajectory_id=trajectory_id,
                episode_key=episode_key,
                sample_index=sample_index,
                turn_index=turn_index,
                policy_global_step=global_step,
                policy_micro_step=micro_step,
                state=state,
                state_hash=sha256_json(state),
                action=action,
                action_hash=sha256_json(action),
                next_state=next_state,
                next_state_hash=None if next_state is None else sha256_json(next_state),
                reward=reward if done else 0.0,
                trajectory_terminal_reward=reward,
                done=done,
                gamma_duration=1,
                trajectory_correct=bool(row.get("correct", False)),
                trajectory_legal=bool(row.get("legal", False)),
                trajectory_has_harness_error=has_error,
                result_reward_source=reward_source,
            ))
    return transitions, issues


def audit_transitions(
    transitions: Iterable[SMDPTransition],
    issues: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Summarize whether offline action-value learning is identifiable enough."""
    rows = list(transitions)
    issue_rows = list(issues)
    by_state: dict[str, list[SMDPTransition]] = defaultdict(list)
    by_state_action: dict[tuple[str, str], list[SMDPTransition]] = defaultdict(list)
    by_trajectory: dict[str, list[SMDPTransition]] = defaultdict(list)
    for transition in rows:
        by_state[transition.state_hash].append(transition)
        by_state_action[(transition.state_hash, transition.action_hash)].append(transition)
        by_trajectory[transition.episode_key].append(transition)

    multi_action_states = [
        values for values in by_state.values()
        if len({value.action_hash for value in values}) > 1
    ]
    mixed_outcome_states = [
        values for values in multi_action_states
        if len({value.trajectory_correct for value in values}) > 1
    ]
    mixed_return_states = [
        values for values in multi_action_states
        if len({round(value.trajectory_terminal_reward, 8) for value in values}) > 1
    ]
    repeated_action_groups = [values for values in by_state_action.values() if len(values) > 1]
    action_outcome_inconsistent = [
        values for values in repeated_action_groups
        if len({value.trajectory_correct for value in values}) > 1
    ]
    terminal_rows = [value for value in rows if value.done]
    task_counts = Counter(value.task_key for value in rows)
    outcome_counts = Counter(
        "correct" if value.trajectory_correct else "incorrect"
        for value in terminal_rows
    )

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "gate": {
            "has_multi_action_state": bool(multi_action_states),
            "has_mixed_outcome_state": bool(mixed_outcome_states),
            "has_mixed_terminal_return_state": bool(mixed_return_states),
            "critic_is_not_proven_useful": not bool(mixed_outcome_states),
            "interpretation": (
                "Proceed to critic training only if repeated exact states include action and outcome "
                "variation; otherwise the critic audit is only a trajectory-success baseline."
            ),
        },
        "observed": {
            "transitions": len(rows),
            "terminal_transitions": len(terminal_rows),
            "tasks": len(task_counts),
            "trajectories": len(by_trajectory),
            "unique_states": len(by_state),
            "states_with_multiple_actions": len(multi_action_states),
            "states_with_mixed_outcomes": len(mixed_outcome_states),
            "states_with_mixed_terminal_returns": len(mixed_return_states),
            "repeated_state_action_groups": len(repeated_action_groups),
            "repeated_state_action_groups_with_mixed_outcomes": len(action_outcome_inconsistent),
            "trajectory_outcomes": dict(sorted(outcome_counts.items())),
            "issues": len(issue_rows),
        },
        "distribution": {
            "turns_per_trajectory": _distribution([len(value) for value in by_trajectory.values()]),
            "actions_per_state": _distribution([
                len({value.action_hash for value in values})
                for values in by_state.values()
            ]),
        },
    }


def _distribution(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": ordered[(len(ordered) - 1) // 2],
        "p90": ordered[min(len(ordered) - 1, int(math.ceil(0.9 * len(ordered)) - 1))],
        "max": ordered[-1],
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".next")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(stable_json(row) + "\n")
        target.flush()
    temporary.replace(path)
