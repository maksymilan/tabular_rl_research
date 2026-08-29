#!/usr/bin/env python3
"""Replay the production SAAM mask over a frozen policy-episode pool.

The command deliberately deserializes only scored policy episodes and never
consults environment metadata, gold SQL, or answer rows.  It uses the exact pure
function imported by the trainer, then verifies that every surviving policy
coefficient equals its vanilla value under the original normalization counts.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from frameworks.trl.fixed_rollout_pool import deserialize_episode  # noqa: E402
from frameworks.trl.state_action_ambiguity import (  # noqa: E402
    SCHEMA_VERSION as SAAM_SCHEMA_VERSION,
    _decision_identities,
    apply_state_action_ambiguity_mask,
)
from frameworks.trl.transition_batch import (  # noqa: E402
    build_transition_updates,
    policy_reduction_advantages,
)


SCHEMA_VERSION = "saam-training-mask-offline-audit-v1"


def _rows(path: Path) -> Iterable[dict[str, Any]]:
    source_handle = (
        gzip.open(path, "rt", encoding="utf-8")
        if path.suffix == ".gz"
        else path.open("r", encoding="utf-8")
    )
    with source_handle as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            yield row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_pool(path: Path, *, policy_reduction: str) -> dict[str, Any]:
    grouped = defaultdict(list)
    for row in _rows(path):
        episode = deserialize_episode(row)
        grouped[int(episode.sample.audit_record["example_index"])].append(episode)
    if not grouped:
        raise ValueError("rollout pool is empty")

    totals = defaultdict(float)
    correct_trajectories: set[str] = set()
    incorrect_trajectories: set[str] = set()
    trajectories_with_surviving_signal: set[str] = set()
    tasks_with_mask: set[int] = set()
    group_sizes: set[int] = set()
    eligible_by_tool: Counter[str] = Counter()
    masked_by_tool: Counter[str] = Counter()
    eligible_by_depth: Counter[int] = Counter()
    masked_by_depth: Counter[int] = Counter()
    removed_coefficient_mass_by_outcome: Counter[str] = Counter()
    surviving_coefficient_mass_by_outcome: Counter[str] = Counter()
    ambiguous_positive_mass = 0.0
    ambiguous_negative_mass = 0.0
    two_sided_conflict_mass = 0.0
    ambiguous_majority_residual_mass = 0.0
    conflict_mass_by_tool: Counter[str] = Counter()
    for example_index, episodes in sorted(grouped.items()):
        group_sizes.add(len(episodes))
        updates = build_transition_updates(
            episodes,
            reward_mode="result-only",
            train_turns="all",
        )
        original_transition_count = len(updates)
        original_trajectory_count = len(
            {update.trajectory_id for update in updates}
        )
        vanilla_coefficients = policy_reduction_advantages(
            updates,
            reduction=policy_reduction,
            normalization_transition_count=original_transition_count,
            normalization_trajectory_count=original_trajectory_count,
        )
        masked, receipt = apply_state_action_ambiguity_mask(
            episodes,
            updates,
            credit_assignment="saam-strict",
        )
        masked_coefficients = policy_reduction_advantages(
            masked,
            reduction=policy_reduction,
            normalization_transition_count=original_transition_count,
            normalization_trajectory_count=original_trajectory_count,
        )
        decision_identities = _decision_identities(episodes)
        decision_groups = defaultdict(list)
        for identity in decision_identities.values():
            if identity.action_signature is None:
                continue
            decision_groups[
                (
                    identity.example_index,
                    identity.state_signature,
                    identity.action_signature,
                )
            ].append(identity)
        vanilla_by_key = {
            (update.trajectory_id, update.turn_index): coefficient
            for update, coefficient in zip(updates, vanilla_coefficients, strict=True)
        }
        for (_, _, action_signature), identities in decision_groups.items():
            if {identity.correct for identity in identities} != {False, True}:
                continue
            values = [
                vanilla_by_key[(identity.trajectory_id, identity.turn_index)]
                for identity in identities
                if (identity.trajectory_id, identity.turn_index) in vanilla_by_key
            ]
            positive = sum(value for value in values if value > 0.0)
            negative = sum(-value for value in values if value < 0.0)
            if positive == 0.0 or negative == 0.0:
                continue
            conflict = 2.0 * min(positive, negative)
            residual = abs(positive - negative)
            ambiguous_positive_mass += positive
            ambiguous_negative_mass += negative
            two_sided_conflict_mass += conflict
            ambiguous_majority_residual_mass += residual
            tool = json.loads(action_signature)["tool"]
            conflict_mass_by_tool[tool] += conflict
        episodes_by_id = {
            str(episode.sample.audit_record["trajectory_id"]): episode
            for episode in episodes
            if episode.sample.process_update
        }
        for original_update, update, before, after in zip(
            updates,
            masked,
            vanilla_coefficients,
            masked_coefficients,
            strict=True,
        ):
            if update.advantage != 0.0 and before != after:
                raise RuntimeError(
                    f"example {example_index}: surviving coefficient was renormalized"
                )
            audited_turn = episodes_by_id[update.trajectory_id].sample.audit_record[
                "turns"
            ][update.turn_index]
            parsed = audited_turn.get("parsed") if isinstance(audited_turn, dict) else None
            tool = (
                parsed.get("tool")
                if isinstance(parsed, dict) and isinstance(parsed.get("tool"), str)
                else "__UNPARSED__"
            )
            eligible_by_tool[tool] += 1
            eligible_by_depth[update.turn_index] += 1
            outcome = "correct" if update.trajectory_correct else "incorrect"
            removed_coefficient_mass_by_outcome[outcome] += abs(before) - abs(after)
            surviving_coefficient_mass_by_outcome[outcome] += abs(after)
            if original_update.advantage != 0.0 and update.advantage == 0.0:
                masked_by_tool[tool] += 1
                masked_by_depth[update.turn_index] += 1
            target = (
                correct_trajectories
                if update.trajectory_correct
                else incorrect_trajectories
            )
            target.add(update.trajectory_id)
            if after != 0.0:
                trajectories_with_surviving_signal.add(update.trajectory_id)

        if receipt.newly_zeroed_transitions:
            tasks_with_mask.add(example_index)
        totals["eligible_episodes"] += receipt.eligible_episodes
        totals["eligible_transitions"] += receipt.eligible_transitions
        totals["matchable_transitions"] += receipt.matchable_transitions
        totals["unmatchable_transitions"] += receipt.unmatchable_transitions
        totals["repeated_state_action_groups"] += receipt.repeated_state_action_groups
        totals["ambiguous_state_action_groups"] += (
            receipt.ambiguous_state_action_groups
        )
        totals["ambiguous_transitions"] += receipt.ambiguous_transitions
        totals["newly_zeroed_transitions"] += receipt.newly_zeroed_transitions
        totals["newly_zeroed_response_tokens"] += (
            receipt.newly_zeroed_response_tokens
        )
        totals["newly_zeroed_initial_transitions"] += (
            receipt.newly_zeroed_initial_transitions
        )
        totals["newly_zeroed_noninitial_transitions"] += (
            receipt.newly_zeroed_noninitial_transitions
        )
        totals["fully_zeroed_trajectories"] += receipt.fully_zeroed_trajectories
        totals["vanilla_absolute_policy_coefficient_mass"] += sum(
            abs(value) for value in vanilla_coefficients
        )
        totals["saam_absolute_policy_coefficient_mass"] += sum(
            abs(value) for value in masked_coefficients
        )
        totals["eligible_response_tokens"] += sum(
            len(update.response_ids) for update in updates
        )

    integer_fields = {
        key: int(value)
        for key, value in totals.items()
        if not key.endswith("coefficient_mass")
    }
    vanilla_mass = totals["vanilla_absolute_policy_coefficient_mass"]
    saam_mass = totals["saam_absolute_policy_coefficient_mass"]
    transition_count = int(totals["eligible_transitions"])
    response_tokens = int(totals["eligible_response_tokens"])
    return {
        "schema_version": SCHEMA_VERSION,
        "saam_schema_version": SAAM_SCHEMA_VERSION,
        "input": str(path),
        "input_sha256": _sha256(path),
        "policy_reduction": policy_reduction,
        "state_identity": "prior parsed full action plus Harness structured outcome",
        "action_identity": (
            "parsed tool plus complete arguments; JSON object order ignored; "
            "describe_table.tables sorted"
        ),
        "reward_authority": "binary terminal correctness only",
        "gold_sql_read": False,
        "normalization": "pre-mask transition and trajectory counts; no renormalization",
        "tasks": len(grouped),
        "group_sizes": sorted(group_sizes),
        **integer_fields,
        "tasks_with_mask": len(tasks_with_mask),
        "correct_trajectories": len(correct_trajectories),
        "incorrect_trajectories": len(incorrect_trajectories),
        "correct_trajectories_with_surviving_signal": len(
            correct_trajectories & trajectories_with_surviving_signal
        ),
        "incorrect_trajectories_with_surviving_signal": len(
            incorrect_trajectories & trajectories_with_surviving_signal
        ),
        "eligible_transitions_by_tool": dict(sorted(eligible_by_tool.items())),
        "masked_transitions_by_tool": dict(sorted(masked_by_tool.items())),
        "mask_fraction_by_tool": {
            tool: masked_by_tool[tool] / count
            for tool, count in sorted(eligible_by_tool.items())
        },
        "eligible_transitions_by_depth": {
            str(depth): count for depth, count in sorted(eligible_by_depth.items())
        },
        "masked_transitions_by_depth": {
            str(depth): count for depth, count in sorted(masked_by_depth.items())
        },
        "removed_absolute_policy_coefficient_mass_by_outcome": dict(
            removed_coefficient_mass_by_outcome
        ),
        "surviving_absolute_policy_coefficient_mass_by_outcome": dict(
            surviving_coefficient_mass_by_outcome
        ),
        "vanilla_ambiguous_positive_policy_coefficient_mass": (
            ambiguous_positive_mass
        ),
        "vanilla_ambiguous_negative_policy_coefficient_mass": (
            ambiguous_negative_mass
        ),
        "vanilla_two_sided_conflict_mass": two_sided_conflict_mass,
        "vanilla_ambiguous_majority_residual_mass": (
            ambiguous_majority_residual_mass
        ),
        "vanilla_two_sided_conflict_fraction_of_ambiguous_mass": (
            two_sided_conflict_mass
            / (ambiguous_positive_mass + ambiguous_negative_mass)
            if ambiguous_positive_mass + ambiguous_negative_mass > 0.0
            else 0.0
        ),
        "vanilla_two_sided_conflict_mass_by_tool": dict(
            sorted(conflict_mass_by_tool.items())
        ),
        "saam_mixed_key_direct_terminal_credit_conflict_mass": 0.0,
        "masked_transition_fraction": (
            totals["newly_zeroed_transitions"] / transition_count
            if transition_count
            else 0.0
        ),
        "masked_response_token_fraction": (
            totals["newly_zeroed_response_tokens"] / response_tokens
            if response_tokens
            else 0.0
        ),
        "vanilla_absolute_policy_coefficient_mass": vanilla_mass,
        "saam_absolute_policy_coefficient_mass": saam_mass,
        "removed_absolute_policy_coefficient_fraction": (
            (vanilla_mass - saam_mass) / vanilla_mass if vanilla_mass else 0.0
        ),
        "surviving_coefficient_equality_check": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pool", type=Path)
    parser.add_argument(
        "--policy-reduction",
        choices=("transition_mean", "trajectory_mean", "trajectory_token_mean"),
        default="trajectory_token_mean",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_pool(args.pool, policy_reduction=args.policy_reduction)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
