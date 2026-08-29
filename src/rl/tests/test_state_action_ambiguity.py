from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from frameworks.trl.state_action_ambiguity import (  # noqa: E402
    apply_asymmetric_error_credit,
    apply_state_action_ambiguity_mask,
)
from frameworks.trl.transition_batch import (  # noqa: E402
    PolicyEpisode,
    PolicyTurn,
    build_transition_updates,
    policy_reduction_advantages,
)


def _policy_turn(seed: int, response_length: int = 2) -> PolicyTurn:
    return PolicyTurn(
        prompt_ids=(seed, seed + 1),
        response_ids=tuple(range(seed + 2, seed + 2 + response_length)),
        sampling_logprobs=tuple(-0.1 for _ in range(response_length)),
    )


def _audited_turn(
    turn_index: int,
    tool: str | None,
    arguments: dict | None = None,
    *,
    output=None,
) -> dict:
    turn = {"turn_index": turn_index}
    if tool is not None:
        turn["parsed"] = {
            "think": "model text is deliberately irrelevant",
            "tool": tool,
            "arguments": arguments or {},
        }
    else:
        turn["execution_error_type"] = "carrier_error"
        turn["execution_error"] = "ProtocolError: malformed"
        turn["error_event"] = {
            "error_type": "carrier_error",
            "error_code": "malformed",
            "message": "ProtocolError: malformed",
        }
    if output is not None:
        turn["tool_output"] = output
    return turn


def _episode(
    trajectory_id: str,
    *,
    correct: bool,
    turns: list[dict],
    example_index: int = 7,
    process_update: bool = True,
    response_lengths: list[int] | None = None,
) -> PolicyEpisode:
    response_lengths = response_lengths or [2] * len(turns)
    policy_turns = [
        _policy_turn(index * 10 + 1, response_length)
        for index, response_length in enumerate(response_lengths)
    ]
    sample = SimpleNamespace(
        reward=1.0 if correct else 0.0,
        correct=correct,
        process_update=process_update,
        step_rewards=None,
        turns=[
            (list(turn.prompt_ids), list(turn.response_ids))
            for turn in policy_turns
        ],
        audit_record={
            "trajectory_id": trajectory_id,
            "example_index": example_index,
            "turns": turns,
            # If implementation accidentally consults this, the tests should not
            # need it and production SAAM would violate its reward contract.
            "gold_sql": "MUST NOT BE READ",
        },
    )
    return PolicyEpisode(sample=sample, policy_turns=policy_turns)


class StateActionAmbiguityTest(unittest.TestCase):
    def test_asymmetric_credit_keeps_correct_shared_and_suppresses_wrong_shared(self):
        common = _audited_turn(
            0,
            "describe_table",
            {"tables": ["orders"]},
            output={"schemas": {"orders": ["id", "amount"]}},
        )
        episodes = [
            _episode(
                "wrong",
                correct=False,
                turns=[common, _audited_turn(1, "project", {"columns": ["id"]})],
            ),
            _episode(
                "correct",
                correct=True,
                turns=[common, _audited_turn(1, "project", {"columns": ["amount"]})],
            ),
        ]
        updates = build_transition_updates(episodes, reward_mode="result-only")
        credited, audit = apply_asymmetric_error_credit(
            episodes, updates, error_penalty=1.0
        )
        by_key = {
            (update.trajectory_id, update.turn_index): update.advantage
            for update in credited
        }
        self.assertEqual(by_key[("wrong", 0)], 0.0)
        self.assertGreater(by_key[("correct", 0)], 0.0)
        self.assertEqual(audit.shared_success_correct_kept, 1)
        self.assertEqual(audit.shared_success_wrong_suppressed, 1)
        self.assertEqual(audit.newly_zeroed_transitions, 1)

    def test_asymmetric_error_is_negative_even_on_correct_terminal_trajectory(self):
        error = _audited_turn(0, None)
        episodes = [
            _episode("wrong", correct=False, turns=[error]),
            _episode("correct", correct=True, turns=[error]),
        ]
        updates = build_transition_updates(episodes, reward_mode="result-only")
        credited, audit = apply_asymmetric_error_credit(
            episodes, updates, error_penalty=1.0
        )
        by_key = {
            (update.trajectory_id, update.turn_index): update.advantage
            for update in credited
        }
        self.assertLess(by_key[("correct", 0)], 0.0)
        self.assertLess(by_key[("wrong", 0)], 0.0)
        self.assertEqual(audit.deterministic_error_transitions, 2)
        self.assertEqual(audit.correct_error_positive_flips, 1)
        self.assertGreater(audit.correct_error_positive_mass_flipped, 0.0)

    def test_common_action_is_zeroed_but_first_divergent_actions_keep_grpo_credit(self):
        common = _audited_turn(
            0,
            "describe_table",
            {"tables": ["orders"]},
            output={"schemas": {"orders": ["id", "amount"]}},
        )
        episodes = [
            _episode(
                "wrong",
                correct=False,
                turns=[
                    common,
                    _audited_turn(1, "project", {"table": "orders", "columns": ["id"]}),
                ],
            ),
            _episode(
                "correct",
                correct=True,
                turns=[
                    common,
                    _audited_turn(
                        1,
                        "project",
                        {"table": "orders", "columns": ["amount"]},
                    ),
                ],
            ),
        ]
        updates = build_transition_updates(episodes, reward_mode="result-only")
        masked, audit = apply_state_action_ambiguity_mask(
            episodes,
            updates,
            credit_assignment="saam-strict",
        )
        by_key = {
            (update.trajectory_id, update.turn_index): update.advantage
            for update in masked
        }
        self.assertEqual(by_key[("wrong", 0)], 0.0)
        self.assertEqual(by_key[("correct", 0)], 0.0)
        self.assertLess(by_key[("wrong", 1)], 0.0)
        self.assertGreater(by_key[("correct", 1)], 0.0)
        self.assertEqual(audit.ambiguous_state_action_groups, 1)
        self.assertEqual(audit.newly_zeroed_transitions, 2)
        self.assertEqual(audit.newly_zeroed_initial_transitions, 2)
        self.assertEqual(audit.fully_zeroed_trajectories, 0)

    def test_full_action_arguments_are_part_of_the_match(self):
        episodes = [
            _episode(
                "wrong",
                correct=False,
                turns=[_audited_turn(0, "describe_table", {"tables": ["orders"]})],
            ),
            _episode(
                "correct",
                correct=True,
                turns=[_audited_turn(0, "describe_table", {"tables": ["users"]})],
            ),
        ]
        updates = build_transition_updates(episodes, reward_mode="result-only")
        masked, audit = apply_state_action_ambiguity_mask(
            episodes, updates, credit_assignment="saam-strict"
        )
        self.assertEqual(
            [update.advantage for update in masked],
            [update.advantage for update in updates],
        )
        self.assertEqual(audit.ambiguous_state_action_groups, 0)

    def test_describe_table_order_is_unordered_only_for_current_action(self):
        episodes = [
            _episode(
                "wrong",
                correct=False,
                turns=[
                    _audited_turn(
                        0,
                        "describe_table",
                        {"tables": ["orders", "users"]},
                        output={"ok": True},
                    ),
                    _audited_turn(1, "plan", {"steps": ["join"]}),
                ],
            ),
            _episode(
                "correct",
                correct=True,
                turns=[
                    _audited_turn(
                        0,
                        "describe_table",
                        {"tables": ["users", "orders"]},
                        output={"ok": True},
                    ),
                    _audited_turn(1, "plan", {"steps": ["join"]}),
                ],
            ),
        ]
        updates = build_transition_updates(episodes, reward_mode="result-only")
        masked, audit = apply_state_action_ambiguity_mask(
            episodes, updates, credit_assignment="saam-strict"
        )
        zeros = [
            (update.trajectory_id, update.turn_index)
            for update in masked
            if update.advantage == 0.0
        ]
        self.assertEqual(zeros, [("wrong", 0), ("correct", 0)])
        self.assertEqual(audit.newly_zeroed_noninitial_transitions, 0)

    def test_same_state_action_never_matches_across_questions(self):
        action = _audited_turn(0, "describe_table", {"tables": ["orders"]})
        episodes = [
            _episode("wrong", correct=False, turns=[action], example_index=7),
            _episode("correct", correct=True, turns=[action], example_index=8),
        ]
        updates = build_transition_updates(episodes, reward_mode="result-only")
        # Each question is homogeneous K=1, so this also checks that SAAM does
        # not manufacture signal or cross-contaminate groups.
        masked, audit = apply_state_action_ambiguity_mask(
            episodes, updates, credit_assignment="saam-strict"
        )
        self.assertEqual([update.advantage for update in masked], [0.0, 0.0])
        self.assertEqual(audit.ambiguous_state_action_groups, 0)

    def test_malformed_action_and_descendants_fail_closed_without_masking(self):
        episodes = [
            _episode(
                "wrong",
                correct=False,
                turns=[
                    _audited_turn(0, None),
                    _audited_turn(1, "describe_table", {"tables": ["orders"]}),
                ],
            ),
            _episode(
                "correct",
                correct=True,
                turns=[
                    _audited_turn(0, None),
                    _audited_turn(1, "describe_table", {"tables": ["orders"]}),
                ],
            ),
        ]
        updates = build_transition_updates(episodes, reward_mode="result-only")
        masked, audit = apply_state_action_ambiguity_mask(
            episodes, updates, credit_assignment="saam-strict"
        )
        self.assertEqual(
            [update.advantage for update in masked],
            [update.advantage for update in updates],
        )
        self.assertEqual(audit.matchable_transitions, 0)
        self.assertEqual(audit.unmatchable_transitions, 4)

    def test_mask_does_not_renormalize_surviving_trajectory_mean_coefficients(self):
        common = _audited_turn(
            0,
            "describe_table",
            {"tables": ["orders"]},
            output={"ok": True},
        )
        episodes = [
            _episode(
                "wrong",
                correct=False,
                turns=[common, _audited_turn(1, "project", {"columns": ["id"]})],
                response_lengths=[2, 6],
            ),
            _episode(
                "correct",
                correct=True,
                turns=[common, _audited_turn(1, "project", {"columns": ["amount"]})],
                response_lengths=[3, 5],
            ),
        ]
        updates = build_transition_updates(episodes, reward_mode="result-only")
        original_count = len(updates)
        original_trajectory_count = len({update.trajectory_id for update in updates})
        masked, _ = apply_state_action_ambiguity_mask(
            episodes, updates, credit_assignment="saam-strict"
        )
        coefficients = policy_reduction_advantages(
            masked,
            reduction="trajectory_mean",
            normalization_transition_count=original_count,
            normalization_trajectory_count=original_trajectory_count,
        )
        # The two surviving turn-1 coefficients retain the original 1/2 turn
        # weight times T/N=2 scale, i.e. exactly their vanilla advantages.
        self.assertEqual(coefficients[0], 0.0)
        self.assertEqual(coefficients[2], 0.0)
        self.assertTrue(math.isclose(coefficients[1], updates[1].advantage))
        self.assertTrue(math.isclose(coefficients[3], updates[3].advantage))

    def test_trajectory_mode_is_an_exact_no_op(self):
        action = _audited_turn(0, "describe_table", {"tables": ["orders"]})
        episodes = [
            _episode("wrong", correct=False, turns=[action]),
            _episode("correct", correct=True, turns=[action]),
        ]
        updates = build_transition_updates(episodes, reward_mode="result-only")
        retained, audit = apply_state_action_ambiguity_mask(
            episodes, updates, credit_assignment="trajectory"
        )
        self.assertEqual(retained, updates)
        self.assertEqual(audit.newly_zeroed_transitions, 0)

    def test_preexisting_homogeneous_zero_group_is_not_attributed_to_saam(self):
        action = _audited_turn(0, "describe_table", {"tables": ["orders"]})
        episodes = [
            _episode("correct_a", correct=True, turns=[action]),
            _episode("correct_b", correct=True, turns=[action]),
        ]
        updates = build_transition_updates(episodes, reward_mode="result-only")
        masked, audit = apply_state_action_ambiguity_mask(
            episodes, updates, credit_assignment="saam-strict"
        )
        self.assertEqual([update.advantage for update in masked], [0.0, 0.0])
        self.assertEqual(audit.newly_zeroed_transitions, 0)
        self.assertEqual(audit.fully_zeroed_trajectories, 0)


if __name__ == "__main__":
    unittest.main()
