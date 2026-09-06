from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError:  # The lightweight local audit environment has no Torch.
    torch = None


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from rl.frameworks.trl.transition_batch import (  # noqa: E402
    PolicyEpisode,
    PolicyTurn,
    build_transition_microbatch_ranges,
    build_transition_updates,
    class_conditional_routing_advantages,
    correctness_primary_clean_secondary_advantages,
    policy_reduction_advantages,
    retain_policy_contributing_updates,
    smc_mode_concentration_advantages,
    standardized_group_advantages,
)


def _turn(seed: int) -> PolicyTurn:
    return PolicyTurn(
        prompt_ids=(seed, seed + 1),
        response_ids=(seed + 2, seed + 3),
        sampling_logprobs=(-0.1, -0.2),
    )


def _sized_turn(seed: int, response_length: int) -> PolicyTurn:
    response_ids = tuple(range(seed + 2, seed + 2 + response_length))
    return PolicyTurn(
        prompt_ids=(seed, seed + 1),
        response_ids=response_ids,
        sampling_logprobs=tuple(-0.1 for _ in response_ids),
    )


def _episode(
    trajectory_id: str,
    reward: float,
    step_rewards: list[float],
    *,
    process_update: bool = True,
    example_index: int = 1,
) -> PolicyEpisode:
    turns = [_turn(index * 10 + 1) for index in range(len(step_rewards))]
    sample = SimpleNamespace(
        reward=reward,
        correct=bool(reward > 0),
        process_update=process_update,
        step_rewards=step_rewards,
        turns=[
            (list(turn.prompt_ids), list(turn.response_ids))
            for turn in turns
        ],
        audit_record={
            "trajectory_id": trajectory_id,
            "example_index": example_index,
            "process_reward": {
                "steps": [
                    {
                        "features": {"legal_success": value >= 0.0},
                        "p_local": max(0.0, -value),
                    }
                    for value in step_rewards
                ]
            },
        },
    )
    return PolicyEpisode(sample=sample, policy_turns=turns)


class TransitionBatchTest(unittest.TestCase):
    def test_smc_selects_highest_probability_correct_and_wrong_modes(self):
        def episode(trajectory_id, correct, logprob):
            turns = [
                PolicyTurn(
                    prompt_ids=(1, 2),
                    response_ids=(3, 4),
                    sampling_logprobs=(logprob, logprob),
                )
            ]
            sample = SimpleNamespace(
                reward=1.0 if correct else -1.0,
                correct=correct,
                process_update=True,
                step_rewards=None,
                turns=[([1, 2], [3, 4])],
                audit_record={
                    "trajectory_id": trajectory_id,
                    "example_index": 11,
                },
            )
            return PolicyEpisode(sample=sample, policy_turns=turns)

        episodes = [
            episode("correct-low", True, -1.2),
            episode("correct-high", True, -0.4),
            episode("wrong-low", False, -0.9),
            episode("wrong-high", False, -0.3),
        ]
        self.assertEqual(
            smc_mode_concentration_advantages(episodes),
            [0.0, 1.0, 0.0, -1.0],
        )

    def test_smc_is_zero_for_homogeneous_groups(self):
        def episode(trajectory_id, correct):
            turn = _turn(10 + len(trajectory_id))
            sample = SimpleNamespace(
                reward=1.0 if correct else -1.0,
                correct=correct,
                process_update=True,
                step_rewards=None,
                turns=[(list(turn.prompt_ids), list(turn.response_ids))],
                audit_record={
                    "trajectory_id": trajectory_id,
                    "example_index": 12,
                },
            )
            return PolicyEpisode(sample=sample, policy_turns=[turn])

        self.assertEqual(
            smc_mode_concentration_advantages(
                [episode("a", True), episode("b", True)]
            ),
            [0.0, 0.0],
        )

    def test_smc_requires_finite_sampled_logprobs_in_mixed_group(self):
        good = _episode("good", 1.0, [1.0])
        bad = _episode("bad", 0.0, [0.0])
        bad.policy_turns[0] = PolicyTurn(
            prompt_ids=bad.policy_turns[0].prompt_ids,
            response_ids=bad.policy_turns[0].response_ids,
            sampling_logprobs=(float("nan"), float("nan")),
        )
        with self.assertRaises(ValueError):
            smc_mode_concentration_advantages([good, bad])

    def test_dynamic_microbatch_budget_uses_padded_prompt_completion_cost(self):
        ranges = build_transition_microbatch_ranges(
            [100, 110, 500, 510, 900],
            [50, 60, 100, 110, 120],
            max_rows=8,
            token_budget=1_500,
        )
        # The first two rows cost 2 * (110 + 60) = 340. Adding row three
        # costs 3 * (500 + 100) = 1,800, so it starts a new micro-batch.
        self.assertEqual(ranges, [(0, 2), (2, 4), (4, 5)])

    def test_dynamic_microbatch_respects_row_cap_and_keeps_oversized_row(self):
        ranges = build_transition_microbatch_ranges(
            [100] * 5 + [2_000],
            [50] * 5 + [2_000],
            max_rows=2,
            token_budget=100,
        )
        # Every normal pair exceeds the deliberately tiny token budget, while
        # the final 4k row is still admitted alone rather than being dropped.
        self.assertEqual(ranges, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])

    def test_zero_token_budget_preserves_fixed_row_ranges(self):
        self.assertEqual(
            build_transition_microbatch_ranges(
                [10] * 5,
                [4] * 5,
                max_rows=2,
                token_budget=0,
            ),
            [(0, 2), (2, 4), (4, 5)],
        )

    def test_decimal_homogeneous_group_has_strict_zero_advantage(self):
        self.assertEqual(
            standardized_group_advantages([0.2] * 8, [True] * 8),
            [0.0] * 8,
        )

    def test_ladder_value_equal_to_group_mean_has_strict_zero_advantage(self):
        rewards = [0.0, 0.0, 0.2, 0.2, 0.0, 1.0, 0.0, 0.2]
        advantages = standardized_group_advantages(rewards, [True] * 8)
        self.assertEqual([advantages[index] for index in (2, 3, 7)], [0.0] * 3)
        self.assertLess(advantages[0], 0.0)
        self.assertGreater(advantages[5], 0.0)

    @unittest.skipIf(torch is None, "Torch is available in the remote TRL runtime")
    def test_all_k8_execution_ladder_compositions_match_torch_float32(self):
        # Exhaust all 45 count compositions of eight rewards drawn from the
        # TRUST-SQL execution ladder.  The trainer constructs coefficients in
        # Python and then materializes them as float32 tensors, so compare at
        # that exact boundary against the author-style Torch computation.
        for zero_count in range(9):
            for legal_wrong_count in range(9 - zero_count):
                correct_count = 8 - zero_count - legal_wrong_count
                rewards = (
                    [0.0] * zero_count
                    + [0.2] * legal_wrong_count
                    + [1.0] * correct_count
                )
                actual = torch.tensor(
                    standardized_group_advantages(rewards, [True] * 8),
                    dtype=torch.float32,
                )
                if len(set(rewards)) == 1:
                    # GRPO's mathematical relative signal is exactly zero.  Raw
                    # Torch reduction of eight 0.2 values has a known numerical
                    # residual, which the trainer deliberately removes.
                    torch.testing.assert_close(
                        actual, torch.zeros_like(actual), rtol=0.0, atol=0.0
                    )
                    continue
                reference_rewards = torch.tensor(rewards, dtype=torch.float32)
                reference = (
                    reference_rewards - reference_rewards.mean()
                ) / (reference_rewards.std(unbiased=False) + 1e-6)
                torch.testing.assert_close(
                    actual,
                    reference,
                    rtol=0.0,
                    atol=0.0,
                    msg=(
                        f"counts=(0:{zero_count},0.2:{legal_wrong_count},"
                        f"1:{correct_count})"
                    ),
                )

    def test_result_only_uses_population_group_advantage_for_every_turn(self):
        episodes = [
            _episode("a", 0.0, [0.0, 0.0]),
            _episode("b", 1.0, [0.0]),
        ]
        updates = build_transition_updates(episodes, reward_mode="result-only")
        self.assertEqual(len(updates), 3)
        expected = abs(
            standardized_group_advantages([0.0, 1.0], [True, True])[0]
        )
        self.assertTrue(math.isclose(updates[0].advantage, -expected))
        self.assertTrue(math.isclose(updates[1].advantage, -expected))
        self.assertTrue(math.isclose(updates[2].advantage, expected))
        self.assertEqual(
            [update.trajectory_turn_weight for update in updates],
            [0.5, 0.5, 1.0],
        )

    def test_correctness_primary_profile_cannot_flip_outcome_sign(self):
        correct_clean = _episode("correct-clean", 1.5, [0.0], example_index=7)
        correct_error = _episode("correct-error", 1.0, [0.0], example_index=7)
        correct_error.sample.audit_record["errors"] = 1
        wrong_clean = _episode("wrong-clean", -0.5, [0.0], example_index=7)
        wrong_clean.sample.correct = False
        wrong_error = _episode("wrong-error", -1.0, [0.0], example_index=7)
        wrong_error.sample.correct = False
        wrong_error.sample.audit_record["error_events"] = [{"error_type": "execution_error"}]
        values = correctness_primary_clean_secondary_advantages(
            [correct_clean, correct_error, wrong_clean, wrong_error],
            clean_weight=0.25,
        )
        self.assertGreater(values[0], 0.0)
        self.assertGreater(values[1], 0.0)
        self.assertLess(values[2], 0.0)
        self.assertLess(values[3], 0.0)

    def test_class_conditional_routing_preserves_correct_efficiency_signal(self):
        correct_clean = _episode("correct-clean", 1.5, [0.0], example_index=7)
        correct_error = _episode("correct-error", 1.0, [0.0], example_index=7)
        correct_error.sample.audit_record["errors"] = 1
        values = class_conditional_routing_advantages(
            [correct_clean, correct_error], clean_weight=0.25
        )
        self.assertGreater(values[0], 0.0)
        self.assertLess(values[1], 0.0)
        self.assertAlmostEqual(abs(values[0]), 0.25, places=4)

    def test_class_conditional_routing_does_not_reward_wrong_clean(self):
        wrong_clean = _episode("wrong-clean", -0.5, [0.0], example_index=7)
        wrong_clean.sample.correct = False
        wrong_error = _episode("wrong-error", -1.0, [0.0], example_index=7)
        wrong_error.sample.correct = False
        wrong_error.sample.audit_record["error_events"] = [
            {"error_type": "execution_error"}
        ]
        values = class_conditional_routing_advantages(
            [wrong_clean, wrong_error], clean_weight=0.25
        )
        self.assertEqual(values, [0.0, 0.0])

    def test_class_conditional_routing_mixed_group_keeps_wrong_side_equal(self):
        correct_clean = _episode("correct-clean", 1.5, [0.0], example_index=7)
        wrong_clean = _episode("wrong-clean", -0.5, [0.0], example_index=7)
        wrong_clean.sample.correct = False
        wrong_error = _episode("wrong-error", -1.0, [0.0], example_index=7)
        wrong_error.sample.correct = False
        wrong_error.sample.audit_record["error_events"] = [
            {"error_type": "execution_error"}
        ]
        values = class_conditional_routing_advantages(
            [correct_clean, wrong_clean, wrong_error], clean_weight=0.25
        )
        self.assertGreater(values[0], 0.0)
        self.assertLess(values[1], 0.0)
        self.assertEqual(values[1], values[2])

    def test_trajectory_mean_gives_long_and_short_trajectories_equal_mass(self):
        updates = build_transition_updates(
            [
                _episode("long", 0.0, [0.0, 0.0]),
                _episode("short", 1.0, [0.0]),
            ],
            reward_mode="result-only",
        )
        values = policy_reduction_advantages(
            updates,
            reduction="trajectory_mean",
        )
        by_trajectory = {}
        for update, value in zip(updates, values, strict=True):
            by_trajectory.setdefault(update.trajectory_id, 0.0)
            by_trajectory[update.trajectory_id] += value
        self.assertTrue(
            math.isclose(
                abs(by_trajectory["long"]),
                abs(by_trajectory["short"]),
            )
        )

    def test_trajectory_token_mean_matches_whole_response_token_weights(self):
        episodes = [
            _episode("a", 0.0, [0.0, 0.0]),
            _episode("b", 1.0, [0.0]),
        ]
        episodes[0].policy_turns = [_sized_turn(1, 2), _sized_turn(11, 6)]
        episodes[1].policy_turns = [_sized_turn(21, 5)]
        for episode in episodes:
            episode.sample.turns = [
                (list(turn.prompt_ids), list(turn.response_ids))
                for turn in episode.policy_turns
            ]
        updates = build_transition_updates(episodes, reward_mode="result-only")
        self.assertEqual(
            [update.trajectory_token_weight for update in updates],
            [0.25, 0.75, 1.0],
        )
        values = policy_reduction_advantages(
            updates,
            reduction="trajectory_token_mean",
        )
        scale = len(updates) / 2
        self.assertTrue(math.isclose(values[0], updates[0].advantage * 0.25 * scale))
        self.assertTrue(math.isclose(values[1], updates[1].advantage * 0.75 * scale))
        self.assertTrue(math.isclose(values[2], updates[2].advantage * scale))
        # ``training_step`` multiplies every one-transition micro-loss by 1/T.
        # The resulting sum must equal one trajectory's whole-response token
        # mean divided by the number of trajectories, as in TRUST-SQL.
        turn_token_means = [1.0, 3.0]
        implemented = sum(
            values[index] * turn_token_means[index] / len(updates)
            for index in (0, 1)
        )
        whole_response_mean = (2 * 1.0 + 6 * 3.0) / 8
        expected = updates[0].advantage * whole_response_mean / 2
        self.assertTrue(math.isclose(implemented, expected))

    def test_policy_only_homogeneous_group_keeps_one_zero_gradient_update(self):
        updates = build_transition_updates(
            [
                _episode("a", 0.2, [0.0, 0.0]),
                _episode("b", 0.2, [0.0]),
            ],
            reward_mode="result-only",
        )
        retained, dropped = retain_policy_contributing_updates(
            updates,
            policy_loss_coefficient=1.0,
            rank_loss_coefficient=0.0,
            kl_beta=0.0,
        )
        self.assertEqual(len(retained), 1)
        self.assertEqual(retained[0].advantage, 0.0)
        self.assertEqual(dropped, len(updates) - 1)

    def test_null_transition_drop_is_disabled_for_other_loss_consumers(self):
        updates = build_transition_updates(
            [
                _episode("a", 0.2, [0.0, 0.0]),
                _episode("b", 0.2, [0.0]),
            ],
            reward_mode="result-only",
        )
        for kwargs in (
            {
                "policy_loss_coefficient": 0.0,
                "rank_loss_coefficient": 0.0,
                "kl_beta": 0.0,
            },
            {
                "policy_loss_coefficient": 1.0,
                "rank_loss_coefficient": 0.5,
                "kl_beta": 0.0,
            },
            {
                "policy_loss_coefficient": 1.0,
                "rank_loss_coefficient": 0.0,
                "kl_beta": 0.1,
            },
        ):
            with self.subTest(kwargs=kwargs):
                retained, dropped = retain_policy_contributing_updates(
                    updates,
                    **kwargs,
                )
                self.assertEqual(retained, updates)
                self.assertEqual(dropped, 0)

    def test_process_rewards_stay_turn_local(self):
        updates = build_transition_updates(
            [_episode("a", 1.0, [-0.25, 0.75, 0.0])],
            reward_mode="process",
        )
        self.assertEqual([update.advantage for update in updates], [-0.25, 0.75, 0.0])
        self.assertEqual(
            [update.legal_success for update in updates],
            [False, True, True],
        )
        self.assertEqual(
            [update.local_penalty for update in updates],
            [0.25, 0.0, 0.0],
        )

    def test_excluded_episode_is_not_optimized(self):
        updates = build_transition_updates(
            [
                _episode("a", 1.0, [1.0]),
                _episode("b", 0.0, [0.0], process_update=False),
            ],
            reward_mode="process",
        )
        self.assertEqual([update.trajectory_id for update in updates], ["a"])

    def test_last_turn_mode_preserves_original_turn_index(self):
        updates = build_transition_updates(
            [_episode("a", 1.0, [0.1, 0.2, 0.3])],
            reward_mode="process",
            train_turns="last",
        )
        self.assertEqual(len(updates), 1)
        self.assertEqual(updates[0].turn_index, 2)
        self.assertEqual(updates[0].advantage, 0.3)

    def test_policy_logprobs_must_align_with_response_tokens(self):
        broken = _episode("a", 1.0, [1.0])
        broken.policy_turns[0] = PolicyTurn(
            prompt_ids=(1,),
            response_ids=(2, 3),
            sampling_logprobs=(-0.1,),
        )
        with self.assertRaisesRegex(ValueError, "sampling logprobs"):
            build_transition_updates([broken], reward_mode="process")

    def test_homogeneous_result_group_has_zero_advantage(self):
        self.assertEqual(
            standardized_group_advantages([1.0, 1.0, 1.0], [True, True, True]),
            [0.0, 0.0, 0.0],
        )

    def test_result_advantages_are_normalized_per_problem_not_across_batch(self):
        updates = build_transition_updates(
            [
                _episode("a0", 0.0, [0.0], example_index=10),
                _episode("a1", 1.0, [0.0], example_index=10),
                _episode("b0", 1.0, [0.0], example_index=20),
                _episode("b1", 1.0, [0.0], example_index=20),
            ],
            reward_mode="result-only",
        )
        by_id = {update.trajectory_id: update.advantage for update in updates}
        self.assertLess(by_id["a0"], 0.0)
        self.assertGreater(by_id["a1"], 0.0)
        self.assertEqual(by_id["b0"], 0.0)
        self.assertEqual(by_id["b1"], 0.0)


if __name__ == "__main__":
    unittest.main()
