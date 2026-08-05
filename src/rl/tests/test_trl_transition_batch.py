from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from frameworks.trl.transition_batch import (  # noqa: E402
    PolicyEpisode,
    PolicyTurn,
    build_transition_updates,
    standardized_group_advantages,
)


def _turn(seed: int) -> PolicyTurn:
    return PolicyTurn(
        prompt_ids=(seed, seed + 1),
        response_ids=(seed + 2, seed + 3),
        sampling_logprobs=(-0.1, -0.2),
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
    def test_result_only_uses_population_group_advantage_for_every_turn(self):
        episodes = [
            _episode("a", 0.0, [0.0, 0.0]),
            _episode("b", 1.0, [0.0]),
        ]
        updates = build_transition_updates(episodes, reward_mode="result-only")
        self.assertEqual(len(updates), 3)
        expected = 1.0 / (1.0 + 2e-6)
        self.assertTrue(math.isclose(updates[0].advantage, -expected))
        self.assertTrue(math.isclose(updates[1].advantage, -expected))
        self.assertTrue(math.isclose(updates[2].advantage, expected))

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
