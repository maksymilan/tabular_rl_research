#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SRC_DIR))

from rl.scenarios.audits.audit_reward_sensitivity import (  # noqa: E402
    build_scenarios,
    reward_map,
    score_episodes,
    summarize_results,
)
from rl.objectives.process_credit import ProcessRewardConfig, StepFeature  # noqa: E402


class RewardSensitivityTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [
            (
                "success",
                True,
                [
                    StepFeature(
                        action_index=1,
                        step_id="step_1",
                        tool="condition_filter",
                        legal_success=True,
                        back_slice=1.0,
                        search_reduction=0.5,
                    ),
                    StepFeature(
                        action_index=2,
                        step_id="step_2",
                        tool="answer_from_context",
                        legal_success=True,
                        is_terminal=True,
                        answer_format=1.0,
                    ),
                ],
                {"replay_correct": True},
            ),
            (
                "failure",
                False,
                [
                    StepFeature(
                        action_index=1,
                        step_id="step_1",
                        tool="describe_table",
                        legal_success=True,
                        target_table_delta=0.5,
                        target_column_delta=0.5,
                    ),
                    StepFeature(
                        action_index=2,
                        step_id="step_2",
                        tool="answer_from_context",
                        legal_success=True,
                        is_terminal=True,
                        answer_format=1.0,
                    ),
                ],
                {"replay_correct": False},
            ),
        ]

    def test_all_declared_scenarios_preserve_reward_sign_bounds(self):
        config = ProcessRewardConfig()
        baseline = score_episodes(self.episodes, config)
        baseline_rewards = reward_map(baseline)
        for name, _, _, candidate in build_scenarios(config):
            results = baseline if name == "baseline" else score_episodes(
                self.episodes,
                candidate,
            )
            summary = summarize_results(
                results,
                baseline_rewards=baseline_rewards,
            )
            self.assertEqual(summary["success_total_reward"]["nonpositive"], 0, name)
            self.assertEqual(summary["failure_total_reward"]["nonnegative"], 0, name)

    def test_baseline_comparison_is_zero(self):
        config = ProcessRewardConfig()
        results = score_episodes(self.episodes, config)
        summary = summarize_results(
            results,
            baseline_rewards=reward_map(results),
        )
        self.assertEqual(summary["reward_l1_vs_baseline"], 0.0)
        self.assertEqual(summary["step_sign_changes_vs_baseline"], 0)


if __name__ == "__main__":
    unittest.main()
