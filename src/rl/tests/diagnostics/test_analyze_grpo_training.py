import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rl.scenarios.diagnostics.analyze_grpo_training import (
    GROUP_ADVANTAGE_SUM_ABS_TOLERANCE,
    analyze,
    checkpoint_precision_audit,
    implementation_provenance,
    standardized_group_advantages,
    torch,
)


class AnalyzeGRPOTrainingTest(unittest.TestCase):
    @unittest.skipIf(torch is None, "checkpoint precision audit requires Torch")
    def test_checkpoint_precision_audit_reads_persisted_fp32_state(self) -> None:
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            save_file(
                {
                    "layer.lora_A.weight": torch.ones(2, 2),
                    "layer.lora_B.weight": torch.ones(2, 2),
                },
                checkpoint / "adapter_model.safetensors",
            )
            torch.save(
                {
                    "state": {
                        0: {
                            "step": torch.tensor(1.0),
                            "exp_avg": torch.zeros(2, 2),
                            "exp_avg_sq": torch.zeros(2, 2),
                        }
                    },
                    "param_groups": [{"params": [0]}],
                },
                checkpoint / "optimizer.pt",
            )
            result = checkpoint_precision_audit(checkpoint)
            self.assertTrue(result["available"])
            self.assertTrue(result["passes"])
            self.assertEqual(result["adapter_tensor_count"], 2)
            self.assertEqual(result["adam_moment_tensors"], 2)
            self.assertEqual(
                result["adam_moment_dtype_counts"], {"torch.float32": 2}
            )

    def test_implementation_snapshot_is_verified_against_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            relative = "src/rl/example.py"
            snapshot = root / "implementation_source_snapshot" / relative
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text("baseline source\n")
            digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
            (root / "implementation_lock.json").write_text(
                json.dumps(
                    {
                        "files": {relative: digest},
                        "initial_adapter_sha256": "adapter-hash",
                    }
                )
            )
            result = implementation_provenance(root, {})
            self.assertTrue(result["source_snapshot_matches"])
            self.assertEqual(result["expected_file_count"], 1)
            self.assertEqual(result["initial_adapter_sha256"], "adapter-hash")
            snapshot.write_text("drifted source\n")
            drifted = implementation_provenance(root, {})
            self.assertFalse(drifted["source_snapshot_matches"])
            self.assertIn(relative, drifted["hash_mismatches"])

    def test_standardized_advantages_match_population_std(self) -> None:
        values = standardized_group_advantages([0.0, 0.2, 1.0], [True] * 3)
        self.assertAlmostEqual(sum(values), 0.0, places=6)
        self.assertLess(values[0], values[1])
        self.assertLess(values[1], values[2])
        self.assertEqual(
            standardized_group_advantages([1.0, 1.0], [True, True]),
            [0.0, 0.0],
        )
        self.assertEqual(
            standardized_group_advantages([0.2] * 8, [True] * 8),
            [0.0] * 8,
        )
        mixed = standardized_group_advantages(
            [0.0, 0.0, 0.2, 0.2, 0.0, 1.0, 0.0, 0.2], [True] * 8
        )
        self.assertEqual([mixed[index] for index in (2, 3, 7)], [0.0] * 3)

    @unittest.skipIf(torch is None, "production FP32 audit requires Torch")
    def test_k8_categorical_fp32_group_sum_residual_fits_tolerance(self) -> None:
        maximum_residual = 0.0
        heterogeneous_compositions = 0
        for zeros in range(9):
            for executable_wrong in range(9 - zeros):
                correct = 8 - zeros - executable_wrong
                if sum(
                    count > 0 for count in (zeros, executable_wrong, correct)
                ) < 2:
                    continue
                rewards = (
                    [0.0] * zeros
                    + [0.2] * executable_wrong
                    + [1.0] * correct
                )
                advantages = standardized_group_advantages(
                    rewards, [True] * 8
                )
                maximum_residual = max(
                    maximum_residual, abs(sum(advantages))
                )
                heterogeneous_compositions += 1
        self.assertEqual(heterogeneous_compositions, 42)
        self.assertGreater(maximum_residual, 1e-6)
        self.assertLessEqual(
            maximum_residual, GROUP_ADVANTAGE_SUM_ABS_TOLERANCE
        )
        self.assertLessEqual(GROUP_ADVANTAGE_SUM_ABS_TOLERANCE, 2e-6)

    def test_completed_and_partial_updates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "protocol_version": "version36",
                "protocol_hash": "hash36",
                "reward_mode": "result-only",
                "result_reward_profile": "execution-ladder",
                "policy_reduction": "trajectory_mean",
                "records": 2,
                "group_size": 2,
                "prompts_per_update": 1,
                "gradient_accumulation_steps": 2,
                "optimizer_steps": 2,
                "learning_rate": 8e-7,
            }
            (root / "run_manifest.json").write_text(json.dumps(manifest))
            rows = []
            for index, rewards in ((10, (0.0, 1.0)), (11, (0.2, 0.2))):
                for sample, reward in enumerate(rewards):
                    rows.append(
                        {
                            "protocol_version": "version36",
                            "protocol_hash": "hash36",
                            "example_index": index,
                            "trajectory_id": f"rl_{index}_sample_{sample}",
                            "result_reward": {"value": reward},
                            "correct": reward == 1.0,
                            "legal": reward > 0.0,
                            "failure_type": None if reward == 1.0 else "wrong_answer",
                            "turns": [{}, {}],
                            "elapsed_seconds": 2.0,
                            "error_events": [],
                        }
                    )
            rows.extend(
                {
                    "protocol_version": "version36",
                    "protocol_hash": "hash36",
                    "example_index": 12,
                    "trajectory_id": f"rl_12_sample_{sample}",
                    "result_reward": {"value": reward},
                    "correct": False,
                    "legal": reward > 0,
                    "failure_type": "wrong_answer",
                    "turns": [{}],
                    "elapsed_seconds": 1.0,
                    "error_events": [],
                }
                for sample, reward in enumerate((0.0, 0.2))
            )
            rows[1]["error_events"] = [
                {
                    "error_type": "timeout_error",
                    "error_code": "tool_execution_timeout",
                    "state_before_hash": "preserved-state",
                    "state_after_hash": "preserved-state",
                    "details": {"state_preserved": True},
                }
            ]
            for row_index, row in enumerate(rows):
                row["policy_global_step"] = row_index // 4
                row["policy_synced_global_step"] = row_index // 4
                row["policy_micro_step"] = row_index // 2
            (root / "rollouts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            checkpoint = root / "checkpoint-1"
            checkpoint.mkdir()
            (checkpoint / "trainer_state.json").write_text(
                json.dumps(
                    {
                        "global_step": 1,
                        "log_history": [
                            {"step": 1, "grad_norm": 0.5, "loss": 0.01}
                        ],
                    }
                )
            )

            result = analyze(root)
            provenance = result["analysis_provenance"]
            self.assertEqual(len(provenance["analyzer_sha256"]), 64)
            self.assertEqual(
                provenance["group_advantage_sum_abs_tolerance"],
                GROUP_ADVANTAGE_SUM_ABS_TOLERANCE,
            )
            self.assertTrue(result["contract"]["matches"])
            self.assertEqual(result["progress"]["completed_optimizer_steps"], 1)
            self.assertEqual(result["progress"]["partial_next_update_rows"], 2)
            update = result["updates"][0]
            self.assertEqual(update["unique_prompt_groups"], 2)
            self.assertEqual(update["duplicate_example_groups"], {})
            self.assertTrue(update["distinct_prompt_groups_passes"])
            self.assertEqual(update["heterogeneous_groups"], 1)
            self.assertEqual(update["homogeneous_groups"], 1)
            self.assertEqual(update["nonzero_advantage_trajectories"], 2)
            self.assertEqual(update["tiny_nonzero_advantage_trajectories"], 0)
            self.assertTrue(update["exact_zero_contract_passes"])
            self.assertEqual(update["contributing_policy_transitions"], 4)
            self.assertEqual(update["null_policy_transitions"], 4)
            self.assertEqual(update["null_policy_transition_fraction"], 0.5)
            self.assertEqual(update["timeout_trajectories"], 1)
            self.assertEqual(update["timeout_events"], 1)
            self.assertEqual(update["timeout_recovered_correct"], 1)
            self.assertEqual(update["timeout_recovered_legal"], 1)
            self.assertEqual(update["timeout_unrecovered"], 0)
            self.assertEqual(update["timeout_missing_structured_events"], 0)
            self.assertEqual(update["timeout_state_preservation_violations"], 0)
            self.assertTrue(update["timeout_contract_passes"])
            first_group = update["group_summaries"][0]
            self.assertEqual(first_group["example_index"], 10)
            self.assertEqual(first_group["rewards"], [0.0, 1.0])
            self.assertLess(first_group["advantages"][0], 0.0)
            self.assertGreater(first_group["advantages"][1], 0.0)
            self.assertEqual(first_group["policy_global_steps"], [0])
            self.assertEqual(first_group["policy_micro_steps"], [0])
            self.assertEqual(first_group["timeout_trajectories"], 1)
            self.assertEqual(first_group["timeout_events"], 1)
            self.assertEqual(first_group["timeout_recovered_correct"], 1)
            tiers = update["reward_tier_advantages"]
            self.assertEqual(tiers["0.0"]["negative_advantages"], 1)
            self.assertEqual(tiers["0.0"]["positive_advantages"], 0)
            self.assertEqual(tiers["1.0"]["positive_advantages"], 1)
            self.assertEqual(tiers["1.0"]["negative_advantages"], 0)
            self.assertEqual(tiers["0.2"]["zero_advantages"], 2)
            self.assertEqual(tiers["0.2"]["turns"], 4)
            credit_checks = update["credit_direction_checks"]
            self.assertTrue(credit_checks["passes"])
            self.assertEqual(
                credit_checks["group_advantage_sum_abs_tolerance"],
                GROUP_ADVANTAGE_SUM_ABS_TOLERANCE,
            )
            self.assertEqual(
                credit_checks["pairwise_reward_order_violations"], 0
            )
            self.assertEqual(
                credit_checks["homogeneous_nonzero_advantage_groups"], 0
            )
            self.assertEqual(update["trainer_metrics"]["grad_norm"], 0.5)
            self.assertFalse(update["checkpoint_precision"]["available"])
            self.assertEqual(
                result["partial_next_update"]["heterogeneous_groups"], 1
            )
            self.assertTrue(
                result["partial_next_update"]["distinct_prompt_groups_passes"]
            )
            partial_tiers = result["partial_next_update"][
                "reward_tier_advantages"
            ]
            self.assertEqual(partial_tiers["0.0"]["negative_advantages"], 1)
            self.assertEqual(partial_tiers["0.2"]["positive_advantages"], 1)
            binding = result["online_policy_binding"]
            self.assertEqual(binding["policy_global_step_coverage"], 1.0)
            self.assertEqual(binding["policy_global_step_counts"], {"0": 4, "1": 2})
            self.assertTrue(binding["policy_global_step_schedule_matches"])
            self.assertTrue(binding["policy_sync_schedule_matches"])
            self.assertTrue(binding["policy_micro_step_schedule_matches"])
            behavior = result["sampled_policy_behavior"]
            self.assertEqual(behavior["explicit_policy_step_coverage"], 1.0)
            self.assertEqual(len(behavior["policy_steps"]), 2)
            self.assertEqual(
                behavior["repeated_task_comparison"]["tasks"], 0
            )

    def test_mixed_example_group_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "protocol_version": "version36",
                        "protocol_hash": "hash36",
                        "reward_mode": "result-only",
                        "result_reward_profile": "binary",
                        "policy_reduction": "trajectory_mean",
                        "records": 1,
                        "group_size": 2,
                        "prompts_per_update": 1,
                        "gradient_accumulation_steps": 1,
                        "optimizer_steps": 1,
                        "learning_rate": 1e-6,
                    }
                )
            )
            rows = [
                {
                    "protocol_version": "version36",
                    "protocol_hash": "hash36",
                    "example_index": index,
                    "result_reward": {"value": 0.0},
                }
                for index in (1, 2)
            ]
            (root / "rollouts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows)
            )
            with self.assertRaisesRegex(ValueError, "mixes example indices"):
                analyze(root)


if __name__ == "__main__":
    unittest.main()
