import unittest

from rl.scenarios.diagnostics.audit_grpo_baseline_readiness import audit


def _artifacts(
    *, accuracy_net: int = 2, legal_net: int = 1, exact_p: float = 0.5
):
    update = {
        "groups": 2,
        "unique_prompt_groups": 2,
        "distinct_prompt_groups_passes": True,
        "heterogeneous_groups": 2,
        "nonzero_advantage_trajectories": 8,
        "exact_zero_contract_passes": True,
        "credit_direction_checks": {"passes": True},
        "timeout_contract_passes": True,
        "trainer_metrics": {"grad_norm": 0.5},
        "checkpoint_precision": {"available": True, "passes": True},
    }
    training = {
        "contract": {
            "matches": True,
            "manifest_protocol_version": "version36",
            "manifest_protocol_hash": "20a8d3b4356d883c",
        },
        "configuration": {
            "reward_mode": "result-only",
            "result_reward_profile": "execution-ladder",
            "policy_reduction": "trajectory_token_mean",
            "group_size": 8,
        },
        "implementation_provenance": {"source_snapshot_matches": True},
        "progress": {
            "completed_optimizer_steps": 2,
            "planned_optimizer_steps": 2,
            "partial_next_update_rows": 0,
        },
        "online_policy_binding": {
            "policy_global_step_coverage": 1.0,
            "policy_synced_global_step_coverage": 1.0,
            "policy_micro_step_coverage": 1.0,
            "policy_global_step_schedule_matches": True,
            "policy_sync_schedule_matches": True,
            "policy_micro_step_schedule_matches": True,
        },
        "sampled_policy_behavior": {
            "repeated_task_comparison": {
                "tasks": 2,
                "signature_sets_changed": 2,
                "mean_signature_jaccard": 0.5,
            }
        },
        "updates": [dict(update), dict(update)],
        "final_precision": {
            "trainable_parameters": {
                "trainable_tensors": 4,
                "non_fp32_tensors": [],
            },
            "optimizer_state": {
                "moment_tensors": 8,
                "non_fp32_moments": [],
            },
        },
    }
    checkpoint = "/checkpoint/final"
    lora = {
        "checkpoints": [checkpoint],
        "checkpoint_artifacts": {
            checkpoint: {"adapter_sha256": "candidate-adapter-hash"}
        },
        "raw_adapter": {checkpoint: {"update_norm": 0.1}},
        "effective_lora": {checkpoint: {"update_norm": 0.05}},
    }
    evaluation = {
        "cohort": {"total": 4},
        "evaluation_contract": {
            "protocol_version": ["version36"],
            "protocol_hash": ["20a8d3b4356d883c"],
            "temperature": [0.0],
            "top_p": [1.0],
            "denotation_comparison": ["bird-set"],
        },
        "arms": {
            "candidate": {
                "total": 4,
                "correct": 3,
                "legal": 4,
                "evaluation_identity": {
                    "schema_version": "evaluation-model-identity-v1",
                    "adapter_path": checkpoint,
                    "adapter_sha256": "candidate-adapter-hash",
                    "protocol_version": "version36",
                    "protocol_hash": "20a8d3b4356d883c",
                },
            },
            "sft2": {"total": 4, "correct": 1, "legal": 3},
        },
        "comparisons": {
            "candidate_vs_sft2": {
                "accuracy": {
                    "gains": max(accuracy_net, 0),
                    "regressions": max(-accuracy_net, 0),
                    "net": accuracy_net,
                    "net_rate": accuracy_net / 4,
                    "net_percentage_points": 25.0 * accuracy_net,
                    "exact_mcnemar_p": exact_p,
                },
                "legal": {
                    "net": legal_net,
                    "net_rate": legal_net / 4,
                    "net_percentage_points": 25.0 * legal_net,
                },
                "sequence_change": {
                    "exact_action_sequence_changed": 3,
                    "first_exact_action_changed": 1,
                    "normalized_exact_edit_distance": {"mean": 0.4},
                },
            }
        },
    }
    return training, lora, evaluation


class AuditGRPOBaselineReadinessTest(unittest.TestCase):
    def test_positive_evaluated_baseline_requires_all_evidence(self) -> None:
        result = audit(
            *_artifacts(),
            candidate="candidate",
            baseline="sft2",
            expected_eval_count=4,
        )
        self.assertTrue(result["status"]["engineering_valid_evaluated_baseline"])
        self.assertTrue(result["status"]["policy_change_proven"])
        self.assertTrue(result["status"]["positive_accuracy_baseline_observed"])
        self.assertTrue(
            result["status"][
                "positive_accuracy_with_legal_nonregression_observed"
            ]
        )
        self.assertFalse(
            result["status"][
                "statistically_supported_positive_accuracy_baseline"
            ]
        )

    def test_statistically_supported_gain_is_separate_status(self) -> None:
        result = audit(
            *_artifacts(exact_p=0.03125),
            candidate="candidate",
            baseline="sft2",
            expected_eval_count=4,
        )
        self.assertTrue(result["checks"]["accuracy_gain_statistically_supported"])
        self.assertTrue(
            result["status"][
                "statistically_supported_positive_accuracy_baseline"
            ]
        )
        self.assertTrue(
            result["status"][
                "statistically_supported_positive_accuracy_with_legal_nonregression"
            ]
        )

    def test_valid_pipeline_does_not_turn_regression_into_success(self) -> None:
        result = audit(
            *_artifacts(accuracy_net=-1),
            candidate="candidate",
            baseline="sft2",
            expected_eval_count=4,
        )
        self.assertTrue(result["status"]["engineering_valid_evaluated_baseline"])
        self.assertTrue(result["status"]["policy_change_proven"])
        self.assertFalse(result["checks"]["accuracy_gain_observed"])
        self.assertFalse(result["status"]["positive_accuracy_baseline_observed"])

    def test_accuracy_gain_and_legal_nonregression_are_reported_separately(self) -> None:
        result = audit(
            *_artifacts(accuracy_net=2, legal_net=-1),
            candidate="candidate",
            baseline="sft2",
            expected_eval_count=4,
        )
        self.assertTrue(result["status"]["positive_accuracy_baseline_observed"])
        self.assertFalse(
            result["status"][
                "positive_accuracy_with_legal_nonregression_observed"
            ]
        )

    def test_missing_fp32_optimizer_evidence_invalidates_pipeline(self) -> None:
        training, lora, evaluation = _artifacts()
        training["final_precision"]["optimizer_state"]["moment_tensors"] = 0
        result = audit(
            training,
            lora,
            evaluation,
            candidate="candidate",
            baseline="sft2",
            expected_eval_count=4,
        )
        self.assertFalse(result["checks"]["fp32_trainable_and_adam_moments"])
        self.assertFalse(result["status"]["engineering_valid_evaluated_baseline"])

    def test_non_fp32_persisted_checkpoint_invalidates_pipeline(self) -> None:
        training, lora, evaluation = _artifacts()
        training["updates"][0]["checkpoint_precision"]["passes"] = False
        result = audit(
            training,
            lora,
            evaluation,
            candidate="candidate",
            baseline="sft2",
            expected_eval_count=4,
        )
        self.assertFalse(
            result["checks"]["fp32_persisted_checkpoints_all_updates"]
        )
        self.assertFalse(result["status"]["engineering_valid_evaluated_baseline"])

    def test_timeout_state_violation_invalidates_pipeline(self) -> None:
        training, lora, evaluation = _artifacts()
        training["updates"][0]["timeout_contract_passes"] = False
        result = audit(
            training,
            lora,
            evaluation,
            candidate="candidate",
            baseline="sft2",
            expected_eval_count=4,
        )
        self.assertFalse(result["checks"]["timeout_contract_passes_all_updates"])
        self.assertFalse(result["status"]["engineering_valid_evaluated_baseline"])

    def test_duplicate_prompt_inside_update_invalidates_pipeline(self) -> None:
        training, lora, evaluation = _artifacts()
        training["updates"][0]["unique_prompt_groups"] = 1
        training["updates"][0]["distinct_prompt_groups_passes"] = False
        result = audit(
            training,
            lora,
            evaluation,
            candidate="candidate",
            baseline="sft2",
            expected_eval_count=4,
        )
        self.assertFalse(result["checks"]["distinct_prompt_groups_all_updates"])
        self.assertFalse(result["status"]["engineering_valid_evaluated_baseline"])

    def test_tiny_nonzero_advantage_invalidates_pipeline(self) -> None:
        training, lora, evaluation = _artifacts()
        training["updates"][0]["exact_zero_contract_passes"] = False
        training["updates"][0]["tiny_nonzero_advantage_trajectories"] = 3
        result = audit(
            training,
            lora,
            evaluation,
            candidate="candidate",
            baseline="sft2",
            expected_eval_count=4,
        )
        self.assertFalse(
            result["checks"]["exact_zero_contract_passes_all_updates"]
        )
        self.assertFalse(result["status"]["engineering_valid_evaluated_baseline"])

    def test_matching_wrong_protocol_is_not_accepted(self) -> None:
        training, lora, evaluation = _artifacts()
        training["contract"]["manifest_protocol_version"] = "version39"
        result = audit(
            training,
            lora,
            evaluation,
            candidate="candidate",
            baseline="sft2",
            expected_eval_count=4,
        )
        self.assertFalse(result["checks"]["protocol_contract_matches"])
        self.assertFalse(result["status"]["engineering_valid_evaluated_baseline"])

    def test_matching_wrong_evaluation_hash_is_not_accepted(self) -> None:
        training, lora, evaluation = _artifacts()
        evaluation["evaluation_contract"]["protocol_hash"] = ["wrong-hash"]
        result = audit(
            training,
            lora,
            evaluation,
            candidate="candidate",
            baseline="sft2",
            expected_eval_count=4,
        )
        self.assertFalse(result["checks"]["evaluation_contract_matches"])
        self.assertFalse(result["status"]["engineering_valid_evaluated_baseline"])

    def test_wrong_evaluated_adapter_is_not_accepted(self) -> None:
        training, lora, evaluation = _artifacts()
        evaluation["arms"]["candidate"]["evaluation_identity"][
            "adapter_sha256"
        ] = "wrong-adapter"
        result = audit(
            training,
            lora,
            evaluation,
            candidate="candidate",
            baseline="sft2",
            expected_eval_count=4,
        )
        self.assertFalse(result["checks"]["evaluation_adapter_identity_matches"])
        self.assertFalse(result["status"]["engineering_valid_evaluated_baseline"])


if __name__ == "__main__":
    unittest.main()
