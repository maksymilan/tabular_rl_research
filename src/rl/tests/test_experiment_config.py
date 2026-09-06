from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from rl.configuration.experiment_config import (
    QWEN3_8B_BASE_MODEL_FILES,
    RLExperimentConfig,
    base_model_aggregate_sha256,
    validate_base_model_identity_contract,
)
from rl.objectives.process_credit import ProcessRewardConfig


QWEN3_8B_BASE_MODEL_AGGREGATE_SHA256 = (
    "85bd3b7d908acb3a9b9c7ec57b98d6b9e3b2fb427685ae808d1c43173279cecc"
)


def test_result_only_matrix_config_maps_to_exact_control_defaults() -> None:
    config = RLExperimentConfig.load(
        ROOT / "src" / "rl" / "configs" / "experiments" / "phase1_result_only.yaml"
    )
    defaults = config.argparse_defaults(ROOT)

    assert config.admission_status == "allowed_result_only_control"
    assert defaults["reward_mode"] == "result-only"
    assert defaults["optimizer_name"] == "adamw_torch"
    assert defaults["optimizer_steps"] == 23
    assert defaults["learning_rate"] == 1e-6
    assert defaults["lr_scheduler_type"] == "cosine"
    assert defaults["warmup_ratio"] == 0.03
    assert defaults["kl_beta"] == 0.0
    assert defaults["group_size"] == 4
    assert defaults["temperature"] == 0.7
    assert defaults["top_p"] == 0.95


def test_trustsql_style_result_baseline_has_hardware_equivalent_batch() -> None:
    config = RLExperimentConfig.load(
        ROOT
        / "src"
        / "rl"
        / "configs"
        / "experiments"
        / "trustsql_result_only_grpo_scale60.yaml"
    )
    defaults = config.argparse_defaults(ROOT)
    assert defaults["result_reward_profile"] == "execution-ladder"
    assert defaults["policy_reduction"] == "trajectory_token_mean"
    assert defaults["group_size"] == 8
    assert defaults["prompts_per_update"] == 1
    assert defaults["gradient_accumulation_steps"] == 30
    assert defaults["optimizer_steps"] == 6
    assert defaults["learning_rate"] == 8e-7
    assert defaults["lr_scheduler_type"] == "constant"
    assert defaults["clip_epsilon"] == 0.2
    assert defaults["clip_epsilon_high"] == 0.28
    assert defaults["adam_beta2"] == 0.98


def test_qwen_thinking_mode_is_an_explicit_optional_rollout_identity(tmp_path: Path) -> None:
    source = ROOT / "src" / "rl" / "configs" / "experiments" / "trustsql_result_only_grpo_scale60.yaml"
    payload = source.read_text(encoding="utf-8")
    payload += "  enable_thinking: true\n"
    path = tmp_path / "qwen.yaml"
    path.write_text(payload, encoding="utf-8")
    defaults = RLExperimentConfig.load(path).argparse_defaults(ROOT)
    assert defaults["enable_thinking"] is True


@pytest.mark.parametrize(
    "name",
    [
        "qwen3_8b_atomic_v26_vanilla_grpo.yaml",
        "qwen3_8b_atomic_v26_vanilla_grpo_boundary300.yaml",
        "qwen3_8b_atomic_v26_vanilla_grpo_arm_b_train320_k16.yaml",
    ],
)
def test_qwen3_v26_vanilla_configs_pin_the_same_complete_base_model(
    name: str,
) -> None:
    config = RLExperimentConfig.load(
        ROOT / "src" / "rl" / "configs" / "experiments" / name
    )
    identity = config.argparse_defaults(ROOT)["expected_base_model_identity"]
    assert identity["schema_version"] == "trl-base-model-identity-v1"
    assert identity["aggregate_sha256"] == QWEN3_8B_BASE_MODEL_AGGREGATE_SHA256
    assert tuple(identity["files_sha256"]) == QWEN3_8B_BASE_MODEL_FILES
    assert len(identity["files_sha256"]) == 12
    assert (
        base_model_aggregate_sha256(identity["files_sha256"])
        == QWEN3_8B_BASE_MODEL_AGGREGATE_SHA256
    )


def test_base_model_identity_rejects_partial_or_self_inconsistent_contracts() -> None:
    config = RLExperimentConfig.load(
        ROOT
        / "src/rl/configs/experiments/qwen3_8b_atomic_v26_vanilla_grpo.yaml"
    )
    identity = config.argparse_defaults(ROOT)["expected_base_model_identity"]

    partial = json.loads(json.dumps(identity))
    partial["files_sha256"].pop("vocab.json")
    with pytest.raises(ValueError, match="exactly the 12 frozen files"):
        validate_base_model_identity_contract(partial)

    inconsistent = json.loads(json.dumps(identity))
    inconsistent["aggregate_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="canonical file records"):
        validate_base_model_identity_contract(inconsistent)


@pytest.mark.parametrize(
    "name",
    [
        "phase1_process_no_backslice.yaml",
        "phase1_process_tool_only.yaml",
        "phase2_process_rank.yaml",
        "phase2_process_no_normalize.yaml",
        "phase3_process_strong_penalty.yaml",
        "phase4_process_rank_action_only.yaml",
        "phase5_process_rank_conservative.yaml",
        "phase6_process_rank_conservative_score_masked.yaml",
        "phase7_process_rank_conservative_action_mean.yaml",
    ],
)
def test_process_matrix_configs_are_admitted_only_after_mandatory_gates(name: str) -> None:
    config = RLExperimentConfig.load(
        ROOT / "src" / "rl" / "configs" / "experiments" / name
    )
    defaults = config.argparse_defaults(ROOT)
    assert config.admission_status == "allowed_process_after_gates"
    assert defaults["reward_mode"] == "process"
    assert defaults["process_admission_policy"] == "counterfactual-completeness"
    assert defaults["exclude_empty_reference_results"] is False


@pytest.mark.parametrize(
    ("name", "normalize_positive"),
    [
        ("phase1_process_no_backslice.yaml", True),
        ("phase1_process_tool_only.yaml", True),
        ("phase2_process_rank.yaml", True),
        ("phase2_process_no_normalize.yaml", False),
        ("phase3_process_strong_penalty.yaml", True),
        ("phase4_process_rank_action_only.yaml", True),
        ("phase5_process_rank_conservative.yaml", True),
        ("phase6_process_rank_conservative_score_masked.yaml", True),
        ("phase7_process_rank_conservative_action_mean.yaml", True),
        ("exp12_fixed_process_only.yaml", True),
        ("exp13_fixed_rank_only_action_mean.yaml", True),
        ("exp14_fixed_process_rank_action_mean.yaml", True),
    ],
)
def test_active_process_ablations_share_no_backslice_base(
    name: str,
    normalize_positive: bool,
) -> None:
    config = RLExperimentConfig.load(
        ROOT / "src" / "rl" / "configs" / "experiments" / name
    )
    reward_config = ProcessRewardConfig(
        **{
            key: value
            for key, value in json.loads(
                (ROOT / config.payload["process_reward_config"]).read_text(
                    encoding="utf-8"
                )
            ).items()
            if not key.startswith("_")
        }
    )
    assert reward_config.w_back_slice == 0.0
    assert reward_config.normalize_positive is normalize_positive


def test_strong_penalty_ablation_uses_audited_failure_profile() -> None:
    config = RLExperimentConfig.load(
        ROOT
        / "src"
        / "rl"
        / "configs"
        / "experiments"
        / "phase3_process_strong_penalty.yaml"
    )
    reward_config = ProcessRewardConfig(
        **{
            key: value
            for key, value in json.loads(
                (ROOT / config.payload["process_reward_config"]).read_text(
                    encoding="utf-8"
                )
            ).items()
            if not key.startswith("_")
        }
    )
    reward_config.validate()
    assert reward_config.w_back_slice == 0.0
    assert reward_config.normalize_positive is True
    assert reward_config.lambda_terminal_failure == 0.0
    assert reward_config.lambda_tool_error == 0.24
    assert reward_config.lambda_adjacent_repeat == 0.18
    assert reward_config.lambda_legal_no_state_change == 0.09
    assert reward_config.lambda_ignored_feedback == 0.0
    assert reward_config.lambda_unsupported_guess == 0.0
    assert reward_config.penalty_cap == 0.95


@pytest.mark.parametrize(
    ("name", "update_scope"),
    [
        ("phase4_process_rank_action_only.yaml", "full_trajectory"),
        ("phase5_process_rank_conservative.yaml", "conservative_legal"),
    ],
)
def test_rank_followups_isolate_action_tokens_and_update_scope(
    name: str,
    update_scope: str,
) -> None:
    config = RLExperimentConfig.load(
        ROOT / "src" / "rl" / "configs" / "experiments" / name
    )
    defaults = config.argparse_defaults(ROOT)
    assert defaults["trainable_part"] == "all"
    assert defaults["rank_loss_coefficient"] == 0.5
    assert defaults["rank_beta"] == 0.1
    assert defaults["rank_score_tokens"] == "tool_only"
    assert defaults["rank_update_scope"] == update_scope


@pytest.mark.parametrize(
    ("name", "score_reduction"),
    [
        ("phase6_process_rank_conservative_score_masked.yaml", "sum_tokens"),
        ("phase7_process_rank_conservative_action_mean.yaml", "mean_action"),
    ],
)
def test_conservative_score_followups_are_a_strict_single_variable_chain(
    name: str,
    score_reduction: str,
) -> None:
    config = RLExperimentConfig.load(
        ROOT / "src" / "rl" / "configs" / "experiments" / name
    )
    defaults = config.argparse_defaults(ROOT)
    assert defaults["trainable_part"] == "all"
    assert defaults["rank_loss_coefficient"] == 0.5
    assert defaults["rank_beta"] == 0.1
    assert defaults["rank_score_tokens"] == "tool_only"
    assert defaults["rank_score_scope"] == "conservative_legal"
    assert defaults["rank_score_reduction"] == score_reduction
    assert defaults["rank_update_scope"] == "conservative_legal"


def test_historical_process_checkpoint_remains_evaluation_only() -> None:
    config = RLExperimentConfig.load(
        ROOT
        / "src"
        / "rl"
        / "configs"
        / "experiments"
        / "phase1_process_current.yaml"
    )
    assert config.admission_status == "evaluation_only_existing_checkpoint"


@pytest.mark.parametrize(
    ("name", "policy_coefficient", "rank_coefficient"),
    [
        ("exp12_fixed_process_only.yaml", 1.0, 0.0),
        ("exp13_fixed_rank_only_action_mean.yaml", 0.0, 0.5),
        ("exp14_fixed_process_rank_action_mean.yaml", 1.0, 0.5),
    ],
)
def test_fixed_pool_objectives_are_strict_process_rank_ablation(
    name: str,
    policy_coefficient: float,
    rank_coefficient: float,
) -> None:
    config = RLExperimentConfig.load(
        ROOT / "src" / "rl" / "configs" / "experiments" / name
    )
    defaults = config.argparse_defaults(ROOT)
    assert defaults["policy_loss_coefficient"] == policy_coefficient
    assert defaults["rank_loss_coefficient"] == rank_coefficient
    assert defaults["trainable_part"] == "tool_only"
    assert defaults["rank_score_reduction"] == "mean_action"
    assert defaults["optimizer_steps"] == 60
    assert defaults["learning_rate"] == 1e-6
    assert defaults["kl_beta"] == 0.0


def test_fixed_rank_only_uses_local_features_without_counterfactual_requirement() -> None:
    config = RLExperimentConfig.load(
        ROOT / "src" / "rl" / "configs" / "experiments" /
        "exp13_fixed_rank_only_action_mean.yaml"
    )
    defaults = config.argparse_defaults(ROOT)
    assert config.admission_status == "allowed_rank_only_control"
    assert defaults["process_admission_policy"] == "rank-local-features"
    assert defaults["policy_loss_coefficient"] == 0.0
    assert defaults["rank_loss_coefficient"] == 0.5


@pytest.mark.parametrize(
    "name",
    ["exp12_fixed_process_only.yaml", "exp14_fixed_process_rank_action_mean.yaml"],
)
def test_fixed_process_views_screen_failed_counterfactuals(name: str) -> None:
    config = RLExperimentConfig.load(
        ROOT / "src" / "rl" / "configs" / "experiments" / name
    )
    defaults = config.argparse_defaults(ROOT)
    assert config.admission_status == "allowed_process_screened"
    assert defaults["process_admission_policy"] == "counterfactual-screened"
    assert defaults["policy_loss_coefficient"] == 1.0


@pytest.mark.parametrize(
    ("name", "allocation_mode", "observation_bonus", "backslice_bonus", "steps"),
    [
        ("exp16_dense_uniform_full_response.yaml", "dense_uniform", 0.0, 0.0, 60),
        ("exp17_dense_strategic_full_response.yaml", "dense_strategic", 0.5, 0.5, 60),
        ("exp18_dense_uniform_full_response_scale120.yaml", "dense_uniform", 0.0, 0.0, 120),
        ("exp18_dense_strategic_full_response_scale120.yaml", "dense_strategic", 0.5, 0.5, 120),
    ],
)
def test_dense_outcome_experiments_train_the_full_response_without_cf_screening(
    name: str,
    allocation_mode: str,
    observation_bonus: float,
    backslice_bonus: float,
    steps: int,
) -> None:
    config = RLExperimentConfig.load(
        ROOT / "src" / "rl" / "configs" / "experiments" / name
    )
    defaults = config.argparse_defaults(ROOT)
    reward_config = ProcessRewardConfig(
        **{
            key: value
            for key, value in json.loads(
                (ROOT / config.payload["process_reward_config"]).read_text(
                    encoding="utf-8"
                )
            ).items()
            if not key.startswith("_")
        }
    )
    reward_config.validate()
    assert config.admission_status == "allowed_process_unscreened"
    assert defaults["process_admission_policy"] == "dense-outcome"
    assert defaults["trainable_part"] == "all"
    assert defaults["policy_loss_coefficient"] == 1.0
    assert defaults["rank_loss_coefficient"] == 0.5
    assert defaults["rank_score_tokens"] == "all"
    assert defaults["rank_score_scope"] == "dense_outcome"
    assert defaults["rank_score_reduction"] == "mean_action"
    assert defaults["rank_update_scope"] == "dense_outcome"
    assert defaults["optimizer_steps"] == steps
    assert defaults["learning_rate"] == 1e-6
    assert defaults["kl_beta"] == 0.0
    assert reward_config.allocation_mode == allocation_mode
    assert reward_config.dense_correct_legal_weight == 1.0
    assert reward_config.dense_incorrect_legal_weight == 0.5
    assert reward_config.dense_severe_penalty_weight == 2.0
    assert reward_config.dense_observation_bonus_weight == observation_bonus
    assert reward_config.dense_backslice_bonus_weight == backslice_bonus
