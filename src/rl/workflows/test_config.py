from pathlib import Path

import pytest

from rl.workflows.config import WorkflowConfig, WorkflowKind, build_plan


def test_rl_scale_changes_only_parameters():
    root = Path("/repo")
    verify = build_plan(WorkflowConfig(WorkflowKind.RL_VERIFY, 20, 14), repo_root=root)
    full = build_plan(WorkflowConfig(WorkflowKind.RL_FULL, 500, 30), repo_root=root)
    assert verify["command"][0] == full["command"][0]
    assert verify["command"][1] == "gate"
    assert full["command"][1] == "run"
    assert verify["environment"]["EXPECTED_RECORDS"] == "20"
    assert full["environment"]["EXPECTED_RECORDS"] == "500"


def test_evaluation_is_dataset_split_and_gpu_parameterized():
    plan = build_plan(
        WorkflowConfig(WorkflowKind.EVALUATE, dataset="spider", split="test", gpu_ids=(1, 3)),
        repo_root=Path("/repo"),
    )
    assert plan["dataset"] == "spider"
    assert plan["split"] == "test"
    assert plan["environment"]["GPU_IDS"] == "1,3"
    assert plan["command"][1].endswith("src/rl/evaluation/runners/eval_plan.py")
    assert "--partition" in plan["command"]


def test_rl_requires_checkpoint_6380_and_k8():
    with pytest.raises(ValueError, match="checkpoint-6380"):
        WorkflowConfig(WorkflowKind.RL_FULL, 500, 30, checkpoint="checkpoint-560").validate()
    with pytest.raises(ValueError, match="K=8"):
        WorkflowConfig(WorkflowKind.RL_FULL, 500, 30, group_size=4).validate()


def test_sft_is_explicitly_frozen():
    plan = build_plan(WorkflowConfig(WorkflowKind.SFT_FROZEN), repo_root=Path("/repo"))
    assert plan["workflow"] == "sft_frozen"
    assert plan["command"][0].endswith("train_qwen3_8b_atomic_sft1_newgnn.sh")
