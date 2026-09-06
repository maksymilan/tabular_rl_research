"""Typed configuration and command planning for the four supported workflows.

This module deliberately does not import TRL, vLLM, or Harness code.  It is a
small orchestration boundary: changing an RL mechanism changes ``mechanism``
and (eventually) the mechanism component passed to the trainer; changing scale
only changes ``question_count`` and ``prompts_per_update``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
import sys
from typing import Any


class WorkflowKind(StrEnum):
    RL_VERIFY = "rl_verify"
    RL_FULL = "rl_full"
    EVALUATE = "evaluate"
    SFT_FROZEN = "sft_frozen"


_CANONICAL_RL_ENTRYPOINT = (
    "src/rl/experiments/"
    "run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_single_gpu_a100.sh"
)
_CANONICAL_EVAL_ENTRYPOINT = "src/rl/evaluation/runners/eval_plan.py"
_CANONICAL_SFT_ENTRYPOINT = "src/sft/train_qwen3_8b_atomic_sft1_newgnn.sh"


@dataclass(frozen=True)
class WorkflowConfig:
    """User-facing workflow parameters shared by all experiment scales."""

    kind: WorkflowKind
    question_count: int | None = None
    prompts_per_update: int | None = None
    group_size: int = 8
    mechanism: str = "saam-asymmetric-error"
    dataset: str = "bird"
    dataset_path: str | None = None
    split: str = "dev"
    partition: str = "round_robin"
    gpu_ids: tuple[int, ...] = field(default_factory=tuple)
    checkpoint: str = "checkpoint-6380"
    output_dir: str | None = None
    optimizer_updates: int | None = None

    def validate(self) -> None:
        if self.kind in (WorkflowKind.RL_VERIFY, WorkflowKind.RL_FULL):
            if self.question_count is None or self.question_count <= 0:
                raise ValueError("RL workflow requires question_count > 0")
            if self.prompts_per_update is None or self.prompts_per_update <= 0:
                raise ValueError("RL workflow requires prompts_per_update > 0")
            if self.group_size != 8:
                raise ValueError("Atomic v26 RL requires K=8 (group_size=8)")
            if self.checkpoint != "checkpoint-6380":
                raise ValueError("current RL must start from checkpoint-6380")
            if self.kind is WorkflowKind.RL_FULL and self.question_count < 100:
                raise ValueError("rl_full is for the large cohort; use rl_verify for small runs")
        elif self.kind is WorkflowKind.EVALUATE:
            if not self.dataset or not self.split:
                raise ValueError("evaluation requires dataset and split")
            if self.question_count is not None and self.question_count <= 0:
                raise ValueError("evaluation question_count must be positive")
            if not self.gpu_ids:
                raise ValueError("evaluation requires at least one GPU id")
            if self.partition not in {"round_robin", "contiguous"}:
                raise ValueError("evaluation partition must be round_robin or contiguous")
        elif self.kind is WorkflowKind.SFT_FROZEN:
            # SFT remains a first-class *reproduction* workflow, but it is
            # intentionally not a mutable training route.  The plan carries
            # the frozen marker so callers can require an explicit approval
            # before invoking the historical entrypoint.
            return
        if any(gpu < 0 for gpu in self.gpu_ids):
            raise ValueError("GPU ids must be non-negative")


def build_plan(config: WorkflowConfig, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Return an executable plan without starting processes or touching GPUs."""

    config.validate()
    root = (repo_root or Path.cwd()).resolve()
    if config.kind is WorkflowKind.RL_VERIFY:
        command = [str(root / _CANONICAL_RL_ENTRYPOINT), "gate"]
        env = {
            "EXPECTED_RECORDS": str(config.question_count),
            "PROMPTS_PER_UPDATE": str(config.prompts_per_update),
            "GROUP_SIZE": str(config.group_size),
            "OPTIMIZER_STEPS": str(config.optimizer_updates or 1),
            "RL_MECHANISM": config.mechanism,
        }
    elif config.kind is WorkflowKind.RL_FULL:
        command = [str(root / _CANONICAL_RL_ENTRYPOINT), "run"]
        env = {
            "EXPECTED_RECORDS": str(config.question_count),
            "PROMPTS_PER_UPDATE": str(config.prompts_per_update),
            "GROUP_SIZE": str(config.group_size),
            "OPTIMIZER_STEPS": str(config.optimizer_updates or 200),
            "RL_MECHANISM": config.mechanism,
        }
    elif config.kind is WorkflowKind.EVALUATE:
        dataset_path = config.dataset_path or config.dataset
        output_dir = config.output_dir or str(root / "evaluation_plan")
        command = [
            sys.executable,
            str(root / _CANONICAL_EVAL_ENTRYPOINT),
            "--dataset", dataset_path,
            "--split", config.split,
            "--checkpoint", config.checkpoint,
            "--gpu-ids", ",".join(str(gpu) for gpu in config.gpu_ids),
            "--output-dir", output_dir,
            "--partition", config.partition,
        ]
        env = {
            "DATASET": config.dataset,
            "SPLIT": config.split,
            "GPU_IDS": ",".join(str(gpu) for gpu in config.gpu_ids),
        }
    else:  # guarded by validate; retained for type checkers
        command = [str(root / _CANONICAL_SFT_ENTRYPOINT)]
        env = {}
    return {
        "workflow": config.kind.value,
        "frozen": config.kind is WorkflowKind.SFT_FROZEN,
        "command": command,
        "environment": env,
        "dataset": config.dataset,
        "split": config.split,
        "mechanism": config.mechanism,
        "checkpoint": config.checkpoint,
    }
