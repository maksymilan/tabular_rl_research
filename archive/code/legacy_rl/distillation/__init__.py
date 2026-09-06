"""Streaming multi-teacher policy distillation for the table agent.

The package is deliberately isolated from the existing process-RL and Action-DPO
entrypoints.  A formal run initializes one student from SFT2 exactly once, then
alternates causal student rollouts and small-batch updates.  Frozen teacher
adapters are scorers and repair generators; they are never optimizer targets.
"""

from .losses import branch_dpo_loss, masked_mopd_loss, weighted_logprob
from .masks import PolicyTokenMaskUnavailable, policy_token_weights
from .repair_plan import (
    RepairCandidate,
    RepairSelection,
    parallel_repair_anchor_turns,
    select_verified_repair,
)

__all__ = [
    "PolicyTokenMaskUnavailable",
    "RepairCandidate",
    "RepairSelection",
    "branch_dpo_loss",
    "masked_mopd_loss",
    "parallel_repair_anchor_turns",
    "policy_token_weights",
    "select_verified_repair",
    "weighted_logprob",
]
