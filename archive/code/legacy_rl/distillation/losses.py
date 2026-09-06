"""Loss primitives for streaming OPD plus verified branch preference."""
from __future__ import annotations

from typing import Sequence


def _torch():
    import torch

    return torch


def _as_float_tensor(value, *, like=None):
    torch = _torch()
    if torch.is_tensor(value):
        return value.float()
    device = like.device if like is not None else None
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def weighted_logprob(token_logps, token_weights, *, reduction: str = "mean"):
    """Reduce token log-probabilities without scoring masked context tokens."""
    logps = _as_float_tensor(token_logps)
    weights = _as_float_tensor(token_weights, like=logps)
    if logps.shape != weights.shape:
        raise ValueError(f"logps/weights shape mismatch: {logps.shape} != {weights.shape}")
    denominator = weights.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        raise ValueError("weighted log-probability requires positive total weight")
    total = (logps * weights).sum()
    if reduction == "sum":
        return total
    if reduction == "mean":
        return total / denominator
    raise ValueError(f"unsupported weighted log-probability reduction: {reduction}")


def masked_mopd_loss(
    current_token_logps,
    old_student_token_logps,
    teacher_token_logps,
    token_weights,
    *,
    advantage_clip: float = 5.0,
):
    """Sampled-token MOPD policy-gradient loss.

    ``old_student_token_logps`` and ``teacher_token_logps`` must come from
    frozen, pre-update forwards over the exact student sample.  Their
    difference is clipped and detached.  Gradients flow only through current
    student log-probabilities.
    """
    torch = _torch()
    current = _as_float_tensor(current_token_logps)
    old = _as_float_tensor(old_student_token_logps, like=current)
    teacher = _as_float_tensor(teacher_token_logps, like=current)
    weights = _as_float_tensor(token_weights, like=current)
    if not (current.shape == old.shape == teacher.shape == weights.shape):
        raise ValueError("MOPD tensors must have identical shapes")
    if advantage_clip <= 0.0:
        raise ValueError("advantage_clip must be positive")
    denominator = weights.sum()
    if float(denominator.detach().cpu()) <= 0.0:
        raise ValueError("MOPD requires at least one active policy token")
    advantage = torch.clamp(teacher - old, -advantage_clip, advantage_clip).detach()
    return -((weights * advantage * current).sum() / denominator)


def branch_dpo_loss(
    policy_chosen_logprob,
    policy_rejected_logprob,
    reference_chosen_logprob,
    reference_rejected_logprob,
    *,
    beta: float = 0.1,
):
    """Sequence/branch-level DPO for two verifier-compared continuations.

    Each scalar may be a sum or mean over all model-authored policy tokens in a
    causal multi-turn branch.  Both branches must start from the same replayed
    harness prefix; their later prompts may contain their own factual harness
    observations.
    """
    torch = _torch()
    if beta <= 0.0:
        raise ValueError("DPO beta must be positive")
    chosen = _as_float_tensor(policy_chosen_logprob)
    rejected = _as_float_tensor(policy_rejected_logprob, like=chosen)
    ref_chosen = _as_float_tensor(reference_chosen_logprob, like=chosen)
    ref_rejected = _as_float_tensor(reference_rejected_logprob, like=chosen)
    policy_margin = chosen - rejected
    reference_margin = (ref_chosen - ref_rejected).detach()
    logits = beta * (policy_margin - reference_margin)
    return -torch.nn.functional.logsigmoid(logits)


def branch_logprob_from_turns(
    turn_token_logps: Sequence,
    turn_token_weights: Sequence,
    *,
    reduction: str = "mean",
):
    """Reduce a causal multi-turn continuation into one branch score."""
    torch = _torch()
    if len(turn_token_logps) != len(turn_token_weights) or not turn_token_logps:
        raise ValueError("branch turns and masks must be non-empty and aligned")
    logps = torch.cat([_as_float_tensor(value) for value in turn_token_logps])
    weights = torch.cat(
        [_as_float_tensor(value, like=logps) for value in turn_token_weights]
    )
    return weighted_logprob(logps, weights, reduction=reduction)
