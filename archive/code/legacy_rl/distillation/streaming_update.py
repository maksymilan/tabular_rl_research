"""Memory-bounded backward passes for one streaming small-batch update."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from src.rl.distillation.losses import branch_dpo_loss, weighted_logprob
from src.rl.distillation.masks import PolicyTokenMaskUnavailable, policy_token_weights


@dataclass(frozen=True)
class ScoredTurn:
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    weights: tuple[float, ...]
    old_logps: Any
    teacher_logps: Any


def _release_cuda_cache(backend) -> None:
    """Release inactive generation/prefill blocks before the next long graph.

    Streaming training alternates inference-only generation with checkpointed
    forwards.  On a 24 GiB card the freed blocks can remain split across CUDA's
    cache even though only one response graph is live.  Emptying that cache is
    semantics-neutral and keeps the turn-by-turn backward contract real.
    """
    cuda = getattr(backend.torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()


def score_turns_for_dense_opd(backend, turns: Sequence, *, teacher_adapter: str):
    scored = []
    for turn in turns:
        try:
            weights = policy_token_weights(backend.tokenizer, turn.response_ids)
        except PolicyTokenMaskUnavailable:
            # Malformed carriers have no trustworthy JSON-token boundary.  They
            # remain visible to verifier-backed repair but are not relabeled as
            # arbitrary token-level negatives.
            continue
        teacher = backend.score_response(
            teacher_adapter,
            turn.prompt_ids,
            turn.response_ids,
            requires_grad=False,
        )
        # The HF rollout callback already performs an exact raw-policy prefill
        # after generation.  Reuse it as pi_old; parameters have not changed
        # until the whole small batch is updated.
        old = backend.torch.tensor(
            turn.sampling_logprobs,
            dtype=backend.torch.float32,
            device=teacher.device,
        )
        scored.append(
            ScoredTurn(
                prompt_ids=tuple(turn.prompt_ids),
                response_ids=tuple(turn.response_ids),
                weights=weights,
                old_logps=old,
                teacher_logps=teacher,
            )
        )
    return scored


def mean_teacher_logprob(backend, turns: Sequence, *, teacher_adapter: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for turn in turns:
        try:
            weights = policy_token_weights(backend.tokenizer, turn.response_ids)
        except PolicyTokenMaskUnavailable:
            continue
        logps = backend.score_response(
            teacher_adapter,
            turn.prompt_ids,
            turn.response_ids,
            requires_grad=False,
        )
        weight_tensor = backend.torch.tensor(weights, device=logps.device)
        numerator += float((logps * weight_tensor).sum().detach().cpu())
        denominator += sum(weights)
    if denominator <= 0.0:
        raise ValueError("teacher route has no active student policy tokens")
    return numerator / denominator


def backward_dense_opd(
    backend,
    scored_turns: Sequence[ScoredTurn],
    *,
    scale: float,
    advantage_clip: float,
) -> dict[str, float]:
    """Backpropagate turn by turn so a long trajectory graph is never retained."""
    if scale < 0.0 or advantage_clip <= 0.0:
        raise ValueError("invalid dense OPD scale or advantage clip")
    total_weight = sum(sum(turn.weights) for turn in scored_turns)
    if total_weight <= 0.0:
        raise ValueError("dense OPD trajectory has no active policy tokens")
    advantage_sum = 0.0
    loss_sum = 0.0
    for turn in scored_turns:
        _release_cuda_cache(backend)
        current = backend.score_response(
            backend.STUDENT,
            turn.prompt_ids,
            turn.response_ids,
            requires_grad=True,
        )
        weights = backend.torch.tensor(turn.weights, device=current.device)
        advantage = backend.torch.clamp(
            turn.teacher_logps - turn.old_logps,
            -advantage_clip,
            advantage_clip,
        ).detach()
        loss = -scale * (weights * advantage * current).sum() / total_weight
        loss.backward()
        loss_sum += float(loss.detach().cpu())
        advantage_sum += float((weights * advantage).sum().detach().cpu())
        del current, weights, advantage, loss
        _release_cuda_cache(backend)
    return {
        "dense_loss": loss_sum,
        "mean_advantage": advantage_sum / total_weight,
        "active_tokens": total_weight,
    }


def branch_score_no_grad(backend, turns: Sequence, *, adapter: str) -> float:
    numerator = 0.0
    denominator = 0.0
    for turn in turns:
        try:
            weights = policy_token_weights(backend.tokenizer, turn.response_ids)
        except PolicyTokenMaskUnavailable:
            continue
        logps = backend.score_response(
            adapter,
            turn.prompt_ids,
            turn.response_ids,
            requires_grad=False,
        )
        weight_tensor = backend.torch.tensor(weights, device=logps.device)
        numerator += float((logps * weight_tensor).sum().detach().cpu())
        denominator += sum(weights)
    if denominator <= 0.0:
        raise ValueError("branch has no active policy tokens")
    return numerator / denominator


def branch_policy_token_count(backend, turns: Sequence) -> int:
    total = 0
    for turn in turns:
        try:
            total += sum(
                weight > 0.0
                for weight in policy_token_weights(backend.tokenizer, turn.response_ids)
            )
        except PolicyTokenMaskUnavailable:
            continue
    return total


def backward_branch_dpo_surrogate(
    backend,
    chosen_turns: Sequence,
    rejected_turns: Sequence,
    *,
    reference_adapter: str,
    beta: float,
    scale: float,
) -> dict[str, float]:
    """Backpropagate exact DPO gradient without retaining all branch graphs.

    The scalar derivative is evaluated at the pre-update student.  Each branch
    turn is then re-forwarded and backpropagated with that detached coefficient.
    This is mathematically the same first-order gradient as sequence-level DPO
    while fitting long tool trajectories on a single 24 GiB card.
    """
    if not chosen_turns or not rejected_turns:
        raise ValueError("repair DPO requires two non-empty continuations")
    policy_chosen = branch_score_no_grad(backend, chosen_turns, adapter=backend.STUDENT)
    policy_rejected = branch_score_no_grad(backend, rejected_turns, adapter=backend.STUDENT)
    ref_chosen = branch_score_no_grad(backend, chosen_turns, adapter=reference_adapter)
    ref_rejected = branch_score_no_grad(backend, rejected_turns, adapter=reference_adapter)
    z = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
    if z >= 0.0:
        tail = math.exp(-z)
        coefficient = -beta * tail / (1.0 + tail)
    else:
        coefficient = -beta / (1.0 + math.exp(z))

    for sign, turns in ((1.0, chosen_turns), (-1.0, rejected_turns)):
        weighted_turns = []
        for turn in turns:
            try:
                weights = policy_token_weights(backend.tokenizer, turn.response_ids)
            except PolicyTokenMaskUnavailable:
                continue
            weighted_turns.append((turn, weights))
        total_weight = sum(sum(weights) for _, weights in weighted_turns)
        for turn, weights in weighted_turns:
            _release_cuda_cache(backend)
            logps = backend.score_response(
                backend.STUDENT,
                turn.prompt_ids,
                turn.response_ids,
                requires_grad=True,
            )
            weight_tensor = backend.torch.tensor(weights, device=logps.device)
            surrogate = scale * coefficient * sign * (logps * weight_tensor).sum() / total_weight
            surrogate.backward()
            del logps, weight_tensor, surrogate
            _release_cuda_cache(backend)

    actual = branch_dpo_loss(
        policy_chosen,
        policy_rejected,
        ref_chosen,
        ref_rejected,
        beta=beta,
    )
    return {
        "repair_dpo_loss": float(actual.detach().cpu()),
        "policy_margin": policy_chosen - policy_rejected,
        "reference_margin": ref_chosen - ref_rejected,
        "dpo_gradient_coefficient": coefficient,
    }
