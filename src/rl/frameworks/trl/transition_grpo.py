#!/usr/bin/env python3
"""TRL GRPO adapter for exact transition-level table-agent rollouts."""
from __future__ import annotations

import json
import gc
import math
import os
import time
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import torch
    import torch.nn.functional as F
    from accelerate.utils import is_peft_model
    from trl import GRPOTrainer
    from trl.trainer.utils import pad
except ImportError as exc:  # pragma: no cover - exercised in the GPU environment
    torch = None
    GRPOTrainer = object
    _TRL_IMPORT_ERROR = exc
else:
    _TRL_IMPORT_ERROR = None

from rl.frameworks.trl.transition_batch import (
    build_transition_microbatch_ranges,
    retain_policy_contributing_updates,
)
from rl.frameworks.trl.state_action_ambiguity import CREDIT_ASSIGNMENTS
from rl.frameworks.trl.gradient_conflict import GradientConflictRecorder
from rl.frameworks.trl.tool_loss_mask import (
    ToolMaskUnavailable,
    tool_token_loss_mask,
)
from rl.frameworks.trl.trajectory_ranking import (
    RANK_SCORE_REDUCTIONS,
    RANK_SCORE_SCOPES,
    RANK_SCORE_TOKENS,
    RANK_UPDATE_SCOPES,
    build_pairs_from_transition_metadata,
    build_trajectory_pairs,
    rank_transition_selected,
)
from rl.frameworks.trl.mechanism import RLMechanism


def _dummy_reward(completions, **kwargs):
    """Satisfy GRPOTrainer's constructor; transition rewards are prepared internally."""
    return [0.0] * len(completions)


def _adapter_trainable_parameters(model, adapter_name: str) -> list[str]:
    marker = f".{adapter_name}."
    return [
        name
        for name, parameter in model.named_parameters()
        if marker in name and parameter.requires_grad
    ]


@dataclass(frozen=True)
class _TransitionMicrobatchPlan:
    """Immutable row/column layout shared by all forwards for one batch.

    The transition tensors are padded once at the batch level.  Every compact
    microbatch can therefore derive its non-padding columns from the Python
    lengths instead of scanning CUDA masks and synchronizing the host for each
    slice.  ``trim_bounds`` contains ``(prompt_start, completion_end)`` for
    each range; ``None`` preserves the old behavior for an all-padding side.
    """

    ranges: tuple[tuple[int, int], ...]
    trim_bounds: tuple[tuple[int | None, int | None], ...]
    prompt_lengths: tuple[int, ...]
    completion_lengths: tuple[int, ...]
    prompt_width: int
    completion_width: int


@contextmanager
def use_frozen_reference_adapter(model, adapter_name: str):
    """Select a reference without PEFT's default trainability side effect."""

    previous = model.active_adapter
    if not isinstance(previous, str):
        raise RuntimeError(
            f"the trainable policy must have one named active adapter, got {previous!r}"
        )
    model.set_adapter(adapter_name, inference_mode=True)
    trainable = _adapter_trainable_parameters(model, adapter_name)
    if trainable:
        model.set_adapter(previous, inference_mode=False)
        raise RuntimeError(
            "the KL reference became trainable while being selected: "
            f"{trainable[:5]}"
        )
    try:
        yield
    finally:
        model.set_adapter(previous, inference_mode=False)
        trainable = _adapter_trainable_parameters(model, adapter_name)
        if trainable:
            raise RuntimeError(
                "the KL reference remained trainable after restoring the policy: "
                f"{trainable[:5]}"
            )


class TransitionGRPOTrainer(GRPOTrainer):
    """Use TRL's clipped tokenwise policy loss over exact causal assistant turns."""

    def __init__(
        self,
        *args,
        rollout_collector,
        reward_mode: str,
        trainer_sharding: str = "replicated",
        train_turns: str = "all",
        trainable_part: str = "all",
        rank_loss_coefficient: float = 0.0,
        rank_beta: float = 0.1,
        rank_score_tokens: str = "all",
        rank_score_scope: str = "full_trajectory",
        rank_score_reduction: str = "sum_tokens",
        rank_update_scope: str = "full_trajectory",
        policy_loss_coefficient: float = 1.0,
        policy_reduction: str = "transition_mean",
        credit_assignment: str = "trajectory",
        error_penalty: float = 1.0,
        result_advantage_profile: str = "stored",
        clean_advantage_weight: float = 0.25,
        span_balance_alpha: float | None = None,
        record_gradient_conflicts: bool = False,
        gradient_conflict_dir: Path | None = None,
        gradient_conflict_save_vectors: bool = False,
        gradient_conflict_max_transitions: int = 0,
        gradient_conflict_carrier_only: bool = False,
        reference_adapter_name: str | None = None,
        mechanism: RLMechanism | None = None,
        transition_micro_batch_size: int = 2,
        transition_micro_batch_tokens: int = 0,
        **kwargs,
    ):
        if _TRL_IMPORT_ERROR is not None:
            raise ImportError(
                "TransitionGRPOTrainer requires trl>=0.29, torch, and accelerate"
            ) from _TRL_IMPORT_ERROR
        if transition_micro_batch_size < 1:
            raise ValueError("transition_micro_batch_size must be positive")
        if transition_micro_batch_tokens < 0:
            raise ValueError("transition_micro_batch_tokens must be non-negative")
        if gradient_conflict_max_transitions < 0:
            raise ValueError("gradient_conflict_max_transitions must be non-negative")
        if trainable_part not in {"all", "tool_only"}:
            raise ValueError(f"unsupported trainable_part: {trainable_part}")
        if rank_loss_coefficient < 0:
            raise ValueError("rank_loss_coefficient must be non-negative")
        if policy_loss_coefficient < 0:
            raise ValueError("policy_loss_coefficient must be non-negative")
        if policy_loss_coefficient == 0.0 and rank_loss_coefficient == 0.0:
            raise ValueError("at least one policy or rank loss must be enabled")
        if policy_reduction not in {
            "transition_mean",
            "trajectory_mean",
            "trajectory_token_mean",
        }:
            raise ValueError(f"unsupported policy_reduction: {policy_reduction}")
        if credit_assignment not in CREDIT_ASSIGNMENTS:
            raise ValueError(f"unsupported credit_assignment: {credit_assignment}")
        if credit_assignment == "saam-strict" and reward_mode != "result-only":
            raise ValueError("saam-strict requires result-only terminal rewards")
        if credit_assignment == "saam-strict" and train_turns != "all":
            raise ValueError("saam-strict requires all causal turns")
        if credit_assignment == "saam-strict" and rank_loss_coefficient != 0.0:
            raise ValueError("saam-strict pilot does not mix a trajectory ranking loss")
        if credit_assignment in {
            "saam-asymmetric-error",
            "saam-asymmetric-error-no-mask",
        } and reward_mode != "result-only":
            raise ValueError(f"{credit_assignment} requires result-only terminal rewards")
        if credit_assignment in {
            "saam-asymmetric-error",
            "saam-asymmetric-error-no-mask",
        } and train_turns != "all":
            raise ValueError(f"{credit_assignment} requires all causal turns")
        if credit_assignment in {
            "saam-asymmetric-error",
            "saam-asymmetric-error-no-mask",
        } and rank_loss_coefficient != 0.0:
            raise ValueError(
                f"{credit_assignment} does not mix a trajectory ranking loss"
            )
        if not math.isfinite(error_penalty) or error_penalty <= 0.0:
            raise ValueError("error_penalty must be finite and positive")
        if result_advantage_profile not in {
            "stored",
            "correctness-primary-clean-secondary",
            "class-conditional-routing",
            "smc-mode-concentration",
        }:
            raise ValueError(
                "unsupported result_advantage_profile: "
                f"{result_advantage_profile}"
            )
        if not math.isfinite(clean_advantage_weight) or not 0.0 <= clean_advantage_weight < 1.0:
            raise ValueError("clean_advantage_weight must be finite and in [0, 1)")
        if result_advantage_profile != "stored" and reward_mode != "result-only":
            raise ValueError(
                "result advantage profiles require result-only terminal rewards"
            )
        if span_balance_alpha is not None:
            if not math.isfinite(span_balance_alpha) or not 0.0 <= span_balance_alpha <= 1.0:
                raise ValueError("span_balance_alpha must be in [0, 1]")
            if trainable_part != "all":
                raise ValueError(
                    "span_balance_alpha is only defined for the full response carrier"
                )
        if record_gradient_conflicts and rank_loss_coefficient != 0.0:
            raise ValueError("gradient conflict recording currently requires rank loss=0")
        if rank_beta <= 0:
            raise ValueError("rank_beta must be positive")
        if rank_score_tokens not in RANK_SCORE_TOKENS:
            raise ValueError(f"unsupported rank_score_tokens: {rank_score_tokens}")
        if rank_score_scope not in RANK_SCORE_SCOPES:
            raise ValueError(f"unsupported rank_score_scope: {rank_score_scope}")
        if rank_score_reduction not in RANK_SCORE_REDUCTIONS:
            raise ValueError(
                f"unsupported rank_score_reduction: {rank_score_reduction}"
            )
        if rank_update_scope not in RANK_UPDATE_SCOPES:
            raise ValueError(f"unsupported rank_update_scope: {rank_update_scope}")
        if (
            rank_update_scope == "conservative_legal"
            and rank_score_tokens != "tool_only"
        ):
            raise ValueError(
                "conservative_legal rank updates require tool_only rank scores"
            )
        if (
            rank_score_scope == "conservative_legal"
            and rank_update_scope != "conservative_legal"
        ):
            raise ValueError(
                "conservative_legal rank scores require conservative_legal updates"
            )
        if rank_update_scope == "dense_outcome" and rank_score_tokens != "all":
            raise ValueError("dense_outcome rank updates require all-token rank scores")
        if (
            rank_score_scope == "dense_outcome"
            and rank_update_scope != "dense_outcome"
        ):
            raise ValueError(
                "dense_outcome rank scores require dense_outcome updates"
            )
        self.rollout_collector = rollout_collector
        self.mechanism = mechanism or RLMechanism.from_legacy_args(
            reward_mode=reward_mode,
            result_advantage_profile=result_advantage_profile,
            clean_advantage_weight=clean_advantage_weight,
            policy_reduction=policy_reduction,
            credit_assignment=credit_assignment,
            error_penalty=error_penalty,
        )
        self.transition_reward_mode = self.mechanism.reward_mode
        if trainer_sharding not in {"replicated", "fsdp"}:
            raise ValueError(f"unsupported trainer_sharding: {trainer_sharding}")
        self.trainer_sharding = trainer_sharding
        self.transition_train_turns = train_turns
        self.transition_trainable_part = trainable_part
        self.rank_loss_coefficient = rank_loss_coefficient
        self.rank_beta = rank_beta
        self.rank_score_tokens = rank_score_tokens
        self.rank_score_scope = rank_score_scope
        self.rank_score_reduction = rank_score_reduction
        self.rank_update_scope = rank_update_scope
        self.policy_loss_coefficient = policy_loss_coefficient
        self.policy_reduction = policy_reduction
        self.credit_assignment = self.mechanism.credit_assignment
        self.error_penalty = float(self.mechanism.error_penalty)
        self.result_advantage_profile = self.mechanism.result_advantage_profile
        self.clean_advantage_weight = float(self.mechanism.clean_advantage_weight)
        self.policy_reduction = self.mechanism.policy_reduction
        self.span_balance_alpha = (
            None
            if self.mechanism.span_balance_alpha is None
            else float(self.mechanism.span_balance_alpha)
        )
        self.record_gradient_conflicts = bool(record_gradient_conflicts)
        self.gradient_conflict_dir = gradient_conflict_dir
        self.gradient_conflict_save_vectors = bool(gradient_conflict_save_vectors)
        self.gradient_conflict_max_transitions = int(gradient_conflict_max_transitions)
        self.gradient_conflict_carrier_only = bool(gradient_conflict_carrier_only)
        self.gradient_conflict_recorder = None
        self.reference_adapter_name = reference_adapter_name
        self.transition_micro_batch_size = int(transition_micro_batch_size)
        self.transition_micro_batch_tokens = int(transition_micro_batch_tokens)
        # Delaying FSDP gradient synchronization until the last transition
        # microbatch is mathematically equivalent to synchronizing every
        # microbatch, while removing one reduce-scatter per microbatch. Keep it
        # opt-in until the A100 fixed-pool gate confirms the extra local
        # gradient residency fits safely.
        self.fsdp_microbatch_no_sync = (
            trainer_sharding == "fsdp"
            and os.environ.get("RL_FSDP_MICROBATCH_NO_SYNC", "0") == "1"
        )
        # Inference-only old-policy/reference scoring can keep full FSDP
        # parameters resident for the complete pass. This trades peak memory
        # for avoiding repeated layer all-gathers and is independently gated.
        self.fsdp_inference_full_params = (
            trainer_sharding == "fsdp"
            and os.environ.get("RL_FSDP_INFERENCE_FULL_PARAMS", "0") == "1"
        )
        kwargs.setdefault("reward_funcs", _dummy_reward)
        super().__init__(*args, **kwargs)
        # TRL creates a policy-copy adapter named ``ref`` for any PEFT model with
        # beta > 0. This trainer uses the separately loaded immutable SFT adapter,
        # so discard the unused copy after upstream initialization.
        if (
            self.reference_adapter_name
            and self.reference_adapter_name != "ref"
            and "ref" in getattr(self.model, "peft_config", {})
        ):
            self.model.delete_adapter("ref")
        self.offline_rollout_pool = bool(
            getattr(self.rollout_collector, "is_offline", False)
        )
        if not self.offline_rollout_pool and (
            not self.use_vllm or self.vllm_mode != "server"
        ):
            raise ValueError(
                "TransitionGRPOTrainer requires use_vllm=True and vllm_mode='server'"
            )
        if self.args.steps_per_generation != 1:
            raise ValueError(
                "variable transition batches currently require steps_per_generation=1"
            )
        if self.beta != 0.0:
            if not is_peft_model(self.model):
                raise ValueError(
                    "nonzero KL requires a named frozen PEFT reference adapter"
                )
            if not self.reference_adapter_name:
                raise ValueError(
                    "nonzero KL requires reference_adapter_name; refusing base-model fallback"
                )
            if self.reference_adapter_name not in self.model.peft_config:
                raise ValueError(
                    "configured frozen KL reference adapter is not loaded: "
                    f"{self.reference_adapter_name}"
                )
        if self.credit_assignment == "saam-strict" and self.beta != 0.0:
            raise ValueError("saam-strict pilot requires KL beta=0")
        if self.record_gradient_conflicts:
            if self.beta != 0.0:
                raise ValueError("gradient conflict recording requires KL beta=0")
            if self.policy_loss_coefficient == 0.0:
                raise ValueError("gradient conflict recording requires an enabled policy loss")
            if int(self.args.gradient_accumulation_steps) != 1:
                raise ValueError(
                    "gradient conflict recording requires gradient_accumulation_steps=1"
                )
            if self.accelerator.is_main_process:
                self.gradient_conflict_recorder = GradientConflictRecorder(
                    self.model,
                    self.gradient_conflict_dir
                    or (Path(self.args.output_dir) / "gradient_conflicts"),
                    save_vectors=self.gradient_conflict_save_vectors,
                )

    def _span_balanced_mask(
        self,
        response_ids,
    ) -> tuple[float, ...]:
        """Return a per-token mask whose two spans receive fixed total mass.

        TRL's GRPO loss takes a weighted mean when ``tool_mask`` is supplied.  A
        weight of ``(1-alpha)/R`` on every reasoning token and ``alpha/T`` on
        every tool token therefore implements a convex combination of the two
        span means without changing the transition-level SAAM coefficient.  A
        malformed carrier is kept on the ordinary full-response mean so the
        diagnostic does not silently drop a sampled transition.
        """

        alpha = self.span_balance_alpha
        if alpha is None:
            return tuple(1.0 for _ in response_ids)
        try:
            raw_tool_mask = tool_token_loss_mask(
                self.processing_class,
                response_ids,
            )
        except ToolMaskUnavailable:
            return tuple(1.0 for _ in response_ids)
        tool_count = sum(raw_tool_mask)
        reason_count = len(raw_tool_mask) - tool_count
        if tool_count < 1 or reason_count < 1:
            return tuple(1.0 for _ in response_ids)
        reason_weight = (1.0 - alpha) / reason_count
        tool_weight = alpha / tool_count
        return tuple(
            tool_weight if active else reason_weight
            for active in raw_tool_mask
        )

    def _counterfactual_policy_gradient(
        self,
        model,
        inputs: dict[str, Any],
        *,
        policy_normalization_transitions: int,
        positive: bool | None,
        carrier: str = "full",
    ):
        """Backpropagate one sign/carrier partition using the exact objective.

        ``carrier=full`` is the ordinary response objective.  ``reason`` and
        ``tool`` are diagnostic counterfactuals only: they select the causal
        response span with a mask while preserving the same prompt, attention
        context, advantages, clipping, and importance correction.  The caller
        restores the real optimizer gradient after these probes, so this never
        changes the training update.
        """

        if carrier not in {"full", "reason", "tool"}:
            raise ValueError(f"unsupported gradient carrier: {carrier}")

        microbatch_plan = self._build_transition_microbatch_plan(inputs)
        for range_index, (start, end) in enumerate(microbatch_plan.ranges):
            micro_inputs = self._slice_batch(
                inputs,
                start,
                end,
                trim_bounds=microbatch_plan.trim_bounds[range_index],
            )
            advantages = micro_inputs["advantages"]
            if positive is None:
                selected = torch.ones_like(advantages, dtype=torch.bool)
            elif positive:
                selected = advantages > 0.0
            else:
                selected = advantages < 0.0
            micro_inputs["advantages"] = torch.where(
                selected,
                advantages,
                torch.zeros_like(advantages),
            )
            if carrier != "full":
                carrier_tool_mask = micro_inputs.get("carrier_tool_mask")
                carrier_valid = micro_inputs.get("carrier_mask_valid")
                if carrier_tool_mask is None or carrier_valid is None:
                    raise RuntimeError(
                        "reason/tool gradient recording requires carrier masks"
                    )
                valid = carrier_valid.to(dtype=micro_inputs["completion_mask"].dtype)
                if carrier == "tool":
                    selected_mask = carrier_tool_mask
                else:
                    selected_mask = (
                        micro_inputs["completion_attention_mask"] - carrier_tool_mask
                    ).clamp_min(0)
                # Invalid carriers are excluded from both span probes.  Treating
                # a malformed response as all-reasoning would create a spurious
                # reason gradient and hide carrier-format failures.
                micro_inputs["tool_mask"] = selected_mask * valid.unsqueeze(1)
            else:
                # Full-response diagnostics must remain independent of the
                # optimization carrier (tool-only/span-balanced experiments may
                # otherwise be mislabeled as full-response measurements).
                micro_inputs["tool_mask"] = micro_inputs["completion_attention_mask"]
            synchronize = range_index == len(microbatch_plan.ranges) - 1
            with self._microbatch_sync_context(
                model,
                synchronize=synchronize,
            ):
                with self.compute_loss_context_manager():
                    micro_loss = super()._compute_loss(model, micro_inputs)
                weight = (end - start) / policy_normalization_transitions
                scaled_loss = micro_loss * weight * self.policy_loss_coefficient
                self.accelerator.backward(scaled_loss)

    @staticmethod
    def _select_gradient_probe_inputs(
        inputs: dict[str, Any],
        max_transitions: int,
    ) -> tuple[dict[str, Any], int, int]:
        """Select a deterministic bounded subset for diagnostic probe passes.

        The optimizer still consumes the complete transition batch.  This helper
        only bounds the six counterfactual passes used for gradient diagnostics,
        preventing a long rollout batch from multiplying activation memory and
        runtime.
        """

        total = int(inputs["completion_ids"].shape[0])
        if max_transitions <= 0 or total <= max_transitions:
            return inputs, total, total
        device = inputs["completion_ids"].device
        indices = torch.linspace(
            0,
            total - 1,
            steps=max_transitions,
            device=device,
            dtype=torch.float64,
        ).round().to(dtype=torch.long)
        indices = torch.unique_consecutive(indices)
        selected: dict[str, Any] = {}
        for key, value in inputs.items():
            if (
                torch.is_tensor(value)
                and value.ndim > 0
                and int(value.shape[0]) == total
            ):
                selected[key] = value.index_select(0, indices)
            else:
                selected[key] = value
        return selected, total, int(indices.numel())

    def _record_current_gradient_conflicts(
        self,
        model,
        inputs: dict[str, Any],
        *,
        policy_normalization_transitions: int,
        total: int,
    ) -> None:
        """Record full-response and reason/tool gradient geometry before clipping."""

        recorder = self.gradient_conflict_recorder
        if not self.record_gradient_conflicts or recorder is None:
            return
        if self.gradient_conflict_carrier_only:
            self._record_carrier_only_gradients(
                model,
                inputs,
                policy_normalization_transitions=policy_normalization_transitions,
                total=total,
            )
            return
        # Keep the optimizer gradient off GPU while the diagnostic probes run.
        # The probes are diagnostic-only, so a CPU copy is sufficient and avoids
        # adding a full trainable-gradient clone to the probe peak.
        saved_gradients = [
            (
                parameter,
                parameter.grad.detach().cpu()
                if parameter.grad is not None
                else None,
            )
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        if recorder is not None:
            combined, combined_layers = recorder.capture(model)
        else:
            combined = combined_layers = None
        started = time.perf_counter()
        probe_inputs, source_total, probe_total = self._select_gradient_probe_inputs(
            inputs,
            self.gradient_conflict_max_transitions,
        )
        model.zero_grad(set_to_none=True)
        self._counterfactual_policy_gradient(
            model,
            probe_inputs,
            policy_normalization_transitions=probe_total,
            positive=True,
            carrier="full",
        )
        if recorder is not None:
            positive, positive_layers = recorder.capture(model)
        else:
            positive = positive_layers = None
        model.zero_grad(set_to_none=True)
        self._counterfactual_policy_gradient(
            model,
            probe_inputs,
            policy_normalization_transitions=probe_total,
            positive=False,
            carrier="full",
        )
        if recorder is not None:
            negative, negative_layers = recorder.capture(model)
        else:
            negative = negative_layers = None
        span_gradients = {}
        span_layers = {}
        for carrier in ("reason", "tool"):
            model.zero_grad(set_to_none=True)
            self._counterfactual_policy_gradient(
                model,
                probe_inputs,
                policy_normalization_transitions=probe_total,
                positive=True,
                carrier=carrier,
            )
            if recorder is not None:
                span_positive, span_positive_layers = recorder.capture(model)
            else:
                span_positive = span_positive_layers = None
            model.zero_grad(set_to_none=True)
            self._counterfactual_policy_gradient(
                model,
                probe_inputs,
                policy_normalization_transitions=probe_total,
                positive=False,
                carrier=carrier,
            )
            if recorder is not None:
                span_negative, span_negative_layers = recorder.capture(model)
            else:
                span_negative = span_negative_layers = None
            if recorder is not None:
                span_gradients[carrier] = (span_positive, span_negative)
                span_layers[carrier] = {
                    "positive": span_positive_layers,
                    "negative": span_negative_layers,
                }
        model.zero_grad(set_to_none=True)
        for parameter, gradient in saved_gradients:
            parameter.grad = (
                gradient.to(device=parameter.device)
                if gradient is not None
                else None
            )
        if recorder is None:
            return
        summary = recorder.record(
            step=self._step + 1,
            combined=combined,
            positive=positive,
            negative=negative,
            span_gradients=span_gradients,
            layers={
                "combined": combined_layers,
                "positive": positive_layers,
                "negative": negative_layers,
                "span": span_layers,
            },
            metadata={
                "stage": "pre_clip",
                "credit_assignment": self.credit_assignment,
                "error_penalty": self.error_penalty,
                "transition_count": probe_total,
                "source_transition_count": source_total,
                "policy_normalization_transitions": probe_total,
                "sampled_probe": probe_total < source_total,
                "carrier_valid_transitions": int(
                    probe_inputs["carrier_mask_valid"].sum().item()
                ),
                "carrier_invalid_transitions": int(
                    (~probe_inputs["carrier_mask_valid"].bool()).sum().item()
                ),
                "reason_tokens": int(
                    (
                        (
                            probe_inputs["completion_attention_mask"]
                            - probe_inputs["carrier_tool_mask"]
                        ).clamp_min(0)
                        * probe_inputs["carrier_mask_valid"].bool().unsqueeze(1)
                    ).sum().item()
                ),
                "tool_tokens": int(
                    (
                        probe_inputs["carrier_tool_mask"]
                        * probe_inputs["carrier_mask_valid"].bool().unsqueeze(1)
                    ).sum().item()
                ),
            },
        )
        elapsed = time.perf_counter() - started
        self._metrics["train"]["gradient_conflict/positive_negative_cosine"].append(
            float(summary["positive_negative_cosine"])
        )
        self._metrics["train"]["gradient_conflict/positive_negative_conflict_mass"].append(
            float(summary["positive_negative_conflict_mass"])
        )
        self._metrics["train"]["gradient_conflict/positive_combined_cosine"].append(
            float(summary["positive_combined_cosine"])
        )
        self._metrics["train"]["gradient_conflict/negative_combined_cosine"].append(
            float(summary["negative_combined_cosine"])
        )
        span_summary = summary.get("span_gradients", {})
        for carrier in ("reason", "tool"):
            values = span_summary.get(carrier)
            if not values:
                continue
            prefix = f"gradient_conflict/{carrier}"
            self._metrics["train"][f"{prefix}_positive_negative_cosine"].append(
                float(values["positive_negative_cosine"])
            )
            self._metrics["train"][f"{prefix}_positive_negative_conflict_mass"].append(
                float(values["positive_negative_conflict_mass"])
            )
            self._metrics["train"][f"{prefix}_positive_norm"].append(
                float(values["positive_norm"])
            )
            self._metrics["train"][f"{prefix}_negative_norm"].append(
                float(values["negative_norm"])
            )
            self._metrics["train"][f"{prefix}_positive_full_cosine"].append(
                float(values["positive_full_cosine"])
            )
            self._metrics["train"][f"{prefix}_negative_full_cosine"].append(
                float(values["negative_full_cosine"])
            )
        if "reason" in span_summary:
            self._metrics["train"]["gradient_conflict/reason_tool_positive_cosine"].append(
                float(span_summary["reason"]["positive_tool_cosine"] or 0.0)
            )
            self._metrics["train"]["gradient_conflict/reason_tool_negative_cosine"].append(
                float(span_summary["reason"]["negative_tool_cosine"] or 0.0)
            )
        self._metrics["train"]["gradient_conflict/record_seconds"].append(elapsed)
        # Release CPU probe vectors and allocator cache before the next rollout.
        del saved_gradients, probe_inputs, span_gradients, span_layers
        del combined, positive, negative, combined_layers, positive_layers, negative_layers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _record_carrier_only_gradients(
        self,
        model,
        inputs: dict[str, Any],
        *,
        policy_normalization_transitions: int,
        total: int,
    ) -> None:
        """Record only actual reason/tool carrier gradients.

        The full gradient is already present from the real optimizer objective.
        Two all-advantage masked probes isolate reason and tool carriers; no
        positive/negative sign probes are performed in this mode.
        """

        recorder = self.gradient_conflict_recorder
        if recorder is None:
            return
        saved_gradients = [
            (
                parameter,
                parameter.grad.detach().cpu()
                if parameter.grad is not None
                else None,
            )
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        combined, combined_layers = recorder.capture(model)
        started = time.perf_counter()
        probe_inputs, source_total, probe_total = self._select_gradient_probe_inputs(
            inputs,
            self.gradient_conflict_max_transitions,
        )
        carriers: dict[str, Any] = {}
        carrier_layers: dict[str, list[dict[str, Any]]] = {}
        for carrier in ("reason", "tool"):
            model.zero_grad(set_to_none=True)
            self._counterfactual_policy_gradient(
                model,
                probe_inputs,
                policy_normalization_transitions=probe_total,
                positive=None,
                carrier=carrier,
            )
            carriers[carrier], carrier_layers[carrier] = recorder.capture(model)
        model.zero_grad(set_to_none=True)
        for parameter, gradient in saved_gradients:
            parameter.grad = (
                gradient.to(device=parameter.device)
                if gradient is not None
                else None
            )
        summary = recorder.record_carrier_gradients(
            step=self._step + 1,
            combined=combined,
            carriers=carriers,
            layers={"combined": combined_layers, "carriers": carrier_layers},
            metadata={
                "stage": "pre_clip",
                "credit_assignment": self.credit_assignment,
                "error_penalty": self.error_penalty,
                "mode": "carrier-only",
                "transition_count": probe_total,
                "source_transition_count": source_total,
                "policy_normalization_transitions": probe_total,
                "sampled_probe": probe_total < source_total,
                "carrier_valid_transitions": int(
                    probe_inputs["carrier_mask_valid"].sum().item()
                ),
                "carrier_invalid_transitions": int(
                    (~probe_inputs["carrier_mask_valid"].bool()).sum().item()
                ),
                "reason_tokens": int(
                    (
                        (
                            probe_inputs["completion_attention_mask"]
                            - probe_inputs["carrier_tool_mask"]
                        ).clamp_min(0)
                        * probe_inputs["carrier_mask_valid"].bool().unsqueeze(1)
                    ).sum().item()
                ),
                "tool_tokens": int(
                    (
                        probe_inputs["carrier_tool_mask"]
                        * probe_inputs["carrier_mask_valid"].bool().unsqueeze(1)
                    ).sum().item()
                ),
            },
        )
        elapsed = time.perf_counter() - started
        self._metrics["train"]["gradient_conflict/carrier_only_record_seconds"].append(
            elapsed
        )
        for carrier in ("reason", "tool"):
            values = summary["carrier_gradients"][carrier]
            prefix = f"gradient_conflict/{carrier}"
            self._metrics["train"][f"{prefix}_norm"].append(float(values["norm"]))
            self._metrics["train"][f"{prefix}_combined_cosine"].append(
                float(values["combined_cosine"])
            )
            self._metrics["train"][f"{prefix}_norm_over_combined"].append(
                float(values["norm_over_combined"])
            )
        self._metrics["train"]["gradient_conflict/reason_tool_cosine"].append(
            float(summary["carrier_gradients"].get("reason_tool_cosine", 0.0))
        )
        del saved_gradients, probe_inputs, carriers, carrier_layers
        del combined, combined_layers
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _slice_batch(
        inputs: dict[str, Any],
        start: int,
        end: int,
        *,
        trim_bounds: tuple[int | None, int | None] | None = None,
    ) -> dict[str, Any]:
        """Slice rows and remove padding columns that are empty in this microbatch.

        The frozen pool contains short early actions and much longer late actions.  Retaining the
        global question-level padding width for every two-transition forward wastes most of the
        attention FLOPs.  Removing columns that are padding for *all* rows is exactly equivalent:
        token order, masks, labels, advantages, and normalization are unchanged.
        """
        sliced = {}
        batch_size = inputs["completion_ids"].shape[0]
        for key, value in inputs.items():
            if (
                torch.is_tensor(value)
                and value.ndim > 0
                and value.shape[0] == batch_size
            ):
                sliced[key] = value[start:end]
            else:
                sliced[key] = value
        prompt_width = int(sliced["prompt_ids"].shape[1])
        if trim_bounds is None:
            prompt_support = sliced["prompt_mask"].bool().any(dim=0)
            prompt_start = (
                int(prompt_support.nonzero(as_tuple=False)[0].item())
                if bool(prompt_support.any())
                else None
            )
        else:
            prompt_start = trim_bounds[0]
        if prompt_start is not None:
            for key in ("prompt_ids", "prompt_mask"):
                sliced[key] = sliced[key][:, prompt_start:prompt_width]

        completion_width = int(sliced["completion_ids"].shape[1])
        if trim_bounds is None:
            completion_support = sliced["completion_attention_mask"].bool().any(dim=0)
            completion_end = (
                int(completion_support.nonzero(as_tuple=False)[-1].item()) + 1
                if bool(completion_support.any())
                else None
            )
        else:
            completion_end = trim_bounds[1]
        if completion_end is not None:
            completion_keys = (
                "completion_ids",
                "completion_attention_mask",
                "completion_mask",
                "tool_mask",
                "carrier_tool_mask",
                "rank_completion_mask",
                "old_per_token_logps",
                "sampling_per_token_logps",
                "importance_sampling_ratio",
                "ref_per_token_logps",
            )
            for key in completion_keys:
                value = sliced.get(key)
                if (
                    torch.is_tensor(value)
                    and value.ndim == 2
                    and int(value.shape[1]) == completion_width
                ):
                    sliced[key] = value[:, :completion_end]
        return sliced

    def _build_transition_microbatch_plan(
        self,
        inputs: dict[str, Any],
        *,
        prompt_lengths: list[int] | tuple[int, ...] | None = None,
        completion_lengths: list[int] | tuple[int, ...] | None = None,
    ) -> _TransitionMicrobatchPlan:
        """Build one reusable row/column plan for all transition forwards.

        ``transition_micro_batch_size`` remains the hard row cap and preserves
        the historical behavior when the token budget is zero.  In dynamic
        mode, rows are expected to have been length-ordered before padding so
        the padded-token estimate is also a useful proxy for actual FLOPs.
        """

        batch_size = int(inputs["completion_ids"].shape[0])
        if prompt_lengths is None:
            prompt_lengths = (
                inputs["prompt_mask"]
                .sum(dim=1)
                .detach()
                .to(device="cpu")
                .tolist()
            )
        if completion_lengths is None:
            completion_lengths = (
                inputs["completion_attention_mask"]
                .sum(dim=1)
                .detach()
                .to(device="cpu")
                .tolist()
            )
        prompt_lengths = tuple(int(value) for value in prompt_lengths)
        completion_lengths = tuple(int(value) for value in completion_lengths)
        if len(prompt_lengths) != batch_size or len(completion_lengths) != batch_size:
            raise RuntimeError(
                "transition length metadata is not aligned with the batch"
            )
        ranges = tuple(
            build_transition_microbatch_ranges(
                prompt_lengths,
                completion_lengths,
                max_rows=self.transition_micro_batch_size,
                token_budget=self.transition_micro_batch_tokens,
            )
        )
        prompt_width = int(inputs["prompt_ids"].shape[1])
        completion_width = int(inputs["completion_ids"].shape[1])
        trim_bounds = []
        for start, end in ranges:
            max_prompt_length = max(prompt_lengths[start:end], default=0)
            max_completion_length = max(completion_lengths[start:end], default=0)
            trim_bounds.append(
                (
                    prompt_width - max_prompt_length
                    if max_prompt_length
                    else None,
                    max_completion_length if max_completion_length else None,
                )
            )
        return _TransitionMicrobatchPlan(
            ranges=ranges,
            trim_bounds=tuple(trim_bounds),
            prompt_lengths=prompt_lengths,
            completion_lengths=completion_lengths,
            prompt_width=prompt_width,
            completion_width=completion_width,
        )

    def _transition_microbatch_ranges(
        self,
        inputs: dict[str, Any],
    ) -> list[tuple[int, int]]:
        """Return deterministic fixed-row or padded-token transition slices."""

        return list(self._build_transition_microbatch_plan(inputs).ranges)

    @staticmethod
    def _pad_transition_rows(
        rows,
        *,
        padding_value,
        padding_side: str,
        device,
    ):
        """Pad on CPU once, then transfer one dense tensor to the trainer device.

        Creating every variable-length row directly on CUDA and padding the list
        there causes one allocator/copy pair per transition.  The values and
        padding policy are unchanged when the same rows are padded on CPU first;
        only the number of host-to-device transfers changes.
        """

        padded = pad(
            rows,
            padding_value=padding_value,
            padding_side=padding_side,
        )
        return padded.to(device=device)

    @staticmethod
    def _transition_length_key(update) -> tuple[int, int, int, str, int]:
        """Return a deterministic key that keeps similarly sized rows adjacent."""

        prompt_length = len(update.prompt_ids)
        completion_length = len(update.response_ids)
        return (
            prompt_length + completion_length,
            prompt_length,
            completion_length,
            str(update.trajectory_id),
            int(update.turn_index),
        )

    @staticmethod
    def _sort_transition_batch_by_length(
        inputs: dict[str, Any],
    ) -> tuple[dict[str, Any], tuple[int, ...], tuple[int, ...]]:
        """Restore the length order before token-budget microbatch planning.

        TRL shuffles every generation-batch tensor before ``training_step``.  That
        is harmless for a fixed row cap, but a greedy padded-token planner is
        order-sensitive: the same rows can produce a different number of
        microbatches after the shuffle.  FSDP ranks must execute the same number
        of model collectives, so recover the deterministic order established by
        the rollout-side scheduler before building the plan.  This only permutes
        rows; it does not change any loss value or normalization term.
        """

        completion_ids = inputs["completion_ids"]
        batch_size = int(completion_ids.shape[0])
        prompt_lengths = tuple(
            int(value)
            for value in inputs["prompt_mask"]
            .sum(dim=1)
            .detach()
            .to(device="cpu")
            .tolist()
        )
        completion_lengths = tuple(
            int(value)
            for value in inputs["completion_attention_mask"]
            .sum(dim=1)
            .detach()
            .to(device="cpu")
            .tolist()
        )
        if len(prompt_lengths) != batch_size or len(completion_lengths) != batch_size:
            raise RuntimeError("transition length metadata is not aligned with the batch")
        order = sorted(
            range(batch_size),
            key=lambda index: (
                prompt_lengths[index] + completion_lengths[index],
                prompt_lengths[index],
                completion_lengths[index],
                index,
            ),
        )
        sorted_prompt_lengths = tuple(prompt_lengths[index] for index in order)
        sorted_completion_lengths = tuple(
            completion_lengths[index] for index in order
        )
        if order == list(range(batch_size)):
            return inputs, sorted_prompt_lengths, sorted_completion_lengths

        indices = torch.tensor(
            order,
            device=completion_ids.device,
            dtype=torch.long,
        )
        sorted_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            if (
                torch.is_tensor(value)
                and value.ndim > 0
                and int(value.shape[0]) == batch_size
            ):
                sorted_inputs[key] = value.index_select(0, indices)
            else:
                sorted_inputs[key] = value
        return sorted_inputs, sorted_prompt_lengths, sorted_completion_lengths

    def _assert_fsdp_microbatch_alignment(
        self,
        microbatch_plan: _TransitionMicrobatchPlan,
        *,
        stage: str,
    ) -> None:
        """Fail closed before a model forward if ranks disagree on slot count."""

        if self.trainer_sharding != "fsdp" or self.accelerator.num_processes < 2:
            return
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            return
        device = self.accelerator.device
        count = torch.tensor(
            [len(microbatch_plan.ranges)],
            device=device,
            dtype=torch.long,
        )
        minimum = count.clone()
        maximum = count.clone()
        dist.all_reduce(minimum, op=dist.ReduceOp.MIN)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if int(minimum.item()) != int(maximum.item()):
            raise RuntimeError(
                "FSDP transition microbatch schedule mismatch before "
                f"{stage}: local={len(microbatch_plan.ranges)}, "
                f"global_min={int(minimum.item())}, "
                f"global_max={int(maximum.item())}"
            )

    def _importance_sampling_ratio(
        self,
        old_per_token_logps,
        sampling_per_token_logps,
        completion_mask,
    ):
        logps_diff = (old_per_token_logps - sampling_per_token_logps) * completion_mask
        sequence_level = self.vllm_importance_sampling_mode in {
            "sequence_mask",
            "sequence_truncate",
        }
        if sequence_level:
            logps_diff = logps_diff.sum(dim=-1, keepdim=True)
        ratio = torch.exp(logps_diff)
        if self.vllm_importance_sampling_mode in {
            "sequence_truncate",
            "token_truncate",
        }:
            ratio = torch.clamp(ratio, max=self.vllm_importance_sampling_cap)
        elif self.vllm_importance_sampling_mode in {"sequence_mask", "token_mask"}:
            ratio = ratio.masked_fill(
                ratio > self.vllm_importance_sampling_cap,
                value=0.0,
            )
        else:
            raise ValueError(
                "unsupported vLLM importance-sampling mode: "
                f"{self.vllm_importance_sampling_mode}"
            )
        return ratio

    def _importance_sampling_diagnostics(
        self,
        old_per_token_logps,
        sampling_per_token_logps,
        completion_mask,
        applied_ratio,
    ) -> dict[str, float]:
        """Summarize the vLLM/trainer policy mismatch on supported tokens.

        QLoRA training and BF16 vLLM rollout do not share identical numerical base
        weights. The importance correction is therefore part of the baseline's
        objective, not merely an implementation detail. Persist both the raw log-ratio
        mismatch and the applied capped ratio so an apparent policy update cannot be
        attributed to an unaudited inference/training mismatch.
        """
        support = completion_mask.bool()
        token_log_ratio = old_per_token_logps - sampling_per_token_logps
        sequence_level = self.vllm_importance_sampling_mode in {
            "sequence_mask",
            "sequence_truncate",
        }
        if sequence_level:
            active = support.any(dim=-1)
            raw_log_ratio = (token_log_ratio * support).sum(dim=-1)[active]
            applied = applied_ratio.squeeze(-1)[active]
        else:
            raw_log_ratio = token_log_ratio[support]
            applied = applied_ratio[support]
        if raw_log_ratio.numel() == 0:
            return {
                "log_ratio_abs_mean": 0.0,
                "log_ratio_abs_max": 0.0,
                "applied_ratio_min": 0.0,
                "applied_ratio_mean": 0.0,
                "applied_ratio_max": 0.0,
                "cap_exceeded_fraction": 0.0,
            }
        cap_exceeded = raw_log_ratio > math.log(
            self.vllm_importance_sampling_cap
        )
        return {
            "log_ratio_abs_mean": float(raw_log_ratio.abs().mean()),
            "log_ratio_abs_max": float(raw_log_ratio.abs().max()),
            "applied_ratio_min": float(applied.min()),
            "applied_ratio_mean": float(applied.mean()),
            "applied_ratio_max": float(applied.max()),
            "cap_exceeded_fraction": float(cap_exceeded.float().mean()),
        }

    def _transition_token_logps_compact(
        self,
        model,
        inputs,
        *,
        microbatch_plan: _TransitionMicrobatchPlan | None = None,
    ):
        """Evaluate transition logprobs in compact, length-bucketed microbatches."""
        microbatch_plan = microbatch_plan or self._build_transition_microbatch_plan(
            inputs
        )
        global_completion_width = microbatch_plan.completion_width
        rows = []
        preallocated = None
        for range_index, (start, end) in enumerate(microbatch_plan.ranges):
            micro_inputs = self._slice_batch(
                inputs,
                start,
                end,
                trim_bounds=microbatch_plan.trim_bounds[range_index],
            )
            input_ids = torch.cat(
                [micro_inputs["prompt_ids"], micro_inputs["completion_ids"]],
                dim=1,
            )
            attention_mask = torch.cat(
                [
                    micro_inputs["prompt_mask"],
                    micro_inputs["completion_attention_mask"],
                ],
                dim=1,
            )
            local_completion_width = int(micro_inputs["completion_ids"].shape[1])
            per_token_logps, _ = self._get_per_token_logps_and_entropies(
                model,
                input_ids,
                attention_mask,
                local_completion_width,
                batch_size=end - start,
            )
            # Old-policy and reference scoring are under no_grad.  Write those
            # compact results directly into the final tensor to avoid one
            # F.pad allocation plus a second cat allocation per microbatch.
            # Keep the concatenation path when gradients are enabled because the
            # ranking backward path needs the original autograd graph.
            if not torch.is_grad_enabled():
                if preallocated is None:
                    preallocated = per_token_logps.new_zeros(
                        (
                            int(inputs["completion_ids"].shape[0]),
                            global_completion_width,
                        )
                    )
                preallocated[start:end, :local_completion_width].copy_(
                    per_token_logps
                )
            else:
                if local_completion_width < global_completion_width:
                    per_token_logps = F.pad(
                        per_token_logps,
                        (0, global_completion_width - local_completion_width),
                        value=0.0,
                    )
                rows.append(per_token_logps)
        if preallocated is not None:
            return preallocated
        return torch.cat(rows, dim=0)

    @contextmanager
    def _microbatch_sync_context(self, model, *, synchronize: bool):
        """Delay FSDP gradient synchronization until the final microbatch.

        ``Accelerator.no_sync`` only delegates to a top-level ``no_sync``
        attribute. A PEFT wrapper can instead contain FSDP roots below it, so
        locate those roots explicitly and enter one context per root. The last
        microbatch remains synchronized and therefore produces the same
        reduced gradient as the eager path, up to floating-point reduction
        order.
        """

        if synchronize or not self.fsdp_microbatch_no_sync:
            yield
            return
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except ImportError:  # pragma: no cover - only reached without FSDP
            with self.accelerator.no_sync(model):
                yield
            return

        roots = []

        def visit(module, inside_fsdp: bool = False):
            is_fsdp = isinstance(module, FSDP)
            if is_fsdp and not inside_fsdp:
                roots.append(module)
            for child in module.children():
                visit(child, inside_fsdp or is_fsdp)

        visit(model)
        if not roots:
            # Keep the helper usable for a DDP-wrapped smoke test as well.
            with self.accelerator.no_sync(model):
                yield
            return
        with ExitStack() as stack:
            for root in roots:
                stack.enter_context(root.no_sync())
            yield

    @staticmethod
    @contextmanager
    def _summon_fsdp_full_params(
        model,
        *,
        rank0_only: bool = True,
        offload_to_cpu: bool = True,
    ):
        """Expose FSDP shards for one bounded, collective-safe pass."""

        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        except ImportError:
            yield
            return
        # ``summon_full_params(recurse=True)`` already traverses every nested
        # FSDP unit.  Calling it once per unit causes repeated all-gathers
        # (and, with NCCL/P2P, can even deadlock while a parent gather is still
        # outstanding).  Find only FSDP roots so each shard is gathered once.
        fsdp_roots = []

        def visit(module, inside_fsdp: bool = False):
            is_fsdp = isinstance(module, FSDP)
            if is_fsdp and not inside_fsdp:
                fsdp_roots.append(module)
            for child in module.children():
                visit(child, inside_fsdp or is_fsdp)

        visit(model)
        if not fsdp_roots:
            yield
            return
        with ExitStack() as stack:
            for module in fsdp_roots:
                stack.enter_context(
                    FSDP.summon_full_params(
                        module,
                        recurse=True,
                        writeback=False,
                        rank0_only=rank0_only,
                        offload_to_cpu=offload_to_cpu,
                    )
                )
            yield

    @contextmanager
    def _inference_full_params_context(self, model):
        """Keep full BF16 params resident during one no-grad scoring pass."""

        if not self.fsdp_inference_full_params:
            yield
            return
        with self._summon_fsdp_full_params(
            model,
            rank0_only=False,
            offload_to_cpu=False,
        ):
            yield

    def _sync_qlora_weights_to_vllm(self, model, *, stream: bool = True) -> int:
        """Stream merged base+LoRA weights without mutating training weights."""
        import bitsandbytes as bnb

        config = model.peft_config["default"]
        if getattr(config, "bias", "none") != "none":
            raise NotImplementedError(
                "QLoRA vLLM synchronization does not support trainable bias"
            )
        if getattr(config, "modules_to_save", None):
            raise NotImplementedError(
                "QLoRA vLLM synchronization does not support modules_to_save"
            )

        targets = []
        for module_name, module in model.named_modules():
            base_layer = getattr(module, "base_layer", None)
            lora_a = getattr(module, "lora_A", {})
            lora_b = getattr(module, "lora_B", {})
            if base_layer is not None and "default" in lora_a and "default" in lora_b:
                targets.append((module_name, module, base_layer))
        if not targets:
            return 0
        if not stream:
            return len(targets)

        client = self.vllm_generation.vllm_client
        with torch.no_grad():
            for module_name, module, base_layer in targets:
                if isinstance(base_layer, bnb.nn.Linear4bit):
                    quant_state = getattr(base_layer.weight, "quant_state", None)
                    if quant_state is None:
                        raise RuntimeError(
                            f"4-bit layer has no quantization state: {module_name}"
                        )
                    base_weight = bnb.functional.dequantize_4bit(
                        base_layer.weight.data,
                        quant_state,
                    )
                else:
                    base_weight = base_layer.weight.detach()
                delta_weight = module.get_delta_weight("default")
                merged_weight = (
                    base_weight.to(dtype=torch.bfloat16)
                    + delta_weight.to(dtype=torch.bfloat16)
                ).contiguous()
                # TRL's NCCL communicator requires the source tensor on the
                # trainer CUDA device.  FSDP full-param gathering is offloaded
                # to CPU to avoid materializing a second full decoder shard;
                # copy one merged layer back to GPU at a time.
                if not merged_weight.is_cuda:
                    merged_weight = merged_weight.to(
                        device=torch.device("cuda", torch.cuda.current_device()),
                        non_blocking=True,
                    )
                name = f"{module_name}.weight"
                name = name.removeprefix("base_model.model.")
                name = name.replace("._fsdp_wrapped_module.", ".")
                name = name.removeprefix("_fsdp_wrapped_module.")
                name = name.replace(".base_layer", "")
                client.update_named_param(name, merged_weight)
                del merged_weight, delta_weight, base_weight
        client.reset_prefix_cache()
        return len(targets)

    def _sync_policy_weights(self, mode: str) -> None:
        # In DDP every rank owns a complete QLoRA replica, but only rank 0 may
        # stream weights to the shared vLLM server. Concurrent parameter-update
        # requests from multiple trainers race inside the server and can leave
        # the rollout policy half-updated. The barrier also guarantees that all
        # ranks start their next rollout from the same policy version.
        distributed = self.accelerator.num_processes > 1
        if distributed and self.trainer_sharding == "fsdp":
            started = time.perf_counter()
            unwrapped = self.accelerator.unwrap_model(self.model)
            # FSDP all-gather is collective.  Non-main ranks participate in the
            # summon but never issue vLLM parameter updates.
            with self._summon_fsdp_full_params(unwrapped):
                synced_qlora_layers = (
                    self._sync_qlora_weights_to_vllm(
                        unwrapped,
                        stream=self.accelerator.is_main_process,
                    )
                    if is_peft_model(unwrapped)
                    else 0
                )
            if self.accelerator.is_main_process:
                self._metrics[mode]["rollout/weight_sync_seconds"].append(
                    time.perf_counter() - started
                )
                self._metrics[mode]["rollout/qlora_layers_synced"].append(
                    float(synced_qlora_layers)
                )
            self.accelerator.wait_for_everyone()
            return
        if distributed and not self.accelerator.is_main_process:
            self.accelerator.wait_for_everyone()
            return
        started = time.perf_counter()
        unwrapped = self.accelerator.unwrap_model(self.model)
        synced_qlora_layers = (
            self._sync_qlora_weights_to_vllm(unwrapped)
            if is_peft_model(unwrapped)
            else 0
        )
        if synced_qlora_layers == 0:
            self.vllm_generation.sync_weights()
        self._metrics[mode]["rollout/weight_sync_seconds"].append(
            time.perf_counter() - started
        )
        self._metrics[mode]["rollout/qlora_layers_synced"].append(
            float(synced_qlora_layers)
        )
        if distributed:
            self.accelerator.wait_for_everyone()

    def _generate_and_score_completions(
        self,
        inputs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        mode = "train" if self.model.training else "eval"
        if mode != "train":
            raise RuntimeError(
                "held-out evaluation must use src/eval rollout runners, not trainer eval"
            )

        if (
            getattr(self.rollout_collector, "requires_policy_sync", True)
            and self.state.global_step != self._last_loaded_step
        ):
            self._sync_policy_weights(mode)
            self._last_loaded_step = self.state.global_step

        preparation_started = time.perf_counter()
        episodes = self.rollout_collector.collect(inputs, self)
        collect_finished = time.perf_counter()
        updates = self.mechanism.build_updates(
            episodes,
            train_turns=self.transition_train_turns,
        )
        if self.offline_rollout_pool:
            # Bucketing is within one immutable question/K group.  It changes no objective term,
            # while making adjacent transition microbatches similar in sequence length.
            updates.sort(
                key=lambda update: (
                    len(update.prompt_ids) + len(update.response_ids),
                    str(update.trajectory_id),
                    update.turn_index,
                )
            )
        original_transition_count = len(updates)
        original_trajectory_count = len(
            {update.trajectory_id for update in updates}
        )
        tiny_nonzero_advantages = [
            update
            for update in updates
            if update.advantage != 0.0 and abs(update.advantage) < 1e-12
        ]
        if self.transition_reward_mode == "result-only" and tiny_nonzero_advantages:
            examples = sorted(
                {
                    int(update.example_index)
                    for update in tiny_nonzero_advantages
                }
            )
            raise RuntimeError(
                "result-only advantage normalization produced numerical "
                f"pseudo-signal for {len(tiny_nonzero_advantages)} transitions "
                f"on examples {examples}; refusing policy forward/backward"
            )
        original_nonzero_advantage_count = sum(
            update.advantage != 0.0 for update in updates
        )
        saam_audit = None
        removed_absolute_policy_coefficient_fraction = 0.0
        absolute_policy_coefficient_delta_fraction = 0.0
        if self.credit_assignment == "saam-strict":
            vanilla_effective_advantages = self.mechanism.effective_advantages(
                updates, transition_count=original_transition_count,
                trajectory_count=original_trajectory_count,
            )
            updates, saam_audit = self.mechanism.apply_credit(episodes, updates)
            masked_effective_advantages = self.mechanism.effective_advantages(
                updates, transition_count=original_transition_count,
                trajectory_count=original_trajectory_count,
            )
            for update, vanilla_value, masked_value in zip(
                updates,
                vanilla_effective_advantages,
                masked_effective_advantages,
                strict=True,
            ):
                if update.advantage != 0.0 and masked_value != vanilla_value:
                    raise RuntimeError(
                        "SAAM changed an unmasked policy coefficient; refusing "
                        "a renormalized credit update"
                    )
            vanilla_mass = sum(abs(value) for value in vanilla_effective_advantages)
            masked_mass = sum(abs(value) for value in masked_effective_advantages)
            if vanilla_mass > 0.0:
                removed_absolute_policy_coefficient_fraction = (
                    vanilla_mass - masked_mass
                ) / vanilla_mass
        elif self.credit_assignment in {
            "saam-asymmetric-error",
            "saam-asymmetric-error-no-mask",
        }:
            vanilla_effective_advantages = self.mechanism.effective_advantages(
                updates, transition_count=original_transition_count,
                trajectory_count=original_trajectory_count,
            )
            updates, saam_audit = self.mechanism.apply_credit(episodes, updates)
            masked_effective_advantages = self.mechanism.effective_advantages(
                updates, transition_count=original_transition_count,
                trajectory_count=original_trajectory_count,
            )
            vanilla_mass = sum(abs(value) for value in vanilla_effective_advantages)
            masked_mass = sum(abs(value) for value in masked_effective_advantages)
            coefficient_delta = sum(
                abs(masked - vanilla)
                for vanilla, masked in zip(
                    vanilla_effective_advantages,
                    masked_effective_advantages,
                    strict=True,
                )
            )
            if vanilla_mass > 0.0:
                removed_absolute_policy_coefficient_fraction = max(
                    0.0, vanilla_mass - masked_mass
                ) / vanilla_mass
                absolute_policy_coefficient_delta_fraction = (
                    coefficient_delta / vanilla_mass
                )
        post_credit_nonzero_advantage_count = sum(
            update.advantage != 0.0 for update in updates
        )
        updates, zero_advantage_transitions_dropped = (
            retain_policy_contributing_updates(
                updates,
                policy_loss_coefficient=self.policy_loss_coefficient,
                rank_loss_coefficient=self.rank_loss_coefficient,
                kl_beta=self.beta,
            )
        )
        if self.transition_micro_batch_tokens > 0:
            # Dynamic packing is most effective when adjacent rows have similar
            # prompt/completion lengths. Transition metadata remains attached
            # to each row, so this changes only batch order, not the objective.
            updates.sort(key=self._transition_length_key)
        tool_loss_masks = None
        skipped_tool_mask_transitions = 0
        if self.transition_trainable_part == "tool_only":
            retained_updates = []
            retained_masks = []
            for update in updates:
                try:
                    mask = tool_token_loss_mask(
                        self.processing_class,
                        update.response_ids,
                    )
                except ToolMaskUnavailable:
                    skipped_tool_mask_transitions += 1
                    continue
                retained_updates.append(update)
                retained_masks.append(mask)
            updates = retained_updates
            tool_loss_masks = retained_masks
        if self.span_balance_alpha is not None:
            if tool_loss_masks is not None:
                raise RuntimeError(
                    "span balancing cannot be combined with the tool-only carrier"
                )
            tool_loss_masks = [
                self._span_balanced_mask(update.response_ids)
                for update in updates
            ]
            # Span balancing returns fractional per-token weights.  Keep the
            # values in floating point all the way into TRL's weighted-mean
            # loss; converting them to an integer mask silently turns every
            # valid weight below one into zero and produces a no-op update.
            invalid_masks = [
                index
                for index, mask in enumerate(tool_loss_masks)
                if not mask
                or any(not math.isfinite(float(weight)) for weight in mask)
                or sum(float(weight) for weight in mask) <= 0.0
            ]
            if invalid_masks:
                raise RuntimeError(
                    "span-balanced loss mask has no positive finite weight for "
                    f"transitions {invalid_masks[:8]}"
                )
        if not updates:
            raise RuntimeError(
                "the rollout batch produced no transitions with trainable tokens; "
                "refusing to advance optimizer or scheduler state"
            )

        # Preserve the actual think/tool boundary separately from the training
        # carrier only when the corresponding diagnostic is enabled.  It is not
        # part of the policy objective; parsing it for every transition in the
        # normal training run is pure host work (and used to duplicate the span
        # mask decoding).
        carrier_tool_masks: list[tuple[int, ...]] | None = None
        carrier_mask_valid: list[bool] | None = None
        if self.record_gradient_conflicts:
            carrier_tool_masks = []
            carrier_mask_valid = []
            for update in updates:
                try:
                    carrier_tool_masks.append(
                        tool_token_loss_mask(
                            self.processing_class,
                            update.response_ids,
                        )
                    )
                    carrier_mask_valid.append(True)
                except (ToolMaskUnavailable, ValueError):
                    carrier_tool_masks.append((0,) * len(update.response_ids))
                    carrier_mask_valid.append(False)

        transition_trajectory_indices = None
        trajectory_pairs = []
        if self.rank_loss_coefficient != 0.0:
            trajectory_to_index: dict[str, int] = {}
            trajectory_example_indices = []
            trajectory_correct = []
            transition_trajectory_indices = []
            for update in updates:
                trajectory_index = trajectory_to_index.get(update.trajectory_id)
                if trajectory_index is None:
                    trajectory_index = len(trajectory_to_index)
                    trajectory_to_index[update.trajectory_id] = trajectory_index
                    trajectory_example_indices.append(update.example_index)
                    trajectory_correct.append(update.trajectory_correct)
                transition_trajectory_indices.append(trajectory_index)
            trajectory_pairs = build_trajectory_pairs(
                trajectory_example_indices,
                trajectory_correct,
            )

        device = self.accelerator.device
        prompt_ids_list = [
            torch.tensor(update.prompt_ids, dtype=torch.long)
            for update in updates
        ]
        completion_ids_list = [
            torch.tensor(update.response_ids, dtype=torch.long)
            for update in updates
        ]
        sampling_logps_list = [
            torch.tensor(update.sampling_logprobs, dtype=torch.float32)
            for update in updates
        ]
        prompt_mask_list = [torch.ones_like(ids) for ids in prompt_ids_list]
        completion_attention_mask_list = [
            torch.ones_like(ids) for ids in completion_ids_list
        ]
        completion_loss_mask_list = (
            [
                torch.tensor(mask, dtype=torch.float32)
                for mask in tool_loss_masks
            ]
            if tool_loss_masks is not None
            else completion_attention_mask_list
        )
        rank_loss_masks = None
        skipped_rank_tool_masks = 0
        if self.rank_loss_coefficient != 0.0:
            rank_loss_masks = []
            if self.rank_score_tokens == "tool_only" and tool_loss_masks is not None:
                # Process+Rank tool-only training selects the same response token
                # carrier for both losses.  Parsing every action twice is pure
                # host overhead.
                rank_loss_masks = list(tool_loss_masks)
            elif self.rank_score_tokens == "tool_only":
                for update in updates:
                    try:
                        rank_loss_masks.append(
                            tool_token_loss_mask(
                                self.processing_class,
                                update.response_ids,
                            )
                        )
                    except ToolMaskUnavailable:
                        skipped_rank_tool_masks += 1
                        rank_loss_masks.append((0,) * len(update.response_ids))
            else:
                rank_loss_masks = [
                    tuple(1 for _ in update.response_ids)
                    for update in updates
                ]

        prompt_ids = self._pad_transition_rows(
            prompt_ids_list,
            padding_value=self.pad_token_id,
            padding_side="left",
            device=device,
        )
        prompt_mask = self._pad_transition_rows(
            prompt_mask_list,
            padding_value=0,
            padding_side="left",
            device=device,
        )
        completion_ids = self._pad_transition_rows(
            completion_ids_list,
            padding_value=self.pad_token_id,
            padding_side="right",
            device=device,
        )
        completion_attention_mask = self._pad_transition_rows(
            completion_attention_mask_list,
            padding_value=0,
            padding_side="right",
            device=device,
        )
        completion_loss_mask = self._pad_transition_rows(
            completion_loss_mask_list,
            padding_value=0,
            padding_side="right",
            device=device,
        )
        carrier_tool_mask = None
        if carrier_tool_masks is not None:
            carrier_tool_mask = self._pad_transition_rows(
                [
                    torch.tensor(mask, dtype=torch.long)
                    for mask in carrier_tool_masks
                ],
                padding_value=0,
                padding_side="right",
                device=device,
            )
        rank_completion_mask = None
        if rank_loss_masks is not None:
            rank_completion_mask = self._pad_transition_rows(
                [
                    torch.tensor(mask, dtype=torch.long)
                    for mask in rank_loss_masks
                ],
                padding_value=0,
                padding_side="right",
                device=device,
            )
        sampling_per_token_logps = self._pad_transition_rows(
            sampling_logps_list,
            padding_value=0.0,
            padding_side="right",
            device=device,
        )
        forward_inputs = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_attention_mask": completion_attention_mask,
        }
        microbatch_plan = self._build_transition_microbatch_plan(
            forward_inputs,
            prompt_lengths=[len(update.prompt_ids) for update in updates],
            completion_lengths=[len(update.response_ids) for update in updates],
        )
        self._assert_fsdp_microbatch_alignment(
            microbatch_plan,
            stage="old-policy scoring",
        )
        microbatch_ranges = microbatch_plan.ranges
        prompt_lengths = microbatch_plan.prompt_lengths
        completion_lengths = microbatch_plan.completion_lengths
        padded_costs = [
            (end - start)
            * (
                max(prompt_lengths[start:end])
                + max(completion_lengths[start:end])
            )
            for start, end in microbatch_ranges
        ]
        self._metrics[mode]["rollout/micro_batch_count"].append(
            float(len(microbatch_ranges))
        )
        self._metrics[mode]["rollout/micro_batch_mean_rows"].append(
            float(len(updates) / len(microbatch_ranges))
        )
        self._metrics[mode]["rollout/micro_batch_max_padded_tokens"].append(
            float(max(padded_costs, default=0))
        )
        self._metrics[mode]["rollout/micro_batch_mean_padded_tokens"].append(
            float(sum(padded_costs) / len(padded_costs))
        )
        old_policy_started = time.perf_counter()
        # ``inference_mode`` creates inference tensors that FSDP's
        # ``summon_full_params`` cannot attach its parameter hooks to.  Keep
        # the compatible no-grad path; it still removes the autograd graph and
        # permits the optional full-param gather to be tested independently.
        with torch.no_grad(), self._inference_full_params_context(self.model):
            old_per_token_logps = self._transition_token_logps_compact(
                self.model,
                forward_inputs,
                microbatch_plan=microbatch_plan,
            )
            if self.beta != 0.0:
                unwrapped = self.accelerator.unwrap_model(self.model)
                if not is_peft_model(unwrapped):
                    raise RuntimeError(
                        "the configured frozen KL reference is not a PEFT model"
                    )
                if self.reference_adapter_name not in unwrapped.peft_config:
                    raise RuntimeError(
                        "the configured frozen KL reference disappeared before scoring: "
                        f"{self.reference_adapter_name}"
                    )
                adapter_context = use_frozen_reference_adapter(
                    unwrapped,
                    self.reference_adapter_name,
                )
                with adapter_context:
                    ref_per_token_logps = self._transition_token_logps_compact(
                        self.model,
                        forward_inputs,
                        microbatch_plan=microbatch_plan,
                    )
            else:
                ref_per_token_logps = None
        old_policy_finished = time.perf_counter()

        importance_sampling_ratio = self._importance_sampling_ratio(
            old_per_token_logps,
            sampling_per_token_logps,
            completion_loss_mask,
        )
        importance_diagnostics = self._importance_sampling_diagnostics(
            old_per_token_logps,
            sampling_per_token_logps,
            completion_loss_mask,
            importance_sampling_ratio,
        )

        effective_advantages = self.mechanism.effective_advantages(
            updates,
            transition_count=original_transition_count,
            trajectory_count=original_trajectory_count,
        )
        advantages = torch.tensor(
            effective_advantages,
            device=device,
            dtype=torch.float32,
        )

        rewards = [float(episode.sample.reward) for episode in episodes]
        step_rewards = [
            reward
            for episode in episodes
            for reward in (episode.sample.step_rewards or [])
        ]
        self._metrics[mode]["rollout/episodes"].append(float(len(episodes)))
        self._metrics[mode]["rollout/transitions"].append(float(len(updates)))
        self._metrics[mode]["rollout/original_transitions"].append(
            float(original_transition_count)
        )
        self._metrics[mode]["rollout/zero_advantage_transitions_dropped"].append(
            float(zero_advantage_transitions_dropped)
        )
        self._metrics[mode]["rollout/correct_rate"].append(
            sum(episode.sample.correct for episode in episodes) / len(episodes)
        )
        self._metrics[mode]["rollout/reward_mean"].append(
            sum(rewards) / len(rewards)
        )
        self._metrics[mode]["rollout/nonzero_advantage_fraction"].append(
            original_nonzero_advantage_count / original_transition_count
        )
        self._metrics[mode]["saam/enabled"].append(
            float(self.credit_assignment != "trajectory")
        )
        self._metrics[mode]["saam/post_mask_nonzero_advantage_fraction"].append(
            post_credit_nonzero_advantage_count / original_transition_count
        )
        self._metrics[mode]["saam/ambiguous_state_action_groups"].append(
            float(saam_audit.ambiguous_state_action_groups if saam_audit else 0)
        )
        self._metrics[mode]["saam/newly_zeroed_transitions"].append(
            float(saam_audit.newly_zeroed_transitions if saam_audit else 0)
        )
        self._metrics[mode]["saam/newly_zeroed_response_tokens"].append(
            float(saam_audit.newly_zeroed_response_tokens if saam_audit else 0)
        )
        self._metrics[mode]["saam/newly_zeroed_initial_transitions"].append(
            float(saam_audit.newly_zeroed_initial_transitions if saam_audit else 0)
        )
        self._metrics[mode]["saam/newly_zeroed_noninitial_transitions"].append(
            float(saam_audit.newly_zeroed_noninitial_transitions if saam_audit else 0)
        )
        self._metrics[mode]["saam/removed_absolute_advantage_fraction"].append(
            float(
                saam_audit.removed_absolute_advantage_mass
                / saam_audit.original_absolute_advantage_mass
                if saam_audit and saam_audit.original_absolute_advantage_mass > 0.0
                else 0.0
            )
        )
        self._metrics[mode][
            "saam/removed_absolute_policy_coefficient_fraction"
        ].append(float(removed_absolute_policy_coefficient_fraction))
        self._metrics[mode][
            "saam/absolute_policy_coefficient_delta_fraction"
        ].append(float(absolute_policy_coefficient_delta_fraction))
        self._metrics[mode]["saam/fully_zeroed_trajectories"].append(
            float(saam_audit.fully_zeroed_trajectories if saam_audit else 0)
        )
        self._metrics[mode]["saam/deterministic_error_transitions"].append(
            float(saam_audit.deterministic_error_transitions if saam_audit else 0)
        )
        self._metrics[mode]["saam/infrastructure_timeout_transitions"].append(
            float(saam_audit.infrastructure_timeout_transitions if saam_audit else 0)
        )
        self._metrics[mode]["saam/timeout_penalized_transitions"].append(
            float(saam_audit.timeout_penalized_transitions if saam_audit else 0)
        )
        self._metrics[mode]["saam/shared_success_correct_kept"].append(
            float(saam_audit.shared_success_correct_kept if saam_audit else 0)
        )
        self._metrics[mode]["saam/shared_success_wrong_suppressed"].append(
            float(saam_audit.shared_success_wrong_suppressed if saam_audit else 0)
        )
        self._metrics[mode]["saam/correct_error_positive_flips"].append(
            float(saam_audit.correct_error_positive_flips if saam_audit else 0)
        )
        self._metrics[mode]["rollout/trainable_token_fraction"].append(
            float(
                completion_loss_mask.sum()
                / completion_attention_mask.sum().clamp_min(1)
            )
        )
        if carrier_mask_valid is not None and carrier_tool_mask is not None:
            self._metrics[mode]["gradient_conflict/carrier_valid_transitions"].append(
                float(sum(carrier_mask_valid))
            )
            self._metrics[mode]["gradient_conflict/carrier_invalid_transitions"].append(
                float(len(carrier_mask_valid) - sum(carrier_mask_valid))
            )
            valid_carrier = torch.tensor(
                carrier_mask_valid, device=device, dtype=torch.bool
            ).unsqueeze(1)
            valid_reason_mask = (
                (completion_attention_mask - carrier_tool_mask).clamp_min(0)
                * valid_carrier
            )
            valid_tool_mask = carrier_tool_mask * valid_carrier
            self._metrics[mode]["gradient_conflict/reason_tokens"].append(
                float(valid_reason_mask.sum())
            )
            self._metrics[mode]["gradient_conflict/tool_tokens"].append(
                float(valid_tool_mask.sum())
            )
            self._metrics[mode]["gradient_conflict/tool_token_fraction"].append(
                float(
                    valid_tool_mask.sum()
                    / (valid_reason_mask.sum() + valid_tool_mask.sum()).clamp_min(1)
                )
            )
        self._metrics[mode]["rollout/tool_mask_skipped_transitions"].append(
            float(skipped_tool_mask_transitions)
        )
        self._metrics[mode]["rollout/rank_tool_mask_skipped_transitions"].append(
            float(skipped_rank_tool_masks)
        )
        self._metrics[mode]["rollout/rank_pairs"].append(
            float(len(trajectory_pairs))
        )
        self._metrics[mode]["rollout/collect_seconds"].append(
            collect_finished - preparation_started
        )
        self._metrics[mode]["rollout/preparation_seconds"].append(
            old_policy_started - collect_finished
        )
        self._metrics[mode]["rollout/old_policy_forward_seconds"].append(
            old_policy_finished - old_policy_started
        )
        for name, value in importance_diagnostics.items():
            self._metrics[mode][f"sampling/vllm_importance/{name}"].append(value)
        self._metrics[mode]["rollout/padded_prompt_token_fraction"].append(
            float(prompt_mask.sum() / prompt_mask.numel())
        )
        self._metrics[mode]["rollout/padded_completion_token_fraction"].append(
            float(
                completion_attention_mask.sum()
                / completion_attention_mask.numel()
            )
        )
        self._metrics[mode]["loss/policy_coefficient"].append(
            float(self.policy_loss_coefficient)
        )
        self._metrics[mode]["loss/trajectory_mean_reduction"].append(
            float(self.policy_reduction == "trajectory_mean")
        )
        self._metrics[mode]["loss/trajectory_token_mean_reduction"].append(
            float(self.policy_reduction == "trajectory_token_mean")
        )
        if step_rewards:
            self._metrics[mode]["rollout/nonzero_step_reward_fraction"].append(
                sum(reward != 0.0 for reward in step_rewards) / len(step_rewards)
            )

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_attention_mask": completion_attention_mask,
            # TRL uses completion_mask both as the model attention mask and the
            # default policy-loss mask.  Keep all generated reasoning/action tokens
            # visible to attention and provide the narrower optimization carrier as
            # tool_mask.  This prevents tool-only updates from changing the causal
            # context relative to rollout/old-policy scoring.
            "completion_mask": completion_attention_mask,
            "tool_mask": completion_loss_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "sampling_per_token_logps": sampling_per_token_logps,
            "importance_sampling_ratio": importance_sampling_ratio,
            "num_items_in_batch": completion_loss_mask.sum(),
            # TRL shuffles every generation-batch value along the transition axis.
            # Keep this constant as aligned batch metadata instead of a Python scalar;
            # a scalar makes shuffle_sequence_dict attempt ``value[i]`` on an int.
            "policy_normalization_transitions": torch.full(
                (len(updates),),
                original_transition_count,
                device=device,
                dtype=torch.long,
            ),
        }
        if carrier_tool_mask is not None and carrier_mask_valid is not None:
            output.update(
                {
                    "carrier_tool_mask": carrier_tool_mask,
                    "carrier_mask_valid": torch.tensor(
                        carrier_mask_valid,
                        device=device,
                        dtype=torch.bool,
                    ),
                }
            )
        if rank_completion_mask is not None and transition_trajectory_indices is not None:
            output.update(
                {
                    "rank_completion_mask": rank_completion_mask,
                    "transition_trajectory_indices": torch.tensor(
                        transition_trajectory_indices,
                        device=device,
                        dtype=torch.long,
                    ),
                    "transition_example_indices": torch.tensor(
                        [update.example_index for update in updates],
                        device=device,
                        dtype=torch.long,
                    ),
                    "transition_trajectory_correct": torch.tensor(
                        [update.trajectory_correct for update in updates],
                        device=device,
                        dtype=torch.bool,
                    ),
                    "transition_legal_success": torch.tensor(
                        [update.legal_success for update in updates],
                        device=device,
                        dtype=torch.bool,
                    ),
                    "transition_local_penalty": torch.tensor(
                        [update.local_penalty for update in updates],
                        device=device,
                        dtype=torch.float32,
                    ),
                }
            )
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output

    def _transition_sequence_logps(
        self,
        model,
        inputs,
        *,
        microbatch_plan: _TransitionMicrobatchPlan | None = None,
    ):
        per_token_logps = self._transition_token_logps_compact(
            model,
            inputs,
            microbatch_plan=microbatch_plan,
        )
        return self._sequence_scores_from_token_logps(per_token_logps, inputs)

    def _sequence_scores_from_token_logps(self, per_token_logps, inputs):
        masked_logps = per_token_logps * inputs["rank_completion_mask"]
        sequence_logps = masked_logps.sum(dim=-1)
        if self.rank_score_reduction == "mean_action":
            sequence_logps = sequence_logps / inputs[
                "rank_completion_mask"
            ].sum(dim=-1).clamp_min(1)
        return sequence_logps

    def _ranking_coefficients(
        self,
        model,
        inputs,
        *,
        microbatch_plan: _TransitionMicrobatchPlan | None = None,
    ):
        if self.rank_loss_coefficient == 0.0:
            return None, None
        transition_trajectory_indices = inputs["transition_trajectory_indices"]
        try:
            trajectory_ids, trajectory_pairs = build_pairs_from_transition_metadata(
                transition_trajectory_indices.tolist(),
                inputs["transition_example_indices"].tolist(),
                inputs["transition_trajectory_correct"].tolist(),
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        if not trajectory_pairs:
            return None, None
        with torch.no_grad():
            reuse_old_policy_scores = bool(
                self.offline_rollout_pool
                and int(self.args.num_iterations) == 1
                and inputs.get("old_per_token_logps") is not None
            )
            transition_scores = (
                self._sequence_scores_from_token_logps(
                    inputs["old_per_token_logps"],
                    inputs,
                )
                if reuse_old_policy_scores
                else self._transition_sequence_logps(
                    model,
                    inputs,
                    microbatch_plan=microbatch_plan,
                )
            )
            token_support = inputs["rank_completion_mask"].any(dim=-1)
            score_selected = torch.tensor(
                [
                    rank_transition_selected(
                        self.rank_score_scope,
                        trajectory_correct=bool(correct),
                        legal_success=bool(legal),
                        local_penalty=float(penalty),
                    )
                    for correct, legal, penalty in zip(
                        inputs["transition_trajectory_correct"].tolist(),
                        inputs["transition_legal_success"].tolist(),
                        inputs["transition_local_penalty"].tolist(),
                        strict=True,
                    )
                ],
                device=transition_scores.device,
                dtype=torch.bool,
            )
            score_selected &= token_support
            trajectory_scores = torch.zeros(
                len(trajectory_ids),
                device=transition_scores.device,
                dtype=transition_scores.dtype,
            )
            trajectory_scores.scatter_add_(
                0,
                transition_trajectory_indices,
                transition_scores * score_selected.to(transition_scores.dtype),
            )
            trajectory_score_counts = torch.zeros(
                len(trajectory_ids),
                device=transition_scores.device,
                dtype=transition_scores.dtype,
            )
            trajectory_score_counts.scatter_add_(
                0,
                transition_trajectory_indices,
                score_selected.to(transition_scores.dtype),
            )
            if self.rank_score_reduction == "mean_action":
                trajectory_scores /= trajectory_score_counts.clamp_min(1)
            trajectory_has_support = trajectory_score_counts > 0
            trajectory_pairs = [
                pair
                for pair in trajectory_pairs
                if bool(trajectory_has_support[pair.positive_index])
                and bool(trajectory_has_support[pair.negative_index])
            ]
            if not trajectory_pairs:
                return None, None
            positive = torch.tensor(
                [pair.positive_index for pair in trajectory_pairs],
                device=transition_trajectory_indices.device,
                dtype=torch.long,
            )
            negative = torch.tensor(
                [pair.negative_index for pair in trajectory_pairs],
                device=transition_trajectory_indices.device,
                dtype=torch.long,
            )
            deltas = trajectory_scores[positive] - trajectory_scores[negative]
            raw_rank_loss = F.softplus(-self.rank_beta * deltas).mean()
            pair_derivative = (
                -self.rank_beta
                * torch.sigmoid(-self.rank_beta * deltas)
                / deltas.numel()
            )
            trajectory_coefficients = torch.zeros_like(trajectory_scores)
            trajectory_coefficients.scatter_add_(0, positive, pair_derivative)
            trajectory_coefficients.scatter_add_(0, negative, -pair_derivative)
            trajectory_coefficients *= self.rank_loss_coefficient
            transition_coefficients = trajectory_coefficients[
                transition_trajectory_indices
            ]
            update_selected = torch.tensor(
                [
                    rank_transition_selected(
                        self.rank_update_scope,
                        trajectory_correct=bool(correct),
                        legal_success=bool(legal),
                        local_penalty=float(penalty),
                    )
                    for correct, legal, penalty in zip(
                        inputs["transition_trajectory_correct"].tolist(),
                        inputs["transition_legal_success"].tolist(),
                        inputs["transition_local_penalty"].tolist(),
                        strict=True,
                    )
                ],
                device=transition_coefficients.device,
                dtype=torch.bool,
            )
            update_selected &= token_support
            if self.rank_score_scope in {"conservative_legal", "dense_outcome"}:
                update_selected &= score_selected
            transition_weights = update_selected.to(transition_coefficients.dtype)
            if self.rank_score_reduction == "mean_action":
                transition_weights /= trajectory_score_counts[
                    transition_trajectory_indices
                ].clamp_min(1)
            transition_coefficients *= transition_weights
        self._metrics["train"]["rank/loss"].append(
            float(raw_rank_loss * self.rank_loss_coefficient)
        )
        self._metrics["train"]["rank/score_gap"].append(float(deltas.mean()))
        self._metrics["train"]["rank/pairs"].append(float(deltas.numel()))
        self._metrics["train"]["rank/selected_transitions"].append(
            float(update_selected.sum())
        )
        self._metrics["train"]["rank/scored_transitions"].append(
            float(score_selected.sum())
        )
        self._metrics["train"]["rank/supported_trajectories"].append(
            float(trajectory_has_support.sum())
        )
        self._metrics["train"]["rank/reused_old_policy_scores"].append(
            float(reuse_old_policy_scores)
        )
        return transition_coefficients, (
            raw_rank_loss * self.rank_loss_coefficient
        ).detach()

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Backpropagate transition microbatches without retaining every long graph."""
        started = time.perf_counter()
        model.train()
        inputs = self._prepare_inputs(inputs)
        total = int(inputs["completion_ids"].shape[0])
        (
            inputs,
            prompt_lengths,
            completion_lengths,
        ) = self._sort_transition_batch_by_length(inputs)
        microbatch_plan = self._build_transition_microbatch_plan(
            inputs,
            prompt_lengths=prompt_lengths,
            completion_lengths=completion_lengths,
        )
        self._assert_fsdp_microbatch_alignment(
            microbatch_plan,
            stage="training backward",
        )
        normalization = inputs.get("policy_normalization_transitions")
        policy_normalization_transitions = (
            int(normalization[0].item()) if normalization is not None else total
        )
        detached_loss = torch.zeros((), device=self.accelerator.device)
        policy_started = time.perf_counter()
        if self.policy_loss_coefficient != 0.0:
            for range_index, (start, end) in enumerate(microbatch_plan.ranges):
                micro_inputs = self._slice_batch(
                    inputs,
                    start,
                    end,
                    trim_bounds=microbatch_plan.trim_bounds[range_index],
                )
                synchronize = range_index == len(microbatch_plan.ranges) - 1
                with self._microbatch_sync_context(
                    model,
                    synchronize=synchronize,
                ):
                    with self.compute_loss_context_manager():
                        micro_loss = super()._compute_loss(model, micro_inputs)
                    weight = (end - start) / policy_normalization_transitions
                    scaled_loss = (
                        micro_loss * weight * self.policy_loss_coefficient
                    )
                    self.accelerator.backward(scaled_loss)
                detached_loss = detached_loss + scaled_loss.detach()
        policy_finished = time.perf_counter()
        transition_coefficients, detached_rank_loss = self._ranking_coefficients(
            model,
            inputs,
            microbatch_plan=microbatch_plan,
        )
        rank_coefficients_finished = time.perf_counter()
        if transition_coefficients is not None:
            for range_index, (start, end) in enumerate(microbatch_plan.ranges):
                micro_inputs = self._slice_batch(
                    inputs,
                    start,
                    end,
                    trim_bounds=microbatch_plan.trim_bounds[range_index],
                )
                synchronize = range_index == len(microbatch_plan.ranges) - 1
                with self._microbatch_sync_context(
                    model,
                    synchronize=synchronize,
                ):
                    with self.compute_loss_context_manager():
                        sequence_logps = self._transition_sequence_logps(
                            model,
                            micro_inputs,
                        )
                        coefficients = transition_coefficients[start:end]
                        rank_surrogate = (sequence_logps * coefficients).sum()
                    self.accelerator.backward(rank_surrogate)
            detached_loss = detached_loss + detached_rank_loss
        rank_backward_finished = time.perf_counter()
        self._record_current_gradient_conflicts(
            model,
            inputs,
            policy_normalization_transitions=policy_normalization_transitions,
            total=total,
        )
        self._metrics["train"]["time/policy_backward_seconds"].append(
            policy_finished - policy_started
        )
        self._metrics["train"]["time/rank_coefficients_seconds"].append(
            rank_coefficients_finished - policy_finished
        )
        self._metrics["train"]["time/rank_backward_seconds"].append(
            rank_backward_finished - rank_coefficients_finished
        )
        self._step += 1
        self._current_train_step_time += time.perf_counter() - started
        if self._step % self.current_gradient_accumulation_steps == 0:
            self._metrics["train"]["step_time"].append(
                self._current_train_step_time
            )
            self._current_train_step_time = 0.0
        return detached_loss
