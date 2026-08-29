#!/usr/bin/env python3
"""TRL GRPO adapter for exact transition-level table-agent rollouts."""
from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
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

from frameworks.trl.transition_batch import (
    build_transition_updates,
    policy_reduction_advantages,
    retain_policy_contributing_updates,
)
from frameworks.trl.state_action_ambiguity import (
    CREDIT_ASSIGNMENTS,
    apply_asymmetric_error_credit,
    apply_state_action_ambiguity_mask,
)
from frameworks.trl.gradient_conflict import GradientConflictRecorder
from frameworks.trl.tool_loss_mask import (
    ToolMaskUnavailable,
    tool_token_loss_mask,
)
from frameworks.trl.trajectory_ranking import (
    RANK_SCORE_REDUCTIONS,
    RANK_SCORE_SCOPES,
    RANK_SCORE_TOKENS,
    RANK_UPDATE_SCOPES,
    build_pairs_from_transition_metadata,
    build_trajectory_pairs,
    rank_transition_selected,
)


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
        record_gradient_conflicts: bool = False,
        gradient_conflict_dir: Path | None = None,
        gradient_conflict_save_vectors: bool = False,
        reference_adapter_name: str | None = None,
        transition_micro_batch_size: int = 2,
        **kwargs,
    ):
        if _TRL_IMPORT_ERROR is not None:
            raise ImportError(
                "TransitionGRPOTrainer requires trl>=0.29, torch, and accelerate"
            ) from _TRL_IMPORT_ERROR
        if transition_micro_batch_size < 1:
            raise ValueError("transition_micro_batch_size must be positive")
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
        if credit_assignment == "saam-asymmetric-error" and reward_mode != "result-only":
            raise ValueError("saam-asymmetric-error requires result-only terminal rewards")
        if credit_assignment == "saam-asymmetric-error" and train_turns != "all":
            raise ValueError("saam-asymmetric-error requires all causal turns")
        if credit_assignment == "saam-asymmetric-error" and rank_loss_coefficient != 0.0:
            raise ValueError(
                "saam-asymmetric-error does not mix a trajectory ranking loss"
            )
        if not math.isfinite(error_penalty) or error_penalty <= 0.0:
            raise ValueError("error_penalty must be finite and positive")
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
        self.transition_reward_mode = reward_mode
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
        self.credit_assignment = credit_assignment
        self.error_penalty = float(error_penalty)
        self.record_gradient_conflicts = bool(record_gradient_conflicts)
        self.gradient_conflict_dir = gradient_conflict_dir
        self.gradient_conflict_save_vectors = bool(gradient_conflict_save_vectors)
        self.gradient_conflict_recorder = None
        self.reference_adapter_name = reference_adapter_name
        self.transition_micro_batch_size = transition_micro_batch_size
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
        if self.credit_assignment == "saam-asymmetric-error" and self.beta != 0.0:
            raise ValueError("saam-asymmetric-error requires KL beta=0")
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

    def _counterfactual_policy_gradient(
        self,
        model,
        inputs: dict[str, Any],
        *,
        policy_normalization_transitions: int,
        positive: bool,
    ):
        """Backpropagate one sign partition using the exact mixed-batch objective."""

        total = int(inputs["completion_ids"].shape[0])
        for start in range(0, total, self.transition_micro_batch_size):
            end = min(total, start + self.transition_micro_batch_size)
            micro_inputs = self._slice_batch(inputs, start, end)
            advantages = micro_inputs["advantages"]
            if positive:
                selected = advantages > 0.0
            else:
                selected = advantages < 0.0
            micro_inputs["advantages"] = torch.where(
                selected,
                advantages,
                torch.zeros_like(advantages),
            )
            with self.compute_loss_context_manager():
                micro_loss = super()._compute_loss(model, micro_inputs)
            weight = (end - start) / policy_normalization_transitions
            scaled_loss = micro_loss * weight * self.policy_loss_coefficient
            self.accelerator.backward(scaled_loss)

    def _record_current_gradient_conflicts(
        self,
        model,
        inputs: dict[str, Any],
        *,
        policy_normalization_transitions: int,
        total: int,
    ) -> None:
        """Record mixed, positive-only, and negative-only gradients before clipping."""

        recorder = self.gradient_conflict_recorder
        if not self.record_gradient_conflicts:
            return
        saved_gradients = [
            (
                parameter,
                parameter.grad.detach().clone()
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
        model.zero_grad(set_to_none=True)
        self._counterfactual_policy_gradient(
            model,
            inputs,
            policy_normalization_transitions=policy_normalization_transitions,
            positive=True,
        )
        if recorder is not None:
            positive, positive_layers = recorder.capture(model)
        else:
            positive = positive_layers = None
        model.zero_grad(set_to_none=True)
        self._counterfactual_policy_gradient(
            model,
            inputs,
            policy_normalization_transitions=policy_normalization_transitions,
            positive=False,
        )
        if recorder is not None:
            negative, negative_layers = recorder.capture(model)
        else:
            negative = negative_layers = None
        model.zero_grad(set_to_none=True)
        for parameter, gradient in saved_gradients:
            parameter.grad = gradient
        if recorder is None:
            return
        summary = recorder.record(
            step=self._step + 1,
            combined=combined,
            positive=positive,
            negative=negative,
            layers={
                "combined": combined_layers,
                "positive": positive_layers,
                "negative": negative_layers,
            },
            metadata={
                "stage": "pre_clip",
                "credit_assignment": self.credit_assignment,
                "error_penalty": self.error_penalty,
                "transition_count": total,
                "policy_normalization_transitions": policy_normalization_transitions,
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
        self._metrics["train"]["gradient_conflict/record_seconds"].append(elapsed)

    @staticmethod
    def _slice_batch(inputs: dict[str, Any], start: int, end: int) -> dict[str, Any]:
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
        prompt_support = sliced["prompt_mask"].bool().any(dim=0)
        if bool(prompt_support.any()):
            prompt_start = int(prompt_support.nonzero(as_tuple=False)[0].item())
            for key in ("prompt_ids", "prompt_mask"):
                sliced[key] = sliced[key][:, prompt_start:prompt_width]

        completion_width = int(sliced["completion_ids"].shape[1])
        completion_support = sliced["completion_attention_mask"].bool().any(dim=0)
        if bool(completion_support.any()):
            completion_end = int(
                completion_support.nonzero(as_tuple=False)[-1].item()
            ) + 1
            completion_keys = (
                "completion_ids",
                "completion_attention_mask",
                "completion_mask",
                "tool_mask",
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

    def _transition_token_logps_compact(self, model, inputs):
        """Evaluate transition logprobs in length-bucketed, padding-trimmed microbatches."""
        total = int(inputs["completion_ids"].shape[0])
        global_completion_width = int(inputs["completion_ids"].shape[1])
        rows = []
        for start in range(0, total, self.transition_micro_batch_size):
            end = min(total, start + self.transition_micro_batch_size)
            micro_inputs = self._slice_batch(inputs, start, end)
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
            if local_completion_width < global_completion_width:
                per_token_logps = F.pad(
                    per_token_logps,
                    (0, global_completion_width - local_completion_width),
                    value=0.0,
                )
            rows.append(per_token_logps)
        return torch.cat(rows, dim=0)

    def _sync_qlora_weights_to_vllm(self, model) -> int:
        """Stream dequantized base+LoRA weights without mutating 4-bit training weights."""
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
            if (
                isinstance(base_layer, bnb.nn.Linear4bit)
                and "default" in lora_a
                and "default" in lora_b
            ):
                targets.append((module_name, module, base_layer))
        if not targets:
            return 0

        client = self.vllm_generation.vllm_client
        with torch.no_grad():
            for module_name, module, base_layer in targets:
                quant_state = getattr(base_layer.weight, "quant_state", None)
                if quant_state is None:
                    raise RuntimeError(
                        f"4-bit layer has no quantization state: {module_name}"
                    )
                base_weight = bnb.functional.dequantize_4bit(
                    base_layer.weight.data,
                    quant_state,
                )
                delta_weight = module.get_delta_weight("default")
                merged_weight = (
                    base_weight.to(dtype=torch.bfloat16)
                    + delta_weight.to(dtype=torch.bfloat16)
                ).contiguous()
                name = f"{module_name}.weight"
                name = name.removeprefix("base_model.model.")
                name = name.replace(".base_layer", "")
                client.update_named_param(name, merged_weight)
                del merged_weight, delta_weight, base_weight
        client.reset_prefix_cache()
        return len(targets)

    def _sync_policy_weights(self, mode: str) -> None:
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
        updates = build_transition_updates(
            episodes,
            reward_mode=self.transition_reward_mode,
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
            vanilla_effective_advantages = policy_reduction_advantages(
                updates,
                reduction=self.policy_reduction,
                normalization_transition_count=original_transition_count,
                normalization_trajectory_count=original_trajectory_count,
            )
            updates, saam_audit = apply_state_action_ambiguity_mask(
                episodes,
                updates,
                credit_assignment=self.credit_assignment,
            )
            masked_effective_advantages = policy_reduction_advantages(
                updates,
                reduction=self.policy_reduction,
                normalization_transition_count=original_transition_count,
                normalization_trajectory_count=original_trajectory_count,
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
        elif self.credit_assignment == "saam-asymmetric-error":
            vanilla_effective_advantages = policy_reduction_advantages(
                updates,
                reduction=self.policy_reduction,
                normalization_transition_count=original_transition_count,
                normalization_trajectory_count=original_trajectory_count,
            )
            updates, saam_audit = apply_asymmetric_error_credit(
                episodes,
                updates,
                error_penalty=self.error_penalty,
            )
            masked_effective_advantages = policy_reduction_advantages(
                updates,
                reduction=self.policy_reduction,
                normalization_transition_count=original_transition_count,
                normalization_trajectory_count=original_trajectory_count,
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
        if not updates:
            raise RuntimeError(
                "the rollout batch produced no transitions with trainable tokens; "
                "refusing to advance optimizer or scheduler state"
            )

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
            torch.tensor(update.prompt_ids, device=device, dtype=torch.long)
            for update in updates
        ]
        completion_ids_list = [
            torch.tensor(update.response_ids, device=device, dtype=torch.long)
            for update in updates
        ]
        sampling_logps_list = [
            torch.tensor(update.sampling_logprobs, device=device, dtype=torch.float32)
            for update in updates
        ]
        prompt_mask_list = [torch.ones_like(ids) for ids in prompt_ids_list]
        completion_attention_mask_list = [
            torch.ones_like(ids) for ids in completion_ids_list
        ]
        completion_loss_mask_list = (
            [
                torch.tensor(mask, device=device, dtype=torch.long)
                for mask in tool_loss_masks
            ]
            if tool_loss_masks is not None
            else completion_attention_mask_list
        )
        rank_loss_masks = []
        skipped_rank_tool_masks = 0
        if (
            self.rank_loss_coefficient != 0.0
            and self.rank_score_tokens == "tool_only"
            and tool_loss_masks is not None
        ):
            # Process+Rank tool-only training selects the same response token carrier for both
            # losses.  Parsing every action twice was pure host overhead.
            rank_loss_masks = list(tool_loss_masks)
        elif self.rank_loss_coefficient != 0.0 and self.rank_score_tokens == "tool_only":
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

        prompt_ids = pad(
            prompt_ids_list,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        prompt_mask = pad(prompt_mask_list, padding_value=0, padding_side="left")
        completion_ids = pad(
            completion_ids_list,
            padding_value=self.pad_token_id,
            padding_side="right",
        )
        completion_attention_mask = pad(
            completion_attention_mask_list,
            padding_value=0,
            padding_side="right",
        )
        completion_loss_mask = pad(
            completion_loss_mask_list,
            padding_value=0,
            padding_side="right",
        )
        rank_completion_mask = pad(
            [
                torch.tensor(mask, device=device, dtype=torch.long)
                for mask in rank_loss_masks
            ],
            padding_value=0,
            padding_side="right",
        )
        sampling_per_token_logps = pad(
            sampling_logps_list,
            padding_value=0.0,
            padding_side="right",
        )
        forward_inputs = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_attention_mask": completion_attention_mask,
        }
        old_policy_started = time.perf_counter()
        with torch.no_grad():
            old_per_token_logps = self._transition_token_logps_compact(
                self.model,
                forward_inputs,
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

        effective_advantages = policy_reduction_advantages(
            updates,
            reduction=self.policy_reduction,
            normalization_transition_count=original_transition_count,
            normalization_trajectory_count=original_trajectory_count,
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
            "rank_completion_mask": rank_completion_mask,
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
        if ref_per_token_logps is not None:
            output["ref_per_token_logps"] = ref_per_token_logps
        return output

    def _transition_sequence_logps(self, model, inputs):
        per_token_logps = self._transition_token_logps_compact(model, inputs)
        return self._sequence_scores_from_token_logps(per_token_logps, inputs)

    def _sequence_scores_from_token_logps(self, per_token_logps, inputs):
        masked_logps = per_token_logps * inputs["rank_completion_mask"]
        sequence_logps = masked_logps.sum(dim=-1)
        if self.rank_score_reduction == "mean_action":
            sequence_logps = sequence_logps / inputs[
                "rank_completion_mask"
            ].sum(dim=-1).clamp_min(1)
        return sequence_logps

    def _ranking_coefficients(self, model, inputs):
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
                else self._transition_sequence_logps(model, inputs)
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
        normalization = inputs.get("policy_normalization_transitions")
        policy_normalization_transitions = (
            int(normalization[0].item()) if normalization is not None else total
        )
        detached_loss = torch.zeros((), device=self.accelerator.device)
        policy_started = time.perf_counter()
        if self.policy_loss_coefficient != 0.0:
            for start in range(0, total, self.transition_micro_batch_size):
                end = min(total, start + self.transition_micro_batch_size)
                micro_inputs = self._slice_batch(inputs, start, end)
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
        )
        rank_coefficients_finished = time.perf_counter()
        if transition_coefficients is not None:
            for start in range(0, total, self.transition_micro_batch_size):
                end = min(total, start + self.transition_micro_batch_size)
                micro_inputs = self._slice_batch(inputs, start, end)
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
