#!/usr/bin/env python3
"""Single-GPU QLoRA group-REINFORCE baseline for interactive table tools.

This trainer supports two controlled conditions over the same causal rollout loop:

* result-only: terminal 0/1 group-relative REINFORCE;
* process: harness-replayed per-turn rewards plus a sampled forward-KL penalty to a frozen copy of
  the initialization adapter.

Neither condition uses a learned reward/value model or token-level authored-credit heuristic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
import bitsandbytes as bnb
from accelerate import Accelerator
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_scheduler,
)

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "src" / "rl"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from tool_environment import create_tool_use_env  # noqa: E402
from tool_modules.registry import (  # noqa: E402
    ACTION_BLOCK_TOOL_SCHEME,
    ATOMIC_TOOL_SCHEME,
    TOOL_SCHEME_NAMES,
    build_tool_scheme,
)
from task_loader import load_rl_task_records  # noqa: E402
from reference_result_filter import filter_training_records  # noqa: E402
from process_credit import ProcessRewardConfig  # noqa: E402
from process_objective import sampled_turn_forward_kl  # noqa: E402
from counterfactual_suite import (  # noqa: E402
    CounterfactualSuiteManifest,
    CounterfactualTaskSuite,
    load_counterfactual_suite_manifest,
)
from rollout_scoring import (  # noqa: E402
    RolloutSample as Sample,
    episode_example,
    score_completed_rollout,
)
from frameworks.accelerate.turn_logprobs import (  # noqa: E402
    response_token_logprobs_batched,
    turn_padding_key,
)
from frameworks.accelerate.training_state import (  # noqa: E402
    load_training_state,
    save_training_state,
)
from protocol import (  # noqa: E402
    student_runtime_system_prompt,
    tool_schema_hash,
)


def activate_adapter(model, adapter_name: str) -> None:
    """Switch adapters while keeping the SFT reference parameters explicitly frozen."""
    model.set_adapter(adapter_name)
    for name, parameter in model.named_parameters():
        if ".sft_reference." in name:
            parameter.requires_grad = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--examples-json", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--exclude-empty-reference-results",
        action="store_true",
        help=(
            "drop whole tasks whose hidden reference query returns zero rows, NULL scalar, "
            "or numeric scalar zero before any actor rollout"
        ),
    )
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--rollout-batch-size", type=int, default=0,
                        help="parallel active episodes during sampling; default = group-size")
    parser.add_argument("--logprob-micro-batch-size", type=int, default=2,
                        help="turns per gradient forward pass; lower this if logprob still OOMs")
    parser.add_argument("--train-turns", choices=("all", "last"), default="all",
                        help="assistant turns receiving REINFORCE loss; use last only for an explicit speed ablation")
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("constant", "linear", "cosine"),
        default="cosine",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--reward-mode", choices=("result-only", "process"), default="result-only")
    parser.add_argument(
        "--tool-scheme",
        choices=TOOL_SCHEME_NAMES,
        default=ATOMIC_TOOL_SCHEME,
        help="exclusive model action protocol; schemes never share one visible action space",
    )
    parser.add_argument("--process-reward-config", type=Path)
    parser.add_argument("--counterfactual-suite-manifest", type=Path)
    parser.add_argument(
        "--process-admission-policy",
        choices=("counterfactual-completeness", "denotation-nonempty"),
        default="counterfactual-completeness",
        help=(
            "process-update admission rule: the default requires a task-keyed "
            "counterfactual suite; denotation-nonempty accepts fresh bird-set correctness "
            "on tasks whose hidden reference result was filtered to be non-empty/non-zero"
        ),
    )
    parser.add_argument("--kl-beta", type=float, default=0.0)
    parser.add_argument(
        "--denotation-comparison",
        choices=("bird-set",),
        default="bird-set",
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-batch-calls", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--context-mode", choices=("rolling-legal-history",),
                        default="rolling-legal-history")
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_model(args: argparse.Namespace, accelerator: Accelerator):
    dtype = torch.bfloat16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quant,
        torch_dtype=dtype,
        device_map={"": accelerator.local_process_index},
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    trainable_adapter_path = args.resume_from_checkpoint or args.adapter_path
    model = PeftModel.from_pretrained(model, trainable_adapter_path, is_trainable=True)
    if args.kl_beta > 0:
        model.load_adapter(
            args.adapter_path,
            adapter_name="sft_reference",
            is_trainable=False,
        )
        activate_adapter(model, "default")
    model.gradient_checkpointing_enable()
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("the adapter exposes no trainable parameters")
    # LoRA has few trainable parameters, but paged 8-bit state keeps the long-context margin on a
    # 24GB card available for activations rather than reserving it for optimizer tensors.
    return model, tokenizer, bnb.optim.PagedAdamW8bit(trainable, lr=args.learning_rate)


def load_process_reward_config(path: Path | None) -> ProcessRewardConfig:
    if path is None:
        return ProcessRewardConfig()
    values = {
        key: value
        for key, value in json.loads(path.read_text(encoding="utf-8")).items()
        if not key.startswith("_")
    }
    return ProcessRewardConfig(**values)


def render_prompt(tokenizer, messages: list[dict[str, str]], device: torch.device | None = None) -> list[int]:
    # Transformers 5.x Qwen tokenizers may return rendered text even when ``tokenize=True``.
    # Render first, then use the ordinary tokenizer API so the runner is version-independent.
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False).input_ids


def _trim_generated_response(ids: list[int], *, eos_token_id: int | None, pad_token_id: int | None) -> list[int]:
    """Keep the generated assistant turn and drop padding added after EOS in batched generation."""
    if eos_token_id is not None:
        for index, token_id in enumerate(ids):
            if token_id == eos_token_id:
                return ids[: index + 1]
    if pad_token_id is not None:
        while ids and ids[-1] == pad_token_id:
            ids.pop()
    return ids


def _generate_rollout_chunk(model, tokenizer, chunk: list[tuple[int, list[int]]], *,
                            device: torch.device, args: argparse.Namespace) -> list[tuple[int, list[int], list[int]]]:
    """Generate one batched assistant turn, splitting on OOM for 24GB cards."""
    max_prompt_len = max(len(prompt_ids) for _, prompt_ids in chunk)
    input_rows = []
    mask_rows = []
    for _, prompt_ids in chunk:
        pad_len = max_prompt_len - len(prompt_ids)
        input_rows.append([tokenizer.pad_token_id] * pad_len + prompt_ids)
        mask_rows.append([0] * pad_len + [1] * len(prompt_ids))
    input_ids = torch.tensor(input_rows, device=device, dtype=torch.long)
    attention_mask = torch.tensor(mask_rows, device=device, dtype=torch.long)
    try:
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    except torch.OutOfMemoryError:
        del input_ids, attention_mask
        torch.cuda.empty_cache()
        if len(chunk) == 1:
            env_index, prompt_ids = chunk[0]
            return [(env_index, prompt_ids, [])]
        midpoint = len(chunk) // 2
        return (
            _generate_rollout_chunk(model, tokenizer, chunk[:midpoint], device=device, args=args)
            + _generate_rollout_chunk(model, tokenizer, chunk[midpoint:], device=device, args=args)
        )
    return [
        (
            env_index,
            prompt_ids,
            _trim_generated_response(
                generated[row_index, max_prompt_len:].tolist(),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            ),
        )
        for row_index, (env_index, prompt_ids) in enumerate(chunk)
    ]


@torch.inference_mode()
def sample_group(
    model,
    tokenizer,
    metadata: dict[str, Any],
    args: argparse.Namespace,
    process_config: ProcessRewardConfig | None = None,
    counterfactual_suite: CounterfactualTaskSuite | None = None,
) -> list[Sample]:
    example = episode_example(metadata)
    device = next(model.parameters()).device
    envs = [
        create_tool_use_env(
            example,
            tool_scheme=args.tool_scheme,
            example_index=int(metadata["example_index"]),
            max_steps=args.max_steps,
            max_batch_calls=args.max_batch_calls,
            context_mode=args.context_mode,
            history_turns=args.history_turns,
            compact_observations=True,
            denotation_comparison=args.denotation_comparison,
        )
        for _ in range(args.group_size)
    ]
    turns: list[list[tuple[list[int], list[int]]]] = [[] for _ in envs]
    rollout_batch_size = args.rollout_batch_size or args.group_size
    model.eval()
    while any(not env.done for env in envs):
        active: list[tuple[int, list[int]]] = []
        for env_index, env in enumerate(envs):
            if env.done:
                continue
            prompt_ids = render_prompt(tokenizer, env.model_messages())
            if len(prompt_ids) + args.max_new_tokens > args.max_context_tokens:
                env.done = True
                env.failure_type = "context_overflow"
                continue
            active.append((env_index, prompt_ids))
        if not active:
            break

        for offset in range(0, len(active), rollout_batch_size):
            chunk = active[offset: offset + rollout_batch_size]
            for env_index, prompt_ids, response_ids in _generate_rollout_chunk(
                model, tokenizer, chunk, device=device, args=args
            ):
                env = envs[env_index]
                if not response_ids:
                    env.done = True
                    env.failure_type = "generation_oom"
                    continue
                text = tokenizer.decode(response_ids, skip_special_tokens=True)
                turns[env_index].append((prompt_ids, response_ids))
                env.apply_model_output(text)

    samples = []
    for sample_index, (env, sample_turns) in enumerate(zip(envs, turns, strict=True)):
        samples.append(
            score_completed_rollout(
                env,
                sample_turns,
                metadata,
                sample_index=sample_index,
                reward_mode=args.reward_mode,
                process_config=process_config,
                process_admission_policy=args.process_admission_policy,
                denotation_comparison=args.denotation_comparison,
                counterfactual_suite=counterfactual_suite,
            )
        )
        env.close()
    return samples


def trainable_turns(sample: Sample, mode: str) -> list[tuple[list[int], list[int]]]:
    turns = [turn for turn in sample.turns if turn[1]]
    if mode == "all":
        return turns
    return turns[-1:] if turns else []


def backward_group_loss(model, tokenizer, samples: list[Sample], logprob_micro_batch_size: int, train_turns: str,
                        kl_beta: float,
                        accelerator: Accelerator) -> tuple[float | None, list[float], float, int]:
    """Backpropagate the group loss incrementally so long episodes do not retain every graph."""
    eligible_indices = [
        index for index, sample in enumerate(samples) if sample.process_update
    ]
    if not eligible_indices:
        return None, [0.0] * len(samples), 0.0, 0
    rewards = torch.tensor(
        [samples[index].reward for index in eligible_indices],
        dtype=torch.float32,
    )
    reward_std = rewards.std(unbiased=False).item()
    advantages = [0.0] * len(samples)
    if reward_std == 0.0:
        if kl_beta == 0:
            return None, advantages, 0.0, 0
    else:
        for sample_index, advantage in zip(
            eligible_indices,
            ((rewards - rewards.mean()) / (reward_std + 1e-6)).tolist(),
            strict=True,
        ):
            advantages[sample_index] = advantage
    device = next(model.parameters()).device
    entries: list[tuple[int, tuple[list[int], list[int]]]] = []
    for sample_index in eligible_indices:
        sample = samples[sample_index]
        turns = trainable_turns(sample, train_turns)
        entries.extend((sample_index, turn) for turn in turns)
    if not entries:
        return None, advantages, 0.0, 0
    # The objective is a sum over independent turns, so ordering does not change its semantics.
    # Bucketing adjacent microbatch items by their exact sequence/response lengths avoids padding
    # a short turn to the longest turn from a different episode.
    entries.sort(key=lambda entry: turn_padding_key(entry[1]))

    loss_value = 0.0
    kl_value = 0.0
    micro_batch_size = max(1, int(logprob_micro_batch_size))
    for offset in range(0, len(entries), micro_batch_size):
        chunk = entries[offset: offset + micro_batch_size]
        turns = [turn for _, turn in chunk]
        reference_token_logps = None
        if kl_beta > 0:
            activate_adapter(model, "sft_reference")
            with torch.no_grad():
                reference_token_logps = response_token_logprobs_batched(
                    model,
                    tokenizer,
                    turns,
                    device,
                )
            activate_adapter(model, "default")
        current_token_logps = response_token_logprobs_batched(
            model,
            tokenizer,
            turns,
            device,
        )
        logps = [token_logps.sum() for token_logps in current_token_logps]
        terms = []
        for index, ((sample_index, _), logp) in enumerate(zip(chunk, logps, strict=True)):
            term = -float(advantages[sample_index]) * logp
            if reference_token_logps is not None:
                kl = sampled_turn_forward_kl(
                    current_token_logps[index],
                    reference_token_logps[index],
                )
                kl_value += float(kl.detach().cpu()) / len(entries)
                term = term + float(kl_beta) * kl
            terms.append(term / len(entries))
        micro_loss = torch.stack(terms).sum()
        loss_value += float(micro_loss.detach().cpu())
        accelerator.backward(micro_loss)
    return loss_value, advantages, kl_value, len(entries)


def backward_group_loss_with_retry(model, tokenizer, samples: list[Sample], logprob_micro_batch_size: int,
                                   train_turns: str,
                                   kl_beta: float,
                                   accelerator: Accelerator, optimizer):
    """Retry the gradient path with smaller logprob microbatches after OOM."""
    micro_batch_size = max(1, int(logprob_micro_batch_size))
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            loss_value, advantages, sampled_kl, trained_turns = backward_group_loss(
                model,
                tokenizer,
                samples,
                micro_batch_size,
                train_turns,
                kl_beta,
                accelerator,
            )
            return (
                loss_value,
                advantages,
                sampled_kl,
                trained_turns,
                micro_batch_size,
                None,
            )
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if micro_batch_size == 1:
                return (
                    None,
                    [0.0] * len(samples),
                    0.0,
                    0,
                    micro_batch_size,
                    "gradient_oom",
                )
            micro_batch_size = max(1, micro_batch_size // 2)


def process_entries(
    samples: list[Sample],
    train_turns: str,
) -> list[tuple[tuple[list[int], list[int]], float]]:
    entries: list[tuple[tuple[list[int], list[int]], float]] = []
    for sample in samples:
        if not sample.process_update:
            continue
        rewards = sample.step_rewards or []
        if len(sample.turns) != len(rewards):
            raise ValueError("process turn/reward lengths must align")
        paired = list(zip(sample.turns, rewards, strict=True))
        if train_turns == "last":
            paired = paired[-1:]
        entries.extend(paired)
    return entries


def backward_process_loss(
    model,
    tokenizer,
    samples: list[Sample],
    logprob_micro_batch_size: int,
    train_turns: str,
    kl_beta: float,
    accelerator: Accelerator,
) -> tuple[float | None, float, int]:
    """Backpropagate ``-(1/M) sum r_t log pi + beta/M sum KL_t`` incrementally."""
    entries = process_entries(samples, train_turns)
    if not entries:
        return None, 0.0, 0
    entries.sort(key=lambda entry: turn_padding_key(entry[0]))
    device = next(model.parameters()).device
    micro_batch_size = max(1, int(logprob_micro_batch_size))
    loss_value = 0.0
    kl_value = 0.0
    for offset in range(0, len(entries), micro_batch_size):
        chunk = entries[offset: offset + micro_batch_size]
        turns = [turn for turn, _ in chunk]
        reference_token_logps = None
        if kl_beta > 0:
            activate_adapter(model, "sft_reference")
            with torch.no_grad():
                reference_token_logps = response_token_logprobs_batched(
                    model,
                    tokenizer,
                    turns,
                    device,
                )
            activate_adapter(model, "default")
        current_token_logps = response_token_logprobs_batched(
            model,
            tokenizer,
            turns,
            device,
        )
        current_logps = [token_logprobs.sum() for token_logprobs in current_token_logps]
        terms = []
        for index, (current_logp, (_, reward)) in enumerate(zip(current_logps, chunk, strict=True)):
            term = -float(reward) * current_logp
            if reference_token_logps is not None:
                kl = sampled_turn_forward_kl(
                    current_token_logps[index],
                    reference_token_logps[index],
                )
                kl_value += float(kl.detach().cpu()) / len(entries)
                term = term + float(kl_beta) * kl
            terms.append(term / len(entries))
        micro_loss = torch.stack(terms).sum()
        loss_value += float(micro_loss.detach().cpu())
        accelerator.backward(micro_loss)
    return loss_value, kl_value, len(entries)


def backward_process_loss_with_retry(
    model,
    tokenizer,
    samples: list[Sample],
    logprob_micro_batch_size: int,
    train_turns: str,
    kl_beta: float,
    accelerator: Accelerator,
    optimizer,
):
    micro_batch_size = max(1, int(logprob_micro_batch_size))
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            loss_value, sampled_kl, trained_turns = backward_process_loss(
                model,
                tokenizer,
                samples,
                micro_batch_size,
                train_turns,
                kl_beta,
                accelerator,
            )
            return loss_value, sampled_kl, trained_turns, micro_batch_size, None
        except torch.OutOfMemoryError:
            activate_adapter(model, "default")
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if micro_batch_size == 1:
                return None, 0.0, 0, micro_batch_size, "gradient_oom"
            micro_batch_size = max(1, micro_batch_size // 2)


def checkpoint_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Fields that must remain fixed when continuing an interrupted controlled run."""
    def artifact_identity(path: Path | None) -> dict[str, str] | None:
        if path is None:
            return None
        resolved = path.resolve()
        digest = hashlib.sha256(resolved.read_bytes()).hexdigest() if resolved.is_file() else ""
        return {"path": str(resolved), "sha256": digest}

    if args.tool_scheme == ATOMIC_TOOL_SCHEME:
        runtime_prompt = student_runtime_system_prompt(
            context_mode=args.context_mode,
            compact=False,
        )
        scheme = build_tool_scheme(
            args.tool_scheme,
            system_prompt=runtime_prompt,
            max_batch_calls=args.max_batch_calls,
        )
        selected_tool_schema_hash = tool_schema_hash()
    else:
        scheme = build_tool_scheme(
            args.tool_scheme,
            max_batch_calls=args.max_batch_calls,
        )
        runtime_prompt = scheme.system_prompt
        selected_tool_schema_hash = None
    metadata = {
        **scheme.manifest_fields(),
        "model_path": str(args.model_path.resolve()),
        "sft_adapter_path": str(args.adapter_path.resolve()),
        "examples_json": artifact_identity(args.examples_json),
        "selection": artifact_identity(args.selection),
        "exclude_empty_reference_results": args.exclude_empty_reference_results,
        "process_reward_config": (
            artifact_identity(args.process_reward_config)
            if args.reward_mode == "process"
            else None
        ),
        "counterfactual_suite_manifest": (
            artifact_identity(args.counterfactual_suite_manifest)
            if (
                args.reward_mode == "process"
                and args.counterfactual_suite_manifest is not None
            )
            else None
        ),
        "process_admission_policy": (
            args.process_admission_policy
            if args.reward_mode == "process"
            else None
        ),
        "reward_mode": args.reward_mode,
        "denotation_comparison": args.denotation_comparison,
        "prompt_role": "student-runtime",
        "student_runtime_prompt_sha256": hashlib.sha256(
            runtime_prompt.encode("utf-8")
        ).hexdigest(),
        "tool_schema_sha256": selected_tool_schema_hash,
        "steps": args.steps,
        "group_size": args.group_size,
        "rollout_batch_size": args.rollout_batch_size,
        "train_turns": args.train_turns,
        "context_mode": args.context_mode,
        "history_turns": args.history_turns,
        "max_steps": args.max_steps,
        "max_new_tokens": args.max_new_tokens,
        "max_context_tokens": args.max_context_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "kl_beta": args.kl_beta,
    }
    if args.tool_scheme == ACTION_BLOCK_TOOL_SCHEME:
        metadata.update({
            "action_block_budget": args.max_steps,
            "max_batch_calls": args.max_batch_calls,
        })
    return metadata


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    output_dir: Path,
    step: int,
    args: argparse.Namespace,
) -> None:
    target = output_dir / f"checkpoint-{step}"
    target.mkdir(parents=True, exist_ok=True)
    activate_adapter(model, "default")
    model.save_pretrained(target, selected_adapters=["default"])
    save_training_state(
        target,
        step=step,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=checkpoint_metadata(args),
    )


def main() -> int:
    args = parse_args()
    if args.group_size < 2:
        raise SystemExit("--group-size must be at least 2 for a group-relative baseline")
    if args.rollout_batch_size < 0:
        raise SystemExit("--rollout-batch-size must be non-negative")
    if args.kl_beta < 0:
        raise SystemExit("--kl-beta must be non-negative")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise SystemExit("--warmup-ratio must be in [0, 1)")
    if args.save_every <= 0:
        raise SystemExit("--save-every must be positive")
    if (
        args.tool_scheme == ACTION_BLOCK_TOOL_SCHEME
        and args.reward_mode == "process"
    ):
        raise SystemExit(
            "action-block currently supports result-only RL only; atomic-local process "
            "credit must not be assigned to an entire authored block"
        )
    if (
        args.reward_mode == "process"
        and args.process_admission_policy == "counterfactual-completeness"
        and args.counterfactual_suite_manifest is None
    ):
        raise SystemExit(
            "--counterfactual-suite-manifest is required for process RL; "
            "single-database correctness cannot pass the dependency-completeness gate"
        )
    if (
        args.reward_mode == "process"
        and args.process_admission_policy == "denotation-nonempty"
        and not args.exclude_empty_reference_results
    ):
        raise SystemExit(
            "--process-admission-policy denotation-nonempty requires "
            "--exclude-empty-reference-results"
        )
    if args.max_batch_calls < 1:
        raise SystemExit("--max-batch-calls must be positive")
    if (
        args.tool_scheme == ACTION_BLOCK_TOOL_SCHEME
        and args.max_batch_calls > 5
    ):
        raise SystemExit("active action-block supports at most 5 calls per block")
    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise SystemExit("this baseline is intentionally single-GPU; launch without accelerate multi-process")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_rl_task_records(
        ROOT, split="train", selection=args.selection, examples_json=args.examples_json,
        limit=args.limit, seed=args.seed, context_mode=args.context_mode,
    )
    reference_result_audits = []
    if args.exclude_empty_reference_results:
        records, reference_result_audits = filter_training_records(records)
    if not records:
        raise SystemExit("no training records selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.exclude_empty_reference_results:
        (args.output_dir / "reference_result_filter.json").write_text(
            json.dumps(
                {
                    "schema_version": "rl-empty-reference-task-filter-v1",
                    "retained_count": len(records),
                    "excluded_count": sum(
                        audit.excluded for audit in reference_result_audits
                    ),
                    "tasks": [audit.to_dict() for audit in reference_result_audits],
                },
                ensure_ascii=False,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )
    process_config = (
        load_process_reward_config(args.process_reward_config)
        if args.reward_mode == "process"
        else None
    )
    if process_config is not None:
        process_config.validate()
    counterfactual_manifest: CounterfactualSuiteManifest | None = (
        load_counterfactual_suite_manifest(args.counterfactual_suite_manifest)
        if (
            args.reward_mode == "process"
            and args.process_admission_policy == "counterfactual-completeness"
        )
        else None
    )
    counterfactual_suites: dict[str, CounterfactualTaskSuite] = {}
    if counterfactual_manifest is not None:
        for record in records:
            metadata = record["environment"]
            suite = counterfactual_manifest.suite_for(metadata)
            counterfactual_suites[str(metadata["task_id"])] = suite
    log_path = args.output_dir / "metrics.jsonl"
    rollout_log_path = args.output_dir / "rollouts.jsonl"
    if args.resume_from_checkpoint is None and (log_path.exists() or rollout_log_path.exists()):
        raise SystemExit(
            "output directory already contains run logs; use --resume-from-checkpoint "
            "or choose an isolated output directory"
        )

    model, tokenizer, optimizer = load_model(args, accelerator)
    scheduler = get_scheduler(
        args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=round(args.steps * args.warmup_ratio),
        num_training_steps=args.steps,
    )
    completed_step = 0
    if args.resume_from_checkpoint is not None:
        completed_step = load_training_state(
            args.resume_from_checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_metadata=checkpoint_metadata(args),
            map_location=next(model.parameters()).device,
        )
        if completed_step >= args.steps:
            raise SystemExit(
                f"checkpoint already completed step {completed_step}, requested steps={args.steps}"
            )
    for step in range(completed_step + 1, args.steps + 1):
        step_started = time.time()
        record = records[(step - 1) % len(records)]
        rollout_started = time.time()
        samples = sample_group(
            model,
            tokenizer,
            record["environment"],
            args,
            process_config,
            (
                counterfactual_suites[str(record["environment"]["task_id"])]
                if (
                    args.reward_mode == "process"
                    and args.process_admission_policy
                    == "counterfactual-completeness"
                )
                else None
            ),
        )
        rollout_seconds = time.time() - rollout_started
        model.train()
        optimization_error = None
        sampled_kl = 0.0
        trained_turns = 0
        if args.reward_mode == "result-only":
            (
                loss_value,
                advantages,
                sampled_kl,
                trained_turns,
                used_logprob_micro_batch_size,
                optimization_error,
            ) = backward_group_loss_with_retry(
                model,
                tokenizer,
                samples,
                args.logprob_micro_batch_size,
                args.train_turns,
                args.kl_beta,
                accelerator,
                optimizer,
            )
        else:
            advantages = [0.0] * len(samples)
            (
                loss_value,
                sampled_kl,
                trained_turns,
                used_logprob_micro_batch_size,
                optimization_error,
            ) = backward_process_loss_with_retry(
                model,
                tokenizer,
                samples,
                args.logprob_micro_batch_size,
                args.train_turns,
                args.kl_beta,
                accelerator,
                optimizer,
            )
        updated = loss_value is not None
        if updated:
            try:
                accelerator.clip_grad_norm_((param for param in model.parameters() if param.requires_grad), 1.0)
                optimizer.step()
                scheduler.step()
            except torch.OutOfMemoryError:
                optimization_error = "optimizer_oom"
                updated = False
                loss_value = None
        if not updated:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        event = {
            "step": step,
            "example_index": record["environment"]["example_index"],
            "rewards": [sample.reward for sample in samples],
            "advantages": advantages,
            "correct_count": sum(sample.correct for sample in samples),
            "failure_types": [sample.failure_type for sample in samples],
            "sample_turns": [len(sample.turns) for sample in samples],
            "step_rewards": [sample.step_rewards for sample in samples],
            "process_update": [sample.process_update for sample in samples],
            "reward_mode": args.reward_mode,
            "tool_scheme": args.tool_scheme,
            "denotation_comparison": args.denotation_comparison,
            "kl_beta": args.kl_beta,
            "sampled_kl": sampled_kl,
            "trained_turns": trained_turns,
            "train_turns": args.train_turns,
            "context_mode": args.context_mode,
            "history_turns": args.history_turns,
            "rolling_observation_style": "resident",
            "loss": loss_value,
            "learning_rate": scheduler.get_last_lr()[0],
            "updated": updated,
            "optimization_error": optimization_error,
            "rollout_batch_size": args.rollout_batch_size or args.group_size,
            "requested_logprob_micro_batch_size": args.logprob_micro_batch_size,
            "used_logprob_micro_batch_size": used_logprob_micro_batch_size,
            "rollout_seconds": round(rollout_seconds, 3),
            "step_seconds": round(time.time() - step_started, 3),
        }
        with log_path.open("a", encoding="utf-8") as sink:
            sink.write(json.dumps(event, ensure_ascii=False) + "\n")
        with rollout_log_path.open("a", encoding="utf-8") as sink:
            for sample_index, sample in enumerate(samples):
                sink.write(json.dumps({
                    "training_step": step,
                    "sample_index": sample_index,
                    **sample.audit_record,
                }, ensure_ascii=False) + "\n")
        accelerator.print(json.dumps(event, ensure_ascii=False))
        if step % args.save_every == 0:
            save_checkpoint(
                model,
                optimizer,
                scheduler,
                args.output_dir,
                step,
                args,
            )
    if args.steps % args.save_every != 0:
        save_checkpoint(
            model,
            optimizer,
            scheduler,
            args.output_dir,
            args.steps,
            args,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
