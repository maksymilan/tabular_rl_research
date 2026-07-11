#!/usr/bin/env python3
"""Single-GPU QLoRA group-REINFORCE baseline for interactive table tools.

This deliberately small trainer is the hardware-compatible baseline for the two non-P2P RTX 3090s.
For each question it samples a group of complete tool episodes, receives only terminal 0/1 execution
rewards, normalizes rewards inside the group, and updates LoRA weights with REINFORCE.  It is a
GRPO-style estimator without PPO clipping, KL shaping, process reward, or vLLM/FSDP dependencies.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import bitsandbytes as bnb
import torch.nn.functional as F
from accelerate import Accelerator
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, LogitsProcessor, LogitsProcessorList

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "src" / "rl"), str(ROOT / "src" / "eval"), str(ROOT / "src" / "harness"), str(ROOT / "src" / "sft")]

from env import ToolUseEnv  # noqa: E402
from task_data import load_result_only_task_records  # noqa: E402


@dataclass
class Sample:
    reward: float
    correct: bool
    failure_type: str | None
    turns: list[tuple[list[int], list[int]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--examples-json", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--rollout-batch-size", type=int, default=0,
                        help="parallel active episodes during sampling; default = group-size")
    parser.add_argument("--logprob-micro-batch-size", type=int, default=2,
                        help="turns per gradient forward pass; lower this if logprob still OOMs")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--save-every", type=int, default=25)
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
    model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    model.gradient_checkpointing_enable()
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("the adapter exposes no trainable parameters")
    # LoRA has few trainable parameters, but paged 8-bit state keeps the long-context margin on a
    # 24GB card available for activations rather than reserving it for optimizer tensors.
    return model, tokenizer, bnb.optim.PagedAdamW8bit(trainable, lr=args.learning_rate)


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


def _has_balanced_tool_call_json(text: str) -> bool:
    tag = text.rfind("<tool_call>")
    if tag < 0:
        return False
    begin = text.find("{", tag + len("<tool_call>"))
    if begin < 0:
        return False
    depth = 0
    in_string = False
    escaped = False
    for ch in text[begin:]:
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return True
    return False


class ForceEosAfterStop(LogitsProcessor):
    """Force per-row EOS after a protocol stop string appears in generated tokens."""

    def __init__(self, tokenizer, stop_ids: list[int], *, prompt_width: int, eos_token_id: int | None):
        self.tokenizer = tokenizer
        self.stop_ids = stop_ids
        self.prompt_width = prompt_width
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.eos_token_id is None or not self.stop_ids:
            return scores
        stop_len = len(self.stop_ids)
        for row_index in range(input_ids.shape[0]):
            generated = input_ids[row_index, self.prompt_width:]
            if generated.numel() < stop_len:
                continue
            should_stop = False
            for end in range(stop_len, generated.numel() + 1):
                if generated[end - stop_len: end].tolist() == self.stop_ids:
                    should_stop = True
                    break
            if not should_stop and _has_balanced_tool_call_json(
                self.tokenizer.decode(generated.tolist(), skip_special_tokens=False)
            ):
                should_stop = True
            if should_stop:
                scores[row_index, :] = -torch.inf
                scores[row_index, self.eos_token_id] = 0
        return scores


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
        stop_ids = tokenizer("</tool_call>", add_special_tokens=False).input_ids
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([
                ForceEosAfterStop(
                    tokenizer,
                    stop_ids,
                    prompt_width=max_prompt_len,
                    eos_token_id=tokenizer.eos_token_id,
                )
            ]),
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
def sample_group(model, tokenizer, metadata: dict[str, Any], args: argparse.Namespace) -> list[Sample]:
    example = {"db_id": metadata["db_id"], "question": metadata["question"], "query": metadata["gold_sql"]}
    device = next(model.parameters()).device
    envs = [
        ToolUseEnv(example, example_index=int(metadata["example_index"]), max_steps=args.max_steps)
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
    for env, sample_turns in zip(envs, turns, strict=True):
        record = env.record()
        samples.append(Sample(
            reward=1.0 if record["correct"] else 0.0,
            correct=bool(record["correct"]),
            failure_type=record["failure_type"],
            turns=sample_turns,
        ))
    return samples


def _model_logits_for_response(model, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                               logits_to_keep: int) -> torch.Tensor:
    """Return only the response-prediction logits when the model supports it.

    Qwen-family causal LM forward methods in recent Transformers accept ``logits_to_keep``.
    Avoiding full-context logits is the difference between a usable long-context RL step and a
    gradient OOM on 24GB cards.  The fallback keeps compatibility with older installs.
    """
    try:
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=logits_to_keep,
        ).logits
    except TypeError:
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    return logits[:, -logits_to_keep:, :]


def response_logprobs_batched(model, tokenizer, turns: list[tuple[list[int], list[int]]],
                              device: torch.device) -> list[torch.Tensor]:
    """Mean log-probabilities for assistant turns, padded into one training microbatch."""
    max_response_len = max(len(response_ids) for _, response_ids in turns)
    sequences = [prompt_ids + response_ids[:-1] for prompt_ids, response_ids in turns]
    max_sequence_len = max(len(seq) for seq in sequences)
    input_rows = []
    mask_rows = []
    target_rows = []
    for seq, (_, response_ids) in zip(sequences, turns, strict=True):
        pad_len = max_sequence_len - len(seq)
        input_rows.append([tokenizer.pad_token_id] * pad_len + seq)
        mask_rows.append([0] * pad_len + [1] * len(seq))
        target_pad = max_response_len - len(response_ids)
        target_rows.append([-100] * target_pad + response_ids)
    input_ids = torch.tensor(input_rows, device=device, dtype=torch.long)
    attention_mask = torch.tensor(mask_rows, device=device, dtype=torch.long)
    targets = torch.tensor(target_rows, device=device, dtype=torch.long)
    logits = _model_logits_for_response(model, input_ids, attention_mask, max_response_len)
    token_losses = F.cross_entropy(
        logits.float().transpose(1, 2),
        targets,
        ignore_index=-100,
        reduction="none",
    )
    mask = targets.ne(-100)
    return [-(token_losses[index][mask[index]].mean()) for index in range(len(turns))]


def group_loss(model, tokenizer, samples: list[Sample], logprob_micro_batch_size: int) -> tuple[torch.Tensor | None, list[float]]:
    rewards = torch.tensor([sample.reward for sample in samples], dtype=torch.float32)
    if rewards.std(unbiased=False).item() == 0.0:
        return None, [0.0] * len(samples)
    advantages = ((rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)).tolist()
    device = next(model.parameters()).device
    entries: list[tuple[int, tuple[list[int], list[int]]]] = []
    for sample_index, sample in enumerate(samples):
        entries.extend((sample_index, turn) for turn in sample.turns if turn[1])
    if not entries:
        return None, advantages

    per_sample_logps: list[list[torch.Tensor]] = [[] for _ in samples]
    micro_batch_size = max(1, int(logprob_micro_batch_size))
    for offset in range(0, len(entries), micro_batch_size):
        chunk = entries[offset: offset + micro_batch_size]
        logps = response_logprobs_batched(model, tokenizer, [turn for _, turn in chunk], device)
        for (sample_index, _), logp in zip(chunk, logps, strict=True):
            per_sample_logps[sample_index].append(logp)

    terms = []
    for sample_logps, advantage in zip(per_sample_logps, advantages, strict=True):
        if sample_logps:
            terms.append(-float(advantage) * torch.stack(sample_logps).mean())
    return (torch.stack(terms).mean() if terms else None), advantages


def backward_group_loss(model, tokenizer, samples: list[Sample], logprob_micro_batch_size: int,
                        accelerator: Accelerator) -> tuple[float | None, list[float]]:
    """Backpropagate the group loss incrementally so long episodes do not retain every graph."""
    rewards = torch.tensor([sample.reward for sample in samples], dtype=torch.float32)
    if rewards.std(unbiased=False).item() == 0.0:
        return None, [0.0] * len(samples)
    advantages = ((rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)).tolist()
    device = next(model.parameters()).device
    entries: list[tuple[int, tuple[list[int], list[int]]]] = []
    turn_counts = []
    for sample_index, sample in enumerate(samples):
        turns = [turn for turn in sample.turns if turn[1]]
        turn_counts.append(len(turns))
        entries.extend((sample_index, turn) for turn in turns)
    if not entries:
        return None, advantages

    loss_value = 0.0
    micro_batch_size = max(1, int(logprob_micro_batch_size))
    for offset in range(0, len(entries), micro_batch_size):
        chunk = entries[offset: offset + micro_batch_size]
        logps = response_logprobs_batched(model, tokenizer, [turn for _, turn in chunk], device)
        terms = []
        for (sample_index, _), logp in zip(chunk, logps, strict=True):
            # Original objective: mean over samples of advantage * mean turn log-probability.
            weight = -float(advantages[sample_index]) / (len(samples) * max(1, turn_counts[sample_index]))
            terms.append(weight * logp)
        micro_loss = torch.stack(terms).sum()
        loss_value += float(micro_loss.detach().cpu())
        accelerator.backward(micro_loss)
    return loss_value, advantages


def backward_group_loss_with_retry(model, tokenizer, samples: list[Sample], logprob_micro_batch_size: int,
                                   accelerator: Accelerator, optimizer):
    """Retry the gradient path with smaller logprob microbatches after OOM."""
    micro_batch_size = max(1, int(logprob_micro_batch_size))
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            loss_value, advantages = backward_group_loss(model, tokenizer, samples, micro_batch_size, accelerator)
            return loss_value, advantages, micro_batch_size, None
        except torch.OutOfMemoryError:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if micro_batch_size == 1:
                return None, [0.0] * len(samples), micro_batch_size, "gradient_oom"
            micro_batch_size = max(1, micro_batch_size // 2)


def save_adapter(model, output_dir: Path, step: int) -> None:
    target = output_dir / f"checkpoint-{step}"
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)


def main() -> int:
    args = parse_args()
    if args.group_size < 2:
        raise SystemExit("--group-size must be at least 2 for a group-relative baseline")
    if args.rollout_batch_size < 0:
        raise SystemExit("--rollout-batch-size must be non-negative")
    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise SystemExit("this baseline is intentionally single-GPU; launch without accelerate multi-process")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    records = load_result_only_task_records(
        ROOT, split="train", selection=args.selection, examples_json=args.examples_json,
        limit=args.limit, seed=args.seed,
    )
    if not records:
        raise SystemExit("no training records selected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer, optimizer = load_model(args, accelerator)
    log_path = args.output_dir / "metrics.jsonl"
    for step in range(1, args.steps + 1):
        step_started = time.time()
        record = records[(step - 1) % len(records)]
        rollout_started = time.time()
        samples = sample_group(model, tokenizer, record["environment"], args)
        rollout_seconds = time.time() - rollout_started
        model.train()
        optimization_error = None
        loss_value, advantages, used_logprob_micro_batch_size, optimization_error = backward_group_loss_with_retry(
            model, tokenizer, samples, args.logprob_micro_batch_size, accelerator, optimizer
        )
        updated = loss_value is not None
        if updated:
            try:
                accelerator.clip_grad_norm_((param for param in model.parameters() if param.requires_grad), 1.0)
                optimizer.step()
            except torch.OutOfMemoryError:
                optimization_error = "backward_oom"
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
            "loss": loss_value,
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
        accelerator.print(json.dumps(event, ensure_ascii=False))
        if step % args.save_every == 0:
            save_adapter(model, args.output_dir, step)
    save_adapter(model, args.output_dir, args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
