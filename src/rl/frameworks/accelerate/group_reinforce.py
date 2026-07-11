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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import bitsandbytes as bnb
from accelerate import Accelerator
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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


def render_prompt(tokenizer, messages: list[dict[str, str]], device: torch.device) -> list[int]:
    # Transformers 5.x Qwen tokenizers may return rendered text even when ``tokenize=True``.
    # Render first, then use the ordinary tokenizer API so the runner is version-independent.
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, add_special_tokens=False).input_ids


@torch.inference_mode()
def sample_episode(model, tokenizer, metadata: dict[str, Any], args: argparse.Namespace) -> Sample:
    example = {"db_id": metadata["db_id"], "question": metadata["question"], "query": metadata["gold_sql"]}
    env = ToolUseEnv(example, example_index=int(metadata["example_index"]), max_steps=args.max_steps)
    device = next(model.parameters()).device
    turns: list[tuple[list[int], list[int]]] = []
    model.eval()
    while not env.done:
        prompt_ids = render_prompt(tokenizer, env.model_messages(), device)
        if len(prompt_ids) + args.max_new_tokens > args.max_context_tokens:
            env.done = True
            env.failure_type = "context_overflow"
            break
        input_ids = torch.tensor([prompt_ids], device=device, dtype=torch.long)
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )[0]
        response_ids = generated[len(prompt_ids):].tolist()
        if not response_ids:
            env.done = True
            env.failure_type = "empty_generation"
            break
        text = tokenizer.decode(response_ids, skip_special_tokens=True)
        turns.append((prompt_ids, response_ids))
        env.apply_model_output(text)
    record = env.record()
    return Sample(
        reward=1.0 if record["correct"] else 0.0,
        correct=bool(record["correct"]),
        failure_type=record["failure_type"],
        turns=turns,
    )


def response_logprob(model, prompt_ids: list[int], response_ids: list[int], device: torch.device) -> torch.Tensor:
    """Mean log-probability of one assistant turn; observations are never optimized."""
    ids = torch.tensor([prompt_ids + response_ids], device=device, dtype=torch.long)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
    start = len(prompt_ids) - 1
    token_logits = logits[:, start : start + len(response_ids), :]
    token_ids = ids[:, len(prompt_ids) :]
    return torch.log_softmax(token_logits, dim=-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1).mean()


def group_loss(model, samples: list[Sample]) -> tuple[torch.Tensor | None, list[float]]:
    rewards = torch.tensor([sample.reward for sample in samples], dtype=torch.float32)
    if rewards.std(unbiased=False).item() == 0.0:
        return None, [0.0] * len(samples)
    advantages = ((rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)).tolist()
    device = next(model.parameters()).device
    terms = []
    for sample, advantage in zip(samples, advantages, strict=True):
        if not sample.turns:
            continue
        turn_logps = torch.stack([response_logprob(model, prompt, response, device) for prompt, response in sample.turns])
        terms.append(-float(advantage) * turn_logps.mean())
    return (torch.stack(terms).mean() if terms else None), advantages


def save_adapter(model, output_dir: Path, step: int) -> None:
    target = output_dir / f"checkpoint-{step}"
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)


def main() -> int:
    args = parse_args()
    if args.group_size < 2:
        raise SystemExit("--group-size must be at least 2 for a group-relative baseline")
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
        record = records[(step - 1) % len(records)]
        samples = [sample_episode(model, tokenizer, record["environment"], args) for _ in range(args.group_size)]
        model.train()
        optimization_error = None
        try:
            loss, advantages = group_loss(model, samples)
            updated = loss is not None
            if updated:
                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                accelerator.clip_grad_norm_((param for param in model.parameters() if param.requires_grad), 1.0)
                optimizer.step()
        except torch.OutOfMemoryError:
            # A rare long interaction must not discard the completed full-train run. Its rollout
            # remains visible in the log, but it contributes no gradient on this 24GB baseline.
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            loss = None
            advantages = [0.0] * len(samples)
            updated = False
            optimization_error = "gradient_oom"
        event = {
            "step": step,
            "example_index": record["environment"]["example_index"],
            "rewards": [sample.reward for sample in samples],
            "advantages": advantages,
            "correct_count": sum(sample.correct for sample in samples),
            "failure_types": [sample.failure_type for sample in samples],
            "loss": float(loss.detach().cpu()) if loss is not None else None,
            "updated": updated,
            "optimization_error": optimization_error,
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
