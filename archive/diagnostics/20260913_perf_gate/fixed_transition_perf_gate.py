#!/usr/bin/env python3
"""Run a bounded, fixed-transition trainer performance and fusion gate.

This diagnostic never calls an optimizer step and never writes model weights.  It
uses one immutable transition reconstructed from an existing rollout record so
that base-storage and old/current-log-prob measurements share exactly the same
token ids.  The script is intentionally kept under archive/diagnostics: it is
evidence-producing code, not a training entry point.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--mode", choices=("4bit", "bf16"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target-total-tokens", type=int, default=2048)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--attention", default="sdpa")
    parser.add_argument("--fusion", action="store_true")
    return parser.parse_args()


def load_rollout_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def select_fixed_transition(
    tokenizer,
    rows: list[dict[str, Any]],
    *,
    target_total: int,
    max_total: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        for turn in row.get("turns", []):
            # A terminal generation-truncation marker is stored as a turn-like
            # record but has no authored prompt/output token pair.
            if "model_input" not in turn or "model_output" not in turn:
                continue
            messages = turn["model_input"]
            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt_ids = list(
                tokenizer(prompt_text, add_special_tokens=False).input_ids
            )
            response_ids = list(
                tokenizer(
                    turn["model_output"], add_special_tokens=False
                ).input_ids
            )
            total = len(prompt_ids) + len(response_ids)
            if len(response_ids) < 16 or total > max_total:
                continue
            candidates.append(
                {
                    "row_index": row_index,
                    "example_index": row.get("example_index"),
                    "turn_index": turn.get("turn_index"),
                    "prompt_ids": prompt_ids,
                    "response_ids": response_ids,
                    "prompt_tokens": len(prompt_ids),
                    "response_tokens": len(response_ids),
                    "total_tokens": total,
                }
            )
    if not candidates:
        raise RuntimeError(
            f"no transition has response>=16 and total<={max_total} tokens"
        )
    candidates.sort(
        key=lambda item: (
            abs(int(item["total_tokens"]) - target_total),
            int(item["row_index"]),
            int(item["turn_index"] or 0),
        )
    )
    return candidates[0]


def load_model(args: argparse.Namespace):
    quantization = None
    if args.mode == "4bit":
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attention,
    }
    if quantization is not None:
        model_kwargs["quantization_config"] = quantization
    base = AutoModelForCausalLM.from_pretrained(str(args.model), **model_kwargs)
    base.config.use_cache = False
    if args.mode == "4bit":
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=True
        )
    model = PeftModel.from_pretrained(
        base, str(args.adapter), is_trainable=True, autocast_adapter_dtype=True
    )
    trainable = 0
    for name, parameter in model.named_parameters():
        if "lora_" in name:
            parameter.requires_grad_(True)
            if parameter.is_floating_point():
                parameter.data = parameter.data.float()
            trainable += parameter.numel()
    if trainable == 0:
        raise RuntimeError("adapter did not expose trainable LoRA parameters")
    model.gradient_checkpointing_enable()
    # PEFT's k-bit preparation enables this hook for the 4-bit path.  The
    # unquantized BF16 path needs the same hook explicitly when activation
    # checkpointing is enabled; otherwise checkpoint() sees no grad-requiring
    # input and the LoRA backward graph is discarded.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()
    # The production trainer disables dropout.  Enforce the same invariant for
    # the two-pass/fused numerical comparison without changing model weights.
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    return model, trainable


def response_logprobs(model, input_ids: torch.Tensor, prompt_tokens: int, response_tokens: int):
    # logits_to_keep=N returns the last N positions.  Keeping R+1 positions
    # starts at prompt_tokens-1, exactly the causal positions predicting the R
    # response tokens, and avoids materializing prompt logits.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            use_cache=False,
            logits_to_keep=response_tokens + 1,
        )
        logits = output.logits[:, :-1, :].float()
        targets = input_ids[:, prompt_tokens : prompt_tokens + response_tokens]
        return torch.log_softmax(logits, dim=-1).gather(
            -1, targets.unsqueeze(-1)
        ).squeeze(-1)


def trainable_grads(model) -> list[torch.Tensor]:
    return [
        parameter.grad.detach().float().cpu().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    ]


def gradient_summary(left: list[torch.Tensor], right: list[torch.Tensor]) -> dict[str, float]:
    left_flat = torch.cat([item.reshape(-1) for item in left])
    right_flat = torch.cat([item.reshape(-1) for item in right])
    delta = left_flat - right_flat
    denominator = left_flat.norm() * right_flat.norm()
    cosine = (
        float(torch.dot(left_flat, right_flat) / denominator)
        if float(denominator) != 0.0
        else 1.0
    )
    return {
        "gradient_max_abs_diff": float(delta.abs().max()),
        "gradient_l2_diff": float(delta.norm()),
        "gradient_cosine": max(-1.0, min(1.0, cosine)),
        "gradient_norm_two_pass": float(left_flat.norm()),
        "gradient_norm_fused": float(right_flat.norm()),
    }


def timed_two_pass(
    model,
    input_ids: torch.Tensor,
    *,
    prompt_tokens: int,
    response_tokens: int,
    advantage: torch.Tensor,
) -> tuple[float, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        old_logps = response_logprobs(
            model, input_ids, prompt_tokens, response_tokens
        )
    current_logps = response_logprobs(
        model, input_ids, prompt_tokens, response_tokens
    )
    loss = -(current_logps * advantage).mean()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return elapsed, trainable_grads(model), old_logps.detach(), current_logps.detach()


def timed_fused(
    model,
    input_ids: torch.Tensor,
    *,
    prompt_tokens: int,
    response_tokens: int,
    advantage: torch.Tensor,
) -> tuple[float, list[torch.Tensor], torch.Tensor, torch.Tensor]:
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    current_logps = response_logprobs(
        model, input_ids, prompt_tokens, response_tokens
    )
    old_logps = current_logps.detach()
    loss = -(current_logps * advantage).mean()
    loss.backward()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return elapsed, trainable_grads(model), old_logps, current_logps.detach()


def main(args: argparse.Namespace | None = None) -> None:
    args = args or parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.iterations < 2:
        raise SystemExit("iterations must be at least 2")
    torch.manual_seed(args.seed)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), trust_remote_code=True
    )
    rows = load_rollout_rows(args.rollouts)
    selected = select_fixed_transition(
        tokenizer,
        rows,
        target_total=args.target_total_tokens,
        max_total=args.max_total_tokens,
    )
    prompt_ids = torch.tensor(selected["prompt_ids"], dtype=torch.long)
    response_ids = torch.tensor(selected["response_ids"], dtype=torch.long)
    input_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).cuda()
    response_tokens = int(response_ids.numel())
    prompt_tokens = int(prompt_ids.numel())
    advantage = torch.linspace(
        -1.0, 1.0, response_tokens, device=input_ids.device, dtype=torch.float32
    ).unsqueeze(0)

    model, trainable_count = load_model(args)
    torch.cuda.reset_peak_memory_stats()
    # Warm up dispatch and allocator, but do not include it in measurements.
    with torch.no_grad():
        response_logprobs(model, input_ids, prompt_tokens, response_tokens)
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    two_pass_times: list[float] = []
    fused_times: list[float] = []
    last_two = None
    last_fused = None
    for _ in range(args.iterations):
        if args.fusion:
            two = timed_two_pass(
                model,
                input_ids,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                advantage=advantage,
            )
            fused = timed_fused(
                model,
                input_ids,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                advantage=advantage,
            )
            two_pass_times.append(two[0])
            fused_times.append(fused[0])
            last_two, last_fused = two, fused
        else:
            started, grads, old, current = timed_fused(
                model,
                input_ids,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens,
                advantage=advantage,
            )
            fused_times.append(started)
            last_fused = (started, grads, old, current)

    result: dict[str, Any] = {
        "schema_version": "fixed-transition-perf-gate-v1",
        "status": "complete",
        "mode": args.mode,
        "fusion_test": bool(args.fusion),
        "model": str(args.model),
        "adapter": str(args.adapter),
        "rollouts": str(args.rollouts),
        "attention": args.attention,
        "gradient_checkpointing": True,
        "autocast": "bfloat16",
        "optimizer_step": False,
        "trainable_parameters": trainable_count,
        "selection": {
            key: value
            for key, value in selected.items()
            if key not in {"prompt_ids", "response_ids"}
        },
        "two_pass_seconds": two_pass_times,
        "fused_seconds": fused_times,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if two_pass_times:
        result["two_pass_median_seconds"] = statistics.median(two_pass_times)
        result["two_pass_steady_median_seconds"] = statistics.median(
            two_pass_times[1:] if len(two_pass_times) > 2 else two_pass_times
        )
    if fused_times:
        result["fused_median_seconds"] = statistics.median(fused_times)
        result["fused_steady_median_seconds"] = statistics.median(
            fused_times[1:] if len(fused_times) > 2 else fused_times
        )
    if two_pass_times and fused_times:
        two_pass_steady = result["two_pass_steady_median_seconds"]
        fused_steady = result["fused_steady_median_seconds"]
        result["fusion_speedup"] = two_pass_steady / fused_steady
        old_two, current_two = last_two[2], last_two[3]
        old_fused, current_fused = last_fused[2], last_fused[3]
        result["logprob_max_abs_diff"] = float(
            (current_two - current_fused).abs().max()
        )
        result["old_detach_max_abs_diff"] = float(
            (old_fused - current_fused).abs().max()
        )
        result.update(gradient_summary(last_two[1], last_fused[1]))
    (output / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli_args = parse_args()
    try:
        main(cli_args)
    except torch.cuda.OutOfMemoryError as exc:
        # Leave an explicit, machine-readable record and let the wrapper move
        # on to the next isolated arm after the process exits.
        output_arg = cli_args.output.resolve()
        output_arg.mkdir(parents=True, exist_ok=True)
        (output_arg / "result.json").write_text(
            json.dumps(
                {
                    "schema_version": "fixed-transition-perf-gate-v1",
                    "status": "oom",
                    "mode": cli_args.mode,
                    "error": str(exc),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
