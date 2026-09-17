#!/usr/bin/env python3
"""Bounded PyTorch profiler for representative Qwen3-4B trainer passes.

This is a diagnostic only: it never calls optimizer.step() and never changes
the model or adapter.  The remote launcher supplies model and adapter paths.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--lengths", default="4096,8192")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--storage", choices=("4bit", "bf16"), default="4bit")
    return p.parse_args()


def load_model(model_path: str, adapter_path: str, device: str, storage: str):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": {"": device},
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    if storage == "4bit":
        kwargs["quantization_config"] = quant
    base = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    base.config.use_cache = False
    if storage == "4bit":
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(
        base, adapter_path, is_trainable=True, autocast_adapter_dtype=True
    )
    trainable = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
            trainable += param.numel()
            if param.is_floating_point():
                param.data = param.data.float()
    if trainable == 0:
        raise RuntimeError("adapter did not expose trainable LoRA parameters")
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.train()
    return model


def profile_length(model, length: int, output_dir: Path, device: str) -> dict:
    dev = torch.device(device)
    ids = torch.randint(0, model.config.vocab_size, (1, length), device=dev)
    mask = torch.ones_like(ids)
    # Warmup establishes allocator and SDPA dispatch without recording it.
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
    def backward():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1).logits
            loss = logits.float().mean()
        loss.backward()
    backward()
    trainable = [p for p in model.parameters() if p.requires_grad]
    grad_coverage = sum(p.grad is not None for p in trainable)
    if grad_coverage != len(trainable):
        raise RuntimeError(f"incomplete gradient coverage: {grad_coverage}/{len(trainable)}")
    if not all(torch.isfinite(p.grad).all().item() for p in trainable):
        raise RuntimeError("non-finite gradient")
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(dev)
    torch.cuda.reset_peak_memory_stats(dev)
    unprofiled = []
    for _ in range(3):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        backward()
        torch.cuda.synchronize(dev)
        unprofiled.append(time.perf_counter() - t0)
        model.zero_grad(set_to_none=True)
    trace_path = output_dir / f"trace_len{length}.json"
    started = time.perf_counter()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        for iteration in range(2):
            with torch.profiler.record_function(f"profile_forward_backward_len{length}"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(
                        input_ids=ids,
                        attention_mask=mask,
                        use_cache=False,
                        logits_to_keep=1,
                    )
                    # A bounded scalar keeps the real logits path and LoRA backward.
                    loss = out.logits.float().mean()
                loss.backward()
            model.zero_grad(set_to_none=True)
            prof.step()
    torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - started
    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=30
    )
    (output_dir / f"top_ops_len{length}.txt").write_text(table, encoding="utf-8")
    return {
        "length": length,
        "iterations": 2,
        "elapsed_seconds": elapsed,
        "seconds_per_iteration": elapsed / 2,
        "unprofiled_seconds": unprofiled,
        "unprofiled_median_seconds": statistics.median(unprofiled),
        "gradient_parameter_coverage": [grad_coverage, len(trainable)],
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(dev),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(dev),
        "trace": str(trace_path),
        "top_ops": str(output_dir / f"top_ops_len{length}.txt"),
    }


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    lengths = [int(x) for x in args.lengths.split(",") if x]
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    model = load_model(args.model, args.adapter, args.device, args.storage)
    results = []
    for length in lengths:
        results.append(profile_length(model, length, out, args.device))
    payload = {
        "schema_version": "qwen3-4b-pytorch-profiler-v1",
        "model": args.model,
        "adapter": args.adapter,
        "device": args.device,
        "storage": args.storage,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "attn_implementation": "sdpa",
        "gradient_checkpointing": True,
        "autocast_dtype": "bfloat16",
        "optimizer_step": False,
        "workload": "synthetic input, last-token mean-logit backward; not GRPO loss or complete update",
        "results": results,
    }
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
