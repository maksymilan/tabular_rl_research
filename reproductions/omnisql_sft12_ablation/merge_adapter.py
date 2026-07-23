#!/usr/bin/env python3
"""Merge one completed QLoRA adapter into its bf16 base for offline evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    args = parser.parse_args()

    for path, required in (
        (args.base_model, "config.json"),
        (args.adapter, "adapter_config.json"),
    ):
        if not (path / required).is_file():
            raise FileNotFoundError(path / required)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty directory: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    adapter_config = json.loads(
        (args.adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    recorded_base = adapter_config.get("base_model_name_or_path")
    if recorded_base and Path(recorded_base).resolve() != args.base_model.resolve():
        raise ValueError(
            f"adapter records base {recorded_base!r}, expected {str(args.base_model)!r}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        local_files_only=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    tuned = PeftModel.from_pretrained(base, args.adapter, local_files_only=True)
    merged = tuned.merge_and_unload(safe_merge=True)
    merged.save_pretrained(
        args.out,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.out)
    (args.out / "merge_manifest.json").write_text(
        json.dumps(
            {
                "base_model": str(args.base_model.resolve()),
                "adapter": str(args.adapter.resolve()),
                "dtype": "bfloat16",
                "safe_merge": True,
                "max_shard_size": args.max_shard_size,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
