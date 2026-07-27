#!/usr/bin/env python3
"""Merge a carrier-repaired PEFT adapter into a standard model directory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-shard-size", default="4GB")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output_dir}")

    adapter_config = LoraConfig.from_pretrained(args.adapter)
    token_targets = adapter_config.trainable_token_indices
    if not isinstance(token_targets, dict) or len(token_targets) != 1:
        raise ValueError(
            "adapter must contain exactly one output-module trainable-token target"
        )
    output_module_name, token_ids = next(iter(token_targets.items()))
    if len(token_ids) == 0:
        raise ValueError("trainable-token target is empty")

    tokenizer = AutoTokenizer.from_pretrained(
        args.adapter,
        local_files_only=True,
        trust_remote_code=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False)
    token_parameters = [
        parameter.detach().float().cpu()
        for name, parameter in model.named_parameters()
        if "trainable_tokens_delta" in name
    ]
    if len(token_parameters) != 1:
        raise ValueError(
            f"expected one trainable-token tensor, found {len(token_parameters)}"
        )
    expected_rows = token_parameters[0]

    merged_model = model.merge_and_unload(safe_merge=True)
    remaining_tuners = [
        name
        for name, module in merged_model.named_modules()
        if isinstance(module, BaseTunerLayer)
    ]
    if remaining_tuners:
        raise ValueError(f"PEFT tuner layers remain after merge: {remaining_tuners[:5]}")

    output_embeddings = merged_model.get_output_embeddings()
    merged_rows = (
        output_embeddings.weight.detach()
        .index_select(
            0,
            torch.tensor(token_ids, device=output_embeddings.weight.device),
        )
        .float()
        .cpu()
    )
    expected_merged_rows = expected_rows.to(output_embeddings.weight.dtype).float()
    if not torch.equal(expected_merged_rows, merged_rows):
        max_difference = float((expected_merged_rows - merged_rows).abs().max())
        raise ValueError(
            f"merged carrier rows differ from adapter rows; max_difference={max_difference}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(args.output_dir)

    source_repair_manifest = args.adapter / "repair_manifest.json"
    manifest = {
        "artifact_type": "merged-carrier-repaired-model-v1",
        "base_model": args.base_model,
        "adapter": str(args.adapter),
        "adapter_model_sha256": sha256_file(
            args.adapter / "adapter_model.safetensors"
        ),
        "adapter_config_sha256": sha256_file(args.adapter / "adapter_config.json"),
        "output_module_name": output_module_name,
        "boundary_token_ids": list(token_ids),
        "carrier_rows_exact_after_merge": True,
        "remaining_peft_tuner_layers": 0,
        "dtype": str(merged_model.dtype),
        "max_shard_size": args.max_shard_size,
    }
    if source_repair_manifest.exists():
        manifest["repair_manifest_sha256"] = sha256_file(source_repair_manifest)
    (args.output_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
