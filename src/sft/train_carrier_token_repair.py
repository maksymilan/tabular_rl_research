#!/usr/bin/env python3
"""Repair exact carrier-token output rows while preserving an existing LoRA.

The source adapter's LoRA weights are copied unchanged. Only the output-head
rows for the canonical carrier boundaries are optimized. This separates a
carrier representation failure from relational/tool-policy learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
)
from peft.utils.save_and_load import (
    get_peft_model_state_dict,
    load_peft_weights,
    set_peft_model_state_dict,
)
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.optimization import get_cosine_schedule_with_warmup

from carrier_repair import (
    assistant_tool_name,
    canonical_boundary_token_ids,
    encode_last_assistant_target,
    read_jsonl,
    select_tool_balanced_records,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_module_name(model: torch.nn.Module, target: torch.nn.Module) -> str:
    matches = [name for name, module in model.named_modules() if module is target]
    if len(matches) != 1:
        raise ValueError(f"expected one module name for output embeddings, found {matches}")
    return matches[0]


class EncodedDataset(Dataset):
    def __init__(self, examples: list[dict[str, list[int]]]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.examples[index]


class CausalCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in examples]
        attention_mask = [
            torch.tensor(item["attention_mask"], dtype=torch.long) for item in examples
        ]
        labels = [torch.tensor(item["labels"], dtype=torch.long) for item in examples]
        return {
            "input_ids": pad_sequence(
                input_ids,
                batch_first=True,
                padding_value=self.pad_token_id,
            ),
            "attention_mask": pad_sequence(
                attention_mask,
                batch_first=True,
                padding_value=0,
            ),
            "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--source-adapter", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=256)
    parser.add_argument("--cutoff-len", type=int, default=6400)
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=4600,
        help="select complete records no longer than this; records are never truncated",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument(
        "--negative-positions-per-record",
        type=int,
        default=8,
        help="non-boundary assistant positions that suppress false carrier emission",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"output directory is not empty: {args.output_dir}")
    if args.max_records <= 0 or args.epochs <= 0:
        raise ValueError("max-records and epochs must be positive")
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient-accumulation-steps must be positive")
    if args.negative_positions_per_record < 0:
        raise ValueError("negative-positions-per-record must be non-negative")
    if not 0 < args.max_sequence_length <= args.cutoff_len:
        raise ValueError("max-sequence-length must be in (0, cutoff-len]")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=True,
    )
    boundary_ids = canonical_boundary_token_ids(tokenizer)
    required_ids = tuple(boundary_ids.values())
    eligible: list[dict[str, Any]] = []
    encoded_by_identity: dict[int, dict[str, list[int]]] = {}
    complete_record_count = 0
    for record in read_jsonl(args.dataset):
        example = encode_last_assistant_target(
            record,
            tokenizer,
            cutoff_len=args.cutoff_len,
            required_token_ids=required_ids,
        )
        complete_record_count += 1
        if len(example["input_ids"]) <= args.max_sequence_length:
            eligible.append(record)
            encoded_by_identity[id(record)] = example
    selected = select_tool_balanced_records(
        eligible,
        limit=args.max_records,
        seed=args.seed,
    )
    if len(selected) < args.max_records:
        raise ValueError(
            f"only {len(selected)} complete records satisfy "
            f"max_sequence_length={args.max_sequence_length}"
        )
    encoded = [encoded_by_identity[id(record)] for record in selected]
    selected_tool_counts = Counter(assistant_tool_name(record) for record in selected)

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=True,
        quantization_config=quantization,
        dtype=torch.bfloat16,
        device_map={"": 0},
    )
    base_model.config.use_cache = False
    model = PeftModel.from_pretrained(
        base_model,
        args.source_adapter,
        is_trainable=False,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    causal_model = model.get_base_model()
    decoder = causal_model.model
    output_embeddings = causal_model.get_output_embeddings()
    output_module_name = resolve_module_name(
        causal_model,
        output_embeddings,
    )
    source_state = load_peft_weights(args.source_adapter, device="cpu")
    boundary_index = torch.tensor(
        required_ids,
        dtype=torch.long,
        device=output_embeddings.weight.device,
    )
    boundary_rows = torch.nn.Parameter(
        output_embeddings.weight.detach()
        .index_select(0, boundary_index)
        .float()
        .clone()
    )
    trainable_count = boundary_rows.numel()
    expected_count = len(required_ids) * base_model.config.hidden_size
    if trainable_count != expected_count:
        raise ValueError(
            f"expected {expected_count} trainable output-row values, got {trainable_count}"
        )

    loader = DataLoader(
        EncodedDataset(encoded),
        batch_size=1,
        shuffle=True,
        collate_fn=CausalCollator(tokenizer.pad_token_id),
        generator=torch.Generator().manual_seed(args.seed),
    )
    updates_per_epoch = math.ceil(
        len(loader) / args.gradient_accumulation_steps
    )
    total_updates = updates_per_epoch * args.epochs
    warmup_steps = round(total_updates * args.warmup_ratio)
    optimizer = torch.optim.AdamW(
        [boundary_rows],
        lr=args.learning_rate,
        weight_decay=0.0,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "train_log.jsonl"
    optimizer.zero_grad(set_to_none=True)
    global_update = 0
    positive_predictions = 0
    positive_examples = 0
    negative_boundary_predictions = 0
    negative_examples = 0
    for epoch in range(args.epochs):
        for batch_index, batch in enumerate(loader, start=1):
            batch = {name: value.to(model.device) for name, value in batch.items()}
            with torch.inference_mode():
                hidden_states = decoder(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                    return_dict=True,
                ).last_hidden_state

            shifted_hidden = hidden_states[:, :-1, :].reshape(
                -1,
                hidden_states.shape[-1],
            )
            shifted_targets = batch["labels"][:, 1:].reshape(-1)
            positive_mask = torch.zeros_like(shifted_targets, dtype=torch.bool)
            for token_id in required_ids:
                positive_mask |= shifted_targets == token_id
            positive_indices = positive_mask.nonzero(as_tuple=False).flatten()
            if len(positive_indices) != len(required_ids):
                raise ValueError(
                    "each encoded target must contain exactly one open and close boundary"
                )

            negative_candidates = (
                (shifted_targets != -100) & ~positive_mask
            ).nonzero(as_tuple=False).flatten()
            negative_count = min(
                args.negative_positions_per_record,
                len(negative_candidates),
            )
            if negative_count:
                offsets = torch.linspace(
                    0,
                    len(negative_candidates) - 1,
                    steps=negative_count,
                    device=negative_candidates.device,
                ).round().long()
                negative_indices = negative_candidates.index_select(0, offsets)
                objective_indices = torch.cat(
                    [positive_indices, negative_indices],
                    dim=0,
                )
            else:
                negative_indices = negative_candidates[:0]
                objective_indices = positive_indices

            objective_hidden = shifted_hidden.index_select(
                0,
                objective_indices,
            )
            objective_targets = shifted_targets.index_select(
                0,
                objective_indices,
            )
            base_logits = torch.nn.functional.linear(
                objective_hidden.to(output_embeddings.weight.dtype),
                output_embeddings.weight,
            ).float()
            repaired_boundary_logits = torch.nn.functional.linear(
                objective_hidden.float(),
                boundary_rows,
            )
            logits = torch.index_copy(
                base_logits,
                1,
                boundary_index,
                repaired_boundary_logits,
            )
            loss = torch.nn.functional.cross_entropy(logits, objective_targets)
            predicted = logits.detach().argmax(dim=-1)
            positive_predictions += int(
                (predicted[: len(positive_indices)] == objective_targets[: len(positive_indices)])
                .sum()
                .cpu()
            )
            positive_examples += len(positive_indices)
            if negative_count:
                negative_predicted = predicted[len(positive_indices) :]
                negative_boundary_predictions += int(
                    torch.isin(negative_predicted, boundary_index).sum().cpu()
                )
                negative_examples += negative_count
            (loss / args.gradient_accumulation_steps).backward()
            is_boundary = (
                batch_index % args.gradient_accumulation_steps == 0
                or batch_index == len(loader)
            )
            if not is_boundary:
                continue
            torch.nn.utils.clip_grad_norm_([boundary_rows], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_update += 1
            event = {
                "epoch": epoch + 1,
                "update": global_update,
                "loss": float(loss.detach().cpu()),
                "learning_rate": scheduler.get_last_lr()[0],
                "cumulative_positive_top1": (
                    positive_predictions / positive_examples
                ),
                "cumulative_negative_boundary_top1": (
                    negative_boundary_predictions / negative_examples
                    if negative_examples
                    else 0.0
                ),
            }
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)

    trained_boundary_rows = boundary_rows.detach().cpu()
    unwrapped_base = model.unload()
    if hasattr(unwrapped_base, "peft_config"):
        delattr(unwrapped_base, "peft_config")
    repair_config = LoraConfig.from_pretrained(args.source_adapter)
    repair_config.inference_mode = False
    repair_config.trainable_token_indices = {
        output_module_name: list(required_ids),
    }
    repaired_model = get_peft_model(unwrapped_base, repair_config)
    initialized_state = get_peft_model_state_dict(repaired_model)
    initialized_state.update(source_state)
    incompatible = set_peft_model_state_dict(repaired_model, initialized_state)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"unexpected source-adapter weights: {incompatible.unexpected_keys}"
        )
    token_parameters = [
        parameter
        for name, parameter in repaired_model.named_parameters()
        if "trainable_tokens_delta" in name
    ]
    if len(token_parameters) != 1:
        raise ValueError(
            f"expected one trainable-token parameter, found {len(token_parameters)}"
        )
    token_parameters[0].data.copy_(
        trained_boundary_rows.to(
            device=token_parameters[0].device,
            dtype=token_parameters[0].dtype,
        )
    )
    repaired_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    saved_state = load_peft_weights(args.output_dir, device="cpu")
    changed_source_keys = [
        name
        for name, value in source_state.items()
        if name not in saved_state or not torch.equal(value.cpu(), saved_state[name].cpu())
    ]
    if changed_source_keys:
        raise ValueError(
            f"source LoRA tensors changed during carrier repair: {changed_source_keys[:5]}"
        )
    saved_token_keys = [
        name for name in saved_state if "trainable_tokens_delta" in name
    ]
    if len(saved_token_keys) != 1:
        raise ValueError(
            f"expected one saved trainable-token tensor, found {saved_token_keys}"
        )
    manifest = {
        "repair_type": "exact-output-token-rows-v1",
        "base_model": args.base_model,
        "source_adapter": args.source_adapter,
        "source_adapter_sha256": sha256_file(
            Path(args.source_adapter) / "adapter_model.safetensors"
        ),
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "selected_records": len(selected),
        "complete_records": complete_record_count,
        "eligible_records": len(eligible),
        "cutoff_len": args.cutoff_len,
        "max_sequence_length": args.max_sequence_length,
        "selected_sequence_length": {
            "min": min(len(example["input_ids"]) for example in encoded),
            "max": max(len(example["input_ids"]) for example in encoded),
            "mean": sum(len(example["input_ids"]) for example in encoded) / len(encoded),
        },
        "selected_tool_counts": dict(sorted(selected_tool_counts.items())),
        "boundary_token_ids": boundary_ids,
        "output_module_name": output_module_name,
        "trainable_parameters": trainable_count,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "negative_positions_per_record": args.negative_positions_per_record,
        "optimizer_updates": global_update,
        "training_positive_top1": positive_predictions / positive_examples,
        "training_negative_boundary_top1": (
            negative_boundary_predictions / negative_examples
            if negative_examples
            else 0.0
        ),
        "seed": args.seed,
        "source_lora_frozen": True,
        "source_lora_exact_tensor_match": True,
        "saved_trainable_token_key": saved_token_keys[0],
    }
    (args.output_dir / "repair_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
