#!/usr/bin/env python3
"""Compare raw carrier generation with and without a PEFT adapter.

The probe consumes the exact initial messages recorded by ``rollout_passk.py``.
It intentionally performs no tool parsing or recovery so malformed carrier
tokens remain visible in the output artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "sft"))

from protocol import ProtocolError, parse_assistant_strict  # noqa: E402


def has_active_action_carrier(completion: str) -> bool:
    without_chat_eos = completion.removesuffix("<|im_end|>").strip()
    try:
        parse_assistant_strict(without_chat_eos)
    except ProtocolError:
        return False
    return True


def load_records(path: Path, example_indices: set[int]) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            example_index = int(record["example_index"])
            if example_index in example_indices:
                records[example_index] = record
    missing = example_indices - records.keys()
    if missing:
        raise ValueError(f"example indices are absent from {path}: {sorted(missing)}")
    return [records[index] for index in sorted(records)]


def generate(
    model: PeftModel,
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    adapter_enabled: bool,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {name: value.to(model.device) for name, value in encoded.items()}
    context = nullcontext() if adapter_enabled else model.disable_adapter()
    with context, torch.inference_mode():
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    completion_ids = generated[0, encoded["input_ids"].shape[1] :].tolist()
    completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
    return {
        "adapter_enabled": adapter_enabled,
        "completion": completion,
        "completion_token_ids": completion_ids,
        "has_active_action_carrier": has_active_action_carrier(completion),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--rollout-jsonl", type=Path, required=True)
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument(
        "--example-indices",
        help="comma-separated indices; overrides --example-index",
    )
    parser.add_argument(
        "--modes",
        default="base,adapter",
        help="comma-separated subset of base,adapter",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    example_indices = (
        {int(value) for value in args.example_indices.split(",")}
        if args.example_indices
        else {args.example_index}
    )
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    if not modes or set(modes) - {"base", "adapter"}:
        raise ValueError("modes must be a non-empty subset of base,adapter")
    records = load_records(args.rollout_jsonl, example_indices)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=True,
    )
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
    model = PeftModel.from_pretrained(base_model, args.adapter, is_trainable=False)
    model.eval()

    result = {
        "base_model": args.base_model,
        "adapter": args.adapter,
        "rollout_jsonl": str(args.rollout_jsonl),
        "probes": [],
    }
    for record in records:
        generations = []
        for mode in modes:
            generations.append(
                generate(
                    model,
                    tokenizer,
                    record["initial_model_input"],
                    adapter_enabled=mode == "adapter",
                    max_new_tokens=args.max_new_tokens,
                )
            )
        result["probes"].append(
            {
                "example_index": record["example_index"],
                "question": record.get("question"),
                "generations": generations,
            }
        )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
