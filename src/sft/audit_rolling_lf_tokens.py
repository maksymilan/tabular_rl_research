#!/usr/bin/env python3
"""Exact LLaMA-Factory token audit for last-turn-only rolling SFT records.

Run this in the same LLaMA-Factory environment used for training. A retained record has its
final assistant target preserved byte-for-token after the real template, masking, and cutoff logic.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams.data_args import DataArguments


ROLE_MAP = {"human": "user", "gpt": "assistant"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def encoded_target(template, tokenizer, messages: list[dict], system: str | None, mask_history: bool) -> list[int]:
    processed = template.mm_plugin.process_messages(messages, [], [], [], None)
    pairs = template.encode_multiturn(
        tokenizer,
        processed,
        system,
        None,
        mask_history and not template.preserve_thinking,
    )
    if mask_history:
        pairs = pairs[::-1]
    if not pairs:
        raise ValueError("template produced no assistant target")
    return pairs[0][1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--template", default="qwen")
    args = parser.parse_args()
    if args.cutoff_len <= 0:
        parser.error("--cutoff-len must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    data_args = DataArguments(template=args.template, cutoff_len=args.cutoff_len, mask_history=True)
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    processor = SupervisedDatasetProcessor(template, tokenizer, None, data_args)

    kept, dropped, lengths, target_lengths = [], [], [], []
    for row in read_jsonl(args.input):
        conversations = row["conversations"]
        messages = [{"role": ROLE_MAP[item["from"]], "content": item["value"]} for item in conversations]
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            raise ValueError(f"invalid final target: {row.get('metadata', {}).get('record_id')}")
        expected = encoded_target(template, tokenizer, messages, row.get("system"), mask_history=True)
        input_ids, labels = processor._encode_data_example(
            prompt=messages[:-1],
            response=[messages[-1]],
            system=row.get("system"),
            tools=None,
            images=[],
            videos=[],
            audios=[],
        )
        observed = [token for token in labels if token != IGNORE_INDEX]
        record_id = row.get("metadata", {}).get("record_id")
        if observed == expected:
            kept.append(row)
            lengths.append(len(input_ids))
            target_lengths.append(len(observed))
        else:
            dropped.append({
                "record_id": record_id,
                "encoded_length": len(input_ids),
                "expected_target_tokens": len(expected),
                "retained_target_tokens": len(observed),
            })

    write_jsonl(args.out, kept)
    manifest = {
        "input": str(args.input),
        "output": str(args.out),
        "model": args.model,
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "mask_history": True,
        "total": len(kept) + len(dropped),
        "kept_complete_final_target": len(kept),
        "dropped_truncated_final_target": len(dropped),
        "encoded_lengths": {
            "min": min(lengths) if lengths else 0,
            "p50": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            "p90": sorted(lengths)[min(len(lengths) - 1, int(.9 * len(lengths)))] if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "target_lengths": {
            "min": min(target_lengths) if target_lengths else 0,
            "max": max(target_lengths) if target_lengths else 0,
        },
        "dropped": dropped,
        "tool_hist": dict(Counter(row.get("metadata", {}).get("source_step_id", "unknown") for row in kept)),
    }
    args.out.with_suffix(".token_audit.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
