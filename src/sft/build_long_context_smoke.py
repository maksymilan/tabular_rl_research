#!/usr/bin/env python3
"""Build a deterministic SFT smoke dataset from the longest tokenized record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def content_tokens(tokenizer, record: dict) -> int:
    parts = [record["system"], *(message["value"] for message in record["conversations"])]
    return sum(
        len(tokenizer.encode(part, add_special_tokens=False))
        for part in parts
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-info", required=True)
    parser.add_argument("--dataset-name", default="spider_tools_v1_long_smoke")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_path = Path(args.input)
    longest: tuple[int, int, dict] | None = None
    with input_path.open(encoding="utf-8") as source:
        for index, line in enumerate(source):
            if not line.strip():
                continue
            record = json.loads(line)
            length = content_tokens(tokenizer, record)
            if longest is None or length > longest[0]:
                longest = (length, index, record)
    if longest is None:
        raise ValueError(f"no records found in {input_path}")

    token_count, source_index, record = longest
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for _ in range(args.repeats):
            output.write(json.dumps(record, ensure_ascii=False) + "\n")

    dataset_info_path = Path(args.dataset_info)
    registry = (
        json.loads(dataset_info_path.read_text(encoding="utf-8"))
        if dataset_info_path.exists()
        else {}
    )
    registry[args.dataset_name] = {
        "file_name": output_path.name,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
            "observation_tag": "observation",
        },
    }
    dataset_info_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "source": str(input_path),
        "source_index": source_index,
        "raw_content_tokens": token_count,
        "repeats": args.repeats,
        "model_tokenizer": args.model,
        "output": str(output_path),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
