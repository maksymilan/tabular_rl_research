#!/usr/bin/env python3
"""Filter ShareGPT SFT records by exact tokenizer content length."""
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


def dataset_entry(file_name: str) -> dict:
    return {
        "file_name": file_name,
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-info", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--max-content-tokens", type=int, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    args = parser.parse_args()

    if args.max_content_tokens < 1:
        parser.error("--max-content-tokens must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    dropped: list[dict] = []
    max_kept = 0
    with input_path.open(encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as output:
        for source_index, line in enumerate(source):
            if not line.strip():
                continue
            record = json.loads(line)
            length = content_tokens(tokenizer, record)
            if length > args.max_content_tokens:
                dropped.append(
                    {"source_index": source_index, "raw_content_tokens": length}
                )
                continue
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            max_kept = max(max_kept, length)

    dataset_info_path = Path(args.dataset_info)
    registry = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    registry[args.dataset_name] = dataset_entry(output_path.name)
    dataset_info_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = {
        "source": str(input_path),
        "model_tokenizer": args.model,
        "max_content_tokens": args.max_content_tokens,
        "kept": kept,
        "dropped": len(dropped),
        "dropped_records": dropped,
        "max_kept_content_tokens": max_kept,
        "output": str(output_path),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
