#!/usr/bin/env python3
"""Build a Qwen3.5-tokenized <=4k SFT subset on the remote server."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def render_record_text(record: dict) -> str:
    system = record.get("system") or ""
    messages = "\n".join(message.get("value", "") for message in record.get("conversations", []))
    return system + "\n" + messages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-info", required=True)
    parser.add_argument("--dataset-name", default="spider_tools_v8_pilot_ready_qwen35_4k")
    parser.add_argument("--model", default="/home/dengyan/models/Qwen3.5-9B")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    dropped: list[dict] = []
    with open(args.input, encoding="utf-8") as source, output.open("w", encoding="utf-8") as sink:
        for index, line in enumerate(source):
            record = json.loads(line)
            token_count = len(
                tokenizer(render_record_text(record), add_special_tokens=False).input_ids
            )
            if token_count <= args.max_tokens:
                sink.write(line)
                kept += 1
            else:
                dropped.append({"index": index, "tokens": token_count})

    dataset_info = {}
    info_path = Path(args.dataset_info)
    if info_path.exists():
        dataset_info = json.loads(info_path.read_text(encoding="utf-8"))
    dataset_info[args.dataset_name] = {
        "file_name": output.name,
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
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n")

    print(json.dumps({"kept": kept, "dropped": len(dropped), "dropped_examples": dropped}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
