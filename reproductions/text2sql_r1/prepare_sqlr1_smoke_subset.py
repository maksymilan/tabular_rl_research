#!/usr/bin/env python3
"""Select a deterministic, short-prompt subset for the 2x24GB GRPO smoke test."""

import argparse
import json
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--max-prompt-tokens", type=int, default=768)
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    lengths = [
        len(tokenizer.apply_chat_template(prompt, add_generation_prompt=True))
        for prompt in frame["prompt"]
    ]
    ranked = sorted(enumerate(lengths), key=lambda item: (item[1], item[0]))
    selected = [
        (index, length)
        for index, length in ranked
        if length <= args.max_prompt_tokens
    ][: args.rows]
    if len(selected) != args.rows:
        raise RuntimeError(
            f"Found only {len(selected)} prompts <= {args.max_prompt_tokens} tokens"
        )

    indices = [index for index, _ in selected]
    output = frame.iloc[indices].copy().reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output, index=False)
    manifest = {
        "source": str(args.input),
        "output": str(args.output),
        "source_indices": indices,
        "prompt_token_lengths": [length for _, length in selected],
        "tokenizer": args.tokenizer,
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
