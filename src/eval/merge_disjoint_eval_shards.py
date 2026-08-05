#!/usr/bin/env python3
"""Merge complete, disjoint evaluation shards without altering sample records."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def atomic_write(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected", required=True, type=int)
    args = parser.parse_args()

    rows: dict[int, dict[str, Any]] = {}
    manifests = []
    for shard in args.shard:
        manifests.append(json.loads((shard / "manifest.json").read_text()))
        for row in load_jsonl(shard / "all.jsonl"):
            index = int(row["example_index"])
            if index in rows:
                raise ValueError(f"duplicate example_index across shards: {index}")
            rows[index] = row
    if len(rows) != args.expected or set(rows) != set(range(args.expected)):
        missing = sorted(set(range(args.expected)) - set(rows))
        raise ValueError(
            f"merged rows are incomplete: {len(rows)}/{args.expected}; missing={missing[:20]}"
        )

    immutable = (
        "n_samples",
        "pass_k",
        "temperature",
        "top_p",
        "protocol_version",
        "denotation_comparison",
        "record_logprobs",
        "top_logprobs",
        "history_turns",
    )
    baseline = manifests[0]
    for manifest in manifests[1:]:
        for key in immutable:
            if manifest.get(key) != baseline.get(key):
                raise ValueError(f"shard manifest mismatch for {key}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_path = args.output_dir / "all.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    if all_path.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite an existing merged evaluation")
    atomic_write(
        all_path,
        "".join(json.dumps(rows[index], ensure_ascii=False) + "\n" for index in sorted(rows)),
    )
    merged_manifest = dict(baseline)
    merged_manifest["selected_indices"] = list(range(args.expected))
    merged_manifest["selection_shards"] = [str(path.resolve()) for path in args.shard]
    merged_manifest["merged_disjoint_shards"] = True
    atomic_write(
        manifest_path,
        json.dumps(merged_manifest, ensure_ascii=False, indent=2) + "\n",
    )
    print(json.dumps({"rows": len(rows), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
