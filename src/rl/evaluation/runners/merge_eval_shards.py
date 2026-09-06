#!/usr/bin/env python3
"""Validate and merge two disjoint pass@1 evaluation shards."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--shard", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--expected", type=int, help="expected records (defaults to source length)")
    args = parser.parse_args()
    examples = [
        json.loads(line)
        for line in args.examples.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_index = {int(row.get("example_index", i)): row for i, row in enumerate(examples)}
    expected_count = args.expected if args.expected is not None else len(examples)
    expected = set(range(expected_count))
    if set(by_index) != expected:
        raise ValueError(f"source input index set is not exactly 0..{expected_count - 1}")
    rows: list[dict] = []
    seen: set[int] = set()
    for path in args.shard:
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        shard_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if shard_manifest.get("requested_size") != len(shard_lines):
            raise ValueError(f"result/manifest count mismatch in {manifest_path}")
        if shard_manifest.get("temperature") != 0.0 or shard_manifest.get("top_p") != 1.0:
            raise ValueError(f"non-greedy decode in {manifest_path}")
        if shard_manifest.get("n_samples") != 1 or shard_manifest.get("pass_k") != [1]:
            raise ValueError(f"unexpected pass@k configuration in {manifest_path}")
        shard_seen: set[int] = set()
        for line in shard_lines:
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row.get("example_index", -1))
            if index in seen:
                raise ValueError(f"duplicate result index: {index}")
            if index not in by_index:
                raise ValueError(f"unexpected result index: {index}")
            if row.get("question") != by_index[index].get("question"):
                raise ValueError(f"question mismatch at index {index}")
            samples = row.get("samples") or []
            if len(samples) != 1:
                raise ValueError(f"expected one sample at index {index}")
            if row.get("protocol_version") != "version26" or row.get("protocol_hash") != "4da19387399bd3a5":
                raise ValueError(f"protocol mismatch at index {index}")
            if row.get("temperature") != 0.0 or row.get("top_p") != 1.0:
                raise ValueError(f"non-greedy result at index {index}")
            if row.get("n_samples") != 1 or row.get("pass_k") != [1]:
                raise ValueError(f"unexpected pass@k result at index {index}")
            if str(samples[0].get("failure_type") or "").casefold() == "api_error":
                raise ValueError(f"API error at index {index}")
            seen.add(index)
            shard_seen.add(index)
            rows.append(row)
        selected = shard_manifest.get("selected_indices")
        if selected is not None and set(map(int, selected)) != shard_seen:
            raise ValueError(f"manifest/result index mismatch in {manifest_path}")
    if seen != expected:
        raise ValueError(f"incomplete coverage: {len(seen)}/{expected_count}")
    rows.sort(key=lambda row: int(row["example_index"]))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "all.jsonl").open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    correct = sum(bool(row["samples"][0].get("correct")) for row in rows)
    legal = sum(bool(row["samples"][0].get("legal")) for row in rows)
    summary = {
        "total": expected_count,
        "correct": correct,
        "average_legal_samples": legal / expected_count,
        "completed_example_indices": list(range(expected_count)),
        "pass_at": {"1": {"correct": correct, "total": expected_count}},
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    adapter_file = args.adapter / "adapter_model.safetensors"
    adapter_sha = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "qwen3-v26-saam-fourlevel-dataparallel-eval-v1",
        "dataset": str(args.examples),
        "records": expected_count,
        "shards": [str(path.parent) for path in args.shard],
        "adapter_path": str(args.adapter),
        "adapter_sha256": adapter_sha,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "max_steps": 30,
        "n_samples": 1,
        "denotation_comparison": "bird-set",
        "correct": correct,
        "legal": legal,
    }
    (args.output / "evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"records": expected_count, "correct": correct, "legal": legal, "accuracy": correct / expected_count}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
