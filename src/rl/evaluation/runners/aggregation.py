"""Reusable validation and aggregation for Atomic v26 pass@1 shards."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def merge_pass1_shards(*, examples: Path, shards: list[Path], output: Path,
                       adapter: Path, expected_count: int | None = None) -> dict[str, Any]:
    source = _jsonl(examples)
    by_index = {int(row.get("example_index", i)): row for i, row in enumerate(source)}
    count = len(source) if expected_count is None else expected_count
    expected = set(range(count))
    if set(by_index) != expected:
        raise ValueError(f"source input index set is not exactly 0..{count - 1}")
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for path in shards:
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest_path = path.parent / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_rows = _jsonl(path)
        if manifest.get("requested_size") != len(shard_rows):
            raise ValueError(f"result/manifest count mismatch in {manifest_path}")
        if manifest.get("temperature") != 0.0 or manifest.get("top_p") != 1.0:
            raise ValueError(f"non-greedy decode in {manifest_path}")
        if manifest.get("n_samples") != 1 or manifest.get("pass_k") != [1]:
            raise ValueError(f"unexpected pass@k configuration in {manifest_path}")
        shard_seen: set[int] = set()
        for row in shard_rows:
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
            if (row.get("protocol_version"), row.get("protocol_hash")) != ("version26", "4da19387399bd3a5"):
                raise ValueError(f"protocol mismatch at index {index}")
            if row.get("temperature") != 0.0 or row.get("top_p") != 1.0 or row.get("n_samples") != 1 or row.get("pass_k") != [1]:
                raise ValueError(f"unexpected decode configuration at index {index}")
            if str(samples[0].get("failure_type") or "").casefold() == "api_error":
                raise ValueError(f"API error at index {index}")
            seen.add(index)
            shard_seen.add(index)
            rows.append(row)
        selected = manifest.get("selected_indices")
        if selected is not None and set(map(int, selected)) != shard_seen:
            raise ValueError(f"manifest/result index mismatch at {manifest_path}")
    if seen != expected:
        raise ValueError(f"incomplete coverage: {len(seen)}/{count}")
    rows.sort(key=lambda row: int(row["example_index"]))
    output.mkdir(parents=True, exist_ok=False)
    (output / "all.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    correct = sum(bool(row["samples"][0].get("correct")) for row in rows)
    legal = sum(bool(row["samples"][0].get("legal")) for row in rows)
    summary = {"total": count, "correct": correct, "average_legal_samples": legal / count,
               "completed_example_indices": list(range(count)),
               "pass_at": {"1": {"correct": correct, "total": count}}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    adapter_file = adapter / "adapter_model.safetensors"
    adapter_sha = hashlib.sha256(adapter_file.read_bytes()).hexdigest()
    eval_manifest = {"schema_version": "qwen3-v26-saam-fourlevel-dataparallel-eval-v1",
                     "dataset": str(examples), "records": count,
                     "shards": [str(path.parent) for path in shards], "adapter_path": str(adapter),
                     "adapter_sha256": adapter_sha, "temperature": 0.0, "top_p": 1.0,
                     "max_tokens": 2048, "max_steps": 30, "n_samples": 1,
                     "denotation_comparison": "bird-set", "correct": correct, "legal": legal}
    (output / "evaluation_manifest.json").write_text(json.dumps(eval_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"records": count, "correct": correct, "legal": legal, "accuracy": correct / count}
