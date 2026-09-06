"""Reusable, data-set agnostic planning for Atomic v26 evaluation.

The actual model serving/evaluation process is intentionally kept outside this
module.  This module owns the part that used to be copied into each launcher:
input validation, deterministic partitioning and the immutable run plan.  A
launcher for a smoke run or a full evaluation can therefore pass different
data, split and GPU counts without carrying another copy of this logic.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

try:
    from rl.shared.io import read_jsonl, sha256_file
except ModuleNotFoundError:  # direct script execution
    from shared.io import read_jsonl, sha256_file


PROTOCOL_VERSION = "version26"
PROTOCOL_HASH = "4da19387399bd3a5"
PROMPT_SHA256 = "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"


def validate_gpu_ids(gpu_ids: Sequence[int], *, allowed: Sequence[int] | None = None) -> tuple[int, ...]:
    ids = tuple(int(value) for value in gpu_ids)
    if not ids:
        raise ValueError("at least one evaluation GPU is required")
    if any(value < 0 for value in ids):
        raise ValueError("GPU ids must be non-negative")
    if len(set(ids)) != len(ids):
        raise ValueError(f"GPU ids must be unique: {ids}")
    if allowed is not None:
        permitted = set(int(value) for value in allowed)
        unknown = sorted(set(ids) - permitted)
        if unknown:
            raise ValueError(f"GPU ids outside allowlist {sorted(permitted)}: {unknown}")
    return ids


def partition_rows(
    rows: Sequence[dict[str, Any]], shard_count: int, *, policy: str = "round_robin"
) -> list[list[dict[str, Any]]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if policy not in {"round_robin", "contiguous"}:
        raise ValueError(f"unsupported partition policy: {policy}")
    shards: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    if policy == "round_robin":
        for ordinal, row in enumerate(rows):
            # Existing example_index values are retained.  Round-robin
            # assignment balances arbitrary data splits.
            shards[ordinal % shard_count].append(row)
    else:
        quotient, remainder = divmod(len(rows), shard_count)
        start = 0
        for index in range(shard_count):
            width = quotient + (index < remainder)
            shards[index].extend(rows[start : start + width])
            start += width
    return shards


@dataclass(frozen=True)
class EvaluationPlan:
    dataset: str
    dataset_sha256: str
    split: str
    records: int
    checkpoint: str
    gpu_ids: tuple[int, ...]
    shard_sizes: tuple[int, ...]
    partition_policy: str = "round_robin"
    protocol_version: str = PROTOCOL_VERSION
    protocol_hash: str = PROTOCOL_HASH
    student_prompt_sha256: str = PROMPT_SHA256
    temperature: float = 0.0
    top_p: float = 1.0
    max_steps: int = 30
    max_tokens: int = 2048
    n_samples: int = 1

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["gpu_ids"] = list(self.gpu_ids)
        value["shard_sizes"] = list(self.shard_sizes)
        return value


def make_plan(
    dataset: Path,
    *,
    split: str,
    checkpoint: Path,
    gpu_ids: Sequence[int],
    allowed_gpu_ids: Sequence[int] | None = None,
    n_samples: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    partition: str = "round_robin",
) -> tuple[EvaluationPlan, list[list[dict[str, Any]]]]:
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    if not split.strip():
        raise ValueError("split must not be empty")
    ids = validate_gpu_ids(gpu_ids, allowed=allowed_gpu_ids)
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rows = read_jsonl(dataset)
    if not rows:
        raise ValueError(f"evaluation input is empty: {dataset}")
    shards = partition_rows(rows, len(ids), policy=partition)
    plan = EvaluationPlan(
        dataset=str(dataset.resolve()),
        dataset_sha256=sha256_file(dataset),
        split=split,
        records=len(rows),
        checkpoint=str(checkpoint),
        gpu_ids=ids,
        shard_sizes=tuple(len(shard) for shard in shards),
        partition_policy=partition,
        n_samples=n_samples,
        temperature=temperature,
        top_p=top_p,
    )
    return plan, shards


def write_plan(
    plan: EvaluationPlan,
    shards: Sequence[Sequence[dict[str, Any]]],
    output_dir: Path,
) -> None:
    if output_dir.exists():
        raise FileExistsError(f"evaluation plan output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    for index, shard in enumerate(shards):
        path = output_dir / f"shard_{index:02d}.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in shard),
            encoding="utf-8",
        )
    (output_dir / "evaluation_plan.json").write_text(
        json.dumps(plan.as_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gpu-ids", required=True, help="comma-separated physical GPU ids")
    parser.add_argument("--allowed-gpu-ids", help="optional comma-separated allowlist")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--partition", choices=("round_robin", "contiguous"), default="round_robin")
    args = parser.parse_args()
    gpu_ids = [int(value) for value in args.gpu_ids.split(",") if value.strip()]
    allowed = None if args.allowed_gpu_ids is None else [int(value) for value in args.allowed_gpu_ids.split(",")]
    plan, shards = make_plan(
        args.dataset,
        split=args.split,
        checkpoint=Path(args.checkpoint),
        gpu_ids=gpu_ids,
        allowed_gpu_ids=allowed,
        n_samples=args.n_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        partition=args.partition,
    )
    write_plan(plan, shards, args.output_dir)
    print(json.dumps(plan.as_dict(), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
