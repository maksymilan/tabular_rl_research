#!/usr/bin/env python3
"""Build a deterministic, episode-complete SFT pilot under the current student prompt.

The source records must already be causal, replay-verified, last-turn-only ShareGPT examples.
This builder does not change structured actions, reasoning, observations, or human-authored
history. It:

1. keeps only episodes with no token-gate omissions and one final terminal action;
2. selects deterministic per-difficulty episode quotas; and
3. replaces the student ``system`` field and deterministically renders the active action carrier.

This is intended for controlled prompt/training diagnostics, not for manufacturing new
trajectories from gold SQL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from protocol import (
    PROTOCOL_VERSION,
    STUDENT_PROMPT_CANONICAL,
    STUDENT_PROMPT_VARIANTS,
    protocol_hash,
    student_runtime_system_prompt,
    tool_schema_hash,
)
from action_carrier import ACTIVE_ACTION_CARRIER
from assemble_student_training_dataset import rerender_conversations
from sft_dataset_registry import write_sharegpt_dataset_info


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_quota(values: list[str]) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for value in values:
        label, separator, raw_count = value.partition("=")
        if not separator or not label or not raw_count.isdigit():
            raise ValueError(f"invalid quota {value!r}; expected LABEL=COUNT")
        count = int(raw_count)
        if label in quotas:
            raise ValueError(f"duplicate quota label: {label}")
        if count <= 0:
            raise ValueError(f"quota must be positive: {value}")
        quotas[label] = count
    if not quotas:
        raise ValueError("at least one quota is required")
    return quotas


def numeric_step_id(value: Any) -> int:
    match = re.fullmatch(r"step_(\d+)", str(value))
    if not match:
        raise ValueError(f"invalid source_step_id: {value!r}")
    return int(match.group(1))


def episode_has_terminal(items: list[tuple[dict[str, Any], dict[str, Any]]]) -> bool:
    ordered = sorted(items, key=lambda item: numeric_step_id(item[1]["source_step_id"]))
    step_numbers = [numeric_step_id(index["source_step_id"]) for _, index in ordered]
    terminal_positions = [
        position
        for position, (_, index) in enumerate(ordered)
        if index.get("tool_name") == "answer_from_context"
    ]
    return (
        step_numbers == sorted(set(step_numbers))
        and terminal_positions == [len(ordered) - 1]
    )


def stable_rank(seed: int, episode_id: str) -> str:
    return sha256_text(f"{seed}:{episode_id}")


def build(
    input_path: Path,
    index_path: Path,
    source_manifest_path: Path,
    out_path: Path,
    index_out_path: Path,
    *,
    dataset_name: str,
    quotas: dict[str, int],
    seed: int,
    student_prompt_variant: str = STUDENT_PROMPT_CANONICAL,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset_name):
        raise ValueError("dataset_name must contain only letters, digits, '.', '_' or '-'")

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    dropped_record_ids = source_manifest.get("dropped_record_ids")
    if not isinstance(dropped_record_ids, list):
        raise ValueError("source manifest must contain dropped_record_ids")
    episodes_with_token_omissions = {
        str(record_id).rsplit("_step_", 1)[0]
        for record_id in dropped_record_ids
        if "_step_" in str(record_id)
    }

    records = read_jsonl(input_path)
    indexes = read_jsonl(index_path)
    if len(records) != len(indexes):
        raise ValueError("SFT records and index rows have different lengths")

    by_episode: defaultdict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    record_ids: set[str] = set()
    source_prompt_hashes: Counter[str] = Counter()
    for record, index in zip(records, indexes):
        metadata = record.get("metadata") or {}
        record_id = metadata.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("record is missing metadata.record_id")
        if record_id in record_ids:
            raise ValueError(f"duplicate record id: {record_id}")
        record_ids.add(record_id)
        if record_id != index.get("record_id"):
            raise ValueError(f"record/index order mismatch at {record_id}")
        episode_id = index.get("source_episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError(f"{record_id}: index is missing source_episode_id")
        system = record.get("system")
        if not isinstance(system, str) or not system:
            raise ValueError(f"{record_id}: record is missing system prompt")
        source_prompt_hashes[sha256_text(system)] += 1
        by_episode[episode_id].append((record, index))

    complete_by_difficulty: defaultdict[str, list[str]] = defaultdict(list)
    for episode_id, items in by_episode.items():
        if episode_id in episodes_with_token_omissions or not episode_has_terminal(items):
            continue
        labels = {str(index.get("source_difficulty") or "unknown") for _, index in items}
        if len(labels) != 1:
            raise ValueError(f"{episode_id}: inconsistent difficulty labels: {sorted(labels)}")
        complete_by_difficulty[next(iter(labels))].append(episode_id)

    selected: set[str] = set()
    availability: dict[str, int] = {}
    for label, quota in quotas.items():
        candidates = complete_by_difficulty.get(label, [])
        availability[label] = len(candidates)
        if len(candidates) < quota:
            raise ValueError(
                f"difficulty {label!r} has {len(candidates)} complete episodes, fewer than {quota}"
            )
        ranked = sorted(candidates, key=lambda episode_id: stable_rank(seed, episode_id))
        selected.update(ranked[:quota])

    system_prompt = student_runtime_system_prompt(
        context_mode="rolling-legal-history",
        student_prompt_variant=student_prompt_variant,
    )
    selected_records: list[dict[str, Any]] = []
    selected_indexes: list[dict[str, Any]] = []
    selected_difficulty: Counter[str] = Counter()
    selected_tools: Counter[str] = Counter()
    source_carriers: Counter[str] = Counter()
    for record, index in zip(records, indexes):
        if index["source_episode_id"] not in selected:
            continue
        rerendered = {
            **record,
            "system": system_prompt,
            "conversations": rerender_conversations(
                record["conversations"],
                source_carriers,
            ),
        }
        selected_records.append(rerendered)
        selected_indexes.append(index)
        selected_difficulty[str(index.get("source_difficulty") or "unknown")] += 1
        selected_tools[str(index.get("tool_name") or "unknown")] += 1
    if len(selected) != sum(quotas.values()):
        raise RuntimeError("selected episode count does not match requested quotas")
    if any(
        episode_id in episodes_with_token_omissions
        or not episode_has_terminal(by_episode[episode_id])
        for episode_id in selected
    ):
        raise RuntimeError("selected cohort contains an incomplete episode")

    write_jsonl_atomic(out_path, selected_records)
    write_jsonl_atomic(index_out_path, selected_indexes)
    snippet_path, registry_path = write_sharegpt_dataset_info(out_path, dataset_name)

    manifest = {
        "selection": "deterministic-stratified-complete-episode-prompt-pilot",
        "selection_policy": {
            "source_requirement": "existing causal replay-verified last-turn-only records",
            "episode_requirement": (
                "no source token-gate omissions and one final answer_from_context target; "
                "step gaps caused by rejected actions are preserved"
            ),
            "mutation": (
                "system field and assistant carrier serialization only; structured actions, "
                "reasoning, observations, and human history unchanged"
            ),
            "rank": "sha256(seed:source_episode_id)",
            "seed": seed,
            "quotas": quotas,
        },
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "index_input": str(index_path),
        "index_input_sha256": sha256_file(index_path),
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "output": str(out_path),
        "output_sha256": sha256_file(out_path),
        "index": str(index_out_path),
        "index_sha256": sha256_file(index_out_path),
        "dataset_name": dataset_name,
        "dataset_info_snippet": str(snippet_path),
        "dataset_info_registry": str(registry_path),
        "protocol_version": PROTOCOL_VERSION,
        "prompt_role": "student-runtime",
        "student_prompt_variant": student_prompt_variant,
        "action_carrier": ACTIVE_ACTION_CARRIER,
        "source_action_carriers": dict(sorted(source_carriers.items())),
        "student_runtime_prompt_characters": len(system_prompt),
        "student_runtime_prompt_sha256": sha256_text(system_prompt),
        "source_system_prompt_sha256s": dict(sorted(source_prompt_hashes.items())),
        "tool_schema_sha256": tool_schema_hash(),
        "protocol_hash": protocol_hash(system_prompt),
        "available_complete_episodes": availability,
        "selected_episodes": len(selected),
        "selected_episode_ids": sorted(selected),
        "records": len(selected_records),
        "difficulty_targets": dict(sorted(selected_difficulty.items())),
        "tool_hist": dict(selected_tools.most_common()),
        "terminal_targets": selected_tools["answer_from_context"],
        "feedback_recovery_targets": sum(
            bool((record.get("metadata") or {}).get("feedback_recovery"))
            for record in selected_records
        ),
        "structured_actions_unchanged": True,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--quota", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--student-prompt-variant",
        choices=STUDENT_PROMPT_VARIANTS,
        default=STUDENT_PROMPT_CANONICAL,
    )
    args = parser.parse_args()
    manifest = build(
        args.input.resolve(),
        args.index.resolve(),
        args.source_manifest.resolve(),
        args.out.resolve(),
        args.index_out.resolve(),
        dataset_name=args.dataset_name,
        quotas=parse_quota(args.quota),
        seed=args.seed,
        student_prompt_variant=args.student_prompt_variant,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
