#!/usr/bin/env python3
"""Build the quality-filtered Qwen3 projection-v3 training corpus.

The source projection is immutable.  This selector combines two exact
LLaMA-Factory full-prefix token audits (8K and 16K), removes only the current
supervision target when its reasoning violates a small frozen quality policy,
and retains every otherwise valid bird-set-correct trajectory regardless of
strict/schema score.  It never rewrites model-visible text.
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

from sft_dataset_registry import write_sharegpt_dataset_info


POLICY_VERSION = "checkpoint-relalg-qwen3-projection-v3-quality-policy-v1"
TARGET_RE = re.compile(
    r"\A\s*<think>(?P<reasoning>.*?)</think>\s*(?P<action>\{.*\})\s*\Z",
    re.DOTALL,
)
PROVIDER_CARRIER_RE = re.compile(r"\breasoning_content\b|\btool_calls\b", re.I)
BENCHMARK_LEAK_RE = re.compile(
    r"\b(?:gold|reference|ground[- ]truth)\s+(?:sql|query|answer|result)\b"
    r"|\b(?:sql|query|answer|result)\s+(?:from|in)\s+(?:the\s+)?"
    r"(?:gold|reference|ground[- ]truth)\b"
    r"|\b(?:I\s+)?(?:remember|recall)(?:ed)?\b.{0,160}\b"
    r"(?:this|the)\s+(?:exact\s+)?(?:question|query|benchmark)\b"
    r"|\b(?:seen|recognize)\s+(?:this|the)\s+(?:exact\s+)?"
    r"(?:question|query|benchmark)\b",
    re.I | re.DOTALL,
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            rows.append(value)
    return rows


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


def record_id(row: dict[str, Any]) -> str:
    value = (row.get("metadata") or {}).get("record_id")
    if not isinstance(value, str) or not value:
        raise ValueError("canonical row is missing metadata.record_id")
    return value


def rejected_by_full_prefix(audit: dict[str, Any]) -> set[str]:
    if audit.get("filter_policy") != "full-prefix":
        raise ValueError("token audit was not produced with full-prefix policy")
    details = audit.get("details")
    if not isinstance(details, list):
        raise ValueError("token audit is missing details")
    rejected: set[str] = set()
    for item in details:
        valid = (
            item.get("target_status") == "complete"
            and item.get("processor_target_complete") is not False
            and item.get("current_source_tokens_kept")
            == item.get("current_source_tokens_original")
            and (
                not item.get("history_pairs_original")
                or item.get("first_pair_complete") is True
            )
        )
        if valid:
            continue
        item_id = item.get("record_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("token audit detail is missing record_id")
        rejected.add(item_id)
    return rejected


def repeated_ngram_ratio(text: str, n: int = 8) -> float:
    words = re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
    if len(words) < n:
        return 0.0
    grams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def parse_target(row: dict[str, Any], *, item_id: str) -> tuple[str, dict[str, Any]]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"{item_id}: conversations are missing")
    target = conversations[-1]
    if target.get("from") != "gpt" or not isinstance(target.get("value"), str):
        raise ValueError(f"{item_id}: final assistant target is missing")
    match = TARGET_RE.fullmatch(target["value"])
    if match is None:
        raise ValueError(f"{item_id}: target carrier is invalid")
    action = json.loads(match.group("action"))
    if set(action) != {"tool", "arguments"}:
        raise ValueError(f"{item_id}: target action envelope is invalid")
    return match.group("reasoning"), action


def quality_reasons(row: dict[str, Any], *, item_id: str) -> list[str]:
    reasoning, action = parse_target(row, item_id=item_id)
    reasons: list[str] = []
    if PROVIDER_CARRIER_RE.search(reasoning):
        reasons.append("provider_carrier_discussion")
    if BENCHMARK_LEAK_RE.search(reasoning):
        reasons.append("benchmark_answer_memory")
    if repeated_ngram_ratio(reasoning) >= 0.20:
        reasons.append("repetitive_reasoning_8gram_ge_0_20")

    conversations = row["conversations"]
    prior_actions: list[dict[str, Any]] = []
    for item in conversations[:-1]:
        if item.get("from") != "gpt" or not isinstance(item.get("value"), str):
            continue
        try:
            value = json.loads(item["value"])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            prior_actions.append(value)
    if prior_actions and prior_actions[-1] == action:
        reasons.append("adjacent_exact_successful_action_repeat")
    return sorted(set(reasons))


def artifact_tier(index: dict[str, Any]) -> tuple[str, float]:
    if index.get("strict_artifact_accuracy") is True:
        return "strict_exact", 1.0
    if index.get("schema_match") is True:
        return "value_correct_schema_exact", 0.75
    return "value_correct_schema_relaxed", 0.35


def _training_view(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"system": row.get("system"), "conversations": row.get("conversations")}
        for row in rows
    ]


def build(
    canonical: Path,
    index_path: Path,
    audit_8192_path: Path,
    audit_16384_path: Path,
    output_dir: Path,
    dataset_name: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reserved = [
        output_dir / f"{dataset_name}.manifest.json",
        output_dir / f"{dataset_name}_quality_excluded.jsonl",
    ]
    for name in (
        dataset_name,
        f"{dataset_name}_core_8192",
        f"{dataset_name}_supplement_8193_16384",
    ):
        reserved.extend(
            (
                output_dir / f"{name}.jsonl",
                output_dir / f"{name}_canonical.jsonl",
                output_dir / f"{name}_index.jsonl",
            )
        )
    collisions = [str(path) for path in reserved if path.exists()]
    if collisions:
        raise FileExistsError(f"refusing existing v3 output: {collisions[0]}")
    rows = read_jsonl(canonical)
    indexes = read_jsonl(index_path)
    if len(rows) != len(indexes):
        raise ValueError("canonical and index counts differ")
    ids = [record_id(row) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("canonical contains duplicate record ids")
    if ids != [item.get("record_id") for item in indexes]:
        raise ValueError("canonical and index order differ")

    audits = [
        json.loads(audit_8192_path.read_text(encoding="utf-8")),
        json.loads(audit_16384_path.read_text(encoding="utf-8")),
    ]
    for expected_cutoff, audit in zip((8192, 16384), audits):
        if audit.get("records") != len(rows) or audit.get("cutoff_len") != expected_cutoff:
            raise ValueError(f"{expected_cutoff} token audit identity differs")
    rejected_8192 = rejected_by_full_prefix(audits[0])
    rejected_16384 = rejected_by_full_prefix(audits[1])
    if not rejected_16384.issubset(rejected_8192):
        raise ValueError("16K full-prefix admission is not a superset of 8K admission")

    quality_exclusions: list[dict[str, Any]] = []
    kept_rows: list[dict[str, Any]] = []
    kept_indexes: list[dict[str, Any]] = []
    bucket_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    bucket_indexes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    reason_hist: Counter[str] = Counter()
    tier_hist: Counter[str] = Counter()
    bucket_hist: Counter[str] = Counter()
    tool_hist: Counter[str] = Counter()
    episodes_by_bucket: defaultdict[str, set[str]] = defaultdict(set)
    episode_tools: defaultdict[str, set[str]] = defaultdict(set)

    for row, item in zip(rows, indexes):
        item_id = item["record_id"]
        if item_id in rejected_16384:
            continue
        reasons = quality_reasons(row, item_id=item_id)
        if reasons:
            quality_exclusions.append(
                {
                    "record_id": item_id,
                    "source_episode_id": item.get("source_episode_id"),
                    "source_turn_index": item.get("source_turn_index"),
                    "tool_name": item.get("tool_name"),
                    "reasons": reasons,
                }
            )
            reason_hist.update(reasons)
            continue
        bucket = "core_8192" if item_id not in rejected_8192 else "supplement_8193_16384"
        tier, weight = artifact_tier(item)
        derived = dict(item)
        derived.update(
            {
                "quality_policy_version": POLICY_VERSION,
                "quality_filter_passed": True,
                "context_bucket": bucket,
                "artifact_tier": tier,
                "recommended_sample_weight": weight,
            }
        )
        kept_rows.append(row)
        kept_indexes.append(derived)
        bucket_rows[bucket].append(row)
        bucket_indexes[bucket].append(derived)
        tier_hist[tier] += 1
        bucket_hist[bucket] += 1
        tool = str(item.get("tool_name"))
        tool_hist[tool] += 1
        episode_id = str(item.get("source_episode_id"))
        episodes_by_bucket[bucket].add(episode_id)
        episode_tools[episode_id].add(tool)

    names = {
        "combined": dataset_name,
        "core_8192": f"{dataset_name}_core_8192",
        "supplement_8193_16384": f"{dataset_name}_supplement_8193_16384",
    }
    outputs: dict[str, dict[str, Any]] = {}
    for bucket, selected_rows, selected_indexes in (
        ("combined", kept_rows, kept_indexes),
        ("core_8192", bucket_rows["core_8192"], bucket_indexes["core_8192"]),
        (
            "supplement_8193_16384",
            bucket_rows["supplement_8193_16384"],
            bucket_indexes["supplement_8193_16384"],
        ),
    ):
        canonical_out = output_dir / f"{names[bucket]}_canonical.jsonl"
        index_out = output_dir / f"{names[bucket]}_index.jsonl"
        training_out = output_dir / f"{names[bucket]}.jsonl"
        write_jsonl_atomic(canonical_out, selected_rows)
        write_jsonl_atomic(index_out, selected_indexes)
        write_jsonl_atomic(training_out, _training_view(selected_rows))
        write_sharegpt_dataset_info(training_out, names[bucket])
        outputs[bucket] = {
            "dataset_name": names[bucket],
            "records": len(selected_rows),
            "canonical": str(canonical_out),
            "canonical_sha256": sha256(canonical_out),
            "index": str(index_out),
            "index_sha256": sha256(index_out),
            "training_view": str(training_out),
            "training_view_sha256": sha256(training_out),
        }

    excluded_path = output_dir / f"{dataset_name}_quality_excluded.jsonl"
    write_jsonl_atomic(excluded_path, quality_exclusions)
    source_episodes = {str(item.get("source_episode_id")) for item in indexes}
    contributing_episodes = {str(item.get("source_episode_id")) for item in kept_indexes}
    episodes_with_answer = sum("answer" in tools for tools in episode_tools.values())
    manifest = {
        "schema_version": "checkpoint-relalg-qwen3-projection-v3-dataset-v1",
        "quality_policy_version": POLICY_VERSION,
        "mutation": "record selection only; model-visible content is byte-preserved",
        "source": {
            "canonical": str(canonical),
            "canonical_sha256": sha256(canonical),
            "index": str(index_path),
            "index_sha256": sha256(index_path),
            "records": len(rows),
            "episodes": len(source_episodes),
        },
        "token_audits": [
            {
                "path": str(path),
                "sha256": sha256(path),
                "cutoff_len": audit.get("cutoff_len"),
                "model": audit.get("model"),
                "template": audit.get("template"),
                "filter_policy": audit.get("filter_policy"),
            }
            for path, audit in zip((audit_8192_path, audit_16384_path), audits)
        ],
        "selection": {
            "eligible_full_prefix_8192_before_quality": len(rows) - len(rejected_8192),
            "eligible_full_prefix_16384_before_quality": len(rows) - len(rejected_16384),
            "selected_records": len(kept_rows),
            "selected_episodes": len(contributing_episodes),
            "episodes_with_answer_target": episodes_with_answer,
            "episodes_without_answer_target": len(contributing_episodes) - episodes_with_answer,
            "context_bucket_records": dict(sorted(bucket_hist.items())),
            "context_bucket_episodes": {
                key: len(value) for key, value in sorted(episodes_by_bucket.items())
            },
            "artifact_tier_records": dict(sorted(tier_hist.items())),
            "tool_hist": dict(tool_hist.most_common()),
        },
        "quality_filter": {
            "record_level_only": True,
            "schema_or_strict_mismatch_is_not_an_exclusion": True,
            "repeated_reasoning_8gram_threshold": 0.20,
            "excluded_records": len(quality_exclusions),
            "reason_hist": dict(sorted(reason_hist.items())),
            "excluded": str(excluded_path),
            "excluded_sha256": sha256(excluded_path),
        },
        "outputs": outputs,
    }
    manifest_path = output_dir / f"{dataset_name}.manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--token-audit-8192", type=Path, required=True)
    parser.add_argument("--token-audit-16384", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()
    result = build(
        args.canonical.resolve(),
        args.index.resolve(),
        args.token_audit_8192.resolve(),
        args.token_audit_16384.resolve(),
        args.output_dir.resolve(),
        args.dataset_name,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
