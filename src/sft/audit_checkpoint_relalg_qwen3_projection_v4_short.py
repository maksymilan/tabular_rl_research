#!/usr/bin/env python3
"""Fail-closed audit for an extractive projection-v4 short-reasoning corpus."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_checkpoint_relalg_qwen3_projection_v3 import TARGET_RE, read_jsonl, repeated_ngram_ratio, sha256
from build_checkpoint_relalg_qwen3_projection_v4_short import (
    MIN_REASONING_TOKENS,
    OUTPUT_SCHEMA,
    POLICY_VERSION,
    PROJECTION_VERSION,
    action_anchors,
    contains_anchor,
    digest_text,
    has_repeated_sentence,
    parse_target,
    token_count,
)


def _output_paths(manifest: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    output = manifest.get("outputs") or {}
    return tuple(Path(output[key]) for key in ("canonical", "index", "training_view", "excluded"))  # type: ignore[return-value]


def audit(dataset_dir: Path, dataset_name: str, tokenizer: Any) -> dict[str, Any]:
    manifest_path = dataset_dir / f"{dataset_name}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != OUTPUT_SCHEMA:
        raise ValueError("unexpected projection-v4 dataset schema")
    if manifest.get("reasoning_projection_version") != PROJECTION_VERSION:
        raise ValueError("reasoning projection identity differs")
    if manifest.get("reasoning_projection_policy_version") != POLICY_VERSION:
        raise ValueError("short-reasoning policy identity differs")

    policy = manifest.get("policy") or {}
    soft_max = policy.get("soft_max_reasoning_tokens")
    hard_max = policy.get("hard_max_reasoning_tokens")
    max_repeat = policy.get("max_repeated_8gram_ratio")
    if not isinstance(soft_max, int) or not isinstance(hard_max, int) or hard_max < soft_max:
        raise ValueError("manifest token limits are invalid")
    if not isinstance(max_repeat, (int, float)) or not 0 <= max_repeat < 1:
        raise ValueError("manifest repetition limit is invalid")
    if policy.get("abstractive_rewrite") is not False or policy.get("external_model_calls") != 0:
        raise ValueError("manifest does not bind extractive zero-call projection")
    if policy.get("gold_or_future_feedback_access") is not False:
        raise ValueError("manifest gold/future access policy differs")
    implementation = manifest.get("implementation") or {}
    builder_path = Path(implementation.get("builder", ""))
    if not builder_path.is_file() or implementation.get("builder_sha256") != sha256(builder_path):
        raise ValueError("builder implementation identity differs")

    canonical, index_path, training, excluded = _output_paths(manifest)
    for key, path in (
        ("canonical_sha256", canonical),
        ("index_sha256", index_path),
        ("training_view_sha256", training),
        ("excluded_sha256", excluded),
    ):
        if manifest["outputs"].get(key) != sha256(path):
            raise ValueError(f"output hash differs: {key}")

    source = manifest.get("source") or {}
    source_manifest = Path(source["manifest"])
    source_canonical = Path(source["canonical"])
    source_index = Path(source["index"])
    for key, path in (
        ("manifest_sha256", source_manifest),
        ("canonical_sha256", source_canonical),
        ("index_sha256", source_index),
    ):
        if source.get(key) != sha256(path):
            raise ValueError(f"source hash differs: {key}")

    source_rows = read_jsonl(source_canonical)
    source_indexes = read_jsonl(source_index)
    if len(source_rows) != len(source_indexes):
        raise ValueError("source canonical/index counts differ")
    source_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for row, item in zip(source_rows, source_indexes, strict=True):
        item_id = (row.get("metadata") or {}).get("record_id")
        if not isinstance(item_id, str) or item.get("record_id") != item_id or item_id in source_by_id:
            raise ValueError("source record identity/order is invalid")
        source_by_id[item_id] = (row, item)

    output_rows = read_jsonl(canonical)
    output_indexes = read_jsonl(index_path)
    training_rows = read_jsonl(training)
    if not (len(output_rows) == len(output_indexes) == len(training_rows)):
        raise ValueError("canonical/index/training counts differ")

    seen: set[str] = set()
    tools: Counter[str] = Counter()
    episodes: set[str] = set()
    answer_episodes: set[str] = set()
    shortened = 0
    source_tokens: list[int] = []
    projected_tokens: list[int] = []
    for row, item, view in zip(output_rows, output_indexes, training_rows, strict=True):
        item_id = (row.get("metadata") or {}).get("record_id")
        if not isinstance(item_id, str) or item_id in seen:
            raise ValueError("invalid or duplicate selected record id")
        seen.add(item_id)
        if item.get("record_id") != item_id or item_id not in source_by_id:
            raise ValueError(f"{item_id}: selected identity differs from source")
        source_row, source_item = source_by_id[item_id]
        if view != {"system": row.get("system"), "conversations": row.get("conversations")}:
            raise ValueError(f"{item_id}: training view differs from canonical")
        if row.get("system") != source_row.get("system"):
            raise ValueError(f"{item_id}: system text changed")
        if row.get("conversations", [])[:-1] != source_row.get("conversations", [])[:-1]:
            raise ValueError(f"{item_id}: causal history changed")
        for key, value in source_item.items():
            if item.get(key) != value:
                raise ValueError(f"{item_id}: source index field changed: {key}")
        if item.get("reasoning_projection_policy_version") != POLICY_VERSION:
            raise ValueError(f"{item_id}: projection policy is missing")

        source_reasoning, source_action_text, source_action = parse_target(source_row, item_id=item_id)
        projected_reasoning, projected_action_text, projected_action = parse_target(row, item_id=item_id)
        if projected_action_text != source_action_text or projected_action != source_action:
            raise ValueError(f"{item_id}: action JSON changed")
        projection = item.get("reasoning_projection")
        if not isinstance(projection, dict):
            raise ValueError(f"{item_id}: projection metadata is missing")
        if (row.get("metadata") or {}).get("reasoning_projection") != projection:
            raise ValueError(f"{item_id}: canonical/index projection metadata differs")
        if projection.get("reasoning_projection_version") != PROJECTION_VERSION:
            raise ValueError(f"{item_id}: projection version differs")
        span = projection.get("source_character_span")
        if not (
            isinstance(span, list)
            and len(span) == 2
            and all(isinstance(value, int) for value in span)
            and 0 <= span[0] <= span[1] <= len(source_reasoning)
        ):
            raise ValueError(f"{item_id}: source character span is invalid")
        if source_reasoning[span[0] : span[1]].strip() != projected_reasoning:
            raise ValueError(f"{item_id}: projected reasoning is not the bound source substring")
        actual_source_tokens = token_count(tokenizer, source_reasoning)
        actual_projected_tokens = token_count(tokenizer, projected_reasoning)
        expected_fields = {
            "source_reasoning_sha256": digest_text(source_reasoning),
            "source_reasoning_characters": len(source_reasoning),
            "source_reasoning_tokens": actual_source_tokens,
            "projected_reasoning_sha256": digest_text(projected_reasoning),
            "projected_reasoning_characters": len(projected_reasoning),
            "projected_reasoning_tokens": actual_projected_tokens,
            "byte_preserved_action": True,
            "contiguous_source_substring": True,
            "was_shortened": projected_reasoning != source_reasoning,
        }
        for key, value in expected_fields.items():
            if projection.get(key) != value:
                raise ValueError(f"{item_id}: projection metadata differs: {key}")
        if actual_projected_tokens < MIN_REASONING_TOKENS or actual_projected_tokens > hard_max:
            raise ValueError(f"{item_id}: projected reasoning token limit differs")
        repeat_ratio = repeated_ngram_ratio(projected_reasoning)
        if repeat_ratio >= max_repeat or projection.get("projected_reasoning_repeat_ratio") != repeat_ratio:
            raise ValueError(f"{item_id}: projected reasoning repetition differs")
        if has_repeated_sentence(projected_reasoning) or projection.get("projected_exact_sentence_repeat") is not False:
            raise ValueError(f"{item_id}: projected reasoning repeats a sentence")
        grounded = contains_anchor(projected_reasoning, action_anchors(source_action))
        if projection.get("action_anchor_present") is not grounded:
            raise ValueError(f"{item_id}: action-anchor metadata differs")
        if projected_reasoning != source_reasoning and not grounded:
            raise ValueError(f"{item_id}: shortened reasoning lacks action anchor")

        tool = str(source_action["tool"])
        episode_id = str(source_item.get("source_episode_id"))
        tools[tool] += 1
        episodes.add(episode_id)
        if tool == "answer":
            answer_episodes.add(episode_id)
        shortened += int(projected_reasoning != source_reasoning)
        source_tokens.append(actual_source_tokens)
        projected_tokens.append(actual_projected_tokens)

    excluded_rows = read_jsonl(excluded)
    excluded_ids = [item.get("record_id") for item in excluded_rows]
    if any(not isinstance(item_id, str) for item_id in excluded_ids):
        raise ValueError("excluded record id is invalid")
    if len(excluded_ids) != len(set(excluded_ids)) or seen.intersection(excluded_ids):
        raise ValueError("selected/excluded record partition is invalid")
    if seen.union(excluded_ids) != set(source_by_id):
        raise ValueError("selected/excluded records do not partition the source")

    selection = manifest.get("selection") or {}
    expected_selection = {
        "source_records": len(source_by_id),
        "selected_records": len(seen),
        "excluded_records": len(excluded_ids),
        "shortened_records": shortened,
        "unchanged_records": len(seen) - shortened,
        "selected_episodes": len(episodes),
        "episodes_with_answer_target": len(answer_episodes),
        "tool_hist": dict(tools.most_common()),
    }
    for key, value in expected_selection.items():
        if selection.get(key) != value:
            raise ValueError(f"manifest selection differs: {key}")

    return {
        "audit": "checkpoint-relalg-qwen3-projection-v4-short-audit-v1",
        "passed": True,
        "records": len(seen),
        "excluded_records": len(excluded_ids),
        "episodes": len(episodes),
        "episodes_with_answer_target": len(answer_episodes),
        "shortened_records": shortened,
        "unchanged_records": len(seen) - shortened,
        "source_reasoning_tokens_total": sum(source_tokens),
        "projected_reasoning_tokens_total": sum(projected_tokens),
        "reasoning_token_reduction": 1 - sum(projected_tokens) / sum(source_tokens),
        "max_projected_reasoning_tokens": max(projected_tokens, default=0),
        "tool_hist": dict(tools.most_common()),
        "projection_version": PROJECTION_VERSION,
        "policy_version": POLICY_VERSION,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()), trust_remote_code=True)
    result = audit(args.dataset_dir.resolve(), args.dataset_name, tokenizer)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
