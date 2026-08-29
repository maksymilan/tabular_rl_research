#!/usr/bin/env python3
"""Fail-closed audit for a projection-v3 Qwen3 training corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_checkpoint_relalg_qwen3_projection_v3 import (
    POLICY_VERSION,
    artifact_tier,
    quality_reasons,
    sha256,
)
from tool_modules.checkpoint_relalg.qwen3_carrier import (
    QWEN3_INLINE_CARRIER_VERSION,
    parse_qwen3_action,
    qwen3_student_prompt_sha256,
    qwen3_student_system_prompt,
)


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(compact(value).encode("utf-8")).hexdigest()


def audit(dataset_dir: Path, dataset_name: str) -> dict[str, Any]:
    manifest_path = dataset_dir / f"{dataset_name}.manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "checkpoint-relalg-qwen3-projection-v3-dataset-v1":
        raise ValueError("unexpected v3 dataset schema")
    if manifest.get("quality_policy_version") != POLICY_VERSION:
        raise ValueError("v3 quality policy identity differs")
    output = manifest.get("outputs", {}).get("combined", {})
    canonical = Path(output["canonical"])
    index_path = Path(output["index"])
    training = Path(output["training_view"])
    for key, path in (
        ("canonical_sha256", canonical),
        ("index_sha256", index_path),
        ("training_view_sha256", training),
    ):
        if output.get(key) != sha256(path):
            raise ValueError(f"combined output hash differs: {key}")

    records = 0
    seen_ids: set[str] = set()
    seen_inputs: set[str] = set()
    tools: Counter[str] = Counter()
    buckets: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    episodes: set[str] = set()
    answer_episodes: set[str] = set()
    expected_system = qwen3_student_system_prompt()
    with (
        canonical.open(encoding="utf-8") as canonical_handle,
        index_path.open(encoding="utf-8") as index_handle,
        training.open(encoding="utf-8") as training_handle,
    ):
        while True:
            lines = (
                canonical_handle.readline(),
                index_handle.readline(),
                training_handle.readline(),
            )
            if not any(lines):
                break
            if not all(lines):
                raise ValueError("canonical/index/training counts differ")
            row, item, view = (json.loads(line) for line in lines)
            metadata = row.get("metadata") or {}
            item_id = metadata.get("record_id")
            if not isinstance(item_id, str) or not item_id or item_id in seen_ids:
                raise ValueError("invalid or duplicate record id")
            seen_ids.add(item_id)
            if item.get("record_id") != item_id:
                raise ValueError(f"{item_id}: index order differs")
            if view != {"system": row.get("system"), "conversations": row.get("conversations")}:
                raise ValueError(f"{item_id}: training view changed model-visible text")
            if row.get("system") != expected_system:
                raise ValueError(f"{item_id}: student prompt identity differs")
            reasons = quality_reasons(row, item_id=item_id)
            if reasons:
                raise ValueError(f"{item_id}: selected target violates quality policy: {reasons}")
            target = row["conversations"][-1]["value"]
            parsed = parse_qwen3_action(target)
            if parsed["tool"] != item.get("tool_name"):
                raise ValueError(f"{item_id}: target tool/index differ")
            if item.get("quality_policy_version") != POLICY_VERSION:
                raise ValueError(f"{item_id}: quality policy identity differs")
            if item.get("quality_filter_passed") is not True:
                raise ValueError(f"{item_id}: quality pass flag is missing")
            expected_tier, expected_weight = artifact_tier(item)
            if (item.get("artifact_tier"), item.get("recommended_sample_weight")) != (
                expected_tier,
                expected_weight,
            ):
                raise ValueError(f"{item_id}: artifact tier/weight differs")
            if item.get("context_bucket") not in {"core_8192", "supplement_8193_16384"}:
                raise ValueError(f"{item_id}: context bucket differs")

            messages = [{"role": "system", "content": row["system"]}]
            for message in row["conversations"][:-1]:
                role = {"human": "user", "gpt": "assistant"}.get(message.get("from"))
                if role is None or not isinstance(message.get("value"), str):
                    raise ValueError(f"{item_id}: invalid historical conversation")
                messages.append({"role": role, "content": message["value"]})
            model_input_hash = digest(messages)
            if model_input_hash in seen_inputs:
                raise ValueError(f"{item_id}: duplicate model input")
            seen_inputs.add(model_input_hash)
            if item.get("model_input_sha256") != model_input_hash:
                raise ValueError(f"{item_id}: model input hash differs")

            episode_id = item.get("source_episode_id")
            if not isinstance(episode_id, str) or not episode_id:
                raise ValueError(f"{item_id}: source episode id missing")
            episodes.add(episode_id)
            if parsed["tool"] == "answer":
                answer_episodes.add(episode_id)
            tools[parsed["tool"]] += 1
            buckets[item["context_bucket"]] += 1
            tiers[item["artifact_tier"]] += 1
            records += 1

    selection = manifest.get("selection", {})
    if output.get("records") != records or selection.get("selected_records") != records:
        raise ValueError("manifest selected record count differs")
    if selection.get("selected_episodes") != len(episodes):
        raise ValueError("manifest selected episode count differs")
    if selection.get("episodes_with_answer_target") != len(answer_episodes):
        raise ValueError("manifest answer episode count differs")
    if selection.get("tool_hist") != dict(tools.most_common()):
        raise ValueError("manifest tool histogram differs")
    if selection.get("context_bucket_records") != dict(sorted(buckets.items())):
        raise ValueError("manifest bucket histogram differs")
    if selection.get("artifact_tier_records") != dict(sorted(tiers.items())):
        raise ValueError("manifest tier histogram differs")

    return {
        "audit": "checkpoint-relalg-qwen3-projection-v3-audit-v1",
        "passed": True,
        "records": records,
        "episodes": len(episodes),
        "episodes_with_answer_target": len(answer_episodes),
        "unique_record_ids": len(seen_ids),
        "unique_model_inputs": len(seen_inputs),
        "tool_hist": dict(tools.most_common()),
        "context_bucket_records": dict(sorted(buckets.items())),
        "artifact_tier_records": dict(sorted(tiers.items())),
        "quality_policy_version": POLICY_VERSION,
        "qwen3_inline_carrier_version": QWEN3_INLINE_CARRIER_VERSION,
        "qwen3_student_prompt_sha256": qwen3_student_prompt_sha256(),
        "provider_carrier_leaks": 0,
        "benchmark_answer_memory_targets": 0,
        "repetitive_reasoning_targets": 0,
        "adjacent_exact_action_repeats": 0,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
        "canonical": str(canonical),
        "canonical_sha256": sha256(canonical),
        "index": str(index_path),
        "index_sha256": sha256(index_path),
        "training_view": str(training),
        "training_view_sha256": sha256(training),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.dataset_dir.resolve(), args.dataset_name)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
