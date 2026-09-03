#!/usr/bin/env python3
"""Audit the corrected Qwen3 training view against its canonical SFT records."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tool_modules.checkpoint_relalg.qwen3_carrier import (
    QWEN3_INLINE_CARRIER_VERSION,
    parse_qwen3_action,
    qwen3_student_prompt_sha256,
    qwen3_student_system_prompt,
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _action(text: Any, *, location: str) -> tuple[dict[str, Any], bool]:
    if not isinstance(text, str) or not text:
        raise ValueError(f"{location}: assistant action is empty")
    if "<think>" in text or "</think>" in text:
        raise ValueError(f"{location}: historical reasoning was not removed")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location}: historical action is not JSON") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: historical action envelope is not an object")
    if not isinstance(raw["tool"], str) or not raw["tool"]:
        raise ValueError(f"{location}: historical tool name is invalid")
    if not isinstance(raw["arguments"], dict):
        raise ValueError(f"{location}: historical arguments are not an object")
    # A rejected semantic action is valid causal history: it must remain visible
    # together with the Harness error that taught the later recovery. Only the
    # supervised final target is required to pass current action validation.
    return raw, set(raw) != {"tool", "arguments"}


def audit(
    canonical_path: Path,
    index_path: Path,
    training_view_path: Path,
    projection_manifest_path: Path,
    training_manifest_path: Path,
) -> dict[str, Any]:
    projection_manifest = json.loads(projection_manifest_path.read_text(encoding="utf-8"))
    training_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    if projection_manifest.get("projection") != (
        "checkpoint-relalg-native-reasoning-to-qwen3-sharegpt-v2"
    ):
        raise ValueError("projection manifest is not Qwen3 projection v2")
    if projection_manifest.get("qwen3_inline_carrier_version") != QWEN3_INLINE_CARRIER_VERSION:
        raise ValueError("projection manifest carrier identity drifted")
    if projection_manifest.get("qwen3_student_prompt_sha256") != qwen3_student_prompt_sha256():
        raise ValueError("projection manifest Qwen3 prompt identity drifted")
    if training_manifest.get("canonical_input_sha256") != sha256(canonical_path):
        raise ValueError("training manifest does not bind canonical input")
    if training_manifest.get("canonical_index_sha256") != sha256(index_path):
        raise ValueError("training manifest does not bind canonical index")
    if training_manifest.get("output_sha256") != sha256(training_view_path):
        raise ValueError("training manifest does not bind training view")

    expected_system = qwen3_student_system_prompt()
    records = 0
    historical_actions = 0
    historical_noncanonical_envelopes = 0
    target_actions = 0
    seen_ids: set[str] = set()
    with (
        canonical_path.open(encoding="utf-8") as canonical,
        index_path.open(encoding="utf-8") as indexes,
        training_view_path.open(encoding="utf-8") as training,
    ):
        while True:
            canonical_line = canonical.readline()
            index_line = indexes.readline()
            training_line = training.readline()
            if not canonical_line and not index_line and not training_line:
                break
            if not canonical_line or not index_line or not training_line:
                raise ValueError("canonical, index, and training-view counts differ")
            source = json.loads(canonical_line)
            index = json.loads(index_line)
            row = json.loads(training_line)
            metadata = source.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"line {records + 1}: canonical metadata is missing")
            record_id = metadata.get("record_id")
            if not isinstance(record_id, str) or not record_id or record_id in seen_ids:
                raise ValueError(f"line {records + 1}: invalid or duplicate record id")
            seen_ids.add(record_id)
            if index.get("record_id") != record_id:
                raise ValueError(f"{record_id}: index order differs")
            expected_row = {
                "system": source.get("system"),
                "conversations": source.get("conversations"),
            }
            if row != expected_row:
                raise ValueError(f"{record_id}: training view changed model-visible content")
            if row["system"] != expected_system:
                raise ValueError(f"{record_id}: Qwen3 system prompt differs")
            if "provider reasoning_content" in row["system"]:
                raise ValueError(f"{record_id}: provider-only carrier leaked into Qwen3 system")
            conversations = row.get("conversations")
            if not isinstance(conversations, list) or len(conversations) < 2:
                raise ValueError(f"{record_id}: conversations are incomplete")
            for item_number, item in enumerate(conversations):
                expected_role = "human" if item_number % 2 == 0 else "gpt"
                if item.get("from") != expected_role or not isinstance(item.get("value"), str):
                    raise ValueError(f"{record_id}: conversation roles do not alternate")
                if expected_role != "gpt":
                    continue
                if item_number == len(conversations) - 1:
                    parsed = parse_qwen3_action(item["value"])
                    if parsed["action_text"] not in item["value"]:
                        raise ValueError(f"{record_id}: final action text was repaired")
                    target_actions += 1
                else:
                    _, noncanonical = _action(
                        item["value"], location=f"{record_id}:conversation[{item_number}]"
                    )
                    historical_actions += 1
                    historical_noncanonical_envelopes += int(noncanonical)
            records += 1

    if training_manifest.get("records") != records:
        raise ValueError("training manifest record count differs")
    result = {
        "audit": "checkpoint-relalg-qwen3-sft-carrier-audit-v1",
        "passed": True,
        "records": records,
        "unique_record_ids": len(seen_ids),
        "historical_json_actions": historical_actions,
        "historical_noncanonical_envelopes": historical_noncanonical_envelopes,
        "inline_think_json_targets": target_actions,
        "provider_carrier_leaks": 0,
        "model_visible_content_mismatches": 0,
        "qwen3_inline_carrier_version": QWEN3_INLINE_CARRIER_VERSION,
        "qwen3_student_prompt_sha256": qwen3_student_prompt_sha256(),
        "canonical": str(canonical_path),
        "canonical_sha256": sha256(canonical_path),
        "index": str(index_path),
        "index_sha256": sha256(index_path),
        "training_view": str(training_view_path),
        "training_view_sha256": sha256(training_view_path),
        "projection_manifest": str(projection_manifest_path),
        "projection_manifest_sha256": sha256(projection_manifest_path),
        "training_manifest": str(training_manifest_path),
        "training_manifest_sha256": sha256(training_manifest_path),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--training-view", type=Path, required=True)
    parser.add_argument("--projection-manifest", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        args.canonical.resolve(),
        args.index.resolve(),
        args.training_view.resolve(),
        args.projection_manifest.resolve(),
        args.training_manifest.resolve(),
    )
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
