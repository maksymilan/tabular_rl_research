#!/usr/bin/env python3
"""Project checkpoint-relalg native-reasoning candidates to Qwen3 ShareGPT.

The checkpoint-relalg Text-JSON provider history can contain adjacent ``user``
messages: a harness feedback envelope followed by the newly rendered dynamic
context.  LLaMA-Factory's ShareGPT converter requires strict user/assistant
alternation.  This projection therefore joins each adjacent run of user
messages with two newline characters, preserving order and every source
character while making the turn structure trainable.

The final assistant ``reasoning_content`` and visible JSON ``content`` are
rendered as one Qwen3 target, ``<think>...`` followed by the exact JSON action.
Historical assistant reasoning is removed explicitly and only its exact JSON
action remains, matching local Qwen history rather than DeepSeek thinking-mode
history. The source, index, and source manifest are hash-bound and a derived
index records the actual projected model-input and target hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sft_dataset_registry import write_sharegpt_dataset_info
from tool_modules.checkpoint_relalg.qwen3_carrier import (
    QWEN3_INLINE_CARRIER_VERSION,
    provider_student_system_prompt,
    qwen3_student_prompt_sha256,
    qwen3_student_system_prompt,
    render_qwen3_action,
)


PROJECTION = "checkpoint-relalg-native-reasoning-to-qwen3-sharegpt-v2"
SOURCE_SCHEMA = "checkpoint-relalg-native-reasoning-sft-candidate-v1"


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(compact(value).encode("utf-8")).hexdigest()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}: line {line_number} is not an object")
            rows.append(value)
    return rows


def validate_action(content: str, *, record_id: str, message_index: int) -> None:
    start = len(content) - len(content.lstrip())
    try:
        action, end = json.JSONDecoder().raw_decode(content, start)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{record_id}: assistant {message_index} content is not JSON: {exc}"
        ) from exc
    if content[end:].strip():
        raise ValueError(f"{record_id}: assistant {message_index} has trailing content")
    if not isinstance(action, dict):
        raise ValueError(f"{record_id}: assistant {message_index} action is not an object")
    if not isinstance(action.get("tool"), str) or not action["tool"]:
        raise ValueError(f"{record_id}: assistant {message_index} has no tool")
    if not isinstance(action.get("arguments"), dict):
        raise ValueError(f"{record_id}: assistant {message_index} arguments is not an object")


def row_hashes(row: dict[str, Any], *, record_id: str) -> tuple[str, str]:
    conversations = row["conversations"]
    messages = [{"role": "system", "content": row["system"]}]
    for item in conversations[:-1]:
        role = {"human": "user", "gpt": "assistant"}.get(item.get("from"))
        if role is None or not isinstance(item.get("value"), str):
            raise ValueError(f"{record_id}: unsupported projected conversation")
        messages.append({"role": role, "content": item["value"]})
    target = conversations[-1]
    if target.get("from") != "gpt" or not isinstance(target.get("value"), str):
        raise ValueError(f"{record_id}: final projected target is invalid")
    return digest(messages), digest(target["value"])


def project(
    source: Path,
    source_index: Path,
    source_manifest: Path,
    output: Path,
    index_output: Path,
    dataset_name: str,
    drop_invalid_history: bool = False,
) -> dict[str, Any]:
    if output.exists() or index_output.exists():
        raise FileExistsError(output if output.exists() else index_output)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("output_sha256") != sha256(source):
        raise ValueError("source manifest does not bind the candidate file")
    if manifest.get("index_sha256") != sha256(source_index):
        raise ValueError("source manifest does not bind the candidate index")

    rows = read_jsonl(source)
    indexes = read_jsonl(source_index)
    if len(rows) != len(indexes) or len(rows) != manifest.get("records"):
        raise ValueError("candidate, index, and manifest record counts differ")

    projected_rows: list[dict[str, Any]] = []
    projected_indexes: list[dict[str, Any]] = []
    seen: set[str] = set()
    merged_user_boundaries = 0
    assistant_messages = 0
    historical_assistant_messages = 0
    historical_assistant_reasoning_removed = 0
    rejected: list[dict[str, Any]] = []
    expected_source_system = provider_student_system_prompt()
    projected_system = qwen3_student_system_prompt()

    for row, source_item in zip(rows, indexes):
        if row.get("schema_version") != SOURCE_SCHEMA:
            raise ValueError("unexpected candidate schema")
        metadata = row.get("metadata")
        messages = row.get("messages")
        if not isinstance(metadata, dict) or not isinstance(messages, list) or len(messages) < 3:
            raise ValueError("candidate is missing metadata or messages")
        record_id = metadata.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in seen:
            raise ValueError(f"invalid or duplicate record id: {record_id!r}")
        seen.add(record_id)
        if source_item.get("record_id") != record_id:
            raise ValueError(f"row/index order mismatch at {record_id}")
        if row.get("loss_message_index") != len(messages) - 1:
            raise ValueError(f"{record_id}: loss target is not the last message")
        if messages[0].get("role") != "system" or not isinstance(messages[0].get("content"), str):
            raise ValueError(f"{record_id}: first message is not a system message")
        if messages[0]["content"] != expected_source_system:
            raise ValueError(f"{record_id}: source student system prompt identity drifted")

        conversations: list[dict[str, str]] = []
        reject_reason: str | None = None
        row_merged_user_boundaries = 0
        row_assistant_messages = 0
        row_historical_assistant_messages = 0
        for message_index, message in enumerate(messages[1:], start=1):
            role = message.get("role")
            content = message.get("content")
            if role == "user":
                if not isinstance(content, str) or not content:
                    raise ValueError(f"{record_id}: user {message_index} is empty")
                if conversations and conversations[-1]["from"] == "human":
                    conversations[-1]["value"] += "\n\n" + content
                    row_merged_user_boundaries += 1
                else:
                    conversations.append({"from": "human", "value": content})
            elif role == "assistant":
                reasoning = message.get("reasoning_content")
                if not isinstance(reasoning, str) or not reasoning:
                    raise ValueError(f"{record_id}: assistant {message_index} has no reasoning")
                if "<think>" in reasoning or "</think>" in reasoning:
                    raise ValueError(f"{record_id}: assistant reasoning contains think tags")
                if not isinstance(content, str) or not content:
                    raise ValueError(f"{record_id}: assistant {message_index} has no content")
                try:
                    validate_action(content, record_id=record_id, message_index=message_index)
                except ValueError as exc:
                    if message_index == len(messages) - 1 or not drop_invalid_history:
                        raise
                    reject_reason = type(exc).__name__ + ":invalid_historical_assistant_carrier"
                    break
                is_target = message_index == len(messages) - 1
                conversations.append(
                    {
                        "from": "gpt",
                        "value": (
                            render_qwen3_action(reasoning, content) if is_target else content
                        ),
                    }
                )
                row_assistant_messages += 1
                row_historical_assistant_messages += int(not is_target)
            else:
                raise ValueError(f"{record_id}: unsupported role {role!r}")

        if reject_reason is not None:
            rejected.append(
                {
                    "record_id": record_id,
                    "source_episode_id": metadata.get("source_episode_id"),
                    "reason": reject_reason,
                }
            )
            continue
        if not conversations or conversations[-1]["from"] != "gpt":
            raise ValueError(f"{record_id}: projected row has no final assistant target")
        for index, item in enumerate(conversations):
            expected = "human" if index % 2 == 0 else "gpt"
            if item["from"] != expected:
                raise ValueError(f"{record_id}: projected roles do not alternate")

        projected = {
            "system": projected_system,
            "conversations": conversations,
            "metadata": metadata,
        }
        model_input_sha, target_sha = row_hashes(projected, record_id=record_id)
        derived_index = dict(source_item)
        derived_index["source_model_input_sha256"] = source_item.get("model_input_sha256")
        derived_index["source_target_visible_content_sha256"] = source_item.get(
            "target_visible_content_sha256"
        )
        derived_index["model_input_sha256"] = model_input_sha
        derived_index["target_sha256"] = target_sha
        derived_index["model_input_projection"] = PROJECTION
        projected_rows.append(projected)
        projected_indexes.append(derived_index)
        merged_user_boundaries += row_merged_user_boundaries
        assistant_messages += row_assistant_messages
        historical_assistant_messages += row_historical_assistant_messages
        historical_assistant_reasoning_removed += row_historical_assistant_messages

    output.parent.mkdir(parents=True, exist_ok=True)
    for path, values in ((output, projected_rows), (index_output, projected_indexes)):
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                for value in values:
                    handle.write(json.dumps(value, ensure_ascii=False) + "\n")
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()

    snippet, registry = write_sharegpt_dataset_info(output, dataset_name)
    rejected_output = output.with_suffix(".rejected.jsonl")
    with rejected_output.open("w", encoding="utf-8") as handle:
        for item in rejected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    result = {
        "projection": PROJECTION,
        "qwen3_inline_carrier_version": QWEN3_INLINE_CARRIER_VERSION,
        "source_student_prompt_sha256": hashlib.sha256(
            expected_source_system.encode("utf-8")
        ).hexdigest(),
        "qwen3_student_prompt_sha256": qwen3_student_prompt_sha256(),
        "source": str(source),
        "source_sha256": sha256(source),
        "source_index": str(source_index),
        "source_index_sha256": sha256(source_index),
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "output": str(output),
        "output_sha256": sha256(output),
        "index_output": str(index_output),
        "index_output_sha256": sha256(index_output),
        "records": len(projected_rows),
        "source_records": len(rows),
        "dropped_invalid_history_records": len(rejected),
        "rejected_output": str(rejected_output),
        "rejected_output_sha256": sha256(rejected_output),
        "unique_record_ids": len(projected_rows),
        "unique_source_record_ids": len(seen),
        "assistant_messages": assistant_messages,
        "historical_assistant_messages": historical_assistant_messages,
        "historical_assistant_reasoning_removed": historical_assistant_reasoning_removed,
        "merged_adjacent_user_boundaries": merged_user_boundaries,
        "visible_action_json_changed": 0,
        "assistant_carrier_projection_applied": True,
        "system_carrier_clause_projection_applied": True,
        "historical_assistant_policy": "json-action-only-v1",
        "adjacent_user_merge_separator": "two newline characters",
        "dataset_name": dataset_name,
        "dataset_info_snippet": str(snippet),
        "dataset_info_registry": str(registry),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--drop-invalid-history",
        action="store_true",
        help="Drop only rows whose historical assistant carrier is not one strict JSON action.",
    )
    args = parser.parse_args()
    result = project(
        args.input.resolve(),
        args.index.resolve(),
        args.source_manifest.resolve(),
        args.out.resolve(),
        args.index_out.resolve(),
        args.dataset_name,
        args.drop_invalid_history,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
