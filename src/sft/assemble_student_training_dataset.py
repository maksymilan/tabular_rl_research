#!/usr/bin/env python3
"""Assemble complete causal teacher episodes under the student runtime prompt.

Inputs are already generated model↔harness records. This module does not synthesize or repair any
assistant action, reasoning, observation, or history. It combines disjoint source lanes, validates
that every episode ends in one grounded terminal target, replaces the system field with the shared
student runtime contract, and deterministically reserializes assistant actions into the active
carrier. Exact model/template token auditing remains a separate post-assembly gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    STUDENT_PROMPT_CANONICAL,
    assistant_message,
    parse_legacy_assistant_strict,
    parse_assistant_strict,
    protocol_hash,
    student_runtime_system_prompt,
    tool_schema_hash,
)
from action_carrier import ACTIVE_ACTION_CARRIER, LEGACY_TAGGED_ACTION_CARRIER
from sft_dataset_registry import write_sharegpt_dataset_info


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
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


def numeric_step_id(value: Any) -> int:
    match = re.fullmatch(r"step_(\d+)", str(value))
    if match is None:
        raise ValueError(f"invalid source_step_id: {value!r}")
    return int(match.group(1))


def require_string(mapping: dict[str, Any], key: str, *, where: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: missing non-empty {key}")
    return value


def validate_record_pair(
    record: dict[str, Any],
    index: dict[str, Any],
) -> tuple[str, str, int, str]:
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("SFT record is missing metadata")
    record_id = require_string(metadata, "record_id", where="record metadata")
    index_record_id = require_string(index, "record_id", where=f"index {record_id}")
    if record_id != index_record_id:
        raise ValueError(f"record/index order mismatch: {record_id} != {index_record_id}")

    episode_id = require_string(index, "source_episode_id", where=record_id)
    step_id = require_string(index, "source_step_id", where=record_id)
    for key, expected in (
        ("source_episode_id", episode_id),
        ("source_step_id", step_id),
    ):
        if metadata.get(key) != expected:
            raise ValueError(f"{record_id}: metadata/index {key} mismatch")

    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"{record_id}: conversations must be a non-empty list")
    final_turn = conversations[-1]
    if not isinstance(final_turn, dict) or final_turn.get("from") != "gpt":
        raise ValueError(f"{record_id}: final conversation turn must be a gpt target")
    _, parsed_tool, _, _ = parse_source_action(str(final_turn.get("value", "")))
    indexed_tool = require_string(index, "tool_name", where=record_id)
    if parsed_tool != indexed_tool:
        raise ValueError(
            f"{record_id}: parsed target tool {parsed_tool!r} != index tool {indexed_tool!r}"
        )
    return record_id, episode_id, numeric_step_id(step_id), parsed_tool


def parse_source_action(value: str) -> tuple[str, str, dict[str, Any], str]:
    try:
        think, tool, arguments = parse_assistant_strict(value)
        return think, tool, arguments, ACTIVE_ACTION_CARRIER
    except ProtocolError as active_error:
        try:
            think, tool, arguments = parse_legacy_assistant_strict(value)
            return think, tool, arguments, LEGACY_TAGGED_ACTION_CARRIER
        except ProtocolError:
            raise active_error


def rerender_conversations(
    conversations: list[dict[str, Any]],
    source_carriers: Counter[str],
) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for turn in conversations:
        copied = dict(turn)
        if turn.get("from") == "gpt":
            think, tool, arguments, source_carrier = parse_source_action(
                str(turn.get("value", ""))
            )
            source_carriers[source_carrier] += 1
            copied["value"] = assistant_message(think, tool, arguments)
            parsed = parse_assistant_strict(copied["value"])
            if parsed != (think, tool, arguments):
                raise RuntimeError("assistant carrier rerender changed structured action")
        rendered.append(copied)
    return rendered


def build(
    input_paths: list[Path],
    index_paths: list[Path],
    out_path: Path,
    index_out_path: Path,
    *,
    dataset_name: str,
) -> dict[str, Any]:
    if len(input_paths) != len(index_paths):
        raise ValueError("each SFT input must have one matching index")
    if not input_paths:
        raise ValueError("at least one SFT input/index pair is required")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset_name):
        raise ValueError("dataset_name contains unsupported characters")

    source_lanes: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    indexes: list[dict[str, Any]] = []
    for input_path, index_path in zip(input_paths, index_paths):
        lane_records = read_jsonl(input_path)
        lane_indexes = read_jsonl(index_path)
        if len(lane_records) != len(lane_indexes):
            raise ValueError(
                f"{input_path}: {len(lane_records)} records != "
                f"{len(lane_indexes)} indexes in {index_path}"
            )
        source_lanes.append(
            {
                "input": str(input_path),
                "input_sha256": sha256_file(input_path),
                "index": str(index_path),
                "index_sha256": sha256_file(index_path),
                "records": len(lane_records),
            }
        )
        records.extend(lane_records)
        indexes.extend(lane_indexes)

    identities: set[str] = set()
    episode_rows: defaultdict[str, list[tuple[int, str, str]]] = defaultdict(list)
    tool_hist: Counter[str] = Counter()
    source_prompt_hashes: Counter[str] = Counter()
    for record, index in zip(records, indexes):
        identity = validate_record_pair(record, index)
        record_id, episode_id, step_number, tool = identity
        if record_id in identities:
            raise ValueError(f"duplicate record_id across source lanes: {record_id}")
        identities.add(record_id)
        episode_rows[episode_id].append((step_number, record_id, tool))
        tool_hist[tool] += 1
        system = record.get("system")
        if not isinstance(system, str) or not system:
            raise ValueError(f"{record_id}: source system prompt is missing")
        source_prompt_hashes[hashlib.sha256(system.encode("utf-8")).hexdigest()] += 1

    for episode_id, items in episode_rows.items():
        ordered = sorted(items)
        step_numbers = [item[0] for item in ordered]
        if step_numbers != sorted(set(step_numbers)):
            raise ValueError(f"{episode_id}: duplicate or unsorted source step ids")
        terminal_positions = [
            position
            for position, (_, _, tool) in enumerate(ordered)
            if tool == "answer_from_context"
        ]
        if terminal_positions != [len(ordered) - 1]:
            raise ValueError(
                f"{episode_id}: expected exactly one final answer_from_context target"
            )

    student_prompt = student_runtime_system_prompt(
        context_mode="rolling-legal-history",
        student_prompt_variant=STUDENT_PROMPT_CANONICAL,
    )
    source_carriers: Counter[str] = Counter()
    rerendered = [
        {
            **record,
            "system": student_prompt,
            "conversations": rerender_conversations(
                record["conversations"],
                source_carriers,
            ),
        }
        for record in records
    ]
    for source, rendered_record in zip(records, rerendered):
        if source["metadata"] != rendered_record["metadata"]:
            raise RuntimeError("record metadata changed during carrier rerender")
        source_human = [
            turn for turn in source["conversations"] if turn.get("from") != "gpt"
        ]
        rendered_human = [
            turn
            for turn in rendered_record["conversations"]
            if turn.get("from") != "gpt"
        ]
        if source_human != rendered_human:
            raise RuntimeError("non-assistant conversation content changed during carrier rerender")

    write_jsonl_atomic(out_path, rerendered)
    write_jsonl_atomic(index_out_path, indexes)
    snippet_path, registry_path = write_sharegpt_dataset_info(out_path, dataset_name)

    manifest = {
        "assembly": "complete-causal-episodes-student-prompt-active-carrier",
        "mutation": (
            "system field and assistant carrier serialization only; structured actions, reasoning, "
            "observations, human history, metadata, and record order are unchanged"
        ),
        "source_lanes": source_lanes,
        "output": str(out_path),
        "output_sha256": sha256_file(out_path),
        "index": str(index_out_path),
        "index_sha256": sha256_file(index_out_path),
        "dataset_name": dataset_name,
        "dataset_info_snippet": str(snippet_path),
        "dataset_info_registry": str(registry_path),
        "records": len(rerendered),
        "episodes": len(episode_rows),
        "terminal_targets": tool_hist["answer_from_context"],
        "feedback_recovery_targets": sum(
            bool((record.get("metadata") or {}).get("feedback_recovery"))
            for record in rerendered
        ),
        "tool_hist": dict(tool_hist.most_common()),
        "protocol_version": PROTOCOL_VERSION,
        "prompt_role": "student-runtime",
        "student_prompt_variant": STUDENT_PROMPT_CANONICAL,
        "action_carrier": ACTIVE_ACTION_CARRIER,
        "source_action_carriers": dict(sorted(source_carriers.items())),
        "student_runtime_prompt_characters": len(student_prompt),
        "student_runtime_prompt_sha256": hashlib.sha256(
            student_prompt.encode("utf-8")
        ).hexdigest(),
        "source_system_prompt_sha256s": dict(sorted(source_prompt_hashes.items())),
        "tool_schema_sha256": tool_schema_hash(),
        "protocol_hash": protocol_hash(student_prompt),
        "structured_actions_unchanged": True,
        "human_context_and_metadata_unchanged": True,
        "all_episodes_have_one_final_terminal": True,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--index", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()
    manifest = build(
        [path.resolve() for path in args.input],
        [path.resolve() for path in args.index],
        args.out.resolve(),
        args.index_out.resolve(),
        dataset_name=args.dataset_name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
