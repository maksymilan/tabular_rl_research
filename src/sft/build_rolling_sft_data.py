#!/usr/bin/env python3
"""Export verified bounded-rolling episodes as last-turn-only ShareGPT SFT records.

Each record is one decision prefix: the initial task, at most N successful assistant/tool pairs,
the current environment state (and optional LAST TOOL ERROR), then one target action. LLaMA-Factory
must use ``mask_history: true`` so only that final target contributes loss.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from bird_sft1_teacher import replay_success_trajectory  # noqa: E402
from protocol import (  # noqa: E402
    ROLLING_COMPACT_PROMPT_VERSION,
    ROLLING_CONTEXT_VERSION,
    SYSTEM_PROMPT,
    assistant_message,
    protocol_hash,
    rolling_legal_history_messages,
    rolling_system_prompt,
    tool_output_message,
)

ROLE_MAP = {"user": "human", "assistant": "gpt"}
STEP_REF = re.compile(r"\bstep_(\d+)\b")
FACTUAL_STEP_REF_KEYS = {
    "created_by",
    "evidence",
    "evidence_step_id",
    "from_step",
    "source_step_id",
    "step_id",
    "value_ref",
}
CHARS_PER_TOKEN = 3.5


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(compact(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def numeric_step_id(step_id: str) -> int:
    match = re.fullmatch(r"step_(\d+)", str(step_id))
    if not match:
        raise ValueError(f"invalid step id: {step_id!r}")
    return int(match.group(1))


def legal_history(steps: list[dict]) -> list[dict]:
    history = []
    for step in steps:
        tool_call = step["tool_call"]
        history.append({
            "assistant": assistant_message(step["think"], tool_call["tool"], tool_call["arguments"]),
            "observation": tool_output_message(step["step_id"], step["tool_output"]),
        })
    return history


def factual_step_refs(value: Any, parent_key: str | None = None):
    """Yield harness provenance references without treating plan item ids as evidence."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from factual_step_refs(item, key)
    elif isinstance(value, list):
        for item in value:
            yield from factual_step_refs(item, parent_key)
    elif parent_key in FACTUAL_STEP_REF_KEYS and isinstance(value, str):
        match = STEP_REF.fullmatch(value)
        if match:
            yield int(match.group(1))


def validate_prefix(
    trajectory: dict,
    step: dict,
    messages: list[dict],
    target: str,
) -> None:
    current = numeric_step_id(step["step_id"])
    if not messages or messages[0].get("role") != "system":
        raise ValueError(f"{trajectory['trajectory_id']} {step['step_id']}: missing system message")
    if messages[-1].get("role") != "user":
        raise ValueError(f"{trajectory['trajectory_id']} {step['step_id']}: target lacks a user context")
    if target in "\n".join(message["content"] for message in messages[1:]):
        raise ValueError(f"{trajectory['trajectory_id']} {step['step_id']}: target leaked into prompt")
    prompt = "\n".join(message["content"] for message in messages[1:])
    gold_sql = trajectory["source"].get("gold_sql")
    if gold_sql and gold_sql in prompt:
        raise ValueError(f"{trajectory['trajectory_id']} {step['step_id']}: gold SQL leaked into prompt")
    # Validate structured harness provenance. Plan item ids and natural-language goals may
    # legitimately contain a future-looking step_N label, but they are not factual evidence.
    factual_context = (
        step.get("environment_state_before"),
        step.get("last_tool_error_before"),
    )
    for context in factual_context:
        for ref in factual_step_refs(context):
            if ref < current:
                continue
            raise ValueError(
                f"{trajectory['trajectory_id']} {step['step_id']}: future/current reference step_{ref}"
            )


def convert_step(
    trajectory: dict,
    step_index: int,
    history_turns: int,
    system: str,
    *,
    compact_observations: bool = True,
) -> tuple[dict, dict]:
    steps = trajectory["steps"]
    step = steps[step_index]
    source = trajectory["source"]
    messages = rolling_legal_history_messages(
        system,
        trajectory["initial_state"]["dataset_overview"],
        trajectory["question"],
        step["environment_state_before"],
        step.get("last_tool_error_before"),
        source.get("external_knowledge"),
        legal_history(steps[:step_index]),
        history_turns,
        compact_observations=compact_observations,
    )
    tool_call = step["tool_call"]
    target = assistant_message(step["think"], tool_call["tool"], tool_call["arguments"])
    validate_prefix(trajectory, step, messages, target)
    conversations = [
        {"from": ROLE_MAP[message["role"]], "value": message["content"]}
        for message in messages[1:]
    ]
    conversations.append({"from": "gpt", "value": target})
    if conversations[-1]["from"] != "gpt" or any(not message["value"] for message in conversations):
        raise ValueError(f"{trajectory['trajectory_id']} {step['step_id']}: invalid ShareGPT record")
    if any(message["from"] not in {"human", "gpt"} for message in conversations):
        raise ValueError(f"{trajectory['trajectory_id']} {step['step_id']}: invalid conversation role")
    record_id = f"{trajectory['trajectory_id']}_{step['step_id']}"
    record = {
        "system": system,
        "conversations": conversations,
        "metadata": {
            "record_id": record_id,
            "source_episode_id": trajectory["trajectory_id"],
            "source_step_id": step["step_id"],
            "context_mode": "rolling-legal-history",
            "history_turns": history_turns,
            "loss_policy": "last_assistant_turn_only",
            "feedback_recovery": bool(step.get("feedback_recovery")),
            "recovered_from_error_type": step.get("recovered_from_error_type"),
        },
    }
    index = {
        **record["metadata"],
        "source_difficulty": trajectory.get("difficulty"),
        "tool_name": tool_call["tool"],
        "model_input_sha256": digest(messages),
        "target_sha256": digest(target),
    }
    return record, index


def write_dataset_info(out_path: Path, dataset_name: str) -> tuple[Path, Path]:
    entry = {
        dataset_name: {
            "file_name": out_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {"role_tag": "from", "content_tag": "value", "user_tag": "human", "assistant_tag": "gpt"},
        }
    }
    snippet = out_path.parent / f"dataset_info.{dataset_name}.snippet.json"
    snippet.write_text(json.dumps(entry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    registry = out_path.parent / "dataset_info.json"
    existing = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else {}
    existing.update(entry)
    registry.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return snippet, registry


def build(
    input_path: Path,
    out_path: Path,
    index_path: Path,
    *,
    history_turns: int,
    prompt_variant: str = "full",
    observation_style: str = "resident",
) -> dict:
    if history_turns <= 0:
        raise ValueError("history_turns must be positive for the bounded rolling training protocol")
    if prompt_variant not in {"full", "compact"}:
        raise ValueError(f"unknown rolling prompt variant: {prompt_variant}")
    if observation_style not in {"resident", "full"}:
        raise ValueError(f"unknown rolling observation style: {observation_style}")
    trajectories = read_jsonl(input_path)
    system = rolling_system_prompt(SYSTEM_PROMPT, compact=prompt_variant == "compact")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_index = index_path.with_suffix(index_path.suffix + ".tmp")
    target_lengths: list[int] = []
    prefix_lengths: list[int] = []
    tool_hist: Counter = Counter()
    difficulty_hist: Counter = Counter()
    recovery_count = replayed = record_count = 0

    try:
        with tmp_out.open("w", encoding="utf-8") as out, tmp_index.open("w", encoding="utf-8") as index:
            for trajectory in trajectories:
                if trajectory.get("label_status") != "verified":
                    raise ValueError(f"{trajectory.get('trajectory_id')}: not verified")
                generation = trajectory.get("rollout_generation") or {}
                if generation.get("context_mode") != "rolling-legal-history":
                    raise ValueError(f"{trajectory['trajectory_id']}: not a rolling episode")
                if generation.get("history_turns") != history_turns:
                    raise ValueError(
                        f"{trajectory['trajectory_id']}: history window {generation.get('history_turns')} != {history_turns}"
                    )
                replay_ok, replay_error = replay_success_trajectory(trajectory)
                if not replay_ok:
                    raise ValueError(f"{trajectory['trajectory_id']}: replay failed: {replay_error}")
                replayed += 1
                for step_index, step in enumerate(trajectory["steps"]):
                    record, index_row = convert_step(
                        trajectory,
                        step_index,
                        history_turns,
                        system,
                        compact_observations=observation_style == "resident",
                    )
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    index.write(json.dumps(index_row, ensure_ascii=False) + "\n")
                    record_count += 1
                    tool_hist[step["tool_call"]["tool"]] += 1
                    difficulty_hist[trajectory.get("difficulty") or "unknown"] += 1
                    recovery_count += bool(step.get("feedback_recovery"))
                    target_lengths.append(len(record["conversations"][-1]["value"]))
                    prefix_lengths.append(len(system) + sum(len(message["value"]) for message in record["conversations"]))
        os.replace(tmp_out, out_path)
        os.replace(tmp_index, index_path)
    finally:
        if tmp_out.exists():
            tmp_out.unlink()
        if tmp_index.exists():
            tmp_index.unlink()

    prefix_lengths.sort()
    percentile = lambda fraction: prefix_lengths[min(len(prefix_lengths) - 1, int(fraction * len(prefix_lengths)))] if prefix_lengths else 0
    return {
        "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output": str(out_path),
        "index": str(index_path),
        "context_mode": "rolling-legal-history",
        "rolling_context_version": (
            ROLLING_CONTEXT_VERSION
            if observation_style == "resident"
            else "v1-bounded-legal-history-full-observations"
        ),
        "rolling_prompt_variant": prompt_variant,
        "rolling_observation_style": observation_style,
        "rolling_compact_prompt_version": (
            ROLLING_COMPACT_PROMPT_VERSION if prompt_variant == "compact" else None
        ),
        "history_turns": history_turns,
        "loss_policy": "last_assistant_turn_only",
        "required_llamafactory_flag": "mask_history: true",
        "system_prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "base_protocol_hash": protocol_hash(system),
        "source_episodes": len(trajectories),
        "replayed_episodes": replayed,
        "records": record_count,
        "feedback_recovery_targets": recovery_count,
        "difficulty_targets": dict(sorted(difficulty_hist.items())),
        "tool_hist": dict(tool_hist.most_common()),
        "prefix_characters": {
            "p50": percentile(.5), "p90": percentile(.9), "p95": percentile(.95),
            "max": prefix_lengths[-1] if prefix_lengths else 0,
            "mean": int(statistics.mean(prefix_lengths)) if prefix_lengths else 0,
        },
        "target_characters": {
            "mean": int(statistics.mean(target_lengths)) if target_lengths else 0,
            "max": max(target_lengths) if target_lengths else 0,
        },
        "token_estimate_method": f"characters / {CHARS_PER_TOKEN}",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="verified rolling success JSONL")
    parser.add_argument("--out", type=Path, required=True, help="ShareGPT train JSONL output")
    parser.add_argument("--index-out", type=Path, default=None)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--rolling-prompt-variant", choices=["full", "compact"], default="full")
    parser.add_argument(
        "--rolling-observation-style",
        choices=["resident", "full"],
        default="resident",
        help="resident is R2; full reproduces pre-R2 historical tool observations for old adapters",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.dataset_name):
        parser.error("--dataset-name must contain only letters, digits, '.', '_' or '-'")
    input_path = args.input.resolve()
    out_path = args.out.resolve()
    index_path = (args.index_out or out_path.with_name(out_path.stem + "_index.jsonl")).resolve()
    manifest = build(
        input_path,
        out_path,
        index_path,
        history_turns=args.history_turns,
        prompt_variant=args.rolling_prompt_variant,
        observation_style=args.rolling_observation_style,
    )
    snippet, registry = write_dataset_info(out_path, args.dataset_name)
    manifest["dataset_name"] = args.dataset_name
    manifest["dataset_info_snippet"] = str(snippet)
    manifest["dataset_info_registry"] = str(registry)
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
