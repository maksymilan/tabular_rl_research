#!/usr/bin/env python3
"""Export verified action-block episodes as last-turn-only ShareGPT records.

The source provider may carry reasoning separately from visible raw JSON. The student carrier is
strict ``<think>...</think>`` followed by that same JSON action. Every target is reconstructed from
the causal recorded model prefix, and every source episode is freshly replayed before export.

This builder intentionally rejects current evaluation artifacts unless they were explicitly marked
``sft_export_eligible`` after an evaluation gate. Verified recovery episodes are allowed, but a
failed or blocked action block is never an SFT target; only later clean recovery blocks may carry
loss.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import statistics
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from tool_modules.action_block.protocol import (  # noqa: E402
    UNIFIED_ACTION_BLOCK_PROTOCOL_VERSION,
)
from executor import Harness  # noqa: E402
from rollout import execute_tool, new_ctx, overview, score, task_db_path  # noqa: E402
from tool_modules.registry import (  # noqa: E402
    ACTION_BLOCK_TOOL_SCHEME,
    TOOL_SCHEME_REGISTRY_VERSION,
    assert_record_tool_scheme,
    build_action_block_tool_scheme,
    parse_scheme_action,
    render_scheme_action,
)


ROLE_MAP = {"user": "human", "assistant": "gpt"}
CHARS_PER_TOKEN = 3.5


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(compact(value).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_tasks(path: Path, split: str) -> dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        rows = (
            [json.loads(line) for line in handle if line.strip()]
            if path.suffix == ".jsonl"
            else json.load(handle)
        )
    if not isinstance(rows, list):
        raise ValueError("tasks file must contain a JSON array or JSONL records")
    indexed: dict[str, dict] = {}
    for index, task in enumerate(rows):
        task_id = (
            task.get("trajectory_id")
            or task.get("task_id")
            or f"{task.get('dataset', 'bird')}_{split}_{index}"
        )
        indexed[str(task_id)] = task
        indexed.setdefault(f"bird_{split}_{index:05d}", task)
        indexed.setdefault(f"{split}_{index}", task)
    return indexed


def task_for_record(record: dict, tasks: dict[str, dict]) -> dict:
    candidates = (
        record.get("trajectory_id"),
        record.get("task_id"),
        f"bird_train_{int(record.get('example_index', -1)):05d}",
        f"train_{record.get('example_index')}",
    )
    for candidate in candidates:
        if candidate is not None and str(candidate) in tasks:
            task = deepcopy(tasks[str(candidate)])
            task.setdefault("db_id", record.get("db_id"))
            task.setdefault("question", record.get("question"))
            task.setdefault("gold_sql", record.get("gold_sql"))
            return task
    raise ValueError(
        f"{record.get('trajectory_id')}: no matching task in replay task file"
    )


def validate_source_episode(record: dict) -> None:
    trajectory_id = record.get("trajectory_id") or "<unknown>"
    assert_record_tool_scheme(record, ACTION_BLOCK_TOOL_SCHEME)
    if record.get("protocol_version") != UNIFIED_ACTION_BLOCK_PROTOCOL_VERSION:
        raise ValueError(
            f"{trajectory_id}: only {UNIFIED_ACTION_BLOCK_PROTOCOL_VERSION} episodes "
            "may enter the active action-block SFT path"
        )
    if record.get("sft_export_eligible") is not True:
        raise ValueError(
            f"{trajectory_id}: action-block episode was not explicitly promoted for SFT export"
        )
    if not record.get("correct") or not record.get("legal"):
        raise ValueError(f"{trajectory_id}: episode is not a verified correct termination")
    if record.get("denotation_comparison") != "bird-set":
        raise ValueError(f"{trajectory_id}: SFT replay metric must be bird-set")
    turns = record.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{trajectory_id}: episode has no model turns")
    for index, turn in enumerate(turns):
        parsed = turn.get("parsed")
        if parsed is not None and (
            not isinstance(parsed, dict)
            or not isinstance(parsed.get("arguments"), dict)
        ):
            raise ValueError(f"{trajectory_id}: turn {index} has malformed parsed action")
        if parsed is not None and not str(
            turn.get("provider_reasoning_content") or ""
        ).strip():
            raise ValueError(f"{trajectory_id}: turn {index} has empty causal reasoning")
    final_parsed = turns[-1]["parsed"]
    if not (
        final_parsed.get("tool") == "answer_from_context"
        and isinstance(final_parsed.get("arguments"), dict)
    ):
        raise ValueError(
            f"{trajectory_id}: final turn must be the standalone terminal action"
        )


def replay_episode(record: dict, task: dict) -> None:
    """Replay successful primitive events and verify terminal denotation.

    Failed and blocked calls are audit-only context: recoverable failures are required to preserve
    state, so replay skips them and re-executes every later successful primitive in causal order.
    """
    trajectory_id = record.get("trajectory_id") or "<unknown>"
    harness = Harness(task_db_path(task))
    ctx = new_ctx(overview(harness))
    created: set[str] = set()
    try:
        atomic_events = record.get("atomic_events")
        if not isinstance(atomic_events, list) or not atomic_events:
            raise ValueError(f"{trajectory_id}: episode has no atomic replay events")
        for event in atomic_events:
            if event.get("status") != "success":
                continue
            tool = event.get("tool")
            step_id = event.get("step_id")
            if tool == "answer_from_context":
                score_arguments = deepcopy(
                    event.get("resolved_evidence_arguments")
                    or event["arguments"]
                )
                correct, _, _ = score(
                    harness,
                    task.get("gold_sql") or record.get("gold_sql"),
                    score_arguments,
                    created,
                    denotation_comparison="bird-set",
                )
                if not correct:
                    raise ValueError(
                        f"{trajectory_id}: fresh terminal replay did not match gold denotation"
                    )
                return
            output, table = execute_tool(
                harness,
                tool,
                deepcopy(event.get("resolved_arguments") or event.get("arguments") or {}),
                ctx,
                step_id,
                table_output_rows=0,
            )
            if table:
                created.add(table)
        raise ValueError(f"{trajectory_id}: replay reached no terminal action")
    finally:
        harness.conn.close()


def inline_model_input(
    record: dict,
    turn_index: int,
    target_system_prompt: str,
) -> list[dict]:
    """Translate provider-native legal history into the student inline carrier."""
    messages = deepcopy(record["turns"][turn_index]["model_input"])
    if not messages or messages[0].get("role") != "system":
        raise ValueError(
            f"{record.get('trajectory_id')} turn {turn_index}: missing system prompt"
        )
    messages[0]["content"] = target_system_prompt
    prior_turns = record["turns"][:turn_index]
    carrier_map = {
        str(turn.get("raw_model_output") or "").strip(): str(
            turn.get("canonical_model_output") or ""
        ).strip()
        for turn in prior_turns
        if str(turn.get("raw_model_output") or "").strip()
    }
    for message in messages:
        if message.get("role") != "assistant":
            continue
        raw = str(message.get("content") or "").strip()
        canonical = carrier_map.get(raw)
        if not canonical:
            raise ValueError(
                f"{record.get('trajectory_id')} turn {turn_index}: "
                "prior assistant history cannot be translated to inline carrier"
            )
        message["content"] = canonical
    return messages


def convert_turn(
    record: dict,
    turn_index: int,
    *,
    max_batch_calls: int,
) -> tuple[dict, dict]:
    scheme = build_action_block_tool_scheme(max_batch_calls=max_batch_calls)
    turn = record["turns"][turn_index]
    parsed = turn["parsed"]
    reasoning = str(turn.get("provider_reasoning_content") or "").strip()
    target = render_scheme_action(
        scheme,
        reasoning,
        parsed["tool"],
        parsed["arguments"],
    )
    parsed_reason, parsed_tool, parsed_arguments = parse_scheme_action(scheme, target)
    if (
        parsed_reason != reasoning
        or parsed_tool != parsed["tool"]
        or parsed_arguments != parsed["arguments"]
    ):
        raise ValueError(
            f"{record.get('trajectory_id')} turn {turn_index}: target carrier round-trip failed"
        )
    messages = inline_model_input(
        record,
        turn_index,
        scheme.system_prompt,
    )
    prefix_text = "\n".join(str(message.get("content") or "") for message in messages)
    if target in prefix_text:
        raise ValueError(
            f"{record.get('trajectory_id')} turn {turn_index}: target leaked into prefix"
        )
    gold_sql = str(record.get("gold_sql") or "")
    if gold_sql and gold_sql in prefix_text:
        raise ValueError(
            f"{record.get('trajectory_id')} turn {turn_index}: gold SQL leaked into prefix"
        )
    conversations = [
        {"from": ROLE_MAP[message["role"]], "value": message["content"]}
        for message in messages[1:]
    ]
    conversations.append({"from": "gpt", "value": target})
    record_id = f"{record['trajectory_id']}_turn_{turn_index + 1}"
    metadata = {
        "record_id": record_id,
        "source_episode_id": record["trajectory_id"],
        "source_turn_index": turn_index,
        "tool_scheme": ACTION_BLOCK_TOOL_SCHEME,
        "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
        "assistant_carrier": scheme.assistant_carrier,
        "protocol_version": scheme.protocol_version,
        "protocol_hash": scheme.protocol_hash,
        "context_mode": "rolling-legal-history",
        "loss_policy": "last_assistant_turn_only",
        "feedback_recovery": bool(turn.get("feedback_recovery")),
    }
    output = {
        "system": scheme.system_prompt,
        "conversations": conversations,
        "metadata": metadata,
    }
    index = {
        **metadata,
        "tool_name": parsed_tool,
        "atomic_calls": (
            len(parsed_arguments["calls"])
            if parsed_tool == "action_block"
            else 1
        ),
        "model_input_sha256": digest(messages),
        "target_sha256": digest(target),
    }
    return output, index


def is_sft_target_turn(turn: dict) -> bool:
    """Only cleanly executed blocks are targets; error blocks remain causal context."""
    parsed = turn.get("parsed")
    if (
        turn.get("execution_error")
        or not isinstance(parsed, dict)
        or not isinstance(parsed.get("arguments"), dict)
        or int(turn.get("root_error_count") or 0)
        or int(turn.get("blocked_count") or 0)
    ):
        return False
    results = turn.get("batch_results")
    if isinstance(results, list) and any(
        result.get("status") != "success"
        for result in results
    ):
        return False
    return True


def percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    return values[min(len(values) - 1, int(fraction * len(values)))]


def build(
    input_path: Path,
    tasks_path: Path,
    out_path: Path,
    index_path: Path,
    *,
    split: str,
    max_batch_calls: int = 5,
) -> dict:
    records = read_jsonl(input_path)
    tasks = load_tasks(tasks_path, split)
    scheme = build_action_block_tool_scheme(max_batch_calls=max_batch_calls)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp_index = index_path.with_suffix(index_path.suffix + ".tmp")
    tool_hist: collections.Counter = collections.Counter()
    batch_size_hist: collections.Counter = collections.Counter()
    prefix_lengths: list[int] = []
    target_lengths: list[int] = []
    emitted = 0
    skipped_error_turns = 0
    recovery_targets = 0
    try:
        with tmp_out.open("w", encoding="utf-8") as out, tmp_index.open(
            "w", encoding="utf-8"
        ) as index:
            for record in records:
                validate_source_episode(record)
                replay_episode(record, task_for_record(record, tasks))
                for turn_index in range(len(record["turns"])):
                    if not is_sft_target_turn(record["turns"][turn_index]):
                        skipped_error_turns += 1
                        continue
                    converted, index_row = convert_turn(
                        record,
                        turn_index,
                        max_batch_calls=max_batch_calls,
                    )
                    out.write(json.dumps(converted, ensure_ascii=False) + "\n")
                    index.write(json.dumps(index_row, ensure_ascii=False) + "\n")
                    emitted += 1
                    recovery_targets += bool(index_row.get("feedback_recovery"))
                    tool_hist[index_row["tool_name"]] += 1
                    batch_size_hist[index_row["atomic_calls"]] += 1
                    prefix_lengths.append(
                        len(converted["system"])
                        + sum(
                            len(message["value"])
                            for message in converted["conversations"][:-1]
                        )
                    )
                    target_lengths.append(len(converted["conversations"][-1]["value"]))
        os.replace(tmp_out, out_path)
        os.replace(tmp_index, index_path)
    finally:
        if tmp_out.exists():
            tmp_out.unlink()
        if tmp_index.exists():
            tmp_index.unlink()
    prefix_lengths.sort()
    target_lengths.sort()
    return {
        **scheme.manifest_fields(),
        "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "replay_tasks": str(tasks_path),
        "replay_tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "split": split,
        "output": str(out_path),
        "index": str(index_path),
        "source_episodes": len(records),
        "fresh_replayed_episodes": len(records),
        "records": emitted,
        "loss_policy": "last_assistant_turn_only",
        "required_llamafactory_flag": "mask_history: true",
        "denotation_comparison": "bird-set",
        "error_actions_are_sft_targets": False,
        "error_turns_retained_as_causal_context": True,
        "skipped_error_turns": skipped_error_turns,
        "feedback_recovery_targets": recovery_targets,
        "tool_hist": dict(tool_hist.most_common()),
        "batch_size_hist": dict(sorted(batch_size_hist.items())),
        "prefix_characters": {
            "p50": percentile(prefix_lengths, 0.5),
            "p95": percentile(prefix_lengths, 0.95),
            "max": max(prefix_lengths, default=0),
            "mean": int(statistics.mean(prefix_lengths)) if prefix_lengths else 0,
        },
        "target_characters": {
            "p50": percentile(target_lengths, 0.5),
            "p95": percentile(target_lengths, 0.95),
            "max": max(target_lengths, default=0),
            "mean": int(statistics.mean(target_lengths)) if target_lengths else 0,
        },
        "token_estimate_method": f"characters / {CHARS_PER_TOKEN}",
    }


def write_dataset_info(out_path: Path, dataset_name: str) -> tuple[Path, Path]:
    entry = {
        dataset_name: {
            "file_name": out_path.name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }
    snippet = out_path.parent / f"dataset_info.{dataset_name}.snippet.json"
    snippet.write_text(
        json.dumps(entry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    registry = out_path.parent / "dataset_info.json"
    existing = (
        json.loads(registry.read_text(encoding="utf-8"))
        if registry.exists()
        else {}
    )
    existing.update(entry)
    registry.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return snippet, registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--tasks-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", choices=("train", "dev"), default="train")
    parser.add_argument("--max-batch-calls", type=int, default=5)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.dataset_name):
        parser.error("--dataset-name must contain only letters, digits, '.', '_' or '-'")
    if not 1 <= args.max_batch_calls <= 5:
        parser.error("--max-batch-calls must be between 1 and 5")
    input_path = args.input.resolve()
    tasks_path = args.tasks_json.resolve()
    out_path = args.out.resolve()
    index_path = (
        args.index_out
        or out_path.with_name(out_path.stem + "_index.jsonl")
    ).resolve()
    manifest = build(
        input_path,
        tasks_path,
        out_path,
        index_path,
        split=args.split,
        max_batch_calls=args.max_batch_calls,
    )
    snippet, registry = write_dataset_info(out_path, args.dataset_name)
    manifest["dataset_name"] = args.dataset_name
    manifest["dataset_info_snippet"] = str(snippet)
    manifest["dataset_info_registry"] = str(registry)
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
