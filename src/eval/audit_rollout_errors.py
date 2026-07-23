#!/usr/bin/env python3
"""Aggregate protocol/execution failures from causal rollout artifacts.

The reader supports three active artifact shapes:

- one flat teacher/evaluation attempt per JSONL row;
- one pass@K task containing ``samples[]`` per JSONL row;
- one normalized accepted trajectory with audit-only errors under
  ``rollout_generation.error_events``.

Protocol failures may not have a parsed action. For audit attribution only, the script extracts a
literal ``"tool":"..."`` value from raw visible content when present. It never repairs or replays
that action.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Iterable


TOOL_RE = re.compile(r'"tool"\s*:\s*"([^"]+)"', re.IGNORECASE)
ERROR_TYPES = {
    "protocol_error",
    "execution_error",
    "nonrecoverable_execution_error",
    "argument_validation_error",
}


def compact_message(value: Any, limit: int = 800) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def protocol_cluster(message: str) -> str:
    text = message.lower()
    if "native reasoning_content was empty" in text:
        return "missing_native_reasoning"
    if "visible content was empty" in text:
        return "empty_visible_content"
    if "visible content contained reasoning/prose" in text:
        return "reasoning_in_visible_content"
    if "visible content was not exactly one valid json" in text:
        return "visible_content_not_json"
    if "visible json must contain exactly" in text:
        return "visible_json_wrong_shape"
    if "exactly one non-empty <think>" in text or "missing <think>" in text:
        return "missing_or_invalid_think"
    if "exactly one <tool_call>" in text or "tool_call" in text and "expected" in text:
        return "missing_or_invalid_tool_call"
    if (
        "invalid json" in text
        or "not valid json" in text
        or "jsondecodeerror" in text
    ):
        return "invalid_action_json"
    if "unknown tool" in text or "not a valid tool" in text:
        return "unknown_tool"
    if "trailing" in text or "outside" in text:
        return "extra_visible_content"
    return "other_protocol"


def execution_cluster(message: str) -> str:
    text = message.lower()
    if "not an introduced logical column" in text:
        return "join_logical_namespace"
    if "no such column" in text or "ambiguous column" in text:
        return "column_reference"
    if "unknown table" in text or "unknown handle" in text:
        return "unknown_table_or_handle"
    if "no such function" in text:
        return "unsupported_sql_function"
    if (
        "syntax error" in text
        or "unrecognized token" in text
        or "near \"" in text
        or "operationalerror: near " in text
    ):
        return "bad_sql_expression"
    if "scalargroundingerror" in text or "value_ref must cite a scalar" in text:
        return "invalid_scalar_value_ref"
    if "in_table" in text and "one column" in text:
        return "invalid_in_table_shape"
    if "aligned columns" in text or "same number of result columns" in text:
        return "set_op_alignment"
    if "invalid plan op" in text or "unknown plan item" in text or "plan evidence" in text:
        return "invalid_plan_update"
    if "unhashable type" in text or "cannot use 'dict' as a dict key" in text:
        return "malformed_condition_value"
    if "unexpected keyword argument" in text or "unexpected arguments" in text:
        return "unsupported_argument"
    if "keyerror: 'value'" in text or "keyerror: \"value\"" in text:
        return "condition_missing_value"
    if "keyerror: 'column'" in text or "keyerror: \"column\"" in text:
        return "condition_missing_column"
    if "could not decode to utf-8" in text:
        return "sqlite_utf8_decode"
    prefix = compact_message(message).split(":", 1)[0]
    return prefix[:100] or "other_execution"


def argument_cluster(message: str) -> str:
    text = message.lower()
    if "unexpected arguments" in text:
        return "unexpected_arguments"
    if "missing arguments" in text or "requires" in text:
        return "missing_arguments"
    if "limit must be" in text:
        return "invalid_limit"
    if "legacy arguments" in text:
        return "legacy_arguments"
    return "other_argument_validation"


def classify(error_type: str, message: str) -> str:
    if error_type == "protocol_error":
        return protocol_cluster(message)
    if error_type in {"execution_error", "nonrecoverable_execution_error"}:
        return execution_cluster(message)
    if error_type == "argument_validation_error":
        return argument_cluster(message)
    return "other"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            yield value


def iter_episodes(path: Path) -> Iterable[dict[str, Any]]:
    for record in iter_jsonl(path):
        samples = record.get("samples")
        if isinstance(samples, list):
            task_id = str(
                record.get("trajectory_id")
                or f"{record.get('dataset_split', 'task')}_{record.get('example_index', 'unknown')}"
            )
            for ordinal, sample in enumerate(samples):
                if not isinstance(sample, dict):
                    continue
                episode = dict(sample)
                sample_index = sample.get("sample_index", ordinal)
                episode["_episode_id"] = f"{task_id}:sample_{sample_index}"
                episode["_question"] = record.get("question")
                yield episode
            continue
        episode = dict(record)
        episode["_episode_id"] = str(
            record.get("trajectory_id")
            or f"{record.get('db_id', 'episode')}:{record.get('example_index', 'unknown')}"
        )
        episode["_question"] = record.get("question")
        yield episode


def episode_error_events(episode: dict[str, Any]) -> list[dict[str, Any]]:
    events = episode.get("error_events")
    if not isinstance(events, list):
        events = (episode.get("rollout_generation") or {}).get("error_events")
    return [event for event in events or [] if isinstance(event, dict)]


def turn_for_event(episode: dict[str, Any], event: dict[str, Any]) -> dict[str, Any]:
    action_index = event.get("action_index")
    for turn in episode.get("turns") or []:
        if not isinstance(turn, dict):
            continue
        if isinstance(action_index, int) and turn.get("turn_index") == action_index - 1:
            return turn
        if turn.get("error_event") == event:
            return turn
    return {}


def raw_tool_name(turn: dict[str, Any]) -> str | None:
    parsed = turn.get("parsed") or {}
    if isinstance(parsed.get("tool"), str):
        return parsed["tool"]
    raw = turn.get("raw_model_output") or turn.get("model_output") or ""
    match = TOOL_RE.search(str(raw))
    return match.group(1) if match else None


def event_tool(episode: dict[str, Any], event: dict[str, Any]) -> tuple[str, str]:
    attempted = event.get("attempted_tool")
    if isinstance(attempted, str) and attempted:
        return attempted, "event"
    turn = turn_for_event(episode, event)
    parsed = (turn.get("parsed") or {}).get("tool")
    if isinstance(parsed, str) and parsed:
        return parsed, "parsed"
    inferred = raw_tool_name(turn)
    if inferred:
        return inferred, "raw_visible_inference"
    return "unparsed", "unparsed"


def event_message(episode: dict[str, Any], event: dict[str, Any]) -> str:
    if event.get("message"):
        return compact_message(event["message"])
    turn = turn_for_event(episode, event)
    return compact_message(
        turn.get("execution_error")
        or turn.get("protocol_error")
        or episode.get("fail")
        or event.get("error_type")
    )


def legal_tool_attempts(episode: dict[str, Any]) -> collections.Counter[str]:
    turns = episode.get("turns")
    if isinstance(turns, list):
        return collections.Counter(
            tool
            for turn in turns
            if isinstance(turn, dict)
            for tool in [raw_tool_name(turn)]
            if tool
        )
    return collections.Counter(
        str((step.get("tool_call") or {}).get("tool"))
        for step in episode.get("steps") or []
        if isinstance(step, dict) and (step.get("tool_call") or {}).get("tool")
    )


def top_counter(counter: collections.Counter, limit: int | None = None) -> dict[str, int]:
    items = counter.most_common(limit)
    return {str(key): int(count) for key, count in items}


def audit(paths: list[Path], label: str) -> dict[str, Any]:
    episodes = 0
    correct = 0
    legal = 0
    final_failures: collections.Counter[str] = collections.Counter()
    tool_attempts: collections.Counter[str] = collections.Counter()
    error_types: collections.Counter[str] = collections.Counter()
    error_tools: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    error_tool_sources: collections.Counter[str] = collections.Counter()
    clusters: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    cluster_tools: dict[
        str, dict[str, collections.Counter[str]]
    ] = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    episodes_with_error: collections.Counter[str] = collections.Counter()
    recovered_correct: collections.Counter[str] = collections.Counter()
    examples: dict[tuple[str, str, str], list[dict[str, Any]]] = collections.defaultdict(list)

    for path in paths:
        for episode in iter_episodes(path):
            episodes += 1
            normalized_verified = episode.get("label_status") == "verified"
            is_correct = bool(episode.get("correct")) or normalized_verified
            is_legal = bool(episode.get("legal")) or normalized_verified
            correct += is_correct
            legal += is_legal
            failure = str(episode.get("failure_type") or ("none" if is_correct else "unknown"))
            final_failures[failure] += 1
            tool_attempts.update(legal_tool_attempts(episode))

            seen_types: set[str] = set()
            for event in episode_error_events(episode):
                error_type = str(event.get("error_type") or "unknown_error")
                if error_type not in ERROR_TYPES:
                    continue
                message = event_message(episode, event)
                tool, source = event_tool(episode, event)
                cluster = classify(error_type, message)
                error_types[error_type] += 1
                error_tools[error_type][tool] += 1
                error_tool_sources[source] += 1
                clusters[error_type][cluster] += 1
                cluster_tools[error_type][cluster][tool] += 1
                seen_types.add(error_type)
                key = (error_type, cluster, tool)
                if len(examples[key]) < 3:
                    examples[key].append({
                        "episode_id": episode["_episode_id"],
                        "question": episode.get("_question"),
                        "tool": tool,
                        "tool_source": source,
                        "message": message,
                    })
            for error_type in seen_types:
                episodes_with_error[error_type] += 1
                if is_correct:
                    recovered_correct[error_type] += 1

    all_tools = sorted(
        set(tool_attempts)
        | {tool for by_tool in error_tools.values() for tool in by_tool}
    )
    per_tool = {}
    for tool in all_tools:
        typed_errors = {
            error_type: int(error_tools[error_type][tool])
            for error_type in sorted(error_tools)
            if error_tools[error_type][tool]
        }
        total_errors = sum(typed_errors.values())
        attempts = int(tool_attempts[tool])
        per_tool[tool] = {
            "observed_attempts": attempts,
            "errors": total_errors,
            "error_rate": round(total_errors / attempts, 6) if attempts else None,
            "by_error_type": typed_errors,
        }

    cluster_report = {}
    for error_type, counter in sorted(clusters.items()):
        cluster_report[error_type] = {}
        for cluster, count in counter.most_common():
            cluster_examples = [
                item
                for (kind, named_cluster, _tool), items in examples.items()
                if kind == error_type and named_cluster == cluster
                for item in items
            ][:3]
            cluster_report[error_type][cluster] = {
                "count": int(count),
                "examples": cluster_examples,
            }

    return {
        "label": label,
        "inputs": [str(path) for path in paths],
        "episodes": episodes,
        "correct": correct,
        "legal": legal,
        "final_failure_histogram": top_counter(final_failures),
        "tool_attempt_histogram": top_counter(tool_attempts),
        "error_event_histogram": top_counter(error_types),
        "episodes_with_error_type": top_counter(episodes_with_error),
        "correct_episodes_recovered_after_error_type": top_counter(recovered_correct),
        "error_tool_attribution_source": top_counter(error_tool_sources),
        "error_tools_by_type": {
            error_type: top_counter(counter)
            for error_type, counter in sorted(error_tools.items())
        },
        "per_tool": per_tool,
        "error_clusters": cluster_report,
        "error_cluster_tools": {
            error_type: {
                cluster: top_counter(tool_counts)
                for cluster, tool_counts in sorted(by_cluster.items())
            }
            for error_type, by_cluster in sorted(cluster_tools.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.input, args.label)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
