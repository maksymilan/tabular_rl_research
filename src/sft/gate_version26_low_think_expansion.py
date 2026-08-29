#!/usr/bin/env python3
"""Audit and admit a completed Low-Think Atomic-v24 teacher expansion.

The input is a directory of ``generate_teacher_rollouts.py`` shards.  This gate is deliberately
non-mutating with respect to model-authored text: it only selects whole causal episodes.  Admission
requires a correct terminal result, official Flash/request identity, no model-visible gold leak,
bounded Qwen3 reasoning tokens, and a fresh replay under the frozen Atomic version26 runtime.

Run this script with ``PYTHONPATH`` pointing at the isolated version26 ``src/sft``, ``src/eval``,
and ``src/harness`` directories so ``replay_success_trajectory`` is identity-bound to that runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from bird_sft1_teacher import replay_success_trajectory


EXPECTED_MODEL = "deepseek-v4-flash"
EXPECTED_PROTOCOL_HASH = "25ac4c10ef96365c"
EXPECTED_REQUEST_OPTIONS = {
    "thinking": {"type": "enabled"},
    "reasoning_effort": "low",
    "response_format": {"type": "json_object"},
}
FORBIDDEN_REASONING_MARKERS = (
    "gold sql",
    "gold_sql",
    "gold answer",
    "reference sql",
    "reference answer",
    "ground truth",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(fraction * len(ordered))))
    return ordered[index]


def token_stats(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 2) if values else None,
        "median": statistics.median(values) if values else None,
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def validate_manifest(manifest: dict[str, Any], *, shard_size: int) -> None:
    expected = {
        "model": EXPECTED_MODEL,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "context_mode": "rolling-legal-history",
        "history_turns": 4,
        "rolling_prompt_variant": "full",
        "policy_prompt_variant": "canonical",
        "plan_policy": "optional",
        "deepseek_carrier": "json-output",
        "denotation_comparison": "bird-set",
        "max_steps": 30,
        "max_tokens": 2048,
        "attempts_per_example": 1,
        "unique_examples": shard_size,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": manifest.get(key)}
        for key, expected_value in expected.items()
        if manifest.get(key) != expected_value
    }
    if manifest.get("provider_request_options") != EXPECTED_REQUEST_OPTIONS:
        mismatches["provider_request_options"] = {
            "expected": EXPECTED_REQUEST_OPTIONS,
            "actual": manifest.get("provider_request_options"),
        }
    if mismatches:
        raise ValueError(f"shard manifest identity mismatch: {mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--expected-tasks-sha256", required=True)
    parser.add_argument("--expected-selection-sha256", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--max-turn-reason-tokens", type=int, default=1024)
    parser.add_argument("--max-episode-reason-tokens", type=int, default=4096)
    args = parser.parse_args()

    result_root = args.result_root.resolve()
    status_path = result_root / "status.json"
    if not status_path.is_file():
        raise FileNotFoundError(status_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("state") != "completed":
        raise RuntimeError(f"generation is not complete: {status.get('state')!r}")
    if status.get("stop_reason") is not None:
        raise RuntimeError(f"generation has a stop reason: {status.get('stop_reason')!r}")
    if sha256(args.tasks) != args.expected_tasks_sha256:
        raise RuntimeError("tasks SHA mismatch")
    if sha256(args.selection_manifest) != args.expected_selection_sha256:
        raise RuntimeError("selection manifest SHA mismatch")

    tasks = read_jsonl(args.tasks)
    if len(tasks) != args.episodes:
        raise RuntimeError(f"task count mismatch: {len(tasks)}")
    task_ids = [str(row.get("example_id")) for row in tasks]
    if len(set(task_ids)) != args.episodes:
        raise RuntimeError("task IDs are not unique")
    if {str(row.get("dataset")) for row in tasks} != {"bird-sql"}:
        raise RuntimeError("tasks are not pure BIRD")

    all_by_id: dict[str, dict[str, Any]] = {}
    trajectory_by_id: dict[str, dict[str, Any]] = {}
    manifest_paths: list[Path] = []
    all_paths: list[Path] = []
    verified_paths: list[Path] = []
    usage = Counter()
    for start in range(0, args.episodes, args.shard_size):
        shard = result_root / f"shard_{start:04d}_{start + args.shard_size - 1:04d}"
        manifest_path = shard / "verified.manifest.json"
        all_path = shard / "verified.all.jsonl"
        verified_path = shard / "verified.jsonl"
        for path in (manifest_path, all_path, verified_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_manifest(manifest, shard_size=args.shard_size)
        manifest_paths.append(manifest_path)
        all_paths.append(all_path)
        verified_paths.append(verified_path)
        for key, value in (manifest.get("usage_total") or {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                usage[key] += value
        shard_all = read_jsonl(all_path)
        if len(shard_all) != args.shard_size:
            raise RuntimeError(f"{shard.name}: all-record count mismatch")
        for row in shard_all:
            trajectory_id = str(row.get("trajectory_id"))
            if trajectory_id in all_by_id:
                raise RuntimeError(f"duplicate generated trajectory ID: {trajectory_id}")
            all_by_id[trajectory_id] = row
        for trajectory in read_jsonl(verified_path):
            trajectory_id = str(trajectory.get("trajectory_id"))
            if trajectory_id in trajectory_by_id:
                raise RuntimeError(f"duplicate verified trajectory ID: {trajectory_id}")
            trajectory_by_id[trajectory_id] = trajectory

    if set(all_by_id) != set(task_ids):
        raise RuntimeError("generated final IDs differ from frozen task selection")
    correct_ids = {item for item, row in all_by_id.items() if row.get("correct") is True}
    if correct_ids != set(trajectory_by_id):
        raise RuntimeError("verified trajectories differ from correct final records")

    response_models = Counter()
    request_options = Counter()
    response_metadata_missing = 0
    identity_rejected: dict[str, list[str]] = {}
    no_leak_rejected: dict[str, list[str]] = {}
    for trajectory_id in task_ids:
        row = all_by_id[trajectory_id]
        identity_issues: list[str] = []
        leak_issues: list[str] = []
        gold_sql = normalized_text(str(row.get("gold_sql") or ""))
        for turn_index, turn in enumerate(row.get("turns") or []):
            options = turn.get("provider_request_options")
            request_options[json.dumps(options, sort_keys=True, ensure_ascii=False)] += 1
            if options != EXPECTED_REQUEST_OPTIONS:
                identity_issues.append(f"turn_{turn_index}:request_options")
            if "model_output" in turn:
                response = turn.get("provider_response_metadata")
                if not isinstance(response, dict):
                    response_metadata_missing += 1
                    identity_issues.append(f"turn_{turn_index}:missing_response_metadata")
                else:
                    response_model = str(response.get("model"))
                    response_models[response_model] += 1
                    if response_model != EXPECTED_MODEL:
                        identity_issues.append(f"turn_{turn_index}:response_model")

            model_input_text = normalized_text("\n".join(strings(turn.get("model_input") or [])))
            if gold_sql and gold_sql in model_input_text:
                leak_issues.append(f"turn_{turn_index}:gold_sql_in_model_input")
            if '"gold_sql"' in model_input_text or '"gold_sample"' in model_input_text:
                leak_issues.append(f"turn_{turn_index}:forbidden_gold_field_in_model_input")
            reasoning = normalized_text(str(turn.get("provider_reasoning_content") or ""))
            if gold_sql and gold_sql in reasoning:
                leak_issues.append(f"turn_{turn_index}:gold_sql_in_reasoning")
            for marker in FORBIDDEN_REASONING_MARKERS:
                if marker in reasoning:
                    leak_issues.append(f"turn_{turn_index}:forbidden_marker:{marker}")
        if identity_issues:
            identity_rejected[trajectory_id] = sorted(set(identity_issues))
        if leak_issues:
            no_leak_rejected[trajectory_id] = sorted(set(leak_issues))

    if response_metadata_missing:
        raise RuntimeError(f"provider response metadata missing on {response_metadata_missing} turns")
    if identity_rejected:
        raise RuntimeError(f"provider identity mismatch in {len(identity_rejected)} episodes")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        local_files_only=True,
    )
    turn_token_values: list[int] = []
    episode_token_values: list[int] = []
    reason_rejected: dict[str, dict[str, Any]] = {}
    token_passed_ids: set[str] = set()
    for trajectory_id, trajectory in trajectory_by_id.items():
        counts = [
            len(tokenizer.encode(str(step.get("think") or ""), add_special_tokens=False))
            for step in trajectory.get("steps") or []
        ]
        turn_token_values.extend(counts)
        episode_total = sum(counts)
        episode_token_values.append(episode_total)
        episode_max = max(counts, default=0)
        if episode_max > args.max_turn_reason_tokens or episode_total > args.max_episode_reason_tokens:
            reason_rejected[trajectory_id] = {
                "episode_total_reason_tokens": episode_total,
                "max_turn_reason_tokens": episode_max,
                "turns_over_limit": sum(value > args.max_turn_reason_tokens for value in counts),
            }
        else:
            token_passed_ids.add(trajectory_id)

    pre_replay_ids = token_passed_ids - set(no_leak_rejected)
    replay_rejected: dict[str, str] = {}
    admitted_ids: set[str] = set()
    for trajectory_id in task_ids:
        if trajectory_id not in pre_replay_ids:
            continue
        passed, error = replay_success_trajectory(
            trajectory_by_id[trajectory_id],
            denotation_comparison="bird-set",
        )
        if passed:
            admitted_ids.add(trajectory_id)
        else:
            replay_rejected[trajectory_id] = str(error)

    ordered_correct = [trajectory_by_id[item] for item in task_ids if item in trajectory_by_id]
    ordered_token_rejected = [
        trajectory_by_id[item] for item in task_ids if item in reason_rejected
    ]
    ordered_admitted = [trajectory_by_id[item] for item in task_ids if item in admitted_ids]
    correct_path = args.out_dir / "correct_generated.jsonl"
    reason_rejected_path = args.out_dir / "reason_token_rejected.jsonl"
    admitted_path = args.out_dir / "admitted_trajectories.jsonl"
    write_jsonl_atomic(correct_path, ordered_correct)
    write_jsonl_atomic(reason_rejected_path, ordered_token_rejected)
    write_jsonl_atomic(admitted_path, ordered_admitted)

    audit = {
        "schema_version": "atomic-version26-low-think-expansion-admission-v1",
        "status": "admitted_trajectories_ready_for_exact_version26_projection",
        "source": {
            "result_root": str(result_root),
            "tasks": str(args.tasks.resolve()),
            "tasks_sha256": sha256(args.tasks),
            "selection_manifest": str(args.selection_manifest.resolve()),
            "selection_manifest_sha256": sha256(args.selection_manifest),
            "shard_manifests": {str(path): sha256(path) for path in manifest_paths},
            "shard_all_records": {str(path): sha256(path) for path in all_paths},
            "shard_verified": {str(path): sha256(path) for path in verified_paths},
        },
        "identity": {
            "model": EXPECTED_MODEL,
            "protocol_hash": EXPECTED_PROTOCOL_HASH,
            "request_options": EXPECTED_REQUEST_OPTIONS,
            "response_model_counts": dict(sorted(response_models.items())),
            "request_option_variants": dict(sorted(request_options.items())),
            "response_metadata_missing": response_metadata_missing,
            "identity_rejected_episodes": len(identity_rejected),
        },
        "counts": {
            "selected_episodes": len(task_ids),
            "terminal_correct_episodes": len(trajectory_by_id),
            "terminal_failed_episodes": len(task_ids) - len(trajectory_by_id),
            "reason_token_passed_episodes": len(token_passed_ids),
            "reason_token_rejected_episodes": len(reason_rejected),
            "no_leak_rejected_episodes": len(no_leak_rejected),
            "fresh_replay_passed_episodes": len(admitted_ids),
            "fresh_replay_rejected_episodes": len(replay_rejected),
        },
        "reason_token_gate": {
            "tokenizer": args.tokenizer,
            "max_per_turn": args.max_turn_reason_tokens,
            "max_per_episode": args.max_episode_reason_tokens,
            "all_correct_turns": token_stats(turn_token_values),
            "all_correct_episodes": token_stats(episode_token_values),
            "rejected": reason_rejected,
        },
        "no_leak": {
            "forbidden_reasoning_markers": list(FORBIDDEN_REASONING_MARKERS),
            "rejected": no_leak_rejected,
        },
        "fresh_replay": {
            "denotation_comparison": "bird-set",
            "rejected": replay_rejected,
        },
        "provider_usage": dict(sorted(usage.items())),
        "outputs": {
            "correct_generated": {"path": str(correct_path), "sha256": sha256(correct_path)},
            "reason_token_rejected": {
                "path": str(reason_rejected_path),
                "sha256": sha256(reason_rejected_path),
            },
            "admitted_trajectories": {
                "path": str(admitted_path),
                "sha256": sha256(admitted_path),
            },
        },
    }
    write_json_atomic(args.out_dir / "ADMISSION_AUDIT.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
