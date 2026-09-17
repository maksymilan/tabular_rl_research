#!/usr/bin/env python3
"""Rebuild verified rolling episodes from recorded K-sample student rollouts.

The K=8 screening artifacts keep, per sample, the rendered ``model_input``, the raw carrier and
the parsed action (``think`` / ``tool`` / ``arguments``), plus the recorded ``tool_output`` --
but no environment-state snapshots.  ``build_rolling_sft_data.py`` requires canonical episodes
carrying ``environment_state_before`` / ``environment_state``, so this converter re-executes the
recorded action sequence on the frozen Harness, captures the snapshots it needs, and admits an
episode only when

  * every turn parses into a non-empty think block plus one typed tool call,
  * the re-executed tool output matches the recorded output exactly, and
  * the terminal ``answer_from_context`` call scores correct under the frozen comparison.

The result is the exact input format of ``build_rolling_sft_data.py``, which then applies its own
independent replay gate and emits the ``rolling-legal-history`` ShareGPT training view.

Scope note: this is the on-policy self-training ("RFT") data path.  Its ``data_selection``
semantics (difficulty band, per-question cap, determinism) live here so the scenario CLI stays a
thin parameter surface.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src"),
    str(HERE),
]

from bird_sft1_teacher import make_step  # noqa: E402
from executor import Harness  # noqa: E402
from rollout import execute_tool, new_ctx, overview, score  # noqa: E402
from tool_modules.registry import (  # noqa: E402
    ATOMIC_TOOL_SCHEME,
    TOOL_SCHEME_REGISTRY_VERSION,
)

SCHEMA_VERSION = "qwen3-atomic-v26-rft-episode-v1"
TERMINAL_TOOL = "answer_from_context"
LAST_TOOL_ERROR_MARKER = "\n\nLAST TOOL ERROR\n"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _row_key(row: Any) -> str:
    return json.dumps(row, sort_keys=True, ensure_ascii=False, default=str)


def observation_matches(recorded: Any, replayed: Any) -> tuple[bool, bool]:
    """Compare a recorded observation against the replayed one.

    The screen recorder kept fewer fields than a later harness fills in (``limit`` / ``offset`` /
    ``order_by`` / ``conditions`` are echoes with defaults), so every field the recorder kept must
    match and the recorder must not contain fields the replay lacks.  Row lists compare as
    multisets: without an explicit ``ORDER BY`` SQLite row order is not reproducible across runs,
    and the episode keeps the recorded order anyway because that is what the model actually saw.

    Returns ``(matched, order_differs)``.
    """
    if not isinstance(recorded, dict) or not isinstance(replayed, dict):
        return canonical(recorded) == canonical(replayed), False
    order_differs = False
    for key, value in recorded.items():
        if key not in replayed:
            return False, False
        other = replayed[key]
        if key == "rows" and isinstance(value, list) and isinstance(other, list):
            if sorted(map(_row_key, value)) != sorted(map(_row_key, other)):
                return False, False
            if [_row_key(item) for item in value] != [_row_key(item) for item in other]:
                order_differs = True
        elif canonical(value) != canonical(other):
            return False, False
    return True, order_differs


def extract_last_tool_error(model_input: Any) -> tuple[dict | None, str | None]:
    """Recover the structured ``LAST TOOL ERROR`` payload from a recorded model input.

    ``protocol.environment_state_message`` renders the latest rejected action as
    ``"\\n\\nLAST TOOL ERROR\\n" + json.dumps(error, separators=(",", ":"))`` appended to the
    mutable-context user message, so the payload is exactly recoverable.  The body is not a
    fixed placeholder: it carries the real ``error.type`` and ``error.message`` (plus the
    ``actionable-error-v1`` hints) for that rejected action.
    """
    messages = model_input
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError:
            return None, "unparsable_model_input"
    if not isinstance(messages, list):
        return None, "unparsable_model_input"
    payload: str | None = None
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str) and LAST_TOOL_ERROR_MARKER in content:
            payload = content.rsplit(LAST_TOOL_ERROR_MARKER, 1)[1]
    if payload is None:
        return None, "missing_last_tool_error"
    try:
        error = json.loads(payload)
    except json.JSONDecodeError:
        return None, "unparsable_last_tool_error"
    if not isinstance(error, dict):
        return None, "unparsable_last_tool_error"
    return error, None


def failed_action_index(error: dict) -> int | None:
    step_id = error.get("step_id")
    if not isinstance(step_id, str) or "_" not in step_id:
        return None
    try:
        return int(step_id.rsplit("_", 1)[1])
    except ValueError:
        return None


def load_tasks(path: Path) -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    for row in read_jsonl(path):
        key = row.get("example_index", row.get("index"))
        if key is None:
            raise ValueError(f"{path}: task row without example_index")
        tasks[str(key)] = row
    return tasks


def resolve_db_path(task: dict, db_root: Path) -> Path:
    raw = Path(str(task["db_path"]))
    return raw if raw.is_absolute() else (db_root / raw).resolve()


def task_difficulty(task: dict) -> str:
    metadata = task.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("private_difficulty") or "unknown")
    return "unknown"


def rebuild_episode(
    record: dict,
    sample: dict,
    task: dict,
    *,
    trajectory_id: str,
    history_turns: int,
    denotation_comparison: str,
    db_root: Path,
) -> tuple[dict | None, str | None, int]:
    """Re-execute one recorded sample and return a canonical verified episode."""
    turns = sample.get("turns") or []
    if not turns:
        return None, "empty_turns", 0
    db_path = resolve_db_path(task, db_root)
    if not db_path.is_file():
        return None, "missing_database", 0
    order_differences = 0
    recovery_turns = 0
    harness = Harness(str(db_path))
    try:
        catalog = overview(harness)
        ctx = new_ctx(catalog)
        created: set[str] = set()
        steps: list[dict] = []
        next_action_index = 1
        for turn in turns:
            parsed = turn.get("parsed")
            if parsed is None:
                # A rejected action is recorded as its own turn with neither a parsed call nor an
                # executed observation.  It contributes no legal-history pair, but it still
                # occupies an action index because the rendered error payload references it.
                next_action_index += 1
                continue
            if not isinstance(parsed, dict):
                return None, "unparsed_turn", order_differences
            tool = parsed.get("tool")
            arguments = parsed.get("arguments")
            think = parsed.get("think") or ""
            if not isinstance(tool, str) or not isinstance(arguments, dict):
                return None, "unparsed_turn", order_differences
            if not think.strip():
                return None, "empty_think", order_differences
            error_before: dict | None = None
            if turn.get("feedback_recovery"):
                error_before, reason = extract_last_tool_error(turn.get("model_input"))
                if error_before is None:
                    return None, reason, order_differences
                recorded_type = turn.get("recovered_from_error_type")
                actual_type = (error_before.get("error") or {}).get("type")
                if recorded_type and actual_type and recorded_type != actual_type:
                    return None, "recovered_error_type_mismatch", order_differences
                failed_index = failed_action_index(error_before)
                # Keep the true action numbering: the rejected action still occupies an index, and
                # the rendered error payload references it, so renumbering would make the prefix
                # reference its own current step.
                next_action_index = failed_index + 1 if failed_index is not None else next_action_index
                recovery_turns += 1
            step_id = f"step_{next_action_index}"
            state_before = deepcopy(ctx["environment"].snapshot())
            recorded = turn.get("tool_output")
            if tool == TERMINAL_TOOL:
                correct, _, _ = score(
                    harness,
                    task["gold_sql"],
                    arguments,
                    created,
                    denotation_comparison=denotation_comparison,
                )
                if not correct:
                    return None, "terminal_denotation_mismatch", order_differences
                output = recorded if isinstance(recorded, dict) else {}
            else:
                try:
                    output, created_table = execute_tool(harness, tool, arguments, ctx, step_id)
                except Exception as exc:  # harness replay must reproduce the recorded turn
                    return None, f"replay_execute_failed:{type(exc).__name__}", order_differences
                if created_table:
                    created.add(created_table)
                matched, order_differs = observation_matches(recorded, output)
                if not matched:
                    return None, "tool_output_mismatch", order_differences
                order_differences += int(order_differs)
                output = recorded
            state_after = deepcopy(ctx["environment"].snapshot())
            steps.append(
                make_step(
                    step_id, think, tool, arguments, output, state_before, state_after, error_before
                )
            )
            next_action_index += 1
    finally:
        harness.conn.close()

    if steps[-1]["tool_call"]["tool"] != TERMINAL_TOOL:
        return None, "no_terminal_answer", order_differences
    episode = {
        "trajectory_id": trajectory_id,
        "schema_version": SCHEMA_VERSION,
        "source": {
            "dataset": task.get("dataset"),
            "split": task.get("split"),
            "db_id": task.get("db_id"),
            "db_path": str(db_path),
            "question": task.get("question"),
            "gold_sql": task.get("gold_sql"),
            "external_knowledge": task.get("external_knowledge"),
            "example_id": task.get("example_id"),
            "example_index": task.get("example_index"),
        },
        "question": task.get("question"),
        "difficulty": task_difficulty(task),
        "label_status": "verified",
        "tool_scheme": ATOMIC_TOOL_SCHEME,
        "initial_state": {"dataset_overview": catalog},
        "steps": steps,
        "rollout_generation": {
            "method": "student_on_policy_screen_replay",
            "model_checkpoint": "checkpoint-6380",
            "sample_index": sample.get("sample_index"),
            "screen_temperature": record.get("temperature"),
            "screen_protocol_hash": record.get("protocol_hash"),
            "screen_protocol_version": record.get("protocol_version"),
            "context_mode": "rolling-legal-history",
            "history_turns": history_turns,
            "denotation_comparison": denotation_comparison,
            "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
            "error_events": [],
        },
    }
    return episode, None, order_differences


def select_candidates(
    records: list[dict],
    tasks: dict[str, dict],
    *,
    correct_min: int,
    correct_max: int,
    exclude_recovery: bool,
) -> tuple[list[tuple[dict, dict, dict]], collections.Counter]:
    """Return deterministic ``(record, sample, task)`` triples plus a rejection histogram."""
    rejected: collections.Counter = collections.Counter()
    candidates: list[tuple[dict, dict, dict]] = []
    for record in records:
        count = record.get("sample_correct_count")
        if count is None:
            rejected["missing_correct_count"] += 1
            continue
        count = int(count)
        if count < correct_min or count > correct_max:
            rejected["outside_band"] += 1
            continue
        task = tasks.get(str(record.get("example_index")))
        if task is None:
            rejected["missing_task"] += 1
            continue
        for sample in sorted(
            record.get("samples") or [], key=lambda item: int(item.get("sample_index") or 0)
        ):
            if not (sample.get("correct") and sample.get("legal")):
                rejected["not_correct_or_illegal"] += 1
                continue
            if exclude_recovery and any(
                turn.get("feedback_recovery") for turn in sample.get("turns") or []
            ):
                rejected["excluded_feedback_recovery"] += 1
                continue
            candidates.append((record, sample, task))
    return candidates, rejected


def apply_question_cap(
    candidates: list[tuple[dict, dict, dict]], max_per_question: int
) -> tuple[list[tuple[dict, dict, dict]], collections.Counter]:
    """Cap trajectories per question, keeping the lowest sample indices (deterministic)."""
    if max_per_question <= 0:
        return candidates, collections.Counter()
    by_question: dict[str, list[tuple[dict, dict, dict]]] = collections.defaultdict(list)
    for triple in candidates:
        by_question[str(triple[2].get("example_id") or triple[0].get("example_index"))].append(triple)
    kept: list[tuple[dict, dict, dict]] = []
    dropped: collections.Counter = collections.Counter()
    for key in sorted(by_question):
        ordered = sorted(by_question[key], key=lambda t: int(t[1].get("sample_index") or 0))
        kept.extend(ordered[:max_per_question])
        dropped["over_question_cap"] += max(0, len(ordered) - max_per_question)
    return kept, dropped


def build(
    screen_paths: list[Path],
    tasks_path: Path,
    out_path: Path,
    manifest_path: Path,
    *,
    db_root: Path,
    correct_min: int,
    correct_max: int,
    max_per_question: int,
    history_turns: int,
    denotation_comparison: str,
    exclude_recovery: bool,
    limit: int,
) -> dict:
    tasks = load_tasks(tasks_path)
    records: list[dict] = []
    for path in screen_paths:
        records.extend(read_jsonl(path))
    candidates, rejected = select_candidates(
        records,
        tasks,
        correct_min=correct_min,
        correct_max=correct_max,
        exclude_recovery=exclude_recovery,
    )
    candidates, dropped = apply_question_cap(candidates, max_per_question)
    if limit > 0:
        candidates = candidates[:limit]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    episodes: list[dict] = []
    order_difference_episodes = 0
    for record, sample, task in candidates:
        example_id = str(task.get("example_id") or record.get("example_index"))
        trajectory_id = f"rft_{example_id}_s{int(sample.get('sample_index') or 0)}"
        episode, reason, order_differences = rebuild_episode(
            record,
            sample,
            task,
            trajectory_id=trajectory_id,
            history_turns=history_turns,
            denotation_comparison=denotation_comparison,
            db_root=db_root,
        )
        if episode is None:
            rejected[f"rebuild:{reason}"] += 1
            continue
        order_difference_episodes += int(order_differences > 0)
        episodes.append(episode)

    with out_path.open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")

    recovery_steps = [
        step
        for episode in episodes
        for step in episode["steps"]
        if step.get("feedback_recovery")
    ]
    manifest = {
        "schema_version": "qwen3-atomic-v26-rft-episode-build-v1",
        "output": str(out_path),
        "output_sha256": file_sha256(out_path) if out_path.exists() else None,
        "episodes": len(episodes),
        "source_screen_records": len(records),
        "screening_sources": [
            {"path": str(path), "sha256": file_sha256(path), "records": len(read_jsonl(path))}
            for path in screen_paths
        ],
        "tasks": {"path": str(tasks_path), "sha256": file_sha256(tasks_path), "records": len(tasks)},
        "selection": {
            "correct_count_band": [correct_min, correct_max],
            "max_per_question": max_per_question,
            "exclude_feedback_recovery_episodes": exclude_recovery,
            "denotation_comparison": denotation_comparison,
            "limit": limit,
            "deterministic_order": "example_index asc, sample_index asc",
        },
        "gates": {
            "label_status": "verified",
            "tool_output_must_match_replay": True,
            "tool_output_comparison": (
                "recorded fields must all match; rows compare as multisets because SQLite row "
                "order without ORDER BY is not reproducible"
            ),
            "episodes_with_row_order_differences": order_difference_episodes,
            "last_tool_error_source": (
                "recovered verbatim from the recorded model_input; the section body is the real "
                "structured error, not a fixed placeholder"
            ),
            "episodes_with_recovery_turns": sum(
                1 for episode in episodes if any(s.get("feedback_recovery") for s in episode["steps"])
            ),
            "recovery_turns": len(recovery_steps),
            "recovered_error_type_histogram": dict(
                sorted(
                    collections.Counter(
                        str(step.get("recovered_from_error_type")) for step in recovery_steps
                    ).items()
                )
            ),
            "recovered_execution_errors_replayed": (
                "not replayed: the artifact records the error payload but not the attempted "
                "tool/arguments, so error_events stays empty and gaps are skipped."
            ),
            "terminal_denotation_scored": True,
        },
        "rejected": dict(sorted(rejected.items())),
        "dropped": dict(sorted(dropped.items())),
        "difficulty_histogram": dict(sorted(collections.Counter(e["difficulty"] for e in episodes).items())),
        "turns_per_episode": {
            "mean": (
                round(
                    sum(len(e["steps"]) for e in episodes) / len(episodes), 3
                )
                if episodes
                else 0
            ),
            "max": max((len(e["steps"]) for e in episodes), default=0),
        },
        "protocol_hashes": sorted(
            {
                str(e["rollout_generation"].get("screen_protocol_hash"))
                for e in episodes
                if e["rollout_generation"].get("screen_protocol_hash")
            }
        ),
        "tool_scheme": ATOMIC_TOOL_SCHEME,
        "tool_scheme_registry_version": TOOL_SCHEME_REGISTRY_VERSION,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", type=Path, action="append", required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--db-root", type=Path, default=ROOT)
    parser.add_argument("--correct-min", type=int, default=1)
    parser.add_argument("--correct-max", type=int, default=3)
    parser.add_argument("--max-per-question", type=int, default=1)
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--denotation-comparison", default="bird-set")
    parser.add_argument(
        "--exclude-feedback-recovery",
        action="store_true",
        help="control arm: drop episodes with a recovered action instead of rebuilding the error",
    )
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
    build(
        args.screen,
        args.tasks,
        args.out,
        manifest_path,
        db_root=args.db_root.resolve(),
        correct_min=args.correct_min,
        correct_max=args.correct_max,
        max_per_question=args.max_per_question,
        history_turns=args.history_turns,
        denotation_comparison=args.denotation_comparison,
        exclude_recovery=args.exclude_feedback_recovery,
        limit=args.limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
