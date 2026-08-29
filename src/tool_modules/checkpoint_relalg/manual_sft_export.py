#!/usr/bin/env python3
"""Audit and export Codex-manual frozen-Atomic episodes as SFT candidates.

Manual episodes were authored causally against the live Harness, but their compact
artifact stores only the current user state on each turn.  This exporter independently
replays every admitted episode and reconstructs the frozen recent-four provider history
from already-observed prefixes before emitting scheme-local training candidates.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from eval.denotation import compare_denotations
from tool_modules.checkpoint_relalg.protocol import (
    ATOMIC_OPERATOR_PROFILE_FROZEN_V24,
    CARRIER_TEXT_JSON,
    CHECKPOINT_GUIDANCE_PROFILE_DISABLED,
    get_system_prompt,
    prompt_hash,
    tool_schema_hash,
    trim_provider_phase_history,
)
from tool_modules.checkpoint_relalg.runner import (
    _hidden_reference_artifact,
    _open_read_only,
    _task_db_path,
)
from tool_modules.checkpoint_relalg.runtime import CheckpointRelalgRuntime, RuntimeConfig
from tool_modules.checkpoint_relalg.sft_export import (
    ADMISSION_POLICY_VERSION,
    LOSS_POLICY,
    TRAINING_RECORD_VERSION,
    is_successful_target_turn,
)
from tool_modules.checkpoint_relalg.text_json_carrier import (
    text_json_result_message,
    validate_text_json_assistant_message,
)
from tool_modules.registry import build_checkpoint_relalg_tool_scheme


MANUAL_SCHEMA_VERSION = "checkpoint-relalg-atomic-v24-frozen-codex-manual-v1"
MANUAL_EXPORT_POLICY = "checkpoint-relalg-atomic-v24-frozen-codex-manual-sft-candidates-v1"
PROFILE = ATOMIC_OPERATOR_PROFILE_FROZEN_V24
MODE = "atomic"
CARRIER = CARRIER_TEXT_JSON
GUIDANCE = CHECKPOINT_GUIDANCE_PROFILE_DISABLED


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl_atomic(path: Path, rows: list[Mapping[str, Any]]) -> None:
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


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(value)
    return rows


def _runtime(task: Mapping[str, Any]) -> tuple[Any, CheckpointRelalgRuntime]:
    connection = _open_read_only(_task_db_path(task))
    runtime = CheckpointRelalgRuntime(
        connection,
        mode=MODE,
        atomic_operator_profile=PROFILE,
        config=RuntimeConfig(
            max_primitive_calls=30,
            max_checkpoints=0,
            max_restores=0,
            sql_timeout_seconds=20.0,
            max_artifact_rows=100_000,
            max_artifact_bytes=64 * 1024 * 1024,
            max_cell_bytes=4 * 1024 * 1024,
        ),
    )
    return connection, runtime


def _load_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = _load_jsonl(path)
    ids = [task.get("example_id") for task in tasks]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("task example_id values must be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("task example_id values must be unique")
    return tasks


def _snapshot_records(snapshot_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = snapshot_dir / "manifest.json"
    records_path = snapshot_dir / "canonical_all.jsonl"
    index_path = snapshot_dir / "canonical.index.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bound = manifest.get("files", {})
    for name, path in {
        "canonical_all.jsonl": records_path,
        "canonical.index.jsonl": index_path,
    }.items():
        expected = (bound.get(name) or {}).get("sha256")
        if expected != _file_sha256(path):
            raise ValueError(f"manual snapshot binding for {name} is stale")
    if manifest.get("atomic_operator_profile") != PROFILE:
        raise ValueError("manual snapshot profile mismatch")
    if manifest.get("carrier") != CARRIER or manifest.get("checkpoint_guidance_profile") != GUIDANCE:
        raise ValueError("manual snapshot carrier/guidance mismatch")
    if manifest.get("denotation_comparison") != "bird-set":
        raise ValueError("manual snapshot denotation mismatch")
    return manifest, _load_jsonl(records_path)


def _episode_identity_reasons(record: Mapping[str, Any], task: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    expected = {
        "schema_version": MANUAL_SCHEMA_VERSION,
        "training_admission": "diagnostic_only_pending_manual_teacher_admission",
        "sft_export_eligible": False,
        "teacher_identity": "codex_manual_agent",
        "example_id": task.get("example_id"),
        "db_id": task.get("db_id"),
        "mode": MODE,
        "atomic_operator_profile": PROFILE,
        "carrier": CARRIER,
        "checkpoint_guidance_profile": GUIDANCE,
        "teacher_prompt_sha256": prompt_hash(
            MODE,
            teacher=True,
            carrier=CARRIER,
            checkpoint_guidance_profile=GUIDANCE,
            atomic_operator_profile=PROFILE,
        ),
        "tool_schema_sha256": tool_schema_hash(MODE, PROFILE),
        "question_sha256": hashlib.sha256(str(task.get("question") or "").encode()).hexdigest(),
        "denotation_comparison": "bird-set",
        "legal": True,
        "correct": True,
        "fresh_replay_passed": True,
    }
    for field, wanted in expected.items():
        if record.get(field) != wanted:
            reasons.append(field)
    turns = record.get("turns")
    if not isinstance(turns, list) or not turns:
        reasons.append("turns")
    return reasons


def _replay_and_convert(
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    snapshot_dir: Path,
    resolved_task_position: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    teacher_prompt = get_system_prompt(
        MODE,
        teacher=True,
        carrier=CARRIER,
        checkpoint_guidance_profile=GUIDANCE,
        atomic_operator_profile=PROFILE,
    )
    student_prompt = get_system_prompt(
        MODE,
        teacher=False,
        carrier=CARRIER,
        checkpoint_guidance_profile=GUIDANCE,
        atomic_operator_profile=PROFILE,
    )
    scheme = build_checkpoint_relalg_tool_scheme(
        mode=MODE,
        carrier=CARRIER,
        atomic_operator_profile=PROFILE,
    )
    source_hash = _sha256_value(record)
    connection, runtime = _runtime(task)
    history: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    indexes: list[dict[str, Any]] = []
    zero_row_context_only = 0
    unsuccessful_context_only = 0
    try:
        turns = record["turns"]
        for turn_index, turn in enumerate(turns):
            context = runtime.render_context(
                str(task.get("question") or ""), task.get("external_knowledge")
            )
            current_user = {"role": "user", "content": context}
            stored_input = turn.get("model_input")
            expected_manual_input = [
                {"role": "system", "content": teacher_prompt},
                copy.deepcopy(current_user),
            ]
            if _canonical(stored_input) != _canonical(expected_manual_input):
                raise ValueError(f"{record['example_id']}: turn {turn_index + 1} manual input mismatch")
            if turn.get("environment_state_hash_before") != runtime.state.logical_hash():
                raise ValueError(f"{record['example_id']}: turn {turn_index + 1} before hash mismatch")
            assistant = copy.deepcopy(turn.get("assistant_message"))
            authored = validate_text_json_assistant_message(
                MODE, assistant, atomic_operator_profile=PROFILE
            )
            action = turn.get("action")
            if not isinstance(action, Mapping):
                raise ValueError(f"{record['example_id']}: turn {turn_index + 1} has no action")
            if authored["tool"] != action.get("tool") or _canonical(authored["arguments"]) != _canonical(action.get("arguments")):
                raise ValueError(f"{record['example_id']}: turn {turn_index + 1} authored/action mismatch")
            result = runtime.apply(authored["tool"], copy.deepcopy(authored["arguments"]))
            if _canonical(result) != _canonical(turn.get("result")):
                raise ValueError(f"{record['example_id']}: turn {turn_index + 1} result mismatch")
            if turn.get("environment_state_hash_after") != runtime.state.logical_hash():
                raise ValueError(f"{record['example_id']}: turn {turn_index + 1} after hash mismatch")

            target, reason = is_successful_target_turn(turn)
            if target:
                model_input = [
                    {"role": "system", "content": student_prompt},
                    *copy.deepcopy(history),
                    copy.deepcopy(current_user),
                ]
                messages = [*model_input, assistant]
                record_id = f"codex_manual_{record['example_id']}_turn_{turn_index + 1:02d}"
                previous_result = turns[turn_index - 1].get("result") if turn_index else None
                feedback_recovery = bool(
                    isinstance(previous_result, Mapping)
                    and previous_result.get("status") != "success"
                )
                strict = record.get("strict_artifact_audit") or {}
                metadata = {
                    "record_id": record_id,
                    "source_episode_id": record.get("example_id"),
                    "source_task_position": resolved_task_position,
                    "source_task_position_claim": record.get("task_position"),
                    "source_task_position_resolution": (
                        "stored_position"
                        if record.get("task_position") == resolved_task_position
                        else "unique_example_id_in_hash_bound_tasks"
                    ),
                    "source_turn_index": turn_index,
                    "source_result_dir": str(snapshot_dir),
                    "source_artifact_sha256": source_hash,
                    "teacher_identity": "codex_manual_agent",
                    "tool_scheme": "checkpoint-relalg",
                    "mode": MODE,
                    "atomic_operator_profile": PROFILE,
                    "carrier": CARRIER,
                    "checkpoint_guidance_profile": GUIDANCE,
                    "admission_policy_version": ADMISSION_POLICY_VERSION,
                    "manual_export_policy_version": MANUAL_EXPORT_POLICY,
                    "loss_policy": LOSS_POLICY,
                    "tool_name": authored["tool"],
                    "feedback_recovery": feedback_recovery,
                    "strict_artifact_accuracy": strict.get("strict_artifact_accuracy"),
                    "schema_match": strict.get("schema_match"),
                    "reasoning_sha256": hashlib.sha256(assistant["reasoning_content"].encode()).hexdigest(),
                    "reasoning_characters": len(assistant["reasoning_content"]),
                    "recent4_history_reconstructed_from_causal_prefix": True,
                }
                candidates.append(
                    {
                        "schema_version": TRAINING_RECORD_VERSION,
                        "messages": messages,
                        "loss_message_index": len(messages) - 1,
                        "metadata": metadata,
                    }
                )
                indexes.append(
                    {
                        **metadata,
                        "model_input_sha256": _sha256_value(model_input),
                        "target_envelope_sha256": _sha256_value(assistant),
                        "target_visible_content_sha256": hashlib.sha256(assistant["content"].encode()).hexdigest(),
                    }
                )
            elif reason == "zero_row_intermediate_artifact":
                zero_row_context_only += 1
            else:
                unsuccessful_context_only += 1

            if not runtime.done:
                history.extend(
                    [
                        copy.deepcopy(current_user),
                        copy.deepcopy(assistant),
                        text_json_result_message(result),
                    ]
                )
                history = trim_provider_phase_history(history, PROFILE)

        if not runtime.done or runtime.terminal_table is None:
            raise ValueError(f"{record['example_id']}: replay has no terminal artifact")
        answer_columns, answer_rows = runtime.answer_rows()
        reference_columns, reference_rows, reference_ordered = _hidden_reference_artifact(
            task, runtime_config=runtime.config
        )
        if not compare_denotations(answer_rows, reference_rows, comparison="bird-set"):
            raise ValueError(f"{record['example_id']}: hidden terminal denotation mismatch")
        if list(answer_columns) != list(record.get("answer_columns") or []):
            raise ValueError(f"{record['example_id']}: answer columns mismatch")
        if len(answer_rows) != record.get("answer_row_count"):
            raise ValueError(f"{record['example_id']}: answer row count mismatch")
        return candidates, indexes, {
            "example_id": record["example_id"],
            "source_task_position": resolved_task_position,
            "source_task_position_claim": record.get("task_position"),
            "source_task_position_resolution": (
                "stored_position"
                if record.get("task_position") == resolved_task_position
                else "unique_example_id_in_hash_bound_tasks"
            ),
            "source_artifact_sha256": source_hash,
            "targets": len(candidates),
            "zero_row_context_only": zero_row_context_only,
            "unsuccessful_context_only": unsuccessful_context_only,
            "strict_artifact_accuracy": (record.get("strict_artifact_audit") or {}).get("strict_artifact_accuracy"),
            "schema_match": (record.get("strict_artifact_audit") or {}).get("schema_match"),
            "protocol_hash": scheme.protocol_hash,
        }
    finally:
        connection.close()


def export_manual_snapshot(
    snapshot_dir: Path,
    tasks_path: Path,
    *,
    out_path: Path,
    index_path: Path,
) -> dict[str, Any]:
    snapshot_dir = snapshot_dir.resolve()
    tasks_path = tasks_path.resolve()
    snapshot_manifest, records = _snapshot_records(snapshot_dir)
    if snapshot_manifest.get("tasks_sha256") != _file_sha256(tasks_path):
        raise ValueError("manual snapshot task binding is stale")
    tasks = _load_tasks(tasks_path)
    task_positions_by_id = {
        str(task["example_id"]): position for position, task in enumerate(tasks)
    }
    candidates: list[dict[str, Any]] = []
    indexes: list[dict[str, Any]] = []
    admitted: list[dict[str, Any]] = []
    rejected: collections.Counter[str] = collections.Counter()
    tools: collections.Counter[str] = collections.Counter()
    for record in records:
        example_id = record.get("example_id")
        if not isinstance(example_id, str) or example_id not in task_positions_by_id:
            rejected.update(["example_id_not_in_hash_bound_tasks"])
            continue
        position = task_positions_by_id[example_id]
        task = tasks[position]
        reasons = _episode_identity_reasons(record, task)
        if reasons:
            rejected.update(reasons)
            continue
        episode_candidates, episode_indexes, episode = _replay_and_convert(
            record,
            task,
            snapshot_dir=snapshot_dir,
            resolved_task_position=position,
        )
        if not episode_candidates:
            rejected.update(["no_successful_targets"])
            continue
        candidates.extend(episode_candidates)
        indexes.extend(episode_indexes)
        admitted.append(episode)
        tools.update(row["tool_name"] for row in episode_indexes)
    if not candidates:
        raise ValueError("manual snapshot produced no eligible SFT candidates")
    record_ids = [row["metadata"]["record_id"] for row in candidates]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("manual SFT candidate record ids are not unique")
    _write_jsonl_atomic(out_path, candidates)
    _write_jsonl_atomic(index_path, indexes)
    scheme = build_checkpoint_relalg_tool_scheme(
        mode=MODE, carrier=CARRIER, atomic_operator_profile=PROFILE
    )
    manifest = {
        "schema_version": "checkpoint-relalg-manual-sft-export-manifest-v1",
        "manual_export_policy_version": MANUAL_EXPORT_POLICY,
        "admission_policy_version": ADMISSION_POLICY_VERSION,
        "training_record_version": TRAINING_RECORD_VERSION,
        "source_snapshot": str(snapshot_dir),
        "source_snapshot_manifest_sha256": _file_sha256(snapshot_dir / "manifest.json"),
        "source_canonical_all_sha256": _file_sha256(snapshot_dir / "canonical_all.jsonl"),
        "tasks_path": str(tasks_path),
        "tasks_sha256": _file_sha256(tasks_path),
        "source_episodes": len(records),
        "admitted_episodes": len(admitted),
        "rejected_episodes": len(records) - len(admitted),
        "episode_rejections": dict(sorted(rejected.items())),
        "records": len(candidates),
        "output": str(out_path.resolve()),
        "output_sha256": _file_sha256(out_path),
        "index": str(index_path.resolve()),
        "index_sha256": _file_sha256(index_path),
        "loss_policy": LOSS_POLICY,
        "teacher_identity": "codex_manual_agent",
        "native_reasoning_content_preserved": True,
        "recent4_history_reconstructed_from_causal_prefix": True,
        "task_position_resolution": "unique_example_id_in_hash_bound_tasks",
        "task_position_claim_mismatches_admitted": sum(
            row["source_task_position"] != row["source_task_position_claim"]
            for row in admitted
        ),
        "strict_artifact_is_admission_gate": False,
        "schema_match_is_admission_gate": False,
        "admitted_strict_artifact_episodes": sum(row["strict_artifact_accuracy"] is True for row in admitted),
        "admitted_schema_match_episodes": sum(row["schema_match"] is True for row in admitted),
        "error_actions_are_targets": False,
        "zero_row_intermediate_artifacts_are_targets": False,
        "tool_hist": dict(tools.most_common()),
        "tool_scheme": "checkpoint-relalg",
        "mode": MODE,
        "atomic_operator_profile": PROFILE,
        "carrier": CARRIER,
        "checkpoint_guidance_profile": GUIDANCE,
        "protocol_hash": scheme.protocol_hash,
        "student_prompt_sha256": scheme.student_prompt_hash,
        "tool_schema_sha256": scheme.tool_schema_hash,
        "training_admission": "scheme_local_sft_candidate_not_yet_merged",
    }
    _write_json_atomic(out_path.with_suffix(".manifest.json"), manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--tasks-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path)
    args = parser.parse_args()
    out_path = args.out.resolve()
    index_path = (args.index_out or out_path.with_name(out_path.stem + "_index.jsonl")).resolve()
    if out_path == index_path:
        parser.error("--out and --index-out must differ")
    manifest = export_manual_snapshot(
        args.snapshot_dir,
        args.tasks_json,
        out_path=out_path,
        index_path=index_path,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
