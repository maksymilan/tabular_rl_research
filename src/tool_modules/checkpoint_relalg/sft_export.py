#!/usr/bin/env python3
"""Export audited frozen-Atomic episodes as native split-message SFT candidates.

This exporter is intentionally scheme-local.  It accepts only the frozen
``atomic-v24-frozen-v1`` Text-JSON profile, selects episodes whose Harness terminal
denotation is correct under ``bird-set``, and independently requires record structure
and fresh replay to pass.  Strict artifact/schema scores are retained as diagnostics;
they are not admission filters.

The output preserves DeepSeek's causal two-channel assistant envelope
(``reasoning_content`` plus exact JSON ``content``).  It does not silently translate
that envelope into an inline ``<think>`` carrier.  A later model-specific projection
may do so explicitly and auditably.
"""
from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from tool_modules.checkpoint_relalg.audit import audit_result_dir, load_jsonl
from tool_modules.checkpoint_relalg.protocol import (
    ATOMIC_OPERATOR_PROFILE_FROZEN_V24,
    CARRIER_TEXT_JSON,
    CHECKPOINT_GUIDANCE_PROFILE_DISABLED,
    get_system_prompt,
)
from tool_modules.checkpoint_relalg.text_json_carrier import (
    validate_text_json_assistant_message,
)
from tool_modules.registry import (
    CHECKPOINT_RELALG_TOOL_SCHEME,
    build_checkpoint_relalg_tool_scheme,
)


ADMISSION_POLICY_VERSION = (
    "checkpoint-relalg-atomic-v24-frozen-bird-set-sft-candidates-v1"
)
TRAINING_RECORD_VERSION = "checkpoint-relalg-native-reasoning-sft-candidate-v1"
LOSS_POLICY = "last-assistant-message-only"
PROFILE = ATOMIC_OPERATOR_PROFILE_FROZEN_V24
MODE = "atomic"
CARRIER = CARRIER_TEXT_JSON
GUIDANCE = CHECKPOINT_GUIDANCE_PROFILE_DISABLED
ZERO_ARTIFACT_TOOLS = {
    "filter_rows",
    "join",
    "shape_rows",
    "group_aggregate",
    "set_operation",
    "scalar_compute",
    "rank_select",
}


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


def _load_bound_audit(result_dir: Path, *, refresh: bool) -> dict[str, Any]:
    """Load an existing fresh-replay attestation only when its artifacts still match."""

    if refresh:
        return audit_result_dir(result_dir)
    audit_path = result_dir / "checkpoint_relalg_audit.json"
    if not audit_path.is_file():
        raise ValueError(f"{result_dir}: current audit report is unavailable")
    report = json.loads(audit_path.read_text(encoding="utf-8"))
    artifacts = report.get("artifact_sha256")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"{result_dir}: audit report has no artifact binding")
    expected_paths = {
        "manifest.json": result_dir / "manifest.json",
        "all.jsonl": result_dir / "all.jsonl",
    }
    if (result_dir / "batch_status.json").is_file():
        expected_paths["batch_status.json"] = result_dir / "batch_status.json"
    for name, path in expected_paths.items():
        if artifacts.get(name) != _file_sha256(path):
            raise ValueError(f"{result_dir}: audit binding for {name} is stale")
    return report


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
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


def _scheme():
    return build_checkpoint_relalg_tool_scheme(
        mode=MODE,
        carrier=CARRIER,
        atomic_operator_profile=PROFILE,
    )


def episode_rejection_reasons(record: Mapping[str, Any]) -> list[str]:
    """Return deterministic admission failures without consulting hidden gold SQL."""

    scheme = _scheme()
    reasons: list[str] = []
    expected = {
        "tool_scheme": CHECKPOINT_RELALG_TOOL_SCHEME,
        "mode": MODE,
        "final_mode": MODE,
        "atomic_operator_profile": PROFILE,
        "carrier": CARRIER,
        "checkpoint_guidance_profile": GUIDANCE,
        "protocol_hash": scheme.protocol_hash,
        "student_prompt_sha256": scheme.student_prompt_hash,
        "tool_schema_sha256": scheme.tool_schema_hash,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            reasons.append(f"identity:{field}")
    runtime = record.get("runtime_config")
    if not isinstance(runtime, Mapping):
        reasons.append("runtime_config")
    else:
        if runtime.get("max_checkpoints") != 0:
            reasons.append("runtime:max_checkpoints")
        if runtime.get("max_restores") != 0:
            reasons.append("runtime:max_restores")
    if record.get("checkpoint_count") != 0:
        reasons.append("checkpoint_count")
    if record.get("restore_count") != 0:
        reasons.append("restore_count")
    if record.get("denotation_comparison") != "bird-set":
        reasons.append("denotation_comparison")
    if record.get("correct") is not True:
        reasons.append("not_correct")
    if record.get("legal") is not True:
        reasons.append("not_legal")
    turns = record.get("turns")
    if not isinstance(turns, list) or not turns:
        reasons.append("no_turns")
    return reasons


def is_successful_target_turn(turn: Mapping[str, Any]) -> tuple[bool, str | None]:
    """Admit only executed successful actions; zero-row derived tables stay context-only."""

    assistant = turn.get("assistant_message")
    action = turn.get("action")
    result = turn.get("result")
    if not isinstance(assistant, Mapping) or not isinstance(action, Mapping):
        return False, "no_authored_action"
    if not isinstance(result, Mapping) or result.get("status") != "success":
        return False, "unsuccessful_action"
    tool = action.get("tool")
    artifact = result.get("artifact")
    if (
        tool in ZERO_ARTIFACT_TOOLS
        and isinstance(artifact, Mapping)
        and artifact.get("row_count") == 0
    ):
        return False, "zero_row_intermediate_artifact"
    return True, None


def convert_turn(
    record: Mapping[str, Any],
    turn_index: int,
    *,
    source_result_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build one causal native split-message target from an already audited turn."""

    turns = record["turns"]
    turn = turns[turn_index]
    model_input = copy.deepcopy(turn.get("model_input"))
    if not isinstance(model_input, list) or len(model_input) < 2:
        raise ValueError("target turn has no causal model input")
    if model_input[0].get("role") != "system":
        raise ValueError("target turn model input has no leading system message")
    student_prompt = get_system_prompt(
        MODE,
        teacher=False,
        carrier=CARRIER,
        checkpoint_guidance_profile=GUIDANCE,
        atomic_operator_profile=PROFILE,
    )
    model_input[0] = {"role": "system", "content": student_prompt}
    assistant = copy.deepcopy(turn["assistant_message"])
    authored = validate_text_json_assistant_message(
        MODE,
        assistant,
        atomic_operator_profile=PROFILE,
    )
    action = turn["action"]
    if authored["tool"] != action.get("tool") or _canonical(
        authored["arguments"]
    ) != _canonical(action.get("arguments")):
        raise ValueError("provider-authored action differs from recorded Harness action")
    messages = [*model_input, assistant]
    record_id = (
        f"{record.get('example_id') or record.get('task_position')}_"
        f"turn_{turn_index + 1:02d}"
    )
    previous_result = turns[turn_index - 1].get("result") if turn_index else None
    feedback_recovery = bool(
        isinstance(previous_result, Mapping)
        and previous_result.get("status") != "success"
    )
    reasoning = assistant["reasoning_content"]
    metadata = {
        "record_id": record_id,
        "source_episode_id": record.get("example_id"),
        "source_task_position": record.get("task_position"),
        "source_turn_index": turn_index,
        "source_result_dir": str(source_result_dir),
        "tool_scheme": CHECKPOINT_RELALG_TOOL_SCHEME,
        "mode": MODE,
        "atomic_operator_profile": PROFILE,
        "carrier": CARRIER,
        "checkpoint_guidance_profile": GUIDANCE,
        "admission_policy_version": ADMISSION_POLICY_VERSION,
        "loss_policy": LOSS_POLICY,
        "tool_name": authored["tool"],
        "feedback_recovery": feedback_recovery,
        "strict_artifact_accuracy": record.get("strict_artifact_accuracy"),
        "schema_match": record.get("schema_match"),
        "reasoning_sha256": hashlib.sha256(reasoning.encode("utf-8")).hexdigest(),
        "reasoning_characters": len(reasoning),
    }
    candidate = {
        "schema_version": TRAINING_RECORD_VERSION,
        "messages": messages,
        "loss_message_index": len(messages) - 1,
        "metadata": metadata,
    }
    index = {
        **metadata,
        "model_input_sha256": _sha256_value(model_input),
        "target_envelope_sha256": _sha256_value(assistant),
        "target_visible_content_sha256": hashlib.sha256(
            assistant["content"].encode("utf-8")
        ).hexdigest(),
    }
    return candidate, index


def export_result_dirs(
    result_dirs: Iterable[Path],
    *,
    out_path: Path,
    index_path: Path,
    refresh_audit: bool = False,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    source_artifacts: list[dict[str, Any]] = []
    episode_rejections: collections.Counter[str] = collections.Counter()
    turn_rejections: collections.Counter[str] = collections.Counter()
    tool_hist: collections.Counter[str] = collections.Counter()
    admitted_episodes = 0
    correct_strict = 0
    correct_schema = 0

    for raw_dir in result_dirs:
        result_dir = raw_dir.resolve()
        report = _load_bound_audit(result_dir, refresh=refresh_audit)
        for gate in (
            "manifest_binding",
            "cohort_identity",
            "provider_budget",
            "batch_control",
        ):
            if not report.get(gate, {}).get("passed"):
                raise ValueError(f"{result_dir}: required audit gate {gate} failed")
        records = load_jsonl(result_dir / "all.jsonl")
        structural = report.get("records_detail") or []
        replay = report.get("fresh_replay", {}).get("records_detail") or []
        if len(records) != len(structural) or len(records) != len(replay):
            raise ValueError(f"{result_dir}: audit detail length differs from records")
        source_artifacts.append(
            {
                "result_dir": str(result_dir),
                "manifest_sha256": _file_sha256(result_dir / "manifest.json"),
                "all_jsonl_sha256": _file_sha256(result_dir / "all.jsonl"),
                "audit_sha256": _file_sha256(result_dir / "checkpoint_relalg_audit.json"),
                "records": len(records),
                "structural_passed_records": report.get("passed_records"),
                "fresh_replay_passed_records": report.get("fresh_replay", {}).get(
                    "passed_records"
                ),
            }
        )
        for record_index, record in enumerate(records):
            reasons = episode_rejection_reasons(record)
            if not structural[record_index].get("passed"):
                reasons.append("structure_audit")
            if not replay[record_index].get("passed"):
                reasons.append("fresh_replay")
            if reasons:
                episode_rejections.update(reasons)
                continue
            admitted_episodes += 1
            correct_strict += record.get("strict_artifact_accuracy") is True
            correct_schema += record.get("schema_match") is True
            for turn_index, turn in enumerate(record["turns"]):
                target, reason = is_successful_target_turn(turn)
                if not target:
                    turn_rejections.update([reason or "unknown"])
                    continue
                candidate, index = convert_turn(
                    record,
                    turn_index,
                    source_result_dir=result_dir,
                )
                candidates.append(candidate)
                index_rows.append(index)
                tool_hist.update([index["tool_name"]])

    if not candidates:
        raise ValueError("no eligible SFT targets were produced")
    record_ids = [row["metadata"]["record_id"] for row in candidates]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("duplicate training record ids")
    _write_jsonl_atomic(out_path, candidates)
    _write_jsonl_atomic(index_path, index_rows)
    scheme = _scheme()
    manifest = {
        "schema_version": "checkpoint-relalg-sft-export-manifest-v1",
        "admission_policy_version": ADMISSION_POLICY_VERSION,
        "training_record_version": TRAINING_RECORD_VERSION,
        "audit_mode": "fresh-replay-now" if refresh_audit else "bound-existing-fresh-replay",
        "source_artifacts": source_artifacts,
        "output": str(out_path.resolve()),
        "output_sha256": _file_sha256(out_path),
        "index": str(index_path.resolve()),
        "index_sha256": _file_sha256(index_path),
        "admitted_episodes": admitted_episodes,
        "records": len(candidates),
        "loss_policy": LOSS_POLICY,
        "native_reasoning_content_preserved": True,
        "inline_think_projection_applied": False,
        "strict_artifact_is_admission_gate": False,
        "schema_match_is_admission_gate": False,
        "admitted_strict_artifact_episodes": correct_strict,
        "admitted_schema_match_episodes": correct_schema,
        "error_actions_are_targets": False,
        "zero_row_intermediate_artifacts_are_targets": False,
        "episode_rejections": dict(sorted(episode_rejections.items())),
        "turn_rejections": dict(sorted(turn_rejections.items())),
        "tool_hist": dict(tool_hist.most_common()),
        "tool_scheme": CHECKPOINT_RELALG_TOOL_SCHEME,
        "mode": MODE,
        "atomic_operator_profile": PROFILE,
        "carrier": CARRIER,
        "checkpoint_guidance_profile": GUIDANCE,
        "protocol_hash": scheme.protocol_hash,
        "student_prompt_sha256": scheme.student_prompt_hash,
        "tool_schema_sha256": scheme.tool_schema_hash,
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    _write_json_atomic(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--index-out", type=Path)
    parser.add_argument(
        "--refresh-audit",
        action="store_true",
        help="rerun structure and fresh replay instead of reusing a hash-bound audit report",
    )
    args = parser.parse_args()
    out_path = args.out.resolve()
    index_path = (
        args.index_out
        or out_path.with_name(out_path.stem + "_index.jsonl")
    ).resolve()
    if out_path == index_path:
        parser.error("--out and --index-out must differ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", out_path.name):
        parser.error("output file name contains unsupported characters")
    manifest = export_result_dirs(
        args.result_dir,
        out_path=out_path,
        index_path=index_path,
        refresh_audit=args.refresh_audit,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
