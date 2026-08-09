#!/usr/bin/env python3
"""Strict, read-only paired analysis for the Qwen3 atomic-v26 BIRD-dev run.

The evaluator writes one task-level record whose sole greedy trajectory lives in
``samples[0]``.  This script intentionally reads those records directly instead of trusting the
incremental ``summary.json``.  It refuses incomplete/mixed runs and writes nothing: JSON or a
compact Markdown report is emitted to stdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EXPECTED_TOTAL = 1534
WILSON_Z_95 = 1.959963984540054
EXPECTED_RUNTIME_COMMIT = "4cd47c957fc6ae791e76a10594c8cd22f4d3b6de"
EXPECTED_STUDENT_PROMPT_SHA256 = (
    "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
)
MODEL_SIZE = "8b"
EXPECTED_QWEN3_REVISIONS = {
    "4b": "1cfa9a7208912126459214e8b04321603b3df60c",
    "8b": "b968826d9c46dd6066d109eabc6255188de91218",
}
# Backward-compatible export for the frozen 8B analyzer tests and external callers.
EXPECTED_QWEN3_REVISION = EXPECTED_QWEN3_REVISIONS["8b"]
EXPECTED_TOKENIZER_CONFIG_SHA256 = (
    "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
)
EXPECTED_CHAT_TEMPLATE_SHA256 = (
    "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"
)
EXPECTED_SOURCE_DATASET_BASENAME = "bird_dev_20240627.jsonl"
EXPECTED_REMOTE_DATASET_BASENAME = "bird_dev_20240627.newgnn.jsonl"
EXPECTED_SOURCE_DATASET_SHA256 = (
    "8bf5a8bfe93ab49788656e2cc789bf80e729e0ec5f7f40159be01a1ab7b923e0"
)
EXPECTED_REMOTE_DATASET_SHA256 = (
    "636e096babe2db9dde096b655ed01a0e1e6aae1ea5cbf4e952770e02841721f5"
)
EXPECTED_REMOTE_ASSET_GATE_SCHEMA = "qwen3-atomic-v26-all-newgnn-assets-v1"

# These are reference levels, not claims of protocol equivalence.  The first is the completed
# local reproduction under the authors' SQL-tool protocol.  The second is the smallest integer
# success count on 1,534 examples that reaches the paper's reported 47.9% point estimate.
REFERENCE_THRESHOLDS = (
    (
        "author_protocol_raw_qwen3_8b_local_reproduction",
        "Local reproduction of the authors' raw Qwen3-8B protocol",
        709,
    ),
    (
        "paper_raw_qwen3_8b_47_9_percent",
        "Smallest integer count on 1,534 tasks meeting the paper's 47.9% point estimate",
        735,
    ),
)
REFERENCE_THRESHOLDS_4B = (
    (
        "paper_raw_qwen3_4b_29_3_percent",
        "Smallest integer count on 1,534 tasks meeting the paper's raw 29.3% point estimate",
        450,
    ),
    (
        "paper_sft_only_qwen3_4b_46_2_percent",
        "Smallest integer count on 1,534 tasks meeting the paper's SFT-only 46.2% point estimate",
        709,
    ),
    (
        "paper_sft_rl_qwen3_4b_64_9_percent",
        "Smallest integer count on 1,534 tasks meeting the paper's SFT+RL 64.9% point estimate",
        996,
    ),
)

MANIFEST_EQUAL_FIELDS = (
    "runner",
    "tool_scheme",
    "tool_scheme_registry_version",
    "assistant_carrier",
    "protocol_version",
    "protocol_hash",
    "requested_size",
    "n_samples",
    "stop_on_success",
    "pass_k",
    "sample_detail",
    "temperature",
    "top_p",
    "max_tokens",
    "max_steps",
    "few_shot",
    "enable_thinking",
    "system_prompt_variant",
    "context_mode",
    "history_turns",
    "rolling_prompt_variant",
    "rolling_observation_style",
    "denotation_comparison",
    "dataset_purpose",
)


@dataclass(frozen=True)
class TaskResult:
    example_index: int
    db_id: str
    question: str
    gold_sql_sha256: str
    correct: bool
    legal: bool
    steps: int
    errors: int
    failure_type: str | None
    error_types: tuple[str, ...]


@dataclass(frozen=True)
class RunResult:
    label: str
    all_jsonl: Path
    all_jsonl_sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    runtime_gate_path: Path
    runtime_gate_sha256: str
    runtime_gate: dict[str, Any]
    model_gate_path: Path
    model_gate_sha256: str
    model_gate: dict[str, Any]
    remote_asset_gate_path: Path | None
    remote_asset_gate_sha256: str | None
    remote_asset_gate: dict[str, Any] | None
    tasks: dict[int, TaskResult]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_number_equal(value: Any, expected: float) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and float(value) == expected


def _nonnegative_int(value: Any, field: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{field} must be a non-negative integer, got {value!r}",
    )
    return int(value)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _resolve_run_path(path: Path) -> tuple[Path, Path, Path, Path, Path | None]:
    if path.is_dir():
        result_dir = path
    else:
        result_dir = path.parent
    all_jsonl = result_dir / "all.jsonl"
    manifest_path = result_dir / "manifest.json"
    runtime_gate_path = result_dir / "version26_runtime_gate.json"
    model_gate_path = result_dir / "qwen3_model_gate.json"
    remote_asset_gate_path = result_dir / "remote_eval_asset_gate.json"
    _require(all_jsonl.is_file(), f"missing result file: {all_jsonl}")
    _require(manifest_path.is_file(), f"missing manifest: {manifest_path}")
    _require(runtime_gate_path.is_file(), f"missing runtime gate: {runtime_gate_path}")
    _require(model_gate_path.is_file(), f"missing Qwen3 model gate: {model_gate_path}")
    return (
        all_jsonl.resolve(),
        manifest_path.resolve(),
        runtime_gate_path.resolve(),
        model_gate_path.resolve(),
        remote_asset_gate_path.resolve() if remote_asset_gate_path.is_file() else None,
    )


def _read_json_object(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is invalid JSON: {path}: {exc}") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object: {path}")
    return payload, value


def _validate_runtime_gate(
    gate: dict[str, Any], *, label: str, expected_total: int
) -> None:
    _require(gate.get("status") == "ok", f"{label} runtime gate did not pass")
    _require(
        gate.get("source_commit") == EXPECTED_RUNTIME_COMMIT,
        f"{label} runtime source commit drifted",
    )
    _require(
        gate.get("isolation") == "git-archive-exact-commit",
        f"{label} runtime is not the isolated Git archive",
    )
    _require(
        gate.get("current_checkout_protocol_modules_imported") is False,
        f"{label} runtime imported current-checkout protocol modules",
    )
    _require(gate.get("reasoning_parser") is None, f"{label} runtime uses a reasoning parser")
    protocol = gate.get("protocol")
    _require(isinstance(protocol, dict), f"{label} runtime gate has no protocol object")
    protocol_expected = {
        "tool_scheme": "atomic",
        "tool_scheme_registry_version": "tool-scheme-registry-v2",
        "assistant_carrier": "think-json-v1",
        "protocol_version": "version26",
        "rolling_system_prompt_sha256": EXPECTED_STUDENT_PROMPT_SHA256,
    }
    for field, expected in protocol_expected.items():
        _require(
            protocol.get(field) == expected,
            f"{label} runtime gate protocol {field} drifted",
        )
    _require(
        isinstance(protocol.get("rolling_protocol_hash"), str)
        and bool(protocol["rolling_protocol_hash"]),
        f"{label} runtime gate has no rolling protocol hash",
    )
    contract = gate.get("evaluation_contract")
    _require(isinstance(contract, dict), f"{label} runtime gate has no evaluation contract")
    contract_expected = {
        "questions": expected_total,
        "rollouts_per_question": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_steps": 30,
        "context_mode": "rolling-legal-history",
        "history_turns": 4,
        "rolling_observation_style": "resident",
        "denotation_comparison": "bird-set",
        "enable_thinking": True,
        "reasoning_parser": None,
    }
    for field, expected in contract_expected.items():
        _require(
            contract.get(field) == expected,
            f"{label} runtime evaluation contract {field} drifted",
        )
    _require(
        _nonnegative_int(
            contract.get("minimum_max_tokens"),
            f"{label} runtime minimum_max_tokens",
        )
        == 2048,
        f"{label} runtime minimum_max_tokens drifted",
    )


def _validate_model_gate(gate: dict[str, Any], *, label: str) -> None:
    expected = {
        "status": "ok",
        "assumed_huggingface_revision": EXPECTED_QWEN3_REVISIONS[MODEL_SIZE],
        "model_type": "qwen3",
        "architecture": "Qwen3ForCausalLM",
        "tokenizer_config_sha256": EXPECTED_TOKENIZER_CONFIG_SHA256,
        "chat_template_sha256": EXPECTED_CHAT_TEMPLATE_SHA256,
        "official_chat_template": True,
        "enable_thinking": True,
        "empty_think_suppression_injected": False,
        "reasoning_parser": None,
    }
    for field, value in expected.items():
        _require(gate.get(field) == value, f"{label} Qwen3 model gate {field} drifted")
    _require(
        isinstance(gate.get("model_root"), str) and bool(gate["model_root"]),
        f"{label} Qwen3 model gate has no model root",
    )
    adapter = gate.get("adapter")
    if label == "base":
        _require(adapter is None, "base Qwen3 model gate unexpectedly contains an adapter")
    elif label == "adapter":
        _require(isinstance(adapter, dict), "adapter Qwen3 model gate has no adapter report")
        _require(adapter.get("peft_type") == "LORA", "adapter gate is not LoRA/QLoRA")
        _require(
            isinstance(adapter.get("rank"), int)
            and not isinstance(adapter.get("rank"), bool)
            and adapter["rank"] > 0,
            "adapter gate has no positive LoRA rank",
        )


def _validate_remote_asset_gate(
    gate: dict[str, Any],
    *,
    label: str,
    runtime_gate: dict[str, Any],
    model_gate: dict[str, Any],
) -> None:
    """Validate the copied all-NewGNN preflight report for one formal evaluation arm."""
    _require(
        gate.get("schema_version") == EXPECTED_REMOTE_ASSET_GATE_SCHEMA,
        f"{label} remote asset gate schema drifted",
    )
    _require(gate.get("status") == "ok", f"{label} remote asset gate did not pass")
    _require(gate.get("mode") == label, f"{label} remote asset gate mode drifted")
    _require(
        gate.get("runtime") == runtime_gate,
        f"{label} remote asset/runtime gate payloads differ",
    )
    _require(
        gate.get("model") == model_gate,
        f"{label} remote asset/model gate payloads differ",
    )

    evaluation_input = gate.get("evaluation_input")
    _require(
        isinstance(evaluation_input, dict),
        f"{label} remote asset gate has no evaluation_input object",
    )
    expected_input = {
        "status": "ok",
        "source_sha256": EXPECTED_SOURCE_DATASET_SHA256,
        "derived_sha256": EXPECTED_REMOTE_DATASET_SHA256,
        "records": EXPECTED_TOTAL,
        "only_db_path_changed": True,
    }
    for field, expected in expected_input.items():
        _require(
            evaluation_input.get(field) == expected,
            f"{label} remote asset gate evaluation_input {field} drifted",
        )
    databases = evaluation_input.get("databases")
    _require(
        isinstance(databases, dict) and len(databases) == 11,
        f"{label} remote asset gate must verify all 11 BIRD databases",
    )
    database_records = 0
    for db_id, database in databases.items():
        _require(
            isinstance(db_id, str) and bool(db_id) and isinstance(database, dict),
            f"{label} remote asset gate has a malformed database entry",
        )
        records = _nonnegative_int(
            database.get("records"), f"{label} remote database {db_id} records"
        )
        _require(
            _is_sha256(database.get("sha256")),
            f"{label} remote database {db_id} has an invalid sha256",
        )
        database_records += records
    _require(
        database_records == EXPECTED_TOTAL,
        f"{label} remote database record counts do not sum to {EXPECTED_TOTAL}",
    )

    contract = gate.get("evaluation_contract")
    _require(
        isinstance(contract, dict),
        f"{label} remote asset gate has no evaluation contract",
    )
    expected_contract = {
        "questions": EXPECTED_TOTAL,
        "n_samples": 1,
        "pass_k": "1",
        "temperature": 0,
        "top_p": 1,
        "max_steps": 30,
        "max_tokens": 2048,
        "context_mode": "rolling-legal-history",
        "history_turns": 4,
        "rolling_prompt_variant": "full",
        "rolling_observation_style": "resident",
        "denotation_comparison": "bird-set",
        "enable_thinking": True,
        "reasoning_parser": None,
        "max_num_seqs": 4,
        "workers": 4,
        "max_inflight_requests": 4,
    }
    for field, expected in expected_contract.items():
        _require(
            contract.get(field) == expected,
            f"{label} remote asset gate evaluation contract {field} drifted",
        )


def _validate_manifest(manifest: dict[str, Any], *, label: str, expected_total: int) -> None:
    expected = {
        "runner": "tool_rollout_passk",
        "tool_scheme": "atomic",
        "assistant_carrier": "think-json-v1",
        "protocol_version": "version26",
        "requested_size": expected_total,
        "n_samples": 1,
        "stop_on_success": False,
        "pass_k": [1],
        "sample_detail": "full",
        "max_steps": 30,
        "context_mode": "rolling-legal-history",
        "history_turns": 4,
        "rolling_prompt_variant": "full",
        "rolling_observation_style": "resident",
        "denotation_comparison": "bird-set",
        "dataset_purpose": "evaluation",
    }
    for field, value in expected.items():
        _require(
            manifest.get(field) == value,
            f"{label} manifest {field} must be {value!r}, got {manifest.get(field)!r}",
        )
    _require(
        _is_number_equal(manifest.get("temperature"), 0.0),
        f"{label} manifest temperature must be 0",
    )
    _require(
        _is_number_equal(manifest.get("top_p"), 1.0),
        f"{label} manifest top_p must be 1",
    )
    max_tokens = _nonnegative_int(manifest.get("max_tokens"), f"{label} manifest max_tokens")
    _require(max_tokens >= 2048, f"{label} manifest max_tokens must be at least 2048")
    _require(
        str(manifest.get("enable_thinking")).lower() in {"1", "true"},
        f"{label} manifest must enable official Qwen3 thinking",
    )
    expected_model = {
        "base": f"qwen3-{MODEL_SIZE}-atomic-v26-base",
        "adapter": f"qwen3-{MODEL_SIZE}-atomic-v26-sft1-qlora",
    }.get(label)
    if expected_model is not None:
        _require(
            manifest.get("model") == expected_model,
            f"{label} manifest model must be {expected_model!r}",
        )
    for field in ("protocol_hash", "dataset", "model", "system_prompt"):
        _require(
            isinstance(manifest.get(field), str) and bool(manifest[field]),
            f"{label} manifest {field} must be a non-empty string",
        )
    dataset_basename = Path(manifest["dataset"]).name
    _require(
        dataset_basename
        in {EXPECTED_SOURCE_DATASET_BASENAME, EXPECTED_REMOTE_DATASET_BASENAME},
        f"{label} manifest has unexpected dataset basename {dataset_basename!r}",
    )
    system_prompt_sha256 = hashlib.sha256(manifest["system_prompt"].encode("utf-8")).hexdigest()
    _require(
        system_prompt_sha256 == EXPECTED_STUDENT_PROMPT_SHA256,
        f"{label} manifest system prompt hash drifted",
    )


def _validate_row(
    row: dict[str, Any],
    *,
    label: str,
    line_number: int,
    protocol_hash: str,
) -> TaskResult:
    location = f"{label} line {line_number}"
    example_index = _nonnegative_int(row.get("example_index"), f"{location} example_index")
    _require(row.get("tool_scheme") == "atomic", f"{location} is not atomic")
    _require(row.get("assistant_carrier") == "think-json-v1", f"{location} carrier drifted")
    _require(row.get("protocol_version") == "version26", f"{location} is not version26")
    _require(row.get("protocol_hash") == protocol_hash, f"{location} protocol hash drifted")
    _require(row.get("denotation_comparison") == "bird-set", f"{location} is not bird-set")
    _require(row.get("n_samples") == 1, f"{location} must have n_samples=1")
    _require(row.get("pass_k") == [1], f"{location} must have pass_k=[1]")
    _require(_is_number_equal(row.get("temperature"), 0.0), f"{location} is not greedy")
    _require(_is_number_equal(row.get("top_p"), 1.0), f"{location} top_p must be 1")

    samples = row.get("samples")
    _require(isinstance(samples, list) and len(samples) == 1, f"{location} needs one sample")
    sample = samples[0]
    _require(isinstance(sample, dict), f"{location} sample must be an object")
    _require(sample.get("sample_index") == 0, f"{location} sample_index must be 0")
    _require(sample.get("protocol_version") == "version26", f"{location} sample is not version26")
    _require(sample.get("protocol_hash") == protocol_hash, f"{location} sample protocol hash drifted")
    _require(sample.get("tool_scheme") == "atomic", f"{location} sample is not atomic")
    _require(
        sample.get("denotation_comparison") == "bird-set",
        f"{location} sample is not bird-set",
    )

    correct = sample.get("correct")
    legal = sample.get("legal")
    _require(type(correct) is bool, f"{location} sample.correct must be boolean")
    _require(type(legal) is bool, f"{location} sample.legal must be boolean")
    _require(not correct or legal, f"{location} cannot be correct without legal termination")
    _require(row.get("correct") is correct, f"{location} task/sample correctness differs")
    pass_at = row.get("pass_at")
    _require(
        isinstance(pass_at, dict) and pass_at.get("1") is correct,
        f"{location} pass_at[1] differs from sample correctness",
    )
    _require(row.get("sample_correct_count") == int(correct), f"{location} correct count differs")
    _require(row.get("sample_legal_count") == int(legal), f"{location} legal count differs")
    _require(row.get("attempted_samples") == 1, f"{location} attempted_samples must be 1")

    steps = _nonnegative_int(sample.get("steps"), f"{location} sample.steps")
    errors = _nonnegative_int(sample.get("errors"), f"{location} sample.errors")
    error_events = sample.get("error_events")
    _require(isinstance(error_events, list), f"{location} sample.error_events must be a list")
    _require(
        len(error_events) == errors,
        f"{location} sample.errors={errors} but has {len(error_events)} error events",
    )
    error_types = []
    for event_index, event in enumerate(error_events):
        _require(isinstance(event, dict), f"{location} error event {event_index} is not an object")
        error_type = event.get("error_type") or event.get("type")
        _require(
            isinstance(error_type, str) and bool(error_type),
            f"{location} error event {event_index} has no type",
        )
        error_types.append(error_type)

    db_id = row.get("db_id")
    question = row.get("question")
    gold_sql = row.get("gold_sql")
    _require(isinstance(db_id, str) and bool(db_id), f"{location} db_id is missing")
    _require(isinstance(question, str) and bool(question), f"{location} question is missing")
    _require(isinstance(gold_sql, str) and bool(gold_sql), f"{location} gold_sql is missing")
    failure_type = sample.get("failure_type")
    _require(
        failure_type is None or isinstance(failure_type, str),
        f"{location} failure_type must be null or string",
    )
    return TaskResult(
        example_index=example_index,
        db_id=db_id,
        question=question,
        gold_sql_sha256=hashlib.sha256(gold_sql.encode("utf-8")).hexdigest(),
        correct=correct,
        legal=legal,
        steps=steps,
        errors=errors,
        failure_type=failure_type,
        error_types=tuple(error_types),
    )


def load_run(path: Path, *, label: str, expected_total: int = EXPECTED_TOTAL) -> RunResult:
    """Load a completed run while detecting incomplete, duplicate, or concurrently changing data."""
    (
        all_jsonl,
        manifest_path,
        runtime_gate_path,
        model_gate_path,
        remote_asset_gate_path,
    ) = _resolve_run_path(path)
    manifest_bytes, manifest = _read_json_object(manifest_path, f"{label} manifest")
    runtime_gate_bytes, runtime_gate = _read_json_object(
        runtime_gate_path, f"{label} runtime gate"
    )
    model_gate_bytes, model_gate = _read_json_object(model_gate_path, f"{label} model gate")
    _validate_manifest(manifest, label=label, expected_total=expected_total)
    _validate_runtime_gate(runtime_gate, label=label, expected_total=expected_total)
    _validate_model_gate(model_gate, label=label)
    runtime_protocol = runtime_gate["protocol"]
    _require(
        manifest["protocol_hash"] == runtime_protocol["rolling_protocol_hash"],
        f"{label} manifest/runtime protocol hash differs",
    )
    remote_asset_gate_bytes: bytes | None = None
    remote_asset_gate: dict[str, Any] | None = None
    dataset_basename = Path(manifest["dataset"]).name
    if remote_asset_gate_path is not None:
        _require(
            dataset_basename == EXPECTED_REMOTE_DATASET_BASENAME,
            f"{label} remote asset gate requires the pinned derived dataset basename",
        )
        remote_asset_gate_bytes, remote_asset_gate = _read_json_object(
            remote_asset_gate_path, f"{label} remote asset gate"
        )
        _validate_remote_asset_gate(
            remote_asset_gate,
            label=label,
            runtime_gate=runtime_gate,
            model_gate=model_gate,
        )
    elif dataset_basename == EXPECTED_REMOTE_DATASET_BASENAME:
        raise ValueError(
            f"{label} formal NewGNN result is missing remote_eval_asset_gate.json"
        )

    before = all_jsonl.stat()
    digest = hashlib.sha256()
    tasks: dict[int, TaskResult] = {}
    with all_jsonl.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label} line {line_number} is invalid JSON: {exc}") from exc
            _require(isinstance(row, dict), f"{label} line {line_number} must be an object")
            task = _validate_row(
                row,
                label=label,
                line_number=line_number,
                protocol_hash=manifest["protocol_hash"],
            )
            _require(
                task.example_index not in tasks,
                f"{label} duplicate example_index {task.example_index}",
            )
            tasks[task.example_index] = task
    after = all_jsonl.stat()
    _require(
        (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
        f"{label} all.jsonl changed while being read; wait for evaluation to finish",
    )
    expected_ids = set(range(expected_total))
    actual_ids = set(tasks)
    missing = sorted(expected_ids - actual_ids)
    unexpected = sorted(actual_ids - expected_ids)
    _require(
        actual_ids == expected_ids,
        f"{label} must contain exactly indices 0..{expected_total - 1}; "
        f"missing={missing[:10]} unexpected={unexpected[:10]}",
    )
    return RunResult(
        label=label,
        all_jsonl=all_jsonl,
        all_jsonl_sha256=digest.hexdigest(),
        manifest_path=manifest_path,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        manifest=manifest,
        runtime_gate_path=runtime_gate_path,
        runtime_gate_sha256=hashlib.sha256(runtime_gate_bytes).hexdigest(),
        runtime_gate=runtime_gate,
        model_gate_path=model_gate_path,
        model_gate_sha256=hashlib.sha256(model_gate_bytes).hexdigest(),
        model_gate=model_gate,
        remote_asset_gate_path=remote_asset_gate_path,
        remote_asset_gate_sha256=(
            hashlib.sha256(remote_asset_gate_bytes).hexdigest()
            if remote_asset_gate_bytes is not None
            else None
        ),
        remote_asset_gate=remote_asset_gate,
        tasks=tasks,
    )


def wilson_interval(successes: int, total: int, z: float = WILSON_Z_95) -> tuple[float, float]:
    """Return the two-sided Wilson score interval for one binomial proportion."""
    _require(total > 0, "Wilson interval requires total > 0")
    _require(0 <= successes <= total, "Wilson successes must lie in [0,total]")
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    center = (proportion + (z * z / (2.0 * total))) / denominator
    spread = (
        z
        * math.sqrt(
            (proportion * (1.0 - proportion) / total)
            + (z * z / (4.0 * total * total))
        )
        / denominator
    )
    return max(0.0, center - spread), min(1.0, center + spread)


def exact_mcnemar_p(gains: int, regressions: int) -> float:
    """Two-sided exact McNemar/sign-test p-value over discordant task pairs."""
    _require(gains >= 0 and regressions >= 0, "discordant counts must be non-negative")
    discordant = gains + regressions
    if discordant == 0:
        return 1.0
    tail = min(gains, regressions)
    probability = sum(math.comb(discordant, index) for index in range(tail + 1))
    return min(1.0, 2.0 * probability / (2**discordant))


def _proportion(successes: int, total: int) -> dict[str, Any]:
    low, high = wilson_interval(successes, total)
    rate = successes / total
    return {
        "count": successes,
        "total": total,
        "rate": rate,
        "percent": 100.0 * rate,
        "wilson_95_ci": {
            "low": low,
            "high": high,
            "low_percent": 100.0 * low,
            "high_percent": 100.0 * high,
            "confidence": 0.95,
            "method": "Wilson score",
        },
    }


def _run_metrics(run: RunResult) -> dict[str, Any]:
    tasks = [run.tasks[index] for index in sorted(run.tasks)]
    total = len(tasks)
    errors_by_type = Counter(error_type for task in tasks for error_type in task.error_types)
    failure_types = Counter(task.failure_type or "none" for task in tasks)
    process_errors = sum(task.errors for task in tasks)
    total_steps = sum(task.steps for task in tasks)
    return {
        "input": {
            "all_jsonl": os.fspath(run.all_jsonl),
            "all_jsonl_sha256": run.all_jsonl_sha256,
            "manifest": os.fspath(run.manifest_path),
            "manifest_sha256": run.manifest_sha256,
            "version26_runtime_gate": os.fspath(run.runtime_gate_path),
            "version26_runtime_gate_sha256": run.runtime_gate_sha256,
            "qwen3_model_gate": os.fspath(run.model_gate_path),
            "qwen3_model_gate_sha256": run.model_gate_sha256,
            "dataset": run.manifest["dataset"],
            "remote_eval_asset_gate": (
                os.fspath(run.remote_asset_gate_path)
                if run.remote_asset_gate_path is not None
                else None
            ),
            "remote_eval_asset_gate_sha256": run.remote_asset_gate_sha256,
            "model": run.manifest["model"],
            "adapter": run.model_gate.get("adapter"),
        },
        "ex": _proportion(sum(task.correct for task in tasks), total),
        "legal_termination": _proportion(sum(task.legal for task in tasks), total),
        "process_errors": {
            "total": process_errors,
            "trajectories_with_any": sum(task.errors > 0 for task in tasks),
            "mean_per_task": process_errors / total,
            "by_type": dict(sorted(errors_by_type.items())),
        },
        "steps": {
            "total": total_steps,
            "mean": statistics.fmean(task.steps for task in tasks),
        },
        "failure_types": dict(sorted(failure_types.items())),
    }


def _pair_boolean(
    base: RunResult,
    adapter: RunResult,
    *,
    attribute: str,
) -> dict[str, Any]:
    gains = []
    regressions = []
    both_true = 0
    both_false = 0
    for example_index in sorted(base.tasks):
        base_value = bool(getattr(base.tasks[example_index], attribute))
        adapter_value = bool(getattr(adapter.tasks[example_index], attribute))
        if base_value and adapter_value:
            both_true += 1
        elif not base_value and not adapter_value:
            both_false += 1
        elif adapter_value:
            gains.append(example_index)
        else:
            regressions.append(example_index)
    total = len(base.tasks)
    return {
        "paired_tasks": total,
        "both_true": both_true,
        "both_false": both_false,
        "gains": len(gains),
        "regressions": len(regressions),
        "net": len(gains) - len(regressions),
        "delta_rate": (len(gains) - len(regressions)) / total,
        "delta_percentage_points": 100.0 * (len(gains) - len(regressions)) / total,
        "exact_mcnemar_p_two_sided": exact_mcnemar_p(len(gains), len(regressions)),
        "gain_example_indices": gains,
        "regression_example_indices": regressions,
    }


def _validate_paired_manifests(base: RunResult, adapter: RunResult) -> dict[str, Any]:
    _require(base.label == "base", "left run must be loaded with label=base")
    _require(adapter.label == "adapter", "right run must be loaded with label=adapter")
    differences = {
        field: {"base": base.manifest.get(field), "adapter": adapter.manifest.get(field)}
        for field in MANIFEST_EQUAL_FIELDS
        if base.manifest.get(field) != adapter.manifest.get(field)
    }
    _require(not differences, f"base/adapter evaluation manifests differ: {differences}")
    base_dataset = base.manifest["dataset"]
    adapter_dataset = adapter.manifest["dataset"]
    base_basename = Path(base_dataset).name
    adapter_basename = Path(adapter_dataset).name
    both_remote_gated = (
        base.remote_asset_gate is not None and adapter.remote_asset_gate is not None
    )
    _require(
        (base.remote_asset_gate is None) == (adapter.remote_asset_gate is None),
        "base/adapter cannot mix remote-asset-gated and legacy local results",
    )
    if base_dataset != adapter_dataset:
        _require(
            both_remote_gated,
            "different base/adapter dataset paths require both remote asset gates",
        )
        _require(
            base_basename == adapter_basename == EXPECTED_REMOTE_DATASET_BASENAME,
            "different dataset paths must name the pinned NewGNN derived JSONL",
        )
    else:
        _require(
            base_basename
            in {EXPECTED_SOURCE_DATASET_BASENAME, EXPECTED_REMOTE_DATASET_BASENAME},
            f"paired dataset basename is not recognized: {base_basename!r}",
        )
    _require(
        base.runtime_gate_sha256 == adapter.runtime_gate_sha256,
        "base/adapter version26 runtime gate files are not byte-identical",
    )
    for field in ("source_commit", "content_tree_sha256", "key_file_sha256"):
        _require(
            base.runtime_gate.get(field) == adapter.runtime_gate.get(field),
            f"base/adapter runtime gates differ at {field}",
        )
    for field in (
        "model_root",
        "assumed_huggingface_revision",
        "tokenizer_config_sha256",
        "chat_template_sha256",
    ):
        _require(
            base.model_gate.get(field) == adapter.model_gate.get(field),
            f"base/adapter Qwen3 model gates differ at {field}",
        )
    for example_index in sorted(base.tasks):
        left = base.tasks[example_index]
        right = adapter.tasks[example_index]
        _require(
            (left.db_id, left.question) == (right.db_id, right.question),
            f"base/adapter task identity differs at example_index {example_index}",
        )
        _require(
            left.gold_sql_sha256 == right.gold_sql_sha256,
            f"base/adapter gold_sql hash differs at example_index {example_index}",
        )
    return {
        "base_path": base_dataset,
        "adapter_path": adapter_dataset,
        "paths_equal": base_dataset == adapter_dataset,
        "basename": base_basename,
        "remote_asset_gated": both_remote_gated,
        "source_sha256": (
            EXPECTED_SOURCE_DATASET_SHA256 if both_remote_gated else None
        ),
        "derived_sha256": (
            EXPECTED_REMOTE_DATASET_SHA256 if both_remote_gated else None
        ),
        "paired_gold_sql_hashes_equal": True,
    }


def _threshold_comparison(correct: int, threshold_count: int, total: int) -> dict[str, Any]:
    return {
        "meets_or_exceeds": correct >= threshold_count,
        "count_margin": correct - threshold_count,
        "percentage_point_margin": 100.0 * (correct - threshold_count) / total,
    }


def build_analysis(base: RunResult, adapter: RunResult) -> dict[str, Any]:
    dataset_identity = _validate_paired_manifests(base, adapter)
    _require(len(base.tasks) == EXPECTED_TOTAL, "reference thresholds require BIRD-dev1534")
    base_metrics = _run_metrics(base)
    adapter_metrics = _run_metrics(adapter)
    thresholds = {}
    reference_thresholds = REFERENCE_THRESHOLDS_4B if MODEL_SIZE == "4b" else REFERENCE_THRESHOLDS
    for key, description, count in reference_thresholds:
        thresholds[key] = {
            "description": description,
            "reference": _proportion(count, EXPECTED_TOTAL),
            "base": _threshold_comparison(base_metrics["ex"]["count"], count, EXPECTED_TOTAL),
            "adapter": _threshold_comparison(
                adapter_metrics["ex"]["count"], count, EXPECTED_TOTAL
            ),
        }
    return {
        "schema_version": "qwen3-atomic-v26-sft1-paired-bird-dev1534-analysis-v1",
        "scope": {
            "dataset": "BIRD-dev1534",
            "dataset_identity": dataset_identity,
            "tool_protocol": "atomic version26",
            "decode": "greedy n=1",
            "denotation_comparison": "bird-set",
            "threshold_scope_note": (
                "Paper/author-protocol values are external reference levels only; the authors' "
                "SQL-tool protocol and this atomic-tool protocol are not identical."
            ),
        },
        "base": base_metrics,
        "adapter": adapter_metrics,
        "paired_accuracy": _pair_boolean(base, adapter, attribute="correct"),
        "paired_legal_termination": _pair_boolean(base, adapter, attribute="legal"),
        "paired_cost_delta": {
            "process_errors": (
                adapter_metrics["process_errors"]["total"]
                - base_metrics["process_errors"]["total"]
            ),
            "mean_steps": adapter_metrics["steps"]["mean"] - base_metrics["steps"]["mean"],
        },
        "reference_thresholds": thresholds,
    }


def _format_markdown(analysis: dict[str, Any]) -> str:
    base = analysis["base"]
    adapter = analysis["adapter"]
    paired = analysis["paired_accuracy"]
    legal_pair = analysis["paired_legal_termination"]
    lines = [
        f"# Qwen3-{MODEL_SIZE.upper()} atomic version26：base / SFT1 QLoRA 配对结果",
        "",
        "| 运行 | EX (Wilson 95% CI) | Legal | Process errors | Mean steps |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metrics in (("Base", base), ("SFT1 QLoRA", adapter)):
        ex = metrics["ex"]
        ci = ex["wilson_95_ci"]
        legal = metrics["legal_termination"]
        lines.append(
            f"| {label} | {ex['count']}/{ex['total']} ({ex['percent']:.2f}%; "
            f"{ci['low_percent']:.2f}%–{ci['high_percent']:.2f}%) | "
            f"{legal['count']}/{legal['total']} ({legal['percent']:.2f}%) | "
            f"{metrics['process_errors']['total']} | {metrics['steps']['mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 配对",
            "",
            f"- Accuracy: {paired['gains']} gains / {paired['regressions']} regressions, "
            f"net {paired['net']:+d} ({paired['delta_percentage_points']:+.2f} pp), "
            f"two-sided exact McNemar p={paired['exact_mcnemar_p_two_sided']:.6g}.",
            f"- Legal termination: {legal_pair['gains']} gains / "
            f"{legal_pair['regressions']} regressions, "
            f"p={legal_pair['exact_mcnemar_p_two_sided']:.6g}.",
            "",
            "## 外部参考门槛",
            "",
            "| 参考 | Count | Base margin | Adapter margin | Adapter meets |",
            "|---|---:|---:|---:|:---:|",
        ]
    )
    for threshold in analysis["reference_thresholds"].values():
        reference = threshold["reference"]
        lines.append(
            f"| {threshold['description']} | {reference['count']}/{reference['total']} | "
            f"{threshold['base']['count_margin']:+d} | "
            f"{threshold['adapter']['count_margin']:+d} | "
            f"{'yes' if threshold['adapter']['meets_or_exceeds'] else 'no'} |"
        )
    lines.extend(["", analysis["scope"]["threshold_scope_note"]])
    return "\n".join(lines) + "\n"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        required=True,
        type=Path,
        help="base result directory or its all.jsonl",
    )
    parser.add_argument(
        "--adapter",
        required=True,
        type=Path,
        help="SFT1 adapter result directory or its all.jsonl",
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--model-size", choices=("4b", "8b"), default="8b")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    global MODEL_SIZE
    args = parse_args(argv)
    MODEL_SIZE = args.model_size
    base = load_run(args.base, label="base")
    adapter = load_run(args.adapter, label="adapter")
    analysis = build_analysis(base, adapter)
    if args.format == "markdown":
        print(_format_markdown(analysis), end="")
    else:
        print(json.dumps(analysis, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
