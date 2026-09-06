#!/usr/bin/env python3
"""Fresh, fail-closed SFT1-versus-final matched evaluation for vanilla GRPO.

The controller deliberately has no queueing, process discovery, or remote-host
operations.  A run is admitted only when the requested physical GPU is already
idle.  It starts one owned vLLM process group at a time, evaluates SFT1 and the
pre-registered final adapter sequentially, and signals only those owned groups.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Sequence


from rl.evaluation.runners.contracts import default_contract
from rl.evaluation.runners.identity import evaluation_identity
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[4]
DEFAULT_CONTRACT = default_contract("qwen3_8b_v26_vanilla_formal_matched_contract.json")
MATCHED_IDENTITY_FIELDS = (
    "base_model",
    "base_model_revision",
    "base_model_identity_sha256",
    "adapter_config_semantic_sha256",
    "dataset",
    "input_sha256",
    "task_identity_sha256_without_db_path",
    "protocol_version",
    "protocol_hash",
    "runtime",
    "runtime_sha256",
    "serving",
    "concurrency",
    "decode",
    "agent",
    "tool_execution_timeout_seconds",
)

GENERATED_RUNTIME_DIRECTORY_NAMES = frozenset(
    {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
)
GENERATED_RUNTIME_FILE_NAMES = frozenset({".DS_Store"})
GENERATED_RUNTIME_FILE_SUFFIXES = frozenset({".pyc", ".pyo"})


class TerminationRequested(RuntimeError):
    """The controller received SIGINT or SIGTERM."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"row {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def normalize_adapter_config(config: dict[str, Any]) -> dict[str, Any]:
    """Canonicalize PEFT serialization drift without weakening semantics.

    PEFT releases may add optional serialization fields whose value is null,
    and serialize ``target_modules`` in a different set iteration order.  A
    missing field and a null field therefore carry the same meaning here, and
    target modules are compared as a set.  Every other value, including every
    non-null newly introduced field, remains part of the exact semantic
    identity.
    """

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: normalize(child)
                for key, child in sorted(value.items())
                if child is not None
            }
        if isinstance(value, list):
            return [normalize(child) for child in value]
        return value

    normalized = normalize(config)
    target_modules = normalized.get("target_modules")
    if isinstance(target_modules, list):
        if not all(isinstance(module, str) for module in target_modules):
            raise ValueError("adapter target_modules must contain only strings")
        normalized["target_modules"] = sorted(set(target_modules))
    return normalized


def adapter_config_semantic_sha256(config: dict[str, Any]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(normalize_adapter_config(config))
    ).hexdigest()


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(payload)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    atomic_bytes(path, json.dumps(
        value, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n")


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} mismatch: expected={expected!r}, actual={actual!r}")


def validate_contract_shape(contract: dict[str, Any]) -> None:
    """Reject edits that would weaken the preregistered two-arm comparison."""

    require_equal(
        contract.get("schema_version"),
        "qwen3-8b-v26-vanilla-formal-matched-eval-v1",
        "contract schema",
    )
    require_equal(contract.get("arms"), ["sft1", "final"], "formal arms")
    require_equal(contract.get("baseline"), "sft1", "formal baseline")
    require_equal(contract.get("primary_candidate"), "final", "primary candidate")
    runtime = contract["runtime"]
    required_runtime = {
        "source_commit": "4cd47c957fc6ae791e76a10594c8cd22f4d3b6de",
        "content_tree_sha256": "5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab",
        "protocol_version": "version26",
        "protocol_hash": "4da19387399bd3a5",
        "student_prompt_sha256": "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316",
        "tool_scheme": "atomic",
        "assistant_carrier": "think-json-v1",
    }
    for field, expected in required_runtime.items():
        require_equal(runtime.get(field), expected, f"runtime.{field}")
    evaluation = contract["evaluation"]
    required_evaluation = {
        "questions": 1534,
        "n_samples": 1,
        "pass_k": [1],
        "workers": 24,
        "sample_workers": 1,
        "first_sample_workers": 0,
        "max_inflight_requests": 24,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_steps": 30,
        "max_tokens": 2048,
        "api_retries": 3,
        "few_shot": 0,
        "sample_detail": "full",
        "context_mode": "rolling-legal-history",
        "history_turns": 4,
        "rolling_prompt_variant": "full",
        "rolling_observation_style": "resident",
        "denotation_comparison": "bird-set",
        "enable_thinking": True,
        "tool_execution_timeout_seconds": 10.0,
    }
    for field, expected in required_evaluation.items():
        require_equal(evaluation.get(field), expected, f"evaluation.{field}")
    serving = contract["serving"]
    required_serving = {
        "host": "127.0.0.1",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "max_model_len": 16384,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 24,
        "gpu_memory_utilization": 0.9,
        "generation_config": "vllm",
        "max_lora_rank": 64,
        "reasoning_parser": None,
        "chat_template_override": None,
    }
    for field, expected in required_serving.items():
        require_equal(serving.get(field), expected, f"serving.{field}")
    gate = contract["promotion_gate"]
    require_equal(gate.get("expected_count"), 1534, "gate expected count")
    require_equal(
        gate.get("minimum_accuracy_gain_percentage_points"),
        1.0,
        "gate accuracy gain",
    )
    require_equal(gate.get("exact_mcnemar_alpha"), 0.05, "gate alpha")
    require_equal(
        gate.get("legal_nonregression_required"), True, "gate legal rule"
    )


def implementation_paths() -> dict[str, Path]:
    return {
        "formal_v26_matched_eval.py": HERE / "formal_v26_matched_eval.py",
        "formal_v26_rollout_passk.py": HERE / "formal_v26_rollout_passk.py",
        "analyze_evaluation_results.py": (
            PROJECT_ROOT / "src/rl/scenarios/diagnostics/analyze_evaluation_results.py"
        ),
        "audit_vanilla_grpo_matched_gate.py": (
            PROJECT_ROOT / "src/rl/scenarios/diagnostics/audit_vanilla_grpo_matched_gate.py"
        ),
    }


def verify_implementation(contract: dict[str, Any]) -> dict[str, str]:
    expected = contract["implementation_sha256"]
    paths = implementation_paths()
    require_equal(set(expected), set(paths), "identity implementation files")
    observed: dict[str, str] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"identity implementation file is missing: {path}")
        observed[name] = sha256_file(path)
        require_equal(observed[name], expected[name], f"implementation {name} SHA-256")
    return observed


def task_identity_without_db_path(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        public = {key: value for key, value in row.items() if key != "db_path"}
        digest.update(canonical_json_bytes(public))
    return digest.hexdigest()


def verify_source_and_databases(
    contract: dict[str, Any], source: Path, db_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_contract = contract["input"]
    require_equal(sha256_file(source), input_contract["source_sha256"], "source input SHA-256")
    rows = load_jsonl(source)
    require_equal(len(rows), input_contract["records"], "source record count")
    counts: Counter[str] = Counter()
    for ordinal, row in enumerate(rows):
        require_equal(row.get("example_index"), ordinal, f"row {ordinal} example_index")
        db_id = row.get("db_id")
        if db_id not in input_contract["databases"]:
            raise ValueError(f"row {ordinal} has unexpected db_id={db_id!r}")
        if not isinstance(row.get("question"), str) or not row["question"]:
            raise ValueError(f"row {ordinal} has no question")
        if not isinstance(row.get("gold_sql"), str) or not row["gold_sql"]:
            raise ValueError(f"row {ordinal} has no gold_sql")
        source_db = row.get("db_path")
        if not isinstance(source_db, str) or Path(source_db).name != f"{db_id}.sqlite":
            raise ValueError(f"row {ordinal} has invalid source db_path={source_db!r}")
        counts[str(db_id)] += 1
    require_equal(
        task_identity_without_db_path(rows),
        input_contract["task_identity_sha256_without_db_path"],
        "task identity without db_path",
    )
    database_report: dict[str, Any] = {}
    for db_id, expected in sorted(input_contract["databases"].items()):
        require_equal(counts[db_id], expected["records"], f"{db_id} task count")
        database = db_root / db_id / f"{db_id}.sqlite"
        if not database.is_file():
            raise ValueError(f"database is missing: {database}")
        observed_hash = sha256_file(database)
        require_equal(observed_hash, expected["sha256"], f"{db_id} database SHA-256")
        database_report[db_id] = {
            "path": str(database),
            "records": counts[db_id],
            "sha256": observed_hash,
            "size_bytes": database.stat().st_size,
        }
    return rows, database_report


def prepare_derived_input(
    contract: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    db_root: Path,
    output: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if output.exists() or manifest_path.exists():
        raise ValueError("formal derived input artifacts already exist; resume is forbidden")
    input_contract = contract["input"]
    remote_root = PurePosixPath(str(db_root))
    remapped_rows = []
    counts: Counter[str] = Counter()
    for row in rows:
        remapped = dict(row)
        db_id = str(row["db_id"])
        remapped["db_path"] = str(remote_root / db_id / f"{db_id}.sqlite")
        remapped_rows.append(remapped)
        counts[db_id] += 1
    output_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        for row in remapped_rows
    )
    derived_hash = hashlib.sha256(output_bytes).hexdigest()
    require_equal(derived_hash, input_contract["derived_sha256"], "derived input SHA-256")
    databases = {
        db_id: {
            "relative_path": f"{db_id}/{db_id}.sqlite",
            "remote_path": str(remote_root / db_id / f"{db_id}.sqlite"),
            "records": counts[db_id],
            "sha256": expected["sha256"],
        }
        for db_id, expected in sorted(input_contract["databases"].items())
    }
    manifest = {
        "schema_version": "qwen3-atomic-v26-remote-eval-input-v1",
        "transformation": "replace-db_path-only",
        "source_logical_name": "bird_dev_20240627.jsonl",
        "source_sha256": input_contract["source_sha256"],
        "derived_logical_name": "bird_dev_20240627.newgnn.jsonl",
        "derived_sha256": derived_hash,
        "records": len(remapped_rows),
        "preserved_fields": "all-except-db_path",
        "remote_db_root": str(remote_root),
        "databases": databases,
    }
    manifest_bytes = canonical_json_bytes(manifest)
    require_equal(
        hashlib.sha256(manifest_bytes).hexdigest(),
        input_contract["derived_manifest_sha256"],
        "derived input manifest SHA-256",
    )
    atomic_bytes(output, output_bytes)
    atomic_bytes(manifest_path, manifest_bytes)
    return manifest


def is_generated_runtime_artifact(path: Path, root: Path) -> bool:
    """Return whether *path* is a disposable runtime-generated artifact."""

    relative = path.relative_to(root)
    return (
        any(part in GENERATED_RUNTIME_DIRECTORY_NAMES for part in relative.parts)
        or path.name in GENERATED_RUNTIME_FILE_NAMES
        or path.suffix in GENERATED_RUNTIME_FILE_SUFFIXES
    )


def content_tree(root: Path, exported_paths: Sequence[str]) -> tuple[int, str]:
    """Hash frozen source content while ignoring known interpreter/tool caches.

    The digest already includes every admitted relative path and its bytes, so
    an independent exact-file-count gate is redundant.  Generated cache files
    must not change runtime identity or block a post-training evaluation.
    """

    files: list[Path] = []
    for relative in exported_paths:
        directory = root / relative
        if not directory.is_dir():
            raise ValueError(f"runtime directory is missing: {directory}")
        files.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and not is_generated_runtime_artifact(path, root)
        )
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return len(files), digest.hexdigest()


def isolated_protocol_audit(runtime: Path) -> dict[str, Any]:
    program = r'''
import hashlib, inspect, json, pathlib, sys
runtime = pathlib.Path(sys.argv[1]).resolve()
for relative in ("src/eval", "src/harness", "src/sft"):
    sys.path.insert(0, str(runtime / relative))
import action_carrier, protocol, tool_schemes
system = protocol.rolling_system_prompt(protocol.get_system_prompt(), compact=False)
paths = {name: pathlib.Path(inspect.getfile(module)).resolve() for name, module in {
    "action_carrier": action_carrier, "protocol": protocol, "tool_schemes": tool_schemes
}.items()}
if any(runtime not in path.parents for path in paths.values()):
    raise SystemExit(f"protocol import escaped runtime: {paths}")
print(json.dumps({
    "protocol_version": protocol.PROTOCOL_VERSION,
    "tool_scheme": tool_schemes.ATOMIC_TOOL_SCHEME,
    "assistant_carrier": tool_schemes.ATOMIC_ASSISTANT_CARRIER,
    "rolling_system_prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
    "rolling_protocol_hash": protocol.protocol_hash(system),
    "module_paths": {name: str(path) for name, path in paths.items()},
}, sort_keys=True))
'''
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program, str(runtime)],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            "isolated version26 protocol audit failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return json.loads(completed.stdout)


def verify_runtime(contract: dict[str, Any], runtime: Path) -> dict[str, Any]:
    expected = contract["runtime"]
    lock_path = runtime / "runtime_lock.json"
    if not lock_path.is_file():
        raise ValueError(f"runtime lock is missing: {lock_path}")
    require_equal(
        sha256_file(lock_path), expected["runtime_lock_sha256"], "runtime lock SHA-256"
    )
    lock = load_object(lock_path)
    for field in ("source_commit", "exported_paths", "exported_file_count", "content_tree_sha256"):
        require_equal(lock.get(field), expected[field], f"runtime lock {field}")
    count, tree_hash = content_tree(runtime, expected["exported_paths"])
    require_equal(tree_hash, expected["content_tree_sha256"], "runtime content tree")
    for relative, expected_hash in expected["key_file_sha256"].items():
        require_equal(
            sha256_file(runtime / relative), expected_hash, f"runtime key file {relative}"
        )
    rollout_source = (runtime / "src/eval/rollout_passk.py").read_text(encoding="utf-8")
    forbidden = [
        marker
        for marker in ("reasoning_parser", "--reasoning-parser", "--enable-reasoning")
        if marker in rollout_source
    ]
    if forbidden:
        raise ValueError(f"frozen evaluator contains forbidden reasoning-parser markers: {forbidden}")
    audit = isolated_protocol_audit(runtime)
    protocol_expectations = {
        "protocol_version": expected["protocol_version"],
        "tool_scheme": expected["tool_scheme"],
        "assistant_carrier": expected["assistant_carrier"],
        "rolling_system_prompt_sha256": expected["student_prompt_sha256"],
        "rolling_protocol_hash": expected["protocol_hash"],
    }
    for field, value in protocol_expectations.items():
        require_equal(audit.get(field), value, f"runtime protocol {field}")
    return {
        "runtime_root": str(runtime),
        "source_commit": expected["source_commit"],
        "exported_file_count": count,
        "content_tree_sha256": tree_hash,
        "protocol": audit,
    }


def verify_model(contract: dict[str, Any], model: Path) -> dict[str, Any]:
    expected = contract["model"]
    expected_identity = expected["base_model_identity"]
    require_equal(
        expected_identity.get("schema_version"),
        "trl-base-model-identity-v1",
        "base model identity schema",
    )
    identity_files = expected_identity.get("files_sha256") or {}
    expected_filenames = {
        "config.json", "generation_config.json", "merges.txt",
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
        "model.safetensors.index.json", "tokenizer.json",
        "tokenizer_config.json", "vocab.json",
    }
    require_equal(set(identity_files), expected_filenames, "base model identity files")
    observed_files = {
        name: sha256_file(model / name) for name in sorted(identity_files)
    }
    require_equal(observed_files, identity_files, "base model identity file hashes")
    aggregate = hashlib.sha256()
    for name in sorted(observed_files, key=lambda value: value.encode("utf-8")):
        aggregate.update(f"{name}\t{observed_files[name]}\n".encode("utf-8"))
    require_equal(
        aggregate.hexdigest(),
        expected_identity.get("aggregate_sha256"),
        "base model identity aggregate",
    )
    require_equal(sha256_file(model / "config.json"), expected["config_sha256"], "model config")
    require_equal(
        sha256_file(model / "model.safetensors.index.json"),
        expected["model_index_sha256"],
        "model index",
    )
    require_equal(
        sha256_file(model / "tokenizer_config.json"),
        expected["tokenizer_config_sha256"],
        "tokenizer config",
    )
    for name, expected_hash in expected["shards_sha256"].items():
        require_equal(sha256_file(model / name), expected_hash, f"model shard {name}")
    config = load_object(model / "config.json")
    require_equal(config.get("model_type"), "qwen3", "model type")
    if "Qwen3ForCausalLM" not in (config.get("architectures") or []):
        raise ValueError("base model is not Qwen3ForCausalLM")
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError("transformers is missing from the evaluation Python") from exc
    tokenizer = AutoTokenizer.from_pretrained(
        str(model), local_files_only=True, trust_remote_code=False
    )
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("Qwen3 tokenizer has no chat template")
    require_equal(
        hashlib.sha256(template.encode("utf-8")).hexdigest(),
        expected["chat_template_sha256"],
        "Qwen3 chat template",
    )
    messages = [
        {"role": "system", "content": "system contract"},
        {"role": "user", "content": "question"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    if not rendered.endswith("<|im_start|>assistant\n"):
        raise ValueError("enable_thinking=true did not preserve Qwen3 generation boundary")
    if rendered.endswith("<think>\n\n</think>\n\n"):
        raise ValueError("enable_thinking=true injected the no-thinking prefix")
    return {
        "path": str(model),
        "repository": expected["repository"],
        "revision": expected["revision"],
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_config_sha256": expected["tokenizer_config_sha256"],
        "chat_template_sha256": expected["chat_template_sha256"],
        "model_shards_sha256": expected["shards_sha256"],
        "base_model_identity": expected_identity,
        "base_model_identity_sha256": expected_identity["aggregate_sha256"],
    }


def verify_adapter(
    adapter: Path,
    base_model: Path,
    *,
    max_rank: int,
    expected_files: dict[str, str] | None = None,
    expected_global_step: int | None = None,
) -> dict[str, Any]:
    weights = adapter / "adapter_model.safetensors"
    config_path = adapter / "adapter_config.json"
    if not weights.is_file() or not config_path.is_file():
        raise ValueError(f"adapter is incomplete: {adapter}")
    config = load_object(config_path)
    require_equal(str(config.get("peft_type", "")).upper(), "LORA", "adapter PEFT type")
    require_equal(str(config.get("task_type", "")).upper(), "CAUSAL_LM", "adapter task type")
    rank = int(config.get("r", 0))
    if rank <= 0 or rank > max_rank:
        raise ValueError(f"adapter rank {rank} is outside max_lora_rank={max_rank}")
    base_reference = config.get("base_model_name_or_path")
    if not isinstance(base_reference, str) or not base_reference:
        raise ValueError("adapter does not bind a base model")
    if Path(base_reference).is_absolute():
        require_equal(Path(base_reference).resolve(), base_model.resolve(), "adapter base model")
    else:
        require_equal(base_reference, "Qwen/Qwen3-8B", "adapter base repository")
    if expected_files:
        for name, expected_hash in expected_files.items():
            require_equal(sha256_file(adapter / name), expected_hash, f"adapter file {name}")
    if expected_global_step is not None:
        state = load_object(adapter / "trainer_state.json")
        require_equal(int(state.get("global_step", -1)), expected_global_step, "adapter step")
    return {
        "path": str(adapter),
        "adapter_sha256": sha256_file(weights),
        "adapter_config_sha256": sha256_file(config_path),
        "adapter_config_semantic_sha256": adapter_config_semantic_sha256(config),
        "rank": rank,
        "base_model_name_or_path": base_reference,
    }


def verify_training_final(
    contract: dict[str, Any], training_run: Path, base_model: Path, sft1: dict[str, Any]
) -> dict[str, Any]:
    expected = contract["training_final"]
    manifest_path = training_run / "run_manifest.json"
    lock_path = training_run / "implementation_lock.json"
    precision_path = training_run / "training_precision.json"
    for path in (manifest_path, lock_path, precision_path):
        if not path.is_file():
            raise ValueError(f"completed training artifact is missing: {path}")
    manifest = load_object(manifest_path)
    for field, value in expected["expected_manifest"].items():
        require_equal(manifest.get(field), value, f"training manifest {field}")
    require_equal(
        manifest.get("experiment_config_sha256"),
        expected["experiment_config_sha256"],
        "training experiment config",
    )
    runtime_identity = manifest.get("runtime_identity_audit") or {}
    for side in ("expected", "actual"):
        require_equal(
            (runtime_identity.get(side) or {}).get("runtime_content_tree_sha256"),
            contract["runtime"]["content_tree_sha256"],
            f"training runtime identity {side}",
        )
    runtime_modules = manifest.get("runtime_module_audit") or {}
    expected_runtime_root = str(Path(contract["host_paths"]["runtime"]))
    require_equal(
        runtime_modules.get("runtime_root"),
        expected_runtime_root,
        "training runtime module root",
    )
    require_equal(
        runtime_modules.get("tool_environment_factory_module"),
        "tool_environment_v26",
        "training tool environment factory",
    )
    require_equal(
        runtime_modules.get("module_paths"),
        {
            "protocol": expected_runtime_root + "/src/sft/protocol.py",
            "rollout": expected_runtime_root + "/src/eval/rollout.py",
            "executor": expected_runtime_root + "/src/harness/executor.py",
            "tool_schemes": expected_runtime_root + "/src/sft/tool_schemes.py",
        },
        "training runtime module paths",
    )
    rollout = manifest.get("rollout_settings") or {}
    for field, value in {
        "max_steps": 30,
        "max_new_tokens": 2048,
        "max_context_tokens": 16384,
        "history_turns": 4,
        "enable_thinking": True,
        "tool_execution_timeout_seconds": 10.0,
    }.items():
        require_equal(rollout.get(field), value, f"training rollout {field}")
    reference = manifest.get("reference_policy") or {}
    for field, value in {
        "enabled": False,
        "kl_beta": 0.0,
        "adapter_name": None,
        "adapter_path": None,
        "adapter_sha256": None,
        "expected_adapter_sha256": sft1["adapter_sha256"],
        "equals_initial_adapter": True,
    }.items():
        require_equal(reference.get(field), value, f"training reference {field}")
    lock = load_object(lock_path)
    require_equal(lock.get("files"), manifest.get("implementation_source_sha256"), "training implementation lock")
    require_equal(
        manifest.get("base_model_identity"),
        contract["model"]["base_model_identity"],
        "training base model identity",
    )
    require_equal(
        lock.get("base_model_identity"),
        manifest.get("base_model_identity"),
        "training lock base model identity",
    )
    for field in (
        "initial_adapter_sha256",
        "protocol_version",
        "protocol_hash",
        "student_prompt_sha256",
        "reference_policy",
        "experiment_config_sha256",
    ):
        require_equal(lock.get(field), manifest.get(field), f"training lock {field}")
    snapshot_root = training_run / "implementation_source_snapshot"
    files = lock.get("files") or {}
    if not files:
        raise ValueError("training implementation lock has no files")
    for relative, expected_hash in sorted(files.items()):
        snapshot = snapshot_root / relative
        if not snapshot.is_file():
            raise ValueError(f"training source snapshot is missing: {snapshot}")
        require_equal(sha256_file(snapshot), expected_hash, f"training snapshot {relative}")
    precision = load_object(precision_path)
    require_equal(
        precision.get("schema_version"),
        "table-agent-trl-training-precision-v1",
        "training precision schema",
    )
    require_equal(precision.get("optimizer_name"), "adamw_torch", "training optimizer")
    trainable = precision.get("trainable_parameters") or {}
    optimizer = precision.get("optimizer_state") or {}
    if int(trainable.get("trainable_tensors") or 0) < 1 or trainable.get("non_fp32_tensors"):
        raise ValueError("final trainable adapter tensors are not all FP32")
    if int(optimizer.get("moment_tensors") or 0) < 1 or optimizer.get("non_fp32_moments"):
        raise ValueError("final Adam moments are missing or not all FP32")
    checkpoint = training_run / expected["checkpoint_name"]
    final = training_run / "final"
    state = load_object(checkpoint / "trainer_state.json")
    require_equal(int(state.get("global_step", -1)), expected["global_step"], "final checkpoint step")
    checkpoint_report = verify_adapter(checkpoint, base_model, max_rank=64)
    final_report = verify_adapter(final, base_model, max_rank=64)
    primary_checkpoint = expected["checkpoint_name"]
    require_equal(
        final_report["adapter_sha256"],
        checkpoint_report["adapter_sha256"],
        f"final/{primary_checkpoint} adapter weights",
    )
    require_equal(
        final_report["adapter_config_sha256"],
        checkpoint_report["adapter_config_sha256"],
        f"final/{primary_checkpoint} adapter config",
    )
    require_equal(
        final_report["adapter_config_semantic_sha256"],
        sft1["adapter_config_semantic_sha256"],
        "SFT1/final adapter config semantics",
    )
    if final_report["adapter_sha256"] == sft1["adapter_sha256"]:
        raise ValueError("final adapter weights equal SFT1; distinct-policy gate failed")
    checkpoints = sorted(
        int(path.name.removeprefix("checkpoint-"))
        for path in training_run.glob("checkpoint-*")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    )
    if not checkpoints or checkpoints[-1] != expected["global_step"]:
        raise ValueError(f"training has an unexpected final checkpoint sequence: {checkpoints}")
    return {
        **final_report,
        "checkpoint_path": str(checkpoint),
        "checkpoint_global_step": expected["global_step"],
        "run_manifest_sha256": sha256_file(manifest_path),
        "implementation_lock_sha256": sha256_file(lock_path),
        "training_precision_sha256": sha256_file(precision_path),
    }


def package_versions() -> dict[str, str]:
    result: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "vllm", "peft"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def gpu_memory_used_mib(gpu_id: int) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"cannot query physical GPU {gpu_id}: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    try:
        return int(completed.stdout.strip())
    except ValueError as exc:
        raise ValueError(f"invalid GPU memory reading: {completed.stdout!r}") from exc


def require_idle_gpu(gpu_id: int, maximum_used_mib: int) -> int:
    used = gpu_memory_used_mib(gpu_id)
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"cannot query compute processes on physical GPU {gpu_id}: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    compute_pids = [
        int(line.strip())
        for line in completed.stdout.splitlines()
        if line.strip().isdigit()
    ]
    if compute_pids:
        raise ValueError(
            f"physical GPU {gpu_id} has compute processes {compute_pids}; "
            "this launcher never waits or preempts"
        )
    if used > maximum_used_mib:
        raise ValueError(
            f"physical GPU {gpu_id} is not safely idle: used={used} MiB, "
            f"limit={maximum_used_mib} MiB; this launcher never waits or preempts"
        )
    return used


@contextmanager
def exclusive_gpu_lock(gpu_id: int) -> Iterator[None]:
    path = Path(f"/tmp/table-agent-formal-v26-eval-gpu-{gpu_id}.lock")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(f"formal evaluation GPU lock is already held: {path}") from exc
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()} started={utc_now()}\n".encode())
        yield
    finally:
        os.close(descriptor)


def require_free_port(port: int) -> None:
    if not 1024 <= port <= 65535:
        raise ValueError(f"port must be in [1024, 65535], got {port}")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        try:
            listener.bind(("127.0.0.1", port))
        except OSError as exc:
            raise ValueError(f"localhost port {port} is already in use") from exc


def wait_for_released_port(
    port: int,
    *,
    timeout_seconds: float = 120.0,
    poll_interval_seconds: float = 1.0,
) -> dict[str, Any]:
    """Wait only for a port used by the preceding owned evaluation arm.

    The initial run admission remains fail-fast through ``require_free_port``.
    This bounded barrier exists solely between the sequential SFT1 and final
    arms, where recently closed loopback connections can transiently retain
    the fixed serving port.  It never discovers, signals, or preempts a
    process; a port that remains occupied fails closed at the deadline.
    """

    if not 1024 <= port <= 65535:
        raise ValueError(f"port must be in [1024, 65535], got {port}")
    if timeout_seconds <= 0:
        raise ValueError("port release timeout_seconds must be positive")
    if poll_interval_seconds <= 0:
        raise ValueError("port release poll_interval_seconds must be positive")
    started_at_utc = utc_now()
    started = time.monotonic()
    deadline = started + timeout_seconds
    attempts = 0
    last_error = ""
    while True:
        attempts += 1
        try:
            require_free_port(port)
        except ValueError as exc:
            last_error = str(exc)
            now = time.monotonic()
            remaining = deadline - now
            if remaining <= 0:
                raise TimeoutError(
                    f"localhost port {port} did not become free within "
                    f"{timeout_seconds:g} seconds after owned-arm cleanup; "
                    "no process was signaled or preempted; "
                    f"last error: {last_error}"
                ) from exc
            time.sleep(min(poll_interval_seconds, remaining))
            continue
        finished = time.monotonic()
        return {
            "schema_version": "qwen3-v26-inter-arm-port-release-v1",
            "started_at_utc": started_at_utc,
            "finished_at_utc": utc_now(),
            "port": port,
            "attempts": attempts,
            "waited_seconds": round(max(0.0, finished - started), 6),
            "timeout_seconds": timeout_seconds,
            "poll_interval_seconds": poll_interval_seconds,
            "policy": "bounded_wait_after_owned_cleanup_no_signal_no_preemption",
        }


def build_vllm_command(
    python: Path,
    model: Path,
    adapter: Path,
    *,
    port: int,
    served_model: str,
    backbone_name: str,
) -> list[str]:
    return [
        str(python),
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(model),
        "--tokenizer",
        str(model),
        "--served-model-name",
        backbone_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "16384",
        "--max-num-batched-tokens",
        "16384",
        "--max-num-seqs",
        "24",
        "--gpu-memory-utilization",
        "0.90",
        "--generation-config",
        "vllm",
        "--enable-lora",
        "--lora-modules",
        f"{served_model}={adapter}",
        "--max-lora-rank",
        "64",
    ]


def build_evaluator_command(
    python: Path,
    wrapper: Path,
    derived: Path,
    result: Path,
    *,
    port: int,
    served_model: str,
) -> list[str]:
    return [
        str(python),
        "-u",
        str(wrapper),
        "--base-url",
        f"http://127.0.0.1:{port}/v1",
        "--model",
        served_model,
        "--examples-json",
        str(derived),
        "--allow-eval-tasks",
        "--n",
        "1534",
        "--n-samples",
        "1",
        "--pass-k",
        "1",
        "--workers",
        "24",
        "--sample-workers",
        "1",
        "--first-sample-workers",
        "0",
        "--max-inflight-requests",
        "24",
        "--max-steps",
        "30",
        "--max-tokens",
        "2048",
        "--temperature",
        "0",
        "--top-p",
        "1",
        "--api-retries",
        "3",
        "--few-shot",
        "0",
        "--sample-detail",
        "full",
        "--summary-every",
        "10",
        "--context-mode",
        "rolling-legal-history",
        "--history-turns",
        "4",
        "--rolling-prompt-variant",
        "full",
        "--rolling-observation-style",
        "resident",
        "--denotation-comparison",
        "bird-set",
        "--result-dir",
        str(result),
    ]


def wait_for_model(
    port: int, served_model: str, process: subprocess.Popen[Any], timeout_seconds: int
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error = "not ready"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"owned vLLM exited before readiness: {process.returncode}")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/v1/models", timeout=5
            ) as response:
                payload = json.loads(response.read())
            models = {
                item.get("id")
                for item in payload.get("data", [])
                if isinstance(item, dict)
            }
            if served_model in models:
                return
            last_error = f"served adapter absent; advertised={sorted(x for x in models if x)}"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(3)
    raise TimeoutError(f"vLLM readiness timed out: {last_error}")


def stop_owned_process_group(process: subprocess.Popen[Any] | None) -> dict[str, Any]:
    if process is None:
        return {"requested": False, "stopped": True}
    report: dict[str, Any] = {"requested": True, "pid": process.pid}
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=15)
            report["forced_kill"] = True
    report["returncode"] = process.poll()
    report["stopped"] = process.poll() is not None
    return report




def sample_api_error(sample: dict[str, Any]) -> str | None:
    failure = str(sample.get("failure_type") or "").casefold()
    error = str(sample.get("error") or "").casefold()
    if failure in {"api_error", "chatapierror"} or "chatapierror" in error:
        return failure or "ChatAPIError"
    return None


def validate_result(
    contract: dict[str, Any],
    result_dir: Path,
    input_rows: Sequence[dict[str, Any]],
    *,
    derived: Path,
    served_model: str,
    port: int,
) -> dict[str, Any]:
    all_path = result_dir / "all.jsonl"
    manifest_path = result_dir / "manifest.json"
    summary_path = result_dir / "summary.json"
    for path in (all_path, manifest_path, summary_path):
        if not path.is_file():
            raise ValueError(f"evaluation artifact is missing: {path}")
    rows = load_jsonl(all_path)
    expected_count = contract["input"]["records"]
    require_equal(len(rows), expected_count, "evaluation result count")
    indexed: dict[int, dict[str, Any]] = {}
    api_errors: list[dict[str, Any]] = []
    for row in rows:
        index = int(row.get("example_index", -1))
        if index in indexed:
            raise ValueError(f"duplicate evaluation result index: {index}")
        if index < 0 or index >= expected_count:
            raise ValueError(f"unexpected evaluation result index: {index}")
        indexed[index] = row
        source = input_rows[index]
        for field in ("db_id", "question", "gold_sql", "trajectory_id"):
            require_equal(row.get(field), source.get(field), f"result q{index} {field}")
        require_equal(
            row.get("dataset_split"),
            source.get("dataset_split") or source.get("split"),
            f"result q{index} dataset_split",
        )
        samples = row.get("samples")
        if not isinstance(samples, list) or len(samples) != 1:
            raise ValueError(f"q{index} does not contain exactly one sample")
        sample = samples[0]
        if not isinstance(sample, dict):
            raise ValueError(f"q{index} sample is not an object")
        require_equal(sample.get("sample_index"), 0, f"q{index} sample index")
        for field in ("correct", "legal", "steps", "failure_type", "turns"):
            if field not in sample:
                raise ValueError(f"q{index} sample lacks {field}")
        reason = sample_api_error(sample)
        if reason:
            api_errors.append({"example_index": index, "reason": reason})
        top_level_reason = sample_api_error(row)
        if top_level_reason:
            api_errors.append(
                {"example_index": index, "reason": f"top_level_{top_level_reason}"}
            )
        for field, value in {
            "protocol_version": "version26",
            "protocol_hash": "4da19387399bd3a5",
            "temperature": 0.0,
            "top_p": 1.0,
            "denotation_comparison": "bird-set",
            "n_samples": 1,
            "pass_k": [1],
            "max_steps": 30,
            "max_tokens": 2048,
        }.items():
            require_equal(row.get(field), value, f"result q{index} {field}")
        require_equal(
            bool(row.get("correct")),
            bool(sample.get("correct")),
            f"result q{index} correct aggregation",
        )
        require_equal(
            row.get("pass_at"), {"1": bool(sample.get("correct"))}, f"result q{index} pass@1"
        )
    require_equal(sorted(indexed), list(range(expected_count)), "result cohort indices")
    if api_errors:
        raise ValueError(f"evaluation contains API errors: {api_errors[:10]}")
    manifest = load_object(manifest_path)
    expected_manifest = {
        "runner": "tool_rollout_passk",
        "tool_scheme": "atomic",
        "assistant_carrier": "think-json-v1",
        "protocol_version": "version26",
        "protocol_hash": "4da19387399bd3a5",
        "model": served_model,
        "base_url": f"http://127.0.0.1:{port}/v1",
        "dataset": str(derived),
        "requested_size": 1534,
        "n_samples": 1,
        "sample_workers": 1,
        "first_sample_workers": 0,
        "max_inflight_requests": 24,
        "stop_on_success": False,
        "pass_k": [1],
        "sample_detail": "full",
        "summary_every": 10,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "max_steps": 30,
        "api_retries": 3,
        "few_shot": 0,
        "enable_thinking": "1",
        "context_mode": "rolling-legal-history",
        "history_turns": 4,
        "rolling_prompt_variant": "full",
        "rolling_observation_style": "resident",
        "denotation_comparison": "bird-set",
    }
    for field, expected in expected_manifest.items():
        require_equal(manifest.get(field), expected, f"evaluation manifest {field}")
    require_equal(
        hashlib.sha256(manifest["system_prompt"].encode("utf-8")).hexdigest(),
        contract["runtime"]["student_prompt_sha256"],
        "evaluation system prompt",
    )
    unhashed_manifest = {
        key: value
        for key, value in manifest.items()
        if key not in {"config_sha256", "created_at_utc"}
    }
    expected_config_hash = hashlib.sha256(
        json.dumps(
            unhashed_manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    require_equal(manifest.get("config_sha256"), expected_config_hash, "manifest config hash")
    summary = load_object(summary_path)
    require_equal(summary.get("total"), expected_count, "evaluation summary total")
    recomputed_correct = sum(bool(row["samples"][0]["correct"]) for row in rows)
    recomputed_legal = sum(bool(row["samples"][0]["legal"]) for row in rows)
    require_equal(summary.get("correct"), recomputed_correct, "evaluation summary correct")
    require_equal(
        summary.get("completed_example_indices"),
        list(range(expected_count)),
        "evaluation summary completed indices",
    )
    require_equal(
        (summary.get("pass_at") or {}).get("1", {}).get("total"),
        expected_count,
        "evaluation pass@1 total",
    )
    require_equal(
        (summary.get("pass_at") or {}).get("1", {}).get("correct"),
        recomputed_correct,
        "evaluation pass@1 correct",
    )
    require_equal(
        summary.get("average_legal_samples"),
        recomputed_legal / expected_count,
        "evaluation summary legal",
    )
    return {
        "records": len(rows),
        "correct": (summary.get("pass_at") or {}).get("1", {}).get("correct"),
        "legal": recomputed_legal,
        "api_errors": 0,
        "all_jsonl_sha256": sha256_file(all_path),
        "manifest_sha256": sha256_file(manifest_path),
        "summary_sha256": sha256_file(summary_path),
    }


def run_arm(
    contract: dict[str, Any],
    *,
    arm: str,
    adapter: dict[str, Any],
    runtime: Path,
    model: Path,
    derived: Path,
    input_rows: Sequence[dict[str, Any]],
    run_root: Path,
    python: Path,
    gpu_id: int,
    port: int,
    maximum_used_mib: int,
    model_ready_timeout: int,
    versions: dict[str, str],
    implementation: dict[str, str],
) -> dict[str, Any]:
    arm_root = run_root / arm
    if arm_root.exists():
        raise ValueError(f"formal arm already exists; resume is forbidden: {arm_root}")
    arm_root.mkdir()
    result_dir = arm_root / "result"
    served_model = f"qwen3-8b-atomic-v26-formal-{arm}"
    backbone = f"qwen3-8b-atomic-v26-formal-{arm}-backbone"
    identity = evaluation_identity(
        contract,
        arm=arm,
        adapter=adapter,
        runtime=runtime,
        model=model,
        derived=derived,
        versions=versions,
        implementation=implementation,
        gpu_id=gpu_id,
        port=port,
        served_model=served_model,
    )
    identity_path = arm_root / "evaluation_identity.json"
    atomic_json(identity_path, identity)
    identity_hash_before = sha256_file(identity_path)
    vllm_command = build_vllm_command(
        python,
        model,
        Path(adapter["path"]),
        port=port,
        served_model=served_model,
        backbone_name=backbone,
    )
    evaluator_command = build_evaluator_command(
        python,
        HERE / "formal_v26_rollout_passk.py",
        derived,
        result_dir,
        port=port,
        served_model=served_model,
    )
    initial_memory = require_idle_gpu(gpu_id, maximum_used_mib)
    require_free_port(port)
    atomic_json(
        arm_root / "launch_manifest.json",
        {
            "schema_version": "qwen3-v26-formal-arm-launch-v1",
            "created_at_utc": utc_now(),
            "arm": arm,
            "physical_gpu_id": gpu_id,
            "initial_gpu_memory_mib": initial_memory,
            "identity_sha256": identity_hash_before,
            "vllm_command": vllm_command,
            "evaluator_command": evaluator_command,
        },
    )
    vllm_environment = os.environ.copy()
    vllm_environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(gpu_id),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    eval_environment = os.environ.copy()
    eval_environment.pop("PYTHONPATH", None)
    eval_environment.pop("EVAL_SYSTEM_PROMPT_VARIANT", None)
    eval_environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "EVAL_ENABLE_THINKING": "1",
            "TABLE_AGENT_PROTOCOL_RUNTIME_ROOT": str(runtime),
            "FORMAL_TOOL_EXECUTION_TIMEOUT_SECONDS": "10",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    process: subprocess.Popen[Any] | None = None
    evaluator_process: subprocess.Popen[Any] | None = None
    try:
        with (arm_root / "vllm.log").open("wb") as log:
            process = subprocess.Popen(
                vllm_command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=vllm_environment,
                start_new_session=True,
            )
        (arm_root / "vllm.pid").write_text(f"{process.pid}\n", encoding="ascii")
        wait_for_model(port, served_model, process, model_ready_timeout)
        with (arm_root / "evaluation.log").open("wb") as log:
            evaluator_process = subprocess.Popen(
                evaluator_command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=eval_environment,
                start_new_session=True,
            )
            evaluator_returncode = evaluator_process.wait()
        if evaluator_returncode != 0:
            raise RuntimeError(
                f"formal evaluator for {arm} exited {evaluator_returncode}; "
                f"see {arm_root / 'evaluation.log'}"
            )
        result = validate_result(
            contract,
            result_dir,
            input_rows,
            derived=derived,
            served_model=served_model,
            port=port,
        )
        require_equal(sha256_file(identity_path), identity_hash_before, f"{arm} identity immutability")
        return {"identity": str(identity_path), "result_dir": str(result_dir), **result}
    finally:
        evaluator_cleanup = stop_owned_process_group(evaluator_process)
        vllm_cleanup = stop_owned_process_group(process)
        cleanup = {
            "evaluator": evaluator_cleanup,
            "vllm": vllm_cleanup,
        }
        atomic_json(arm_root / "owned_process_cleanup.json", cleanup)
        if not evaluator_cleanup.get("stopped") or not vllm_cleanup.get("stopped"):
            raise RuntimeError(f"owned process cleanup failed for {arm}: {cleanup}")


def run_analysis(
    contract: dict[str, Any], run_root: Path, derived: Path, adapters: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    analysis_path = run_root / "formal_matched_analysis.json"
    gate_path = run_root / "formal_matched_gate.json"
    command = [
        sys.executable,
        "-m",
        "rl.scenarios.diagnostics.analyze_evaluation_results",
        "--examples",
        str(derived),
        "--arm",
        f"sft1={run_root / 'sft1/result/all.jsonl'}",
        "--arm",
        f"final={run_root / 'final/result/all.jsonl'}",
        "--identity",
        f"sft1={run_root / 'sft1/evaluation_identity.json'}",
        "--identity",
        f"final={run_root / 'final/evaluation_identity.json'}",
        "--adapter-sha",
        f"sft1={adapters['sft1']['adapter_sha256']}",
        "--adapter-sha",
        f"final={adapters['final']['adapter_sha256']}",
        "--require-identities",
        "--require-distinct-adapters",
    ]
    for field in MATCHED_IDENTITY_FIELDS:
        command.extend(("--match-identity-field", field))
    command.extend(
        (
            "--compare",
            "final:sft1",
            "--expected-count",
            "1534",
            "--protocol-version",
            "version26",
            "--protocol-hash",
            "4da19387399bd3a5",
            "--temperature",
            "0",
            "--top-p",
            "1",
            "--denotation-comparison",
            "bird-set",
            "--output",
            str(analysis_path),
        )
    )
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        command, check=False, env=environment, cwd=PROJECT_ROOT
    )
    if completed.returncode != 0:
        raise RuntimeError(f"formal matched analyzer exited {completed.returncode}")
    gate_contract = contract["promotion_gate"]
    gate_command = [
        sys.executable,
        "-m",
        "rl.scenarios.diagnostics.audit_vanilla_grpo_matched_gate",
        "--analysis",
        str(analysis_path),
        "--baseline",
        "sft1",
        "--candidate",
        "final",
        "--primary-candidate",
        "final",
        "--expected-count",
        str(gate_contract["expected_count"]),
        "--min-accuracy-gain-pp",
        str(gate_contract["minimum_accuracy_gain_percentage_points"]),
        "--alpha",
        str(gate_contract["exact_mcnemar_alpha"]),
        "--output",
        str(gate_path),
    ]
    completed = subprocess.run(
        gate_command, check=False, env=environment, cwd=PROJECT_ROOT
    )
    if completed.returncode != 0:
        raise RuntimeError(f"formal matched gate exited {completed.returncode}")
    analysis = load_object(analysis_path)
    gate = load_object(gate_path)
    if (gate.get("status") or {}).get("matched_evaluation_valid") is not True:
        raise RuntimeError("formal matched gate did not validate evaluation identity")
    return analysis, gate


def verify_host_assets(
    contract: dict[str, Any], *, gpu_id: int | None = None, maximum_used_mib: int = 512
) -> dict[str, Any]:
    paths = {name: Path(value) for name, value in contract["host_paths"].items()}
    python = paths["python"]
    if not python.is_file():
        raise ValueError(f"evaluation Python is missing: {python}")
    require_equal(Path(sys.executable).resolve(), python.resolve(), "evaluation Python executable")
    runtime = verify_runtime(contract, paths["runtime"])
    model = verify_model(contract, paths["base_model"])
    source_rows, databases = verify_source_and_databases(
        contract, paths["source_input"], paths["database_root"]
    )
    sft1 = verify_adapter(
        paths["sft1_adapter"],
        paths["base_model"],
        max_rank=contract["serving"]["max_lora_rank"],
        expected_files=contract["sft1"]["files_sha256"],
        expected_global_step=contract["sft1"]["global_step"],
    )
    require_equal(sft1["rank"], contract["sft1"]["rank"], "SFT1 LoRA rank")
    final = verify_training_final(
        contract, paths["training_run"], paths["base_model"], sft1
    )
    report: dict[str, Any] = {
        "runtime": runtime,
        "model": model,
        "source_records": len(source_rows),
        "source_rows": source_rows,
        "databases": databases,
        "adapters": {"sft1": sft1, "final": final},
        "package_versions": package_versions(),
    }
    if gpu_id is not None:
        report["physical_gpu_id"] = gpu_id
        report["gpu_memory_used_mib"] = require_idle_gpu(gpu_id, maximum_used_mib)
    return report


def static_plan(contract: dict[str, Any], implementation: dict[str, str]) -> dict[str, Any]:
    return {
        "schema_version": "qwen3-v26-formal-matched-eval-plan-v1",
        "status": "launcher_generated_not_started",
        "remote_operations_performed": False,
        "arms": contract["arms"],
        "baseline": contract["baseline"],
        "primary_candidate": contract["primary_candidate"],
        "questions_per_arm": contract["evaluation"]["questions"],
        "execution": "sequential_same_physical_gpu",
        "no_resume": True,
        "gpu_policy": "fail immediately unless explicitly selected GPU is already idle",
        "serving": contract["serving"],
        "evaluation": contract["evaluation"],
        "matched_identity_fields": list(MATCHED_IDENTITY_FIELDS),
        "implementation_sha256": implementation,
        "host_paths": contract["host_paths"],
    }


def execute_run(
    contract: dict[str, Any],
    *,
    run_dir: Path,
    gpu_id: int,
    port: int,
    maximum_used_mib: int,
    model_ready_timeout: int,
    implementation: dict[str, str],
) -> int:
    paths = {name: Path(value) for name, value in contract["host_paths"].items()}
    allowed_parent = paths["allowed_run_parent"].resolve()
    run_dir = run_dir.resolve()
    if run_dir.parent != allowed_parent:
        raise ValueError(f"run directory must be a direct child of {allowed_parent}: {run_dir}")
    if run_dir.exists():
        raise ValueError(f"formal run directory already exists; resume is forbidden: {run_dir}")
    assets = verify_host_assets(
        contract, gpu_id=gpu_id, maximum_used_mib=maximum_used_mib
    )
    require_free_port(port)
    allowed_parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir()
    status_path = run_dir / "status.json"
    status: dict[str, Any] = {
        "schema_version": "qwen3-v26-formal-matched-eval-status-v1",
        "state": "running",
        "success": False,
        "stage": "preparing_input",
        "started_at_utc": utc_now(),
        "run_dir": str(run_dir),
        "physical_gpu_id": gpu_id,
        "port": port,
    }
    atomic_json(status_path, status)

    def update(**values: Any) -> None:
        status.update(values)
        atomic_json(status_path, status)

    try:
        input_dir = run_dir / "input"
        derived = input_dir / "bird_dev_20240627.newgnn.jsonl"
        input_manifest = input_dir / "bird_dev_20240627.newgnn.manifest.json"
        prepare_derived_input(
            contract,
            assets.pop("source_rows"),
            paths["database_root"],
            derived,
            input_manifest,
        )
        atomic_json(
            run_dir / "preflight_gate.json",
            {
                key: value
                for key, value in assets.items()
                if key not in {"adapters"}
            }
            | {
                "adapters": assets["adapters"],
                "contract_canonical_sha256": hashlib.sha256(
                    canonical_json_bytes(contract)
                ).hexdigest(),
                "implementation_sha256": implementation,
                "derived_input_sha256": sha256_file(derived),
                "derived_manifest_sha256": sha256_file(input_manifest),
            },
        )
        arm_results: dict[str, Any] = {}
        for arm_index, arm in enumerate(contract["arms"]):
            if arm_index:
                update(stage=f"waiting_for_port_release_before_{arm}")
                atomic_json(
                    run_dir / f"inter_arm_port_release_before_{arm}.json",
                    wait_for_released_port(port),
                )
            update(stage=f"evaluating_{arm}")
            arm_results[arm] = run_arm(
                contract,
                arm=arm,
                adapter=assets["adapters"][arm],
                runtime=paths["runtime"],
                model=paths["base_model"],
                derived=derived,
                input_rows=load_jsonl(derived),
                run_root=run_dir,
                python=paths["python"],
                gpu_id=gpu_id,
                port=port,
                maximum_used_mib=maximum_used_mib,
                model_ready_timeout=model_ready_timeout,
                versions=assets["package_versions"],
                implementation=implementation,
            )
        update(stage="analyzing")
        analysis, gate = run_analysis(
            contract, run_dir, derived, assets["adapters"]
        )
        update(
            state="completed",
            success=True,
            stage="completed",
            finished_at_utc=utc_now(),
            arms=arm_results,
            accuracy={
                name: {
                    "correct": values["correct"],
                    "total": values["total"],
                }
                for name, values in analysis["arms"].items()
            },
            gate_status=gate["status"],
            decision=(
                "promote_final"
                if gate["status"].get("promote") is True
                else "do_not_promote_final"
            ),
        )
        return 0
    except Exception as exc:
        update(
            state="failed",
            success=False,
            finished_at_utc=utc_now(),
            failure={
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, preflight, or run the formal Qwen3-8B SFT1-vs-final matched gate."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-plan", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--port", type=int, default=8087)
    parser.add_argument("--maximum-used-mib", type=int, default=512)
    parser.add_argument("--model-ready-timeout", type=int, default=900)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    contract = load_object(args.contract)
    validate_contract_shape(contract)
    implementation = verify_implementation(contract)
    if args.print_plan:
        print(json.dumps(static_plan(contract, implementation), ensure_ascii=False, indent=2))
        return 0
    if args.gpu_id is None or args.gpu_id < 0:
        raise SystemExit("--gpu-id is required and must be non-negative")
    with exclusive_gpu_lock(args.gpu_id):
        if args.preflight:
            report = verify_host_assets(
                contract,
                gpu_id=args.gpu_id,
                maximum_used_mib=args.maximum_used_mib,
            )
            report.pop("source_rows", None)
            print(json.dumps({"status": "preflight_ok", **report}, ensure_ascii=False, indent=2))
            return 0
        if args.run_dir is None:
            raise SystemExit("--run requires --run-dir")

        def terminate(signum: int, _frame: Any) -> None:
            raise TerminationRequested(f"received {signal.Signals(signum).name}")

        signal.signal(signal.SIGTERM, terminate)
        signal.signal(signal.SIGINT, terminate)
        return execute_run(
            contract,
            run_dir=args.run_dir,
            gpu_id=args.gpu_id,
            port=args.port,
            maximum_used_mib=args.maximum_used_mib,
            model_ready_timeout=args.model_ready_timeout,
            implementation=implementation,
        )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # fail closed before a status artifact exists
        print(f"formal matched evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
