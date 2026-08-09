#!/usr/bin/env python3
"""Fail-closed asset gate for the all-NewGNN Qwen3/version26 evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_LOCK = HERE / "remote_eval_lock.json"
DEFAULT_RUNTIME_LOCK = HERE / "runtime_lock.json"
DEFAULT_RUNTIME_VERIFIER = HERE / "verify_version26_runtime.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"row {line_number} in {path} is not an object")
            rows.append(row)
    return rows


def verify_runtime(
    runtime_root: Path,
    runtime_lock: Path,
    runtime_verifier: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            str(runtime_verifier),
            "--runtime-root",
            str(runtime_root),
            "--lock",
            str(runtime_lock),
            "--json",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ValueError(f"exact version26 runtime gate failed: {detail}")
    report = json.loads(completed.stdout)
    checks = {
        "source_commit": report.get("source_commit"),
        "content_tree_sha256": report.get("content_tree_sha256"),
        "rolling_system_prompt_sha256": (report.get("protocol") or {}).get(
            "rolling_system_prompt_sha256"
        ),
    }
    for key, actual in checks.items():
        if actual != expected[key]:
            raise ValueError(
                f"runtime {key} mismatch: expected {expected[key]!r}, got {actual!r}"
            )
    if (report.get("protocol") or {}).get("protocol_version") != "version26":
        raise ValueError("runtime protocol_version is not version26")
    if (report.get("protocol") or {}).get("tool_scheme") != "atomic":
        raise ValueError("runtime tool_scheme is not atomic")
    return report


def verify_input(
    source: Path,
    derived: Path,
    manifest_path: Path,
    db_root: Path,
    expected: dict[str, Any],
) -> dict[str, Any]:
    hashes = {
        "source_sha256": sha256(source),
        "derived_sha256": sha256(derived),
        "manifest_sha256": sha256(manifest_path),
    }
    for key, actual in hashes.items():
        if actual != expected[key]:
            raise ValueError(
                f"evaluation {key} mismatch: expected {expected[key]!r}, got {actual!r}"
            )
    source_rows = load_jsonl(source)
    derived_rows = load_jsonl(derived)
    if len(source_rows) != expected["records"] or len(derived_rows) != expected["records"]:
        raise ValueError(
            f"evaluation row-count mismatch: source={len(source_rows)} "
            f"derived={len(derived_rows)} expected={expected['records']}"
        )
    expected_root = Path(expected["remote_db_root"])
    # Keep the model-visible/eval-JSON path lexical.  On NewGNN this stable /home path is a
    # symlink into /data; resolving it would incorrectly reject the intended mount alias.
    normalized_db_root = Path(os.path.abspath(db_root))
    if normalized_db_root != expected_root:
        raise ValueError(f"DB root mismatch: expected {expected_root}, got {normalized_db_root}")

    observed_counts: dict[str, int] = {name: 0 for name in expected["databases"]}
    for ordinal, (source_row, derived_row) in enumerate(zip(source_rows, derived_rows, strict=True)):
        source_without_path = {k: v for k, v in source_row.items() if k != "db_path"}
        derived_without_path = {k: v for k, v in derived_row.items() if k != "db_path"}
        if source_without_path != derived_without_path:
            raise ValueError(f"non-db_path field drift at row {ordinal}")
        db_id = derived_row.get("db_id")
        if db_id not in expected["databases"]:
            raise ValueError(f"unexpected db_id at row {ordinal}: {db_id!r}")
        expected_path = str(expected_root / db_id / f"{db_id}.sqlite")
        if derived_row.get("db_path") != expected_path:
            raise ValueError(
                f"derived db_path mismatch at row {ordinal}: "
                f"expected {expected_path!r}, got {derived_row.get('db_path')!r}"
            )
        observed_counts[db_id] += 1

    manifest = load_object(manifest_path)
    if manifest.get("source_sha256") != expected["source_sha256"]:
        raise ValueError("input manifest source hash is not pinned")
    if manifest.get("derived_sha256") != expected["derived_sha256"]:
        raise ValueError("input manifest derived hash is not pinned")
    if manifest.get("records") != expected["records"]:
        raise ValueError("input manifest record count is not pinned")
    if manifest.get("remote_db_root") != expected["remote_db_root"]:
        raise ValueError("input manifest DB root is not pinned")

    databases: dict[str, Any] = {}
    for db_id, contract in sorted(expected["databases"].items()):
        path = db_root / db_id / f"{db_id}.sqlite"
        if not path.is_file():
            raise ValueError(f"database is missing: {path}")
        actual_hash = sha256(path)
        if actual_hash != contract["sha256"]:
            raise ValueError(
                f"database hash mismatch for {db_id}: expected {contract['sha256']}, "
                f"got {actual_hash}"
            )
        if observed_counts[db_id] != contract["records"]:
            raise ValueError(
                f"database row-count mismatch for {db_id}: expected {contract['records']}, "
                f"got {observed_counts[db_id]}"
            )
        manifest_entry = (manifest.get("databases") or {}).get(db_id)
        if not isinstance(manifest_entry, dict):
            raise ValueError(f"input manifest lacks database entry: {db_id}")
        if manifest_entry.get("sha256") != actual_hash:
            raise ValueError(f"input manifest database hash mismatch: {db_id}")
        databases[db_id] = {
            "path": str(path),
            "records": observed_counts[db_id],
            "sha256": actual_hash,
            "size_bytes": path.stat().st_size,
        }
    return {
        "status": "ok",
        **hashes,
        "records": len(derived_rows),
        "only_db_path_changed": True,
        "remote_db_root": str(db_root),
        "databases": databases,
    }


def verify_model_and_adapter(
    model_root: Path,
    adapter: Path | None,
    lock: dict[str, Any],
    max_lora_rank: int,
    *,
    model_size: str = "8b",
    model_specs: Path | None = None,
    adapter_lock: Path | None = None,
) -> dict[str, Any]:
    if model_size == "8b":
        from verify_qwen3_model import verify_model

        expected_model = lock["model"]
        if model_root.resolve() != Path(expected_model["path"]):
            raise ValueError(
                f"base model path mismatch: expected {expected_model['path']}, "
                f"got {model_root.resolve()}"
            )
        base_report = verify_model(model_root, adapter, max_lora_rank)
        if base_report.get("assumed_huggingface_revision") != expected_model["revision"]:
            raise ValueError("Qwen3 revision gate failed")
        shard_expectations = expected_model["shards_sha256"]
        expected_adapter = lock.get("adapter")
    elif model_size == "4b":
        if model_specs is None:
            raise ValueError("4B evaluation requires --model-specs")
        from verify_pinned_qwen3_model import verify

        specs = load_object(model_specs)
        expected_model = (specs.get("models") or {}).get("4b")
        if not isinstance(expected_model, dict):
            raise ValueError("4B model spec is missing")
        if model_root.resolve() != Path(expected_model["default_path"]):
            raise ValueError(
                f"base model path mismatch: expected {expected_model['default_path']}, "
                f"got {model_root.resolve()}"
            )
        base_report = verify(model_root, "4b", model_specs)
        base_report["assumed_huggingface_revision"] = expected_model["revision"]
        base_report["model_type"] = "qwen3"
        base_report["architecture"] = "Qwen3ForCausalLM"
        base_report["official_chat_template"] = True
        base_report["empty_think_suppression_injected"] = False
        shard_expectations = expected_model["shards_sha256"]
        expected_adapter = load_object(adapter_lock) if adapter_lock is not None else None
    else:
        raise ValueError(f"unsupported model_size: {model_size}")

    shards: dict[str, Any] = {}
    for name, expected_hash in sorted(shard_expectations.items()):
        path = model_root / name
        if not path.is_file():
            raise ValueError(f"model shard is missing: {path}")
        actual_hash = sha256(path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"model shard hash mismatch for {name}: expected {expected_hash}, got {actual_hash}"
            )
        shards[name] = {"sha256": actual_hash, "size_bytes": path.stat().st_size}

    adapter_report = None
    if adapter is not None:
        if not isinstance(expected_adapter, dict):
            raise ValueError("adapter evaluation requires a pinned adapter lock")
        resolved = adapter.resolve()
        if resolved != Path(expected_adapter["path"]):
            raise ValueError(
                f"adapter path mismatch: expected {expected_adapter['path']}, got {resolved}"
            )
        if resolved.name != expected_adapter["checkpoint_name"]:
            raise ValueError(
                f"adapter must be {expected_adapter['checkpoint_name']}, got {resolved.name}"
            )
        trainer_state = load_object(resolved / "trainer_state.json")
        if int(trainer_state.get("global_step", -1)) != expected_adapter["global_step"]:
            raise ValueError(
                f"adapter global_step mismatch: expected {expected_adapter['global_step']}, "
                f"got {trainer_state.get('global_step')!r}"
            )
        adapter_config = load_object(resolved / "adapter_config.json")
        if str(adapter_config.get("peft_type", "")).upper() != "LORA":
            raise ValueError("adapter peft_type is not LORA")
        if str(adapter_config.get("task_type", "")).upper() != "CAUSAL_LM":
            raise ValueError("adapter task_type is not CAUSAL_LM")
        rank = int(adapter_config.get("r", 0))
        if rank <= 0 or rank > max_lora_rank:
            raise ValueError(f"adapter rank {rank} exceeds max_lora_rank={max_lora_rank}")
        base_reference = adapter_config.get("base_model_name_or_path")
        if not isinstance(base_reference, str) or not base_reference:
            raise ValueError("adapter has no base_model_name_or_path")
        if Path(base_reference).is_absolute():
            if Path(base_reference).resolve() != model_root.resolve():
                raise ValueError("adapter was trained against a different local base")
        else:
            expected_repo = expected_model["repository"]
            if base_reference != expected_repo:
                raise ValueError(
                    f"adapter base repository mismatch: {base_reference} != {expected_repo}"
                )
        files: dict[str, Any] = {}
        for name, expected_hash in sorted(expected_adapter["files_sha256"].items()):
            path = resolved / name
            if not path.is_file():
                raise ValueError(f"adapter artifact is missing: {path}")
            actual_hash = sha256(path)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"adapter artifact hash mismatch for {name}: expected {expected_hash}, "
                    f"got {actual_hash}"
                )
            files[name] = {"sha256": actual_hash, "size_bytes": path.stat().st_size}
        adapter_report = {
            "path": str(resolved),
            "peft_type": "LORA",
            "rank": rank,
            "base_model_name_or_path": base_reference,
            "checkpoint_name": resolved.name,
            "global_step": expected_adapter["global_step"],
            "files": files,
        }

    base_report["model_repository"] = expected_model["repository"]
    base_report["model_revision"] = expected_model["revision"]
    base_report["model_shards"] = shards
    base_report["adapter"] = adapter_report
    return base_report


def verify_all(
    *,
    mode: str,
    source: Path,
    derived: Path,
    input_manifest: Path,
    db_root: Path,
    runtime_root: Path,
    model_root: Path,
    adapter: Path | None,
    lock_path: Path,
    runtime_lock: Path,
    runtime_verifier: Path,
    max_lora_rank: int,
    model_size: str = "8b",
    model_specs: Path | None = None,
    adapter_lock: Path | None = None,
) -> dict[str, Any]:
    lock = load_object(lock_path)
    if mode not in {"base", "adapter"}:
        raise ValueError(f"mode must be base or adapter, got {mode!r}")
    if mode == "base" and adapter is not None:
        raise ValueError("base mode must not receive an adapter")
    if mode == "adapter" and adapter is None:
        raise ValueError("adapter mode requires the pinned checkpoint-560")

    runtime_report = verify_runtime(
        runtime_root, runtime_lock, runtime_verifier, lock["runtime"]
    )
    input_report = verify_input(
        source, derived, input_manifest, db_root, lock["evaluation_input"]
    )
    model_report = verify_model_and_adapter(
        model_root,
        adapter,
        lock,
        max_lora_rank,
        model_size=model_size,
        model_specs=model_specs,
        adapter_lock=adapter_lock,
    )
    return {
        "status": "ok",
        "schema_version": "qwen3-atomic-v26-all-newgnn-assets-v1",
        "mode": mode,
        "runtime": runtime_report,
        "evaluation_input": input_report,
        "model": model_report,
        "evaluation_contract": lock["evaluation_contract"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("base", "adapter"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--derived", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--db-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--runtime-lock", type=Path, default=DEFAULT_RUNTIME_LOCK)
    parser.add_argument("--runtime-verifier", type=Path, default=DEFAULT_RUNTIME_VERIFIER)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--model-size", choices=("4b", "8b"), default="8b")
    parser.add_argument("--model-specs", type=Path)
    parser.add_argument("--adapter-lock", type=Path)
    args = parser.parse_args()
    try:
        report = verify_all(
            mode=args.mode,
            source=args.source,
            derived=args.derived,
            input_manifest=args.input_manifest,
            db_root=args.db_root,
            runtime_root=args.runtime_root,
            model_root=args.model_root,
            adapter=args.adapter,
            lock_path=args.lock,
            runtime_lock=args.runtime_lock,
            runtime_verifier=args.runtime_verifier,
            max_lora_rank=args.max_lora_rank,
            model_size=args.model_size,
            model_specs=args.model_specs,
            adapter_lock=args.adapter_lock,
        )
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"remote evaluation asset gate failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
