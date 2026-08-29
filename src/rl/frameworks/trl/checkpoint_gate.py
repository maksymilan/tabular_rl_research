#!/usr/bin/env python3
"""Synchronous, source-locked checkpoint gate execution helpers."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file: {path}")


def _publish_immutable_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require_regular(path, "checkpoint gate invocation evidence")
        if path.read_bytes() != encoded:
            raise RuntimeError(f"checkpoint gate evidence already differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require_regular(path, "checkpoint gate invocation evidence")
            if path.read_bytes() != encoded:
                raise RuntimeError(f"checkpoint gate evidence raced with different bytes: {path}")
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True)
class CheckpointGateSpec:
    step: int
    script: Path
    script_relative_path: str
    script_sha256: str
    receipt: Path
    tasks: Path
    tasks_manifest: Path
    initial_adapter: Path
    output_dir: Path

    def with_script(self, script: Path) -> "CheckpointGateSpec":
        return replace(self, script=script)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "trl-synchronous-checkpoint-gate-v1",
            "step": self.step,
            "script_relative_path": self.script_relative_path,
            "script_sha256": self.script_sha256,
            "receipt": str(self.receipt),
            "tasks_manifest": str(self.tasks_manifest),
            "tasks_manifest_sha256": sha256_file(self.tasks_manifest),
            "execution": "synchronous_on_save_before_next_optimizer_step",
            "resume_policy": "checkpoint_at_or_after_gate_requires_verified_receipt",
        }


def build_checkpoint_gate_spec(
    *,
    project_root: Path,
    output_dir: Path,
    optimizer_steps: int,
    save_steps: int,
    step: int | None,
    script: Path | None,
    receipt: Path | None,
    tasks: Path | None,
    tasks_manifest: Path | None,
    initial_adapter: Path,
) -> CheckpointGateSpec | None:
    values = (step, script, receipt, tasks_manifest)
    enabled = any(value not in (None, 0) for value in values)
    if not enabled:
        if any(value is not None for value in (script, receipt, tasks_manifest)) or step not in (None, 0):
            raise ValueError("checkpoint gate arguments must be omitted together")
        return None
    if step is None or step <= 0 or step >= optimizer_steps:
        raise ValueError("checkpoint gate step must be positive and precede optimizer completion")
    if step % save_steps != 0:
        raise ValueError("checkpoint gate step must coincide with a saved checkpoint")
    if script is None or receipt is None or tasks is None or tasks_manifest is None:
        raise ValueError("checkpoint gate requires script, receipt, tasks, and tasks manifest")

    root = project_root.resolve()
    script = script.resolve()
    output_dir = output_dir.resolve()
    receipt = receipt.resolve()
    tasks = tasks.resolve()
    tasks_manifest = tasks_manifest.resolve()
    initial_adapter = initial_adapter.resolve()
    _require_regular(script, "checkpoint gate script")
    _require_regular(tasks, "checkpoint gate tasks")
    _require_regular(tasks_manifest, "checkpoint gate tasks manifest")
    _require_regular(initial_adapter / "adapter_model.safetensors", "initial adapter weights")
    try:
        relative = script.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("checkpoint gate script must live inside the source-locked project") from exc
    if receipt.parent != output_dir:
        raise ValueError("checkpoint gate receipt must be a direct child of output-dir")
    if receipt.exists() and (not receipt.is_file() or receipt.is_symlink()):
        raise ValueError(f"invalid checkpoint gate receipt: {receipt}")
    return CheckpointGateSpec(
        step=step,
        script=script,
        script_relative_path=relative,
        script_sha256=sha256_file(script),
        receipt=receipt,
        tasks=tasks,
        tasks_manifest=tasks_manifest,
        initial_adapter=initial_adapter,
        output_dir=output_dir,
    )


def run_checkpoint_gate(
    spec: CheckpointGateSpec,
    *,
    verify_existing: bool,
) -> dict[str, Any]:
    _require_regular(spec.script, "checkpoint gate executable snapshot")
    actual_script_sha = sha256_file(spec.script)
    if actual_script_sha != spec.script_sha256:
        raise RuntimeError(
            "checkpoint gate script hash changed: "
            f"{actual_script_sha} != {spec.script_sha256}"
        )
    command = [
        sys.executable,
        str(spec.script),
        "--run-dir",
        str(spec.output_dir),
        "--tasks",
        str(spec.tasks),
        "--tasks-manifest",
        str(spec.tasks_manifest),
        "--initial-adapter",
        str(spec.initial_adapter),
        "--output",
        str(spec.receipt),
    ]
    if verify_existing:
        command.append("--verify-existing")
    completed = subprocess.run(
        command,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=False,
    )
    receipt_sha256 = None
    if spec.receipt.is_file() and not spec.receipt.is_symlink():
        receipt_sha256 = sha256_file(spec.receipt)
    evidence = {
        "schema_version": "trl-checkpoint-gate-invocation-v1",
        "step": spec.step,
        "verify_existing": verify_existing,
        "script_relative_path": spec.script_relative_path,
        "script_sha256": spec.script_sha256,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "receipt": str(spec.receipt),
        "receipt_sha256": receipt_sha256,
    }
    suffix = "resume_verification" if verify_existing else "invocation"
    evidence_path = spec.output_dir / f"checkpoint_gate_step{spec.step}_{suffix}.json"
    _publish_immutable_json(evidence_path, evidence)
    if completed.returncode != 0:
        raise RuntimeError(
            f"checkpoint gate failed at step {spec.step} with exit {completed.returncode}; "
            f"see {evidence_path}"
        )
    _require_regular(spec.receipt, "checkpoint gate receipt")
    if evidence["receipt_sha256"] != sha256_file(spec.receipt):
        raise RuntimeError("checkpoint gate receipt changed while recording evidence")
    return evidence
