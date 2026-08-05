"""Atomic, bounded checkpoints for streaming teacher-union training."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "streaming-teacher-union-checkpoint-v1"


def jsonl_line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as source:
        return sum(1 for line in source if line.strip())


def truncate_jsonl(path: Path, retained_lines: int) -> None:
    """Atomically restore a JSONL log to an already checkpointed prefix."""
    if retained_lines < 0:
        raise ValueError("retained JSONL line count cannot be negative")
    temporary = path.with_name(f"{path.name}.resume.{os.getpid()}")
    observed = 0
    with temporary.open("w", encoding="utf-8") as target:
        if path.is_file():
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    if observed < retained_lines:
                        target.write(line)
                    observed += 1
                    if observed >= retained_lines:
                        break
    if observed < retained_lines:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"JSONL log is shorter than checkpoint: {path} "
            f"has {observed}, expected at least {retained_lines}"
        )
    temporary.replace(path)


def complete_checkpoints(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    values = []
    for path in root.glob("checkpoint-batch-*"):
        if path.is_dir() and (path / "COMPLETE").is_file():
            values.append(path)
    return sorted(values, key=lambda path: path.name)


def latest_complete_checkpoint(root: Path) -> Path | None:
    values = complete_checkpoints(root)
    return values[-1] if values else None


def load_checkpoint_state(path: Path) -> dict[str, Any]:
    if not (path / "COMPLETE").is_file():
        raise ValueError(f"checkpoint is incomplete: {path}")
    state = json.loads((path / "checkpoint_state.json").read_text(encoding="utf-8"))
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported checkpoint schema: {path}")
    required = {
        "run_identity_sha256",
        "completed_batches",
        "completed_tasks",
        "optimizer_steps",
        "generator_call_index",
        "updates_lines",
        "tasks_lines",
        "student_adapter_relative_path",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"checkpoint state is missing fields {missing}: {path}")
    adapter = path / state["student_adapter_relative_path"]
    if not any(
        (adapter / name).is_file()
        for name in ("adapter_model.safetensors", "adapter_model.bin")
    ):
        raise ValueError(f"checkpoint student adapter is missing: {adapter}")
    if not (path / "optimizer.pt").is_file():
        raise ValueError(f"checkpoint optimizer is missing: {path}")
    return state


def prune_complete_checkpoints(root: Path, *, keep: int) -> list[Path]:
    if keep < 1:
        raise ValueError("checkpoint retention must be positive")
    values = complete_checkpoints(root)
    removed = values[:-keep]
    for path in removed:
        shutil.rmtree(path)
    return removed
