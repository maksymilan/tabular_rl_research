"""Small shared I/O and identity primitives for fixed-pool scenarios."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_id(row: dict[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id") or row.get("trajectory_id")
    if not value:
        raise ValueError("task lacks a stable id")
    return str(value)
