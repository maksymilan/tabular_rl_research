"""Lightweight LLaMA-Factory registry I/O for ShareGPT SFT datasets."""
from __future__ import annotations

import json
import os
from pathlib import Path


def sharegpt_dataset_entry(dataset_name: str, file_name: str) -> dict:
    return {
        dataset_name: {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    }


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_sharegpt_dataset_info(
    dataset_path: Path,
    dataset_name: str,
) -> tuple[Path, Path]:
    """Register one JSONL file beside the dataset without importing rollout code."""
    entry = sharegpt_dataset_entry(dataset_name, dataset_path.name)
    snippet = dataset_path.parent / f"dataset_info.{dataset_name}.snippet.json"
    registry = dataset_path.parent / "dataset_info.json"
    existing = (
        json.loads(registry.read_text(encoding="utf-8"))
        if registry.exists()
        else {}
    )
    if not isinstance(existing, dict):
        raise ValueError(f"dataset registry must contain a JSON object: {registry}")
    existing.update(entry)
    _write_json_atomic(snippet, entry)
    _write_json_atomic(registry, existing)
    return snippet, registry
