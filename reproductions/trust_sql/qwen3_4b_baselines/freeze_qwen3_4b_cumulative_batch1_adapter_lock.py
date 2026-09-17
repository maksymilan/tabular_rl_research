#!/usr/bin/env python3
"""Freeze the completed Qwen3-4B cumulative checkpoint-6380 identity for evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_BASE = Path("/home/dengyan/models/Qwen3-4B-TrustSQL-baseline")
EXPECTED_ADAPTER = Path(
    "/home/dengyan/tabular_rl_outputs/checkpoints/"
    "qwen3-4b-atomic-v26-cumulative-fresh4ep-table-rl-formal-b1/checkpoint-6380"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def freeze(adapter: Path) -> dict[str, Any]:
    adapter = adapter.resolve()
    if adapter != EXPECTED_ADAPTER:
        raise ValueError(f"adapter must be the fixed epoch-4 checkpoint: {adapter}")
    state = load_object(adapter / "trainer_state.json")
    if int(state.get("global_step", -1)) != 6380:
        raise ValueError(f"trainer global_step is not 6380: {state.get('global_step')!r}")
    config = load_object(adapter / "adapter_config.json")
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("adapter peft_type is not LORA")
    if str(config.get("task_type", "")).upper() != "CAUSAL_LM":
        raise ValueError("adapter task_type is not CAUSAL_LM")
    if int(config.get("r", 0)) != 16:
        raise ValueError("adapter rank is not 16")
    base = config.get("base_model_name_or_path")
    if not isinstance(base, str) or Path(base).resolve() != EXPECTED_BASE:
        raise ValueError(f"adapter base path mismatch: {base!r}")
    names = ("adapter_model.safetensors", "adapter_config.json", "trainer_state.json")
    files: dict[str, str] = {}
    for name in names:
        path = adapter / name
        if not path.is_file():
            raise ValueError(f"missing adapter artifact: {path}")
        files[name] = sha256(path)
    return {
        "schema_version": "qwen3-4b-atomic-v26-sft1-adapter-lock-v1",
        "path": str(adapter),
        "checkpoint_name": "checkpoint-6380",
        "global_step": 6380,
        "files_sha256": files,
    }


def atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=EXPECTED_ADAPTER)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite existing lock: {args.output}")
    try:
        payload = freeze(args.adapter)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    atomic_write(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
