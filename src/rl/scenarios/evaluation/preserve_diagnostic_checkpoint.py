#!/usr/bin/env python3
"""Read-only watcher that atomically preserves an ephemeral trainer checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "diagnostic-checkpoint-preservation-v1"
REQUIRED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "trainer_state.json",
)
OPTIONAL_STATE_FILES = (
    "optimizer.pt",
    "rng_state.pth",
    "scheduler.pt",
    "training_args.bin",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot(source: Path, *, expected_step: int, expected_max_steps: int) -> dict[str, Any]:
    if not source.is_dir() or source.is_symlink():
        raise FileNotFoundError(source)
    entries = list(source.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in entries):
        raise ValueError("checkpoint contains a symlink or non-file entry")
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint is incomplete: {missing}")
    state = json.loads((source / "trainer_state.json").read_bytes())
    if int(state.get("global_step", -1)) != expected_step:
        raise ValueError("trainer_state global_step mismatch")
    if int(state.get("max_steps", -1)) != expected_max_steps:
        raise ValueError("trainer_state max_steps mismatch")
    # Preserve every file written by the trainer.  Optional optimizer state is
    # not required for diagnostic inference, but is included when present.
    files = {
        path.name: {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(entries, key=lambda item: item.name)
    }
    if any(record["size_bytes"] <= 0 for record in files.values()):
        raise ValueError("checkpoint contains an empty file")
    return {"global_step": expected_step, "max_steps": expected_max_steps, "files": files}


def preserve(
    source: Path,
    destination: Path,
    *,
    expected_step: int,
    expected_max_steps: int,
    stable_seconds: float,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination must differ")
    if destination.exists() or destination.is_symlink():
        raise ValueError(f"refusing existing destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    first: dict[str, Any] | None = None
    while time.monotonic() <= deadline:
        try:
            current = snapshot(
                source, expected_step=expected_step, expected_max_steps=expected_max_steps
            )
        except (OSError, ValueError, json.JSONDecodeError):
            first = None
            time.sleep(poll_seconds)
            continue
        if first == current:
            break
        first = current
        time.sleep(stable_seconds)
    else:
        raise TimeoutError("timed out before checkpoint became complete and stable")
    if first is None:
        raise RuntimeError("stable checkpoint snapshot is unavailable")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.next-", dir=destination.parent)
    )
    try:
        for name in first["files"]:
            shutil.copy2(source / name, temporary / name, follow_symlinks=False)
        copied = snapshot(
            temporary, expected_step=expected_step, expected_max_steps=expected_max_steps
        )
        if copied != first:
            raise ValueError("copied checkpoint differs from the stable source snapshot")
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "status": "preserved",
            "source": str(source.resolve()),
            "destination": str(destination.resolve()),
            "source_mutated": False,
            "global_step": expected_step,
            "max_steps": expected_max_steps,
            "stability_observations": 2,
            "files": first["files"],
        }
        receipt_path = temporary / "preservation_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for path in temporary.iterdir():
            os.chmod(path, 0o444)
        temporary.rename(destination)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=6)
    parser.add_argument("--expected-max-steps", type=int, default=12)
    parser.add_argument("--stable-seconds", type=float, default=5.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--timeout-seconds", type=float, default=172800.0)
    args = parser.parse_args(argv)
    try:
        result = preserve(
            args.source.resolve(),
            args.destination.resolve(),
            expected_step=args.expected_step,
            expected_max_steps=args.expected_max_steps,
            stable_seconds=args.stable_seconds,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
    except (OSError, ValueError, TypeError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"checkpoint preservation blocked: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps({"status": result["status"], "destination": result["destination"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
