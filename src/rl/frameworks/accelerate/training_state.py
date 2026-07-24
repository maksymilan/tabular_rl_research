#!/usr/bin/env python3
"""Recoverable optimizer, scheduler, and RNG state for the Accelerate backend."""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch


TRAINER_STATE_NAME = "trainer_state.pt"


def save_training_state(
    checkpoint_dir: Path,
    *,
    step: int,
    optimizer: Any,
    scheduler: Any,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically save all state required to continue the same on-policy run."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / TRAINER_STATE_NAME
    temporary = checkpoint_dir / f".{TRAINER_STATE_NAME}.tmp"
    payload = {
        "format_version": 1,
        "step": int(step),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_states": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, temporary)
    os.replace(temporary, target)
    return target


def load_training_state(
    checkpoint_dir: Path,
    *,
    optimizer: Any,
    scheduler: Any,
    expected_metadata: dict[str, Any] | None = None,
    map_location: Any = "cpu",
) -> int:
    """Restore a checkpoint and return its completed training-loop step."""
    source = checkpoint_dir / TRAINER_STATE_NAME
    if not source.exists():
        raise FileNotFoundError(f"missing resumable trainer state: {source}")
    try:
        payload = torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch versions before ``weights_only`` was introduced.
        payload = torch.load(source, map_location=map_location)

    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported trainer-state format: {payload.get('format_version')}")
    actual_metadata = payload.get("metadata") or {}
    for key, expected in (expected_metadata or {}).items():
        actual = actual_metadata.get(key)
        if actual != expected:
            raise ValueError(
                f"resume metadata mismatch for {key}: {actual!r} != {expected!r}"
            )

    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    random.setstate(payload["python_random_state"])
    torch.set_rng_state(payload["torch_random_state"].cpu())
    cuda_states = payload.get("cuda_random_states")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)
    return int(payload["step"])
