#!/usr/bin/env python3
"""Low-overhead JSONL tracing helpers for an isolated RL performance smoke."""
from __future__ import annotations

import atexit
import json
import os
import threading
import time
from pathlib import Path
from typing import Any


_PATH = os.environ.get("RL_PERF_TRACE_PATH")
_SYNC = os.environ.get("RL_PERF_TRACE_SYNC_CUDA", "0") == "1"
_LOCK = threading.Lock()
_HANDLE = None


def tracing_enabled() -> bool:
    return bool(_PATH)


def _handle():
    global _HANDLE
    if not _PATH:
        return None
    with _LOCK:
        if _HANDLE is None:
            path = Path(_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            _HANDLE = path.open("a", encoding="utf-8", buffering=1)
    return _HANDLE


def _close() -> None:
    global _HANDLE
    with _LOCK:
        if _HANDLE is not None:
            _HANDLE.flush()
            _HANDLE.close()
            _HANDLE = None


atexit.register(_close)


def cuda_sync() -> None:
    """Synchronize only when the diagnostic explicitly requests it."""

    if not _SYNC:
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def cuda_snapshot() -> dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "cuda_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_reserved_bytes": int(torch.cuda.memory_reserved()),
            "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }
    except Exception:
        return {}


def trace_event(stage: str, **fields: Any) -> None:
    handle = _handle()
    if handle is None:
        return
    payload: dict[str, Any] = {
        "schema_version": "rl-performance-trace-v1",
        "timestamp_unix": time.time(),
        "monotonic": time.perf_counter(),
        "pid": os.getpid(),
        "rank": int(os.environ.get("RANK", "0")),
        "stage": stage,
    }
    payload.update(fields)
    with _LOCK:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
