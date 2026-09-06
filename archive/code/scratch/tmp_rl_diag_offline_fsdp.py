#!/usr/bin/env python3
"""Run one isolated offline fixed-pool FSDP update.

This diagnostic intentionally monkey-patches only the *production frozen-pool
admission validator* in the imported module.  The production validator is for a
different frozen K4/SFT2 pool; this wrapper validates the fresh K8 artifact's
basic shape/hash and leaves the trainer, loss, FSDP, and transition code intact.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _diagnostic_validate(args):
    if args.fixed_pool_manifest is None or args.fixed_rollout_pool is None:
        return None
    manifest_path = args.fixed_pool_manifest.resolve()
    pool_path = args.fixed_rollout_pool.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "table-agent-diagnostic-rollout-pool-v1":
        raise ValueError("diagnostic pool schema mismatch")
    if payload.get("status") != "diagnostic_generated":
        raise ValueError("diagnostic pool is not marked generated")
    if payload.get("result_reward_profile") != "four-level":
        raise ValueError("diagnostic pool is not four-level")
    tasks = int(payload.get("tasks") or 0)
    group = int(payload.get("group_size") or 0)
    trajectories = int(payload.get("trajectories") or 0)
    if tasks < 1 or group != args.group_size or trajectories != tasks * group:
        raise ValueError(
            f"diagnostic pool shape mismatch: tasks={tasks}, group={group}, "
            f"trajectories={trajectories}, runtime_group={args.group_size}"
        )
    recorded = (
        payload.get("files", {})
        .get("validated_trajectories", {})
    )
    if Path(str(recorded.get("path", ""))).resolve() != pool_path:
        raise ValueError("diagnostic pool path mismatch")
    actual = _sha256(pool_path)
    if recorded.get("sha256") != actual:
        raise ValueError("diagnostic pool SHA256 mismatch")
    reuse = payload.get("reuse_contract") or {}
    if not all(
        reuse.get(key) is True
        for key in ("same_trajectories", "same_sequence", "online_resampling_forbidden")
    ):
        raise ValueError("diagnostic pool reuse contract is incomplete")
    return payload


def main() -> None:
    # Import through the runtime package path supplied by the launcher.  The
    # imported module computes its ROOT from its own immutable runtime source,
    # preserving all normal identity and implementation snapshots.
    runner = importlib.import_module("frameworks.trl.run_transition_grpo")
    runner.validate_fixed_pool_manifest = _diagnostic_validate

    # Transformers 4.57 rejects a non-None ``optimizers`` tuple when FSDP is
    # enabled, because the optimizer must be built after Accelerate wraps the
    # sharded model.  The production runner currently constructs AdamW before
    # Trainer for its replicated path.  For this isolated FSDP probe, discard
    # that pre-wrap object and let Trainer create the equivalent AdamW lazily.
    _OriginalTrainer = runner.TransitionGRPOTrainer

    class DiagnosticFSDPTrainer(_OriginalTrainer):
        def __init__(self, *args, **kwargs):
            sharding = kwargs.get("trainer_sharding")
            if sharding == "fsdp":
                kwargs.pop("optimizers", None)
                trainer_args = kwargs.get("args")
                if trainer_args is not None:
                    trainer_args.optim = "adamw_torch"
            super().__init__(*args, **kwargs)

    runner.TransitionGRPOTrainer = DiagnosticFSDPTrainer
    runner.main()


if __name__ == "__main__":
    main()
