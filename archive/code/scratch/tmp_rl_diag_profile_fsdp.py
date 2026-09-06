#!/usr/bin/env python3
"""Isolated runtime instrumentation for one short, real FSDP RL update.

The production runner/trainer is imported unchanged.  This wrapper only adds
rank-local timing events and the same diagnostic manifest/optimizer patches as
the prior offline probe; it never alters the formal RL output directory.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import time
import types
from pathlib import Path


PROFILE_DIR = Path(os.environ.get("RL_PROFILE_DIR", ".")).resolve()
RANK = int(os.environ.get("RANK", "0"))
PROFILE_PATH = PROFILE_DIR / f"rank{RANK}.jsonl"
PROFILE_DIR.mkdir(parents=True, exist_ok=True)

# Set by the isolated FSDP padding shim below and consumed by stage events.
# Keeping this state in the wrapper makes the local/global schedule visible in
# the JSONL profile without changing the production trainer contract.
PAD_STATS: dict[str, int] = {}


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def _event(event: str, **fields) -> None:
    row = {"time": time.time(), "rank": RANK, "event": event}
    row.update(fields)
    try:
        import torch

        if torch.cuda.is_available():
            row["memory_allocated_mib"] = round(torch.cuda.memory_allocated() / 2**20, 1)
            row["memory_reserved_mib"] = round(torch.cuda.memory_reserved() / 2**20, 1)
            row["max_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            row["max_memory_reserved_mib"] = round(torch.cuda.max_memory_reserved() / 2**20, 1)
    except Exception:
        pass
    with PROFILE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
        raise ValueError("diagnostic pool shape mismatch")
    recorded = payload.get("files", {}).get("validated_trajectories", {})
    if Path(str(recorded.get("path", ""))).resolve() != pool_path:
        raise ValueError("diagnostic pool path mismatch")
    if recorded.get("sha256") != _sha256(pool_path):
        raise ValueError("diagnostic pool SHA256 mismatch")
    reuse = payload.get("reuse_contract") or {}
    if not all(reuse.get(key) is True for key in ("same_trajectories", "same_sequence", "online_resampling_forbidden")):
        raise ValueError("diagnostic pool reuse contract is incomplete")
    return payload


def main() -> None:
    import torch

    runner = importlib.import_module("frameworks.trl.run_transition_grpo")
    transition = importlib.import_module("frameworks.trl.transition_grpo")
    runner.validate_fixed_pool_manifest = _diagnostic_validate

    # FSDP collective safety diagnostic: retain every transition, including
    # groups whose standardized GRPO advantage is exactly zero.  The production
    # helper intentionally drops those rows to save a no-op forward/backward,
    # but doing so independently on each rank can produce different local
    # transition counts (e.g. one rank receives a homogeneous-reward K group),
    # which in turn gives FSDP different numbers of parameter all-gathers.  A
    # zero advantage still contributes an exactly-zero loss while preserving
    # the fixed row/micro schedule.  This replacement is local to this
    # diagnostic wrapper and never changes the runtime source.
    def _keep_zero_advantage_rows(
        updates,
        *,
        policy_loss_coefficient,
        rank_loss_coefficient,
        kl_beta,
    ):
        # All ranks first reach this point after local SAAM/credit processing.
        # Use one NCCL scalar reduction to derive a common transition count,
        # then append zero-advantage copies on shorter ranks.  The copies have
        # real prompt/response tensor shapes, so the ordinary fixed-row
        # microbatch builder emits the same number/order of FSDP collectives on
        # every rank.  Their coefficient is exactly zero; they are communication
        # placeholders only and do not alter the RL objective.
        import dataclasses

        local_count = len(updates)
        global_count = local_count
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                device = torch.device("cuda", torch.cuda.current_device())
                count = torch.tensor([local_count], device=device, dtype=torch.long)
                dist.all_reduce(count, op=dist.ReduceOp.MAX)
                global_count = int(count.item())
        except Exception:
            # The diagnostic must never silently fabricate a distributed result;
            # propagate failures from an initialized process group.  This
            # fallback is only for single-process wrapper sanity checks.
            if os.environ.get("WORLD_SIZE", "1") != "1":
                raise
        padded = list(updates)
        dummy_count = max(0, global_count - local_count)
        if dummy_count:
            if not padded:
                raise RuntimeError("cannot pad an empty transition batch")
            template = min(
                padded,
                key=lambda update: len(update.prompt_ids) + len(update.response_ids),
            )
            for index in range(dummy_count):
                padded.append(
                    dataclasses.replace(
                        template,
                        advantage=0.0,
                        trajectory_id=(
                            f"{template.trajectory_id}__globalpad_r{RANK}_{index}"
                        ),
                        example_index=-1,
                        turn_index=0,
                        trajectory_correct=False,
                    )
                )
        PAD_STATS.clear()
        PAD_STATS.update(
            {
                "local_transition_count": local_count,
                "global_transition_count": global_count,
                "dummy_transition_count": dummy_count,
            }
        )
        _event(
            "global_transition_pad",
            local_transition_count=local_count,
            global_transition_count=global_count,
            dummy_transition_count=dummy_count,
        )
        return padded, 0

    transition.retain_policy_contributing_updates = _keep_zero_advantage_rows
    TrainerBase = runner.TransitionGRPOTrainer

    original_compact = TrainerBase._transition_token_logps_compact
    original_generate = TrainerBase._generate_and_score_completions
    original_train_step = TrainerBase.training_step
    original_get_logps = TrainerBase._get_per_token_logps_and_entropies

    def prof_compact(self, model, inputs):
        _sync()
        started = time.perf_counter()
        ranges = self._transition_microbatch_ranges(inputs)
        result = None
        try:
            result = original_compact(self, model, inputs)
            return result
        finally:
            _sync()
            _event(
                "transition_token_logps_compact",
                stage=getattr(self, "_diag_stage", "unknown"),
                rows=int(inputs["completion_ids"].shape[0]),
                micro_batch_count=len(ranges),
                local_micro_batch_count=(
                    (PAD_STATS.get("local_transition_count", int(inputs["completion_ids"].shape[0]))
                     + self.transition_micro_batch_size - 1)
                    // self.transition_micro_batch_size
                ),
                global_micro_batch_count=len(ranges),
                local_transition_count=PAD_STATS.get(
                    "local_transition_count", int(inputs["completion_ids"].shape[0])
                ),
                global_transition_count=PAD_STATS.get(
                    "global_transition_count", int(inputs["completion_ids"].shape[0])
                ),
                dummy_transition_count=PAD_STATS.get("dummy_transition_count", 0),
                seconds=time.perf_counter() - started,
            )

    def prof_generate(self, inputs):
        _sync()
        started = time.perf_counter()
        self._diag_stage = "collect_and_old_policy"
        try:
            return original_generate(self, inputs)
        finally:
            _sync()
            _event("generate_and_score_total", seconds=time.perf_counter() - started)

    def prof_train_step(self, model, inputs, num_items_in_batch=None):
        _sync()
        started = time.perf_counter()
        self._diag_stage = "policy_backward"
        try:
            return original_train_step(self, model, inputs, num_items_in_batch)
        finally:
            _sync()
            _event("training_step_total", seconds=time.perf_counter() - started)

    def prof_get_logps(self, *args, **kwargs):
        _sync()
        started = time.perf_counter()
        input_ids = args[2] if len(args) > 2 else kwargs.get("input_ids")
        try:
            return original_get_logps(self, *args, **kwargs)
        finally:
            _sync()
            _event(
                "model_forward_micro",
                stage=getattr(self, "_diag_stage", "unknown"),
                rows=int(input_ids.shape[0]) if input_ids is not None else None,
                seq_len=int(input_ids.shape[1]) if input_ids is not None else None,
                seconds=time.perf_counter() - started,
            )

    TrainerBase._transition_token_logps_compact = prof_compact
    TrainerBase._generate_and_score_completions = prof_generate
    TrainerBase.training_step = prof_train_step
    TrainerBase._get_per_token_logps_and_entropies = prof_get_logps

    OriginalTrainer = TrainerBase

    class DiagnosticFSDPTrainer(OriginalTrainer):
        def __init__(self, *args, **kwargs):
            if kwargs.get("trainer_sharding") == "fsdp":
                kwargs.pop("optimizers", None)
                trainer_args = kwargs.get("args")
                if trainer_args is not None:
                    trainer_args.optim = "adamw_torch"
            super().__init__(*args, **kwargs)
            _event("trainer_init")
            original_backward = self.accelerator.backward

            def prof_backward(*args, **kwargs):
                _sync()
                started = time.perf_counter()
                try:
                    return original_backward(*args, **kwargs)
                finally:
                    _sync()
                    _event("accelerator_backward", seconds=time.perf_counter() - started)

            self.accelerator.backward = prof_backward

        def create_optimizer(self):
            result = super().create_optimizer()
            if self.optimizer is not None and not getattr(self, "_diag_optimizer_wrapped", False):
                original_step = self.optimizer.step

                def prof_optimizer_step(_optimizer, *args, **kwargs):
                    _sync()
                    started = time.perf_counter()
                    try:
                        return original_step(*args, **kwargs)
                    finally:
                        _sync()
                        _event("optimizer_step", seconds=time.perf_counter() - started)

                # Optimizer/scheduler integration in recent Transformers
                # inspects ``step.__func__``.  Bind the wrapper as a real
                # method so that bookkeeping remains compatible.
                self.optimizer.step = types.MethodType(
                    prof_optimizer_step, self.optimizer
                )
                self._diag_optimizer_wrapped = True
            return result

    runner.TransitionGRPOTrainer = DiagnosticFSDPTrainer
    _event("process_start")
    try:
        runner.main()
    finally:
        _event("process_end")


if __name__ == "__main__":
    main()
