#!/usr/bin/env python3
"""Prepare one interrupted online-GRPO output for exact checkpoint resume.

The trainer appends rollout audit rows before an optimizer step is committed.
Consequently, an interrupted run can contain rows newer than its last durable
checkpoint.  This helper selects the latest complete periodic checkpoint,
quarantines other checkpoint-shaped directories, validates the committed
rollout prefix, and atomically removes only the uncommitted suffix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = "vanilla-grpo-resume-preparation-v1"
CHECKPOINT_RE = re.compile(r"checkpoint-([0-9]+)")
REQUIRED_CHECKPOINT_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
    "training_args.bin",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_indices(path: Path, *, expected_records: int) -> list[int]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indices = [row.get("example_index") for row in rows]
    if (
        len(rows) != expected_records
        or any(type(value) is not int for value in indices)
        or len(set(indices)) != expected_records
    ):
        raise ValueError(
            f"tasks must contain exactly {expected_records} unique integer example_index values"
        )
    return [int(value) for value in indices]


def _checkpoint_state(
    path: Path,
    *,
    optimizer_steps: int,
    save_steps: int,
) -> tuple[int, dict[str, Any]] | None:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match is None or not path.is_dir() or path.is_symlink():
        raise ValueError(f"invalid checkpoint entry: {path}")
    step = int(match.group(1))
    if not 0 < step <= optimizer_steps or step % save_steps != 0:
        raise ValueError(f"checkpoint step violates periodic save contract: {path}")
    if not all(
        (path / name).is_file() and not (path / name).is_symlink()
        for name in REQUIRED_CHECKPOINT_FILES
    ):
        return None
    try:
        state = json.loads((path / "trainer_state.json").read_text())
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    if type(state.get("global_step")) is not int or int(state["global_step"]) != step:
        return None
    if state.get("max_steps") is not None and int(state["max_steps"]) != optimizer_steps:
        return None
    return step, state


def _quarantine(
    paths: Sequence[Path],
    *,
    root: Path,
    retained_step: int,
) -> list[dict[str, str]]:
    if not paths:
        return []
    quarantine = root / "resume_quarantine"
    quarantine.mkdir(exist_ok=True)
    if quarantine.is_symlink():
        raise ValueError(f"resume quarantine may not be a symlink: {quarantine}")
    ordered = sorted(paths, key=lambda path: path.name)
    destinations = []
    reserved: set[Path] = set()
    for source in ordered:
        stem = f"{source.name}.before-{retained_step}"
        destination = quarantine / stem
        suffix = 1
        while destination.exists() or destination in reserved:
            destination = quarantine / f"{stem}.{suffix}"
            suffix += 1
        reserved.add(destination)
        destinations.append(destination)
    moved = []
    for source, destination in zip(ordered, destinations, strict=True):
        os.replace(source, destination)
        moved.append({"from": str(source.resolve()), "to": str(destination.resolve())})
    return moved


def _validate_and_trim_rollouts(
    path: Path,
    *,
    committed_step: int,
    prompts_per_update: int,
    group_size: int,
    task_indices: Sequence[int],
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing/non-regular rollout log: {path}")
    original = path.read_bytes()
    lines = original.splitlines(keepends=True)
    if any(not line.strip() for line in lines):
        raise ValueError("rollout log contains empty rows")
    expected_rows = committed_step * prompts_per_update * group_size
    if len(lines) < expected_rows:
        raise ValueError(
            f"rollout log is behind checkpoint: {len(lines)} < {expected_rows}"
        )
    prefix = lines[:expected_rows]
    if prefix and not prefix[-1].endswith(b"\n"):
        raise ValueError("committed rollout prefix has an unterminated JSON row")
    task_set = set(task_indices)
    parsed = []
    for position, line in enumerate(prefix):
        try:
            row = json.loads(line)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid committed rollout row {position}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"committed rollout row {position} is not an object")
        parsed.append(row)
    rows_per_step = prompts_per_update * group_size
    for step in range(committed_step):
        block = parsed[step * rows_per_step : (step + 1) * rows_per_step]
        if {row.get("policy_global_step") for row in block} != {step}:
            raise ValueError(f"rollout block {step} has the wrong policy_global_step")
        indices = [row.get("example_index") for row in block]
        if any(type(value) is not int or value not in task_set for value in indices):
            raise ValueError(f"rollout block {step} contains an unknown example_index")
        counts = Counter(int(value) for value in indices)
        if len(counts) != prompts_per_update or set(counts.values()) != {group_size}:
            raise ValueError(f"rollout block {step} is not {prompts_per_update} x K{group_size}")
    steps_per_pass = len(task_indices) // prompts_per_update
    for pass_index in range((committed_step + steps_per_pass - 1) // steps_per_pass):
        pass_start = pass_index * steps_per_pass
        committed_in_pass = min(steps_per_pass, committed_step - pass_start)
        block = parsed[
            pass_start * rows_per_step :
            (pass_start + committed_in_pass) * rows_per_step
        ]
        counts = Counter(int(row["example_index"]) for row in block)
        expected_unique = committed_in_pass * prompts_per_update
        if len(counts) != expected_unique or set(counts.values()) != {group_size}:
            raise ValueError(
                f"committed rollout pass {pass_index} repeats or omits prompt identities"
            )
        if committed_in_pass == steps_per_pass and counts != Counter(
            {index: group_size for index in task_indices}
        ):
            raise ValueError(f"committed rollout pass {pass_index} is not complete")

    retained = b"".join(prefix)
    removed_rows = len(lines) - expected_rows
    if removed_rows:
        temporary = path.with_suffix(path.suffix + ".resume-next")
        temporary.write_bytes(retained)
        os.replace(temporary, path)
    if path.read_bytes() != retained:
        raise RuntimeError("atomic rollout-prefix preparation did not persist exactly")
    return {
        "path": str(path.resolve()),
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "original_rows": len(lines),
        "retained_sha256": hashlib.sha256(retained).hexdigest(),
        "retained_rows": expected_rows,
        "removed_uncommitted_rows": removed_rows,
    }


def prepare_resume(
    *,
    train_output: Path,
    tasks: Path,
    expected_records: int,
    optimizer_steps: int,
    prompts_per_update: int,
    group_size: int,
    save_steps: int,
    retain_checkpoint_steps: Sequence[int] = (),
) -> dict[str, Any]:
    if not train_output.is_dir() or train_output.is_symlink():
        raise ValueError(f"training output must be a regular directory: {train_output}")
    if (
        expected_records < 1
        or optimizer_steps < 1
        or prompts_per_update < 1
        or group_size < 2
        or save_steps < 1
        or optimizer_steps % save_steps != 0
        or expected_records % prompts_per_update != 0
    ):
        raise ValueError("invalid periodic resume dimensions")
    task_indices = _task_indices(tasks, expected_records=expected_records)
    retained_audit_steps = {int(value) for value in retain_checkpoint_steps}
    if any(
        value <= 0 or value >= optimizer_steps or value % save_steps != 0
        for value in retained_audit_steps
    ):
        raise ValueError("retained checkpoint steps violate the periodic save contract")
    checkpoint_entries = sorted(train_output.glob("checkpoint-*"))
    complete: list[tuple[int, Path, dict[str, Any]]] = []
    incomplete: list[Path] = []
    for path in checkpoint_entries:
        value = _checkpoint_state(
            path,
            optimizer_steps=optimizer_steps,
            save_steps=save_steps,
        )
        if value is None:
            incomplete.append(path)
        else:
            step, state = value
            complete.append((step, path, state))
    if not complete:
        raise ValueError("nonempty training output has no complete periodic checkpoint")
    complete.sort(key=lambda item: item[0])
    committed_step, checkpoint, state = complete[-1]
    if committed_step >= optimizer_steps:
        raise ValueError(
            "checkpoint reached the final optimizer step but final output is absent; "
            "manual finalization recovery is required"
        )
    complete_by_step = {step: path for step, path, _ in complete}
    required_retained_steps = {
        value for value in retained_audit_steps if value <= committed_step
    }
    missing_retained_steps = sorted(required_retained_steps - set(complete_by_step))
    if missing_retained_steps:
        raise ValueError(
            "missing complete retained audit checkpoints: "
            f"{missing_retained_steps}"
        )
    retained_audit_paths = {
        complete_by_step[value] for value in required_retained_steps
    }
    rollout = _validate_and_trim_rollouts(
        train_output / "rollouts.jsonl",
        committed_step=committed_step,
        prompts_per_update=prompts_per_update,
        group_size=group_size,
        task_indices=task_indices,
    )
    superseded = [
        path
        for _, path, _ in complete[:-1]
        if path not in retained_audit_paths
    ] + incomplete
    moved = _quarantine(
        superseded,
        root=train_output,
        retained_step=committed_step,
    )
    remaining = sorted(train_output.glob("checkpoint-*"))
    expected_remaining = sorted(retained_audit_paths | {checkpoint})
    if remaining != expected_remaining:
        raise RuntimeError(
            "resume preparation retained the wrong checkpoint set: "
            f"observed={remaining} expected={expected_remaining}"
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "resume_ready",
        "train_output": str(train_output.resolve()),
        "tasks": {
            "path": str(tasks.resolve()),
            "sha256": _sha(tasks),
            "records": expected_records,
        },
        "contract": {
            "optimizer_steps": optimizer_steps,
            "prompts_per_update": prompts_per_update,
            "group_size": group_size,
            "save_steps": save_steps,
            "committed_rollouts_per_step": prompts_per_update * group_size,
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "global_step": committed_step,
            "trainer_state_sha256": _sha(checkpoint / "trainer_state.json"),
            "max_steps": state.get("max_steps"),
        },
        "retained_audit_checkpoints": [
            str(path.resolve()) for path in sorted(retained_audit_paths)
        ],
        "quarantined_checkpoints": moved,
        "rollouts": rollout,
    }


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".next")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--optimizer-steps", type=int, required=True)
    parser.add_argument("--prompts-per-update", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--save-steps", type=int, required=True)
    parser.add_argument(
        "--retain-checkpoint-step",
        action="append",
        type=int,
        default=[],
        help="keep a complete earlier checkpoint for an immutable audit receipt",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare_resume(
            train_output=args.train_output,
            tasks=args.tasks,
            expected_records=args.expected_records,
            optimizer_steps=args.optimizer_steps,
            prompts_per_update=args.prompts_per_update,
            group_size=args.group_size,
            save_steps=args.save_steps,
            retain_checkpoint_steps=args.retain_checkpoint_step,
        )
        _write_atomic(args.output, result)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"GRPO resume preparation failed: {exc}", file=__import__("sys").stderr)
        return 1
    print(json.dumps({"checkpoint": result["checkpoint"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
