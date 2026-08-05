#!/usr/bin/env python3
"""Summarize Hugging Face Trainer loss logs and epoch checkpoints.

The output is deterministic and contains both machine-readable JSON and a compact
CSV. Logged losses are assigned to ``ceil(epoch)`` and weighted by the optimizer
step span represented by each log event.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)$")


def read_state(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def choose_state(output_dir: Path) -> tuple[Path, dict[str, Any]]:
    candidates = []
    root_state = output_dir / "trainer_state.json"
    if root_state.is_file():
        candidates.append((10**18, root_state))
    for path in output_dir.glob("checkpoint-*/trainer_state.json"):
        match = CHECKPOINT_RE.match(path.parent.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"no trainer_state.json under {output_dir}")
    _, path = max(candidates, key=lambda item: item[0])
    return path, read_state(path)


def summarize(output_dir: Path) -> dict[str, Any]:
    state_path, state = choose_state(output_dir)
    loss_events = [
        item
        for item in state.get("log_history", [])
        if isinstance(item.get("loss"), (int, float))
        and isinstance(item.get("step"), int)
        and isinstance(item.get("epoch"), (int, float))
    ]
    loss_events.sort(key=lambda item: item["step"])

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    previous_step = 0
    for event in loss_events:
        epoch_number = max(1, int(math.ceil(float(event["epoch"]) - 1e-9)))
        weight = max(1, int(event["step"]) - previous_step)
        grouped[epoch_number].append({**event, "optimizer_steps_represented": weight})
        previous_step = int(event["step"])

    checkpoints = []
    for path in sorted(
        output_dir.glob("checkpoint-*/trainer_state.json"),
        key=lambda item: int(CHECKPOINT_RE.match(item.parent.name).group(1)),
    ):
        checkpoint_state = read_state(path)
        checkpoints.append(
            {
                "checkpoint": path.parent.name,
                "global_step": checkpoint_state.get("global_step"),
                "epoch": checkpoint_state.get("epoch"),
            }
        )

    epochs = []
    for epoch_number in sorted(grouped):
        events = grouped[epoch_number]
        total_weight = sum(item["optimizer_steps_represented"] for item in events)
        weighted_loss = sum(
            float(item["loss"]) * item["optimizer_steps_represented"] for item in events
        ) / total_weight
        epochs.append(
            {
                "epoch": epoch_number,
                "mean_logged_loss": weighted_loss,
                "first_step": events[0]["step"],
                "last_step": events[-1]["step"],
                "logged_events": len(events),
                "optimizer_steps_represented": total_weight,
                "last_logged_loss": events[-1]["loss"],
                "last_learning_rate": events[-1].get("learning_rate"),
            }
        )

    final_metrics = {}
    for item in reversed(state.get("log_history", [])):
        if "train_loss" in item:
            final_metrics = {
                key: item.get(key)
                for key in (
                    "epoch",
                    "step",
                    "train_loss",
                    "train_runtime",
                    "train_samples_per_second",
                    "train_steps_per_second",
                    "total_flos",
                )
                if key in item
            }
            break

    return {
        "output_dir": str(output_dir),
        "source_trainer_state": str(state_path),
        "global_step": state.get("global_step"),
        "completed_epoch": state.get("epoch"),
        "checkpoints": checkpoints,
        "epochs": epochs,
        "final_metrics": final_metrics,
    }


def write_summary(summary: dict[str, Any], output_dir: Path) -> None:
    json_path = output_dir / "loss_by_epoch.json"
    csv_path = output_dir / "loss_by_epoch.csv"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fieldnames = [
        "epoch",
        "mean_logged_loss",
        "first_step",
        "last_step",
        "logged_events",
        "optimizer_steps_represented",
        "last_logged_loss",
        "last_learning_rate",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary["epochs"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    summary = summarize(output_dir)
    write_summary(summary, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
