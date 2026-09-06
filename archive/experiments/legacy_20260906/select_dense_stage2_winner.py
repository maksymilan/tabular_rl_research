#!/usr/bin/env python3
"""Select one dense Stage1 method for a preregistered scale/seed follow-up."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_requirement(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "stage1-checkpoint-selection-requirements-v2":
        raise ValueError(f"unsupported requirement manifest for {label}: {path}")
    greedy = (payload.get("candidate_metrics") or {}).get("greedy") or {}
    baseline = (payload.get("sft2") or {}).get("greedy") or {}
    if greedy.get("questions") != 1534 or baseline.get("questions") != 1534:
        raise ValueError(f"{label} does not contain a complete BIRD-dev comparison")
    return payload


def atomic_frozen_json(path: Path, payload: dict[str, Any]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to replace a different selection: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".next")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uniform", required=True, type=Path)
    parser.add_argument("--strategic", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    requirements = {
        "uniform": load_requirement(args.uniform, "uniform"),
        "strategic": load_requirement(args.strategic, "strategic"),
    }
    baseline_counts = {
        int(payload["sft2"]["greedy"]["correct"])
        for payload in requirements.values()
    }
    if baseline_counts != {762}:
        raise ValueError(f"unexpected or inconsistent SFT2 baseline: {baseline_counts}")

    candidates: dict[str, dict[str, Any]] = {}
    for label, payload in requirements.items():
        greedy = payload["candidate_metrics"]["greedy"]
        prefix = payload["candidate_metrics"]["fixed_prefix"]
        correct = int(greedy["correct"])
        candidates[label] = {
            "candidate": payload["candidate"],
            "requirements_status": payload["status"],
            "correct": correct,
            "accuracy": float(greedy["greedy_at_1"]),
            "valid_rate": float(greedy["valid_rate"]),
            "fixed_prefix_mean_margin": float(prefix["mean_margin"]),
            "strictly_above_sft2": correct > 762,
            "eligible": payload["status"] == "passed" and correct > 762,
        }

    eligible = [label for label, record in candidates.items() if record["eligible"]]
    selected = None
    if eligible:
        selected = max(
            eligible,
            key=lambda label: (
                candidates[label]["correct"],
                candidates[label]["valid_rate"],
                candidates[label]["fixed_prefix_mean_margin"],
                label == "uniform",
            ),
        )
    payload = {
        "schema_version": "dense-stage2-winner-selection-v1",
        "status": "selected" if selected is not None else "no_eligible_candidate",
        "baseline": {"name": "SFT2 checkpoint-1682", "correct": 762, "questions": 1534},
        "selection_rule": (
            "require Stage1 test requirements passed and full-dev greedy correct > 762; "
            "then maximize correct, valid rate, fixed-prefix mean margin; exact ties prefer "
            "the simpler uniform allocation"
        ),
        "candidates": candidates,
        "selected": selected,
        "selected_config": (
            f"exp18_dense_{selected}_full_response_scale120.yaml"
            if selected is not None
            else None
        ),
        "next_step": (
            "balanced mixed120, independent SFT2 initialization, train seeds 101/202/303"
            if selected is not None
            else "stop without additional RL training"
        ),
    }
    atomic_frozen_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
