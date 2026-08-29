#!/usr/bin/env python3
"""One-step outcome-gradient projection diagnostic on a real Gate60 K8 group.

This is an offline mechanism probe only.  It uses the ordinary binary group
advantage, never reads Gold SQL, performs no rollout, and compares the usual
mixed full-response update with a deterministic PCGrad-style update.  A gated
variant enables projection only when both full-response and tool-only outcome
gradients have a negative cosine; otherwise it falls back to vanilla.  The
projection is between positive- and negative-outcome gradients; reasoning and
tool token spans are not treated as competing objectives here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from diagnose_gate60_gradient_conflict import GradientScorer, build_rows, load_jsonl


def score_rows(scorer: GradientScorer, rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {"full": [], "think": [], "tool": [], "span": []}
    with torch.no_grad():
        for start in range(0, len(rows), 2):
            batch = scorer.score_batch_modes(rows[start : start + 2])
            for mode, tensor in batch.items():
                values[mode].extend(float(value) for value in tensor.detach().cpu())
    return values


def branch_gradient(
    scorer: GradientScorer,
    rows: list[dict[str, Any]],
    mode: str,
    sign: str,
) -> list[torch.Tensor]:
    selected = [
        row for row in rows if (row["advantage"] > 0) == (sign == "positive")
    ]
    if not selected:
        raise ValueError(f"no {sign} rows")
    scorer.model.zero_grad(set_to_none=True)
    scale = 1.0 / len(rows)
    for start in range(0, len(selected), 2):
        batch = selected[start : start + 2]
        coefficients = torch.tensor(
            [-float(row["advantage"]) * scale for row in batch],
            dtype=torch.float32,
            device=scorer.device,
        )
        (scorer.score_batch(batch, mode=mode) * coefficients).sum().backward()
    gradients = [
        parameter.grad.detach().float().cpu().clone()
        if parameter.grad is not None
        else torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        for parameter in scorer.parameters
    ]
    scorer.model.zero_grad(set_to_none=True)
    return gradients


def add(left: list[torch.Tensor], right: list[torch.Tensor]) -> list[torch.Tensor]:
    return [a + b for a, b in zip(left, right, strict=True)]


def dot(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    return sum(float((a * b).sum()) for a, b in zip(left, right, strict=True))


def norm(vector: list[torch.Tensor]) -> float:
    return math.sqrt(max(dot(vector, vector), 0.0))


def project_negative(
    positive: list[torch.Tensor], negative: list[torch.Tensor]
) -> tuple[list[torch.Tensor], dict[str, float]]:
    """Project the negative loss gradient off the conflicting positive component."""
    positive_norm_sq = dot(positive, positive)
    negative_dot = dot(negative, positive)
    coefficient = (
        min(0.0, negative_dot) / positive_norm_sq if positive_norm_sq > 0.0 else 0.0
    )
    projected = [b - coefficient * a for a, b in zip(positive, negative, strict=True)]
    return projected, {
        "positive_norm": math.sqrt(max(positive_norm_sq, 0.0)),
        "negative_norm": norm(negative),
        "negative_positive_dot": negative_dot,
        "negative_positive_cosine": (
            negative_dot / (math.sqrt(max(positive_norm_sq, 0.0)) * norm(negative))
            if positive_norm_sq > 0.0 and norm(negative) > 0.0
            else 0.0
        ),
        "projection_coefficient": coefficient,
        "projected_negative_norm": norm(projected),
        "projected_negative_positive_cosine": (
            dot(projected, positive)
            / (norm(projected) * math.sqrt(max(positive_norm_sq, 0.0)))
            if norm(projected) > 0.0 and positive_norm_sq > 0.0
            else 0.0
        ),
    }


def apply_update(scorer: GradientScorer, gradients: list[torch.Tensor], learning_rate: float) -> None:
    with torch.no_grad():
        for parameter, gradient in zip(scorer.parameters, gradients, strict=True):
            parameter.add_(-learning_rate * gradient.to(parameter.device, dtype=parameter.dtype))


def restore(scorer: GradientScorer, snapshot: list[torch.Tensor]) -> None:
    with torch.no_grad():
        for parameter, value in zip(scorer.parameters, snapshot, strict=True):
            parameter.copy_(value.to(parameter.device, dtype=parameter.dtype))


def alignment(rows: list[dict[str, Any]], before: list[float], after: list[float]) -> dict[str, float]:
    deltas = [a - b for a, b in zip(after, before, strict=True)]
    positive = [delta for row, delta in zip(rows, deltas, strict=True) if row["advantage"] > 0]
    negative = [delta for row, delta in zip(rows, deltas, strict=True) if row["advantage"] < 0]
    return {
        "rows": len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "mean_abs_delta": sum(abs(delta) for delta in deltas) / len(deltas),
        "positive_rows": len(positive),
        "positive_aligned": sum(delta > 0 for delta in positive) / len(positive) if positive else 0.0,
        "positive_mean_delta": sum(positive) / len(positive) if positive else 0.0,
        "negative_rows": len(negative),
        "negative_aligned": sum(delta < 0 for delta in negative) / len(negative) if negative else 0.0,
        "negative_mean_delta": sum(negative) / len(negative) if negative else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--example-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--learning-rate", type=float, default=4e-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_jsonl(args.rollouts)
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    rows = build_rows(records, tokenizer, args.step, 0, args.example_index).get(args.example_index, [])
    if not rows or not any(row["advantage"] > 0 for row in rows) or not any(row["advantage"] < 0 for row in rows):
        raise SystemExit("selected group must contain both positive and negative transitions")

    scorer = GradientScorer(args.base_model, args.adapter, args.device)
    before = score_rows(scorer, rows)
    snapshot = [parameter.detach().clone() for parameter in scorer.parameters]
    positive = branch_gradient(scorer, rows, "full", "positive")
    negative = branch_gradient(scorer, rows, "full", "negative")
    tool_positive = branch_gradient(scorer, rows, "tool", "positive")
    tool_negative = branch_gradient(scorer, rows, "tool", "negative")
    vanilla = add(positive, negative)
    projected_negative, projection = project_negative(positive, negative)
    projected = add(positive, projected_negative)
    tool_positive_dot = dot(tool_positive, tool_negative)
    tool_positive_norm = norm(tool_positive)
    tool_negative_norm = norm(tool_negative)
    tool_cosine = (
        tool_positive_dot / (tool_positive_norm * tool_negative_norm)
        if tool_positive_norm and tool_negative_norm
        else 0.0
    )
    full_cosine = float(projection["negative_positive_cosine"])
    gate_enabled = full_cosine < -0.1 and tool_cosine < -0.1
    gated = projected if gate_enabled else vanilla

    apply_update(scorer, vanilla, args.learning_rate)
    vanilla_after = score_rows(scorer, rows)
    vanilla_change = {
        mode: alignment(rows, before[mode], vanilla_after[mode])
        for mode in ("full", "think", "tool", "span")
    }
    restore(scorer, snapshot)
    apply_update(scorer, projected, args.learning_rate)
    projected_after = score_rows(scorer, rows)
    projected_change = {
        mode: alignment(rows, before[mode], projected_after[mode])
        for mode in ("full", "think", "tool", "span")
    }
    restore(scorer, snapshot)
    apply_update(scorer, gated, args.learning_rate)
    gated_after = score_rows(scorer, rows)
    gated_change = {
        mode: alignment(rows, before[mode], gated_after[mode])
        for mode in ("full", "think", "tool", "span")
    }
    output = {
        "schema_version": "gate60-pcgrad-one-step-audit-v1",
        "gold_sql_read": False,
        "measurement": "ordinary binary group GRPO; full carrier; vanilla versus positive-anchor PCGrad and full/tool conflict gate; one SGD-style step; no rollout",
        "step": args.step,
        "example_index": args.example_index,
        "rows": len(rows),
        "positive_rows": sum(row["advantage"] > 0 for row in rows),
        "negative_rows": sum(row["advantage"] < 0 for row in rows),
        "learning_rate": args.learning_rate,
        "gate": {
            "full_negative_positive_cosine": full_cosine,
            "tool_positive_negative_cosine": tool_cosine,
            "threshold": -0.1,
            "enabled": gate_enabled,
        },
        "projection": projection,
        "vanilla": vanilla_change,
        "pcgrad": projected_change,
        "gated": gated_change,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
