#!/usr/bin/env python3
"""One-step carrier ablation on a real Gate60 K8 group.

The credit is fixed to the ordinary binary group GRPO advantage.  This diagnostic
only changes the response-token carrier: full response, tool-only, or a span
coupling (0.5 * mean-think + 1.0 * mean-tool).  Each mode starts from a fresh
adapter, takes one small SGD-style policy step, and measures the exact sampled
full/tool/span log-prob changes.  It never reads Gold SQL and never changes the
saved rollout or the production trainer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rl.scenarios.diagnostics.diagnose_gate60_gradient_conflict import GradientScorer, build_rows, load_jsonl


def scores(scorer: GradientScorer, rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {
        "full": [],
        "think": [],
        "tool": [],
        "span": [],
    }
    with torch.no_grad():
        for start in range(0, len(rows), 2):
            batch_scores = scorer.score_batch_modes(rows[start : start + 2])
            for mode, tensors in batch_scores.items():
                values[mode].extend(float(value) for value in tensors.detach().cpu())
    return values


def aggregate(scorer: GradientScorer, rows: list[dict[str, Any]], mode: str) -> list[torch.Tensor]:
    """Compute the fixed binary-GRPO update in one pass over all nonzero rows."""
    scorer.model.zero_grad(set_to_none=True)
    scale = 1.0 / len(rows)
    for start in range(0, len(rows), 2):
        batch = rows[start : start + 2]
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
    grouped = build_rows(records, tokenizer, args.step, 0, args.example_index)
    rows = grouped.get(args.example_index, [])
    if not rows:
        raise SystemExit("selected group has no nonzero mixed GRPO transitions")
    if not any(row["advantage"] > 0 for row in rows) or not any(row["advantage"] < 0 for row in rows):
        raise SystemExit("selected group does not contain both positive and negative transitions")

    results = {}
    base_scores: dict[str, list[float]] | None = None
    for mode in ("full", "tool", "span"):
        scorer = GradientScorer(args.base_model, args.adapter, args.device)
        if base_scores is None:
            base_scores = scores(scorer, rows)
        gradients = aggregate(scorer, rows, mode)
        with torch.no_grad():
            for parameter, gradient in zip(scorer.parameters, gradients, strict=True):
                parameter.add_(-args.learning_rate * gradient.to(device=parameter.device, dtype=parameter.dtype))
        after = scores(scorer, rows)
        results[mode] = {
            "learning_rate": args.learning_rate,
            "update_carrier": mode,
            "full_score_change": alignment(rows, base_scores["full"], after["full"]),
            "tool_score_change": alignment(rows, base_scores["tool"], after["tool"]),
            "span_score_change": alignment(rows, base_scores["span"], after["span"]),
        }
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(json.dumps({"mode": mode, **results[mode]}, ensure_ascii=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "schema_version": "gate60-carrier-one-step-audit-v1",
                "gold_sql_read": False,
                "measurement": "ordinary binary group GRPO advantage; fresh adapter per carrier; one SGD-style step; no rollout",
                "step": args.step,
                "example_index": args.example_index,
                "rows": len(rows),
                "positive_rows": sum(row["advantage"] > 0 for row in rows),
                "negative_rows": sum(row["advantage"] < 0 for row in rows),
                "learning_rate": args.learning_rate,
                "modes": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
