#!/usr/bin/env python3
"""Direct reasoning/tool gradient geometry on one real Gate60 mixed K8 group."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from rl.diagnostics.gradient import GradientScorer, build_rows
from rl.diagnostics.records import load_jsonl


def dot(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    return sum(float((a * b).sum()) for a, b in zip(left, right, strict=True))


def geometry(left: list[torch.Tensor], right: list[torch.Tensor]) -> dict[str, float]:
    left_norm = math.sqrt(max(dot(left, left), 0.0))
    right_norm = math.sqrt(max(dot(right, right), 0.0))
    cross = dot(left, right)
    cosine = cross / (left_norm * right_norm) if left_norm and right_norm else 0.0
    return {
        "left_norm": left_norm,
        "right_norm": right_norm,
        "cosine": cosine,
        "sum_over_norm_sum": (
            math.sqrt(max(left_norm * left_norm + right_norm * right_norm + 2.0 * cross, 0.0))
            / (left_norm + right_norm)
            if left_norm + right_norm
            else 0.0
        ),
    }


def aggregate(
    scorer: GradientScorer, rows: list[dict[str, Any]], sign: str, mode: str
) -> list[torch.Tensor]:
    selected = [row for row in rows if (row["advantage"] > 0) == (sign == "positive")]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--example-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
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
    gradients: dict[str, dict[str, list[torch.Tensor]]] = {}
    for sign in ("positive", "negative"):
        gradients[sign] = {
            mode: aggregate(scorer, rows, sign, mode) for mode in ("think", "tool", "full")
        }
    output = {
        "schema_version": "gate60-reason-tool-gradient-geometry-v1",
        "gold_sql_read": False,
        "measurement": "ordinary binary group GRPO advantage; saved causal prompts/responses; no optimizer update",
        "step": args.step,
        "example_index": args.example_index,
        "rows": len(rows),
        "positive_rows": sum(row["advantage"] > 0 for row in rows),
        "negative_rows": sum(row["advantage"] < 0 for row in rows),
        "reason_tool": {
            sign: geometry(gradients[sign]["think"], gradients[sign]["tool"])
            for sign in ("positive", "negative")
        },
        "full_reason": {
            sign: geometry(gradients[sign]["full"], gradients[sign]["think"])
            for sign in ("positive", "negative")
        },
        "full_tool": {
            sign: geometry(gradients[sign]["full"], gradients[sign]["tool"])
            for sign in ("positive", "negative")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
