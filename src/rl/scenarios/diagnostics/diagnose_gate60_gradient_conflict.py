#!/usr/bin/env python3
"""Measure positive/negative and think/action gradient geometry on Gate60 rollouts.

This is a read-only diagnostic.  It reconstructs the exact causal chat prompt and
response text from the saved rollout, computes ordinary binary group GRPO
advantages, and measures gradients on the adapter parameters only.  Gold SQL is
never read.  The diagnostic is intentionally separate from the trainer: it does
not perform optimizer steps or alter any rollout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from rl.diagnostics.gradient import (
    GradientScorer,
    build_rows,
    geometry,
    group_advantages,
    tool_mask,
)
from rl.diagnostics.records import load_jsonl


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--example-index", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_jsonl(args.rollouts)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    grouped = build_rows(
        records, tokenizer, args.step, args.limit_groups, args.example_index
    )
    scorer = GradientScorer(args.base_model, args.adapter, args.device)
    groups = []
    for example_index, rows in grouped.items():
        positives = sum(row["advantage"] > 0 for row in rows)
        negatives = sum(row["advantage"] < 0 for row in rows)
        if not positives or not negatives:
            continue
        pos_full = scorer.aggregate(rows, only_tool=False, sign="positive")
        neg_full = scorer.aggregate(rows, only_tool=False, sign="negative")
        pos_tool = scorer.aggregate(rows, only_tool=True, sign="positive")
        neg_tool = scorer.aggregate(rows, only_tool=True, sign="negative")
        groups.append(
            {
                "example_index": example_index,
                "rows": len(rows),
                "positive_rows": positives,
                "negative_rows": negatives,
                "response_tokens": sum(len(row["response_ids"]) for row in rows),
                "tool_tokens": sum(sum(row["tool_mask"]) for row in rows),
                "full_positive_negative": geometry(pos_full, neg_full),
                "tool_positive_negative": geometry(pos_tool, neg_tool),
                "positive_full_tool": geometry(pos_full, pos_tool),
                "negative_full_tool": geometry(neg_full, neg_tool),
            }
        )
        print(json.dumps(groups[-1], ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "gate60-gradient-conflict-audit-v1",
        "gold_sql_read": False,
        "measurement": "saved causal prompts/responses; adapter gradients only; no rollout or optimizer update",
        "step": args.step,
        "groups": groups,
        "group_count": len(groups),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
