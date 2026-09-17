#!/usr/bin/env python3
"""Retokenize persisted rollout audits to diagnose long training transitions.

Normal turns are reconstructed from their exact model-visible messages and
decoded response. Length-truncated synthetic audit turns use the exact token
counts persisted by the rollout collector. The result is suitable for deciding
an OOM recovery boundary; it is not a replacement for persisted policy token
IDs in a formal numerical-equivalence audit.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--policy-global-step",
        type=int,
        default=-1,
        help="negative audits every persisted policy step",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rows = [json.loads(line) for line in args.rollouts.open() if line.strip()]
    selected = [
        row
        for row in rows
        if args.policy_global_step < 0
        or int(row.get("policy_global_step", -1)) == args.policy_global_step
    ]
    trainable = [row for row in selected if bool(row.get("process_update"))]

    transitions: list[dict[str, object]] = []
    for row in trainable:
        for turn in row.get("turns") or []:
            if "model_input" in turn and "model_output" in turn:
                prompt_ids = tokenizer.apply_chat_template(
                    turn["model_input"],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                response_ids = tokenizer(
                    turn["model_output"], add_special_tokens=False
                ).input_ids
                prompt_tokens = len(prompt_ids)
                response_tokens = len(response_ids)
                length_source = "retokenized_model_text"
            else:
                truncation = turn.get("generation_truncation") or {}
                if not truncation:
                    raise KeyError(
                        "turn lacks both model text and generation truncation evidence: "
                        f"{row.get('trajectory_id')} turn={turn.get('turn_index')}"
                    )
                prompt_tokens = int(truncation["prompt_tokens"])
                response_tokens = int(truncation["completion_tokens"])
                length_source = "exact_generation_truncation_evidence"
            transitions.append(
                {
                    "example_index": int(row["example_index"]),
                    "policy_global_step": int(row.get("policy_global_step", -1)),
                    "trajectory_id": row["trajectory_id"],
                    "correct": bool(row["correct"]),
                    "legal": bool(row["legal"]),
                    "turn_index": int(turn["turn_index"]),
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens,
                    "total_tokens": prompt_tokens + response_tokens,
                    "length_source": length_source,
                }
            )

    totals = [int(item["total_tokens"]) for item in transitions]
    thresholds = (4096, 6144, 7168, 8192, 9216, 10240, 11264, 12288)
    affected = {
        str(threshold): {
            "transitions": sum(value > threshold for value in totals),
            "trajectories": len(
                {
                    str(item["trajectory_id"])
                    for item in transitions
                    if int(item["total_tokens"]) > threshold
                }
            ),
            "examples": len(
                {
                    int(item["example_index"])
                    for item in transitions
                    if int(item["total_tokens"]) > threshold
                }
            ),
        }
        for threshold in thresholds
    }
    context_caps: dict[str, dict[str, object]] = {}
    for cap in (12288, 13312, 14336, 15360, 16000):
        overflowing = [
            item
            for item in transitions
            if int(item["prompt_tokens"]) + args.max_new_tokens > cap
        ]
        context_caps[str(cap)] = {
            "transitions_if_independently_counted": len(overflowing),
            "trajectories_excluded_at_first_overflow": len(
                {str(item["trajectory_id"]) for item in overflowing}
            ),
            "examples": len(
                {int(item["example_index"]) for item in overflowing}
            ),
            "trajectory_ids": sorted(
                {str(item["trajectory_id"]) for item in overflowing}
            ),
        }

    result_distribution = Counter(
        (bool(row.get("correct")), bool(row.get("legal"))) for row in selected
    )
    result = {
        "schema_version": "rollout-transition-length-audit-v1",
        "length_contract": (
            "normal turns retokenized from persisted model text; generation-length "
            "turns use exact persisted token counts"
        ),
        "policy_global_step": args.policy_global_step,
        "max_new_tokens": args.max_new_tokens,
        "rollout_rows": len(selected),
        "trainable_trajectories": len(trainable),
        "excluded_trajectories": len(selected) - len(trainable),
        "transition_count": len(transitions),
        "result_distribution": {
            f"correct={correct},legal={legal}": count
            for (correct, legal), count in sorted(result_distribution.items())
        },
        "max_total_tokens": max(totals, default=0),
        "max_prompt_tokens": max(
            (int(item["prompt_tokens"]) for item in transitions), default=0
        ),
        "max_response_tokens": max(
            (int(item["response_tokens"]) for item in transitions), default=0
        ),
        "thresholds": affected,
        "context_caps": context_caps,
        "top20": sorted(
            transitions,
            key=lambda item: int(item["total_tokens"]),
            reverse=True,
        )[:20],
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
