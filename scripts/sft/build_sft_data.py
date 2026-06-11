#!/usr/bin/env python3
"""Convert verified harness trajectories into LLaMA-Factory sharegpt SFT data.

Input : data/trajectories/spider_{split}.jsonl   (execution-verified + legal trajectories)
Output: data/sft/spider_v0_{split}.jsonl         (one sharegpt record per trajectory)
        data/sft/spider_v0_{split}.manifest.json (counts, token estimates, tool histogram)
        data/sft/dataset_info.snippet.json       (entry to merge into LLaMA-Factory's dataset_info.json)

Record shape: {"system": SYSTEM_PROMPT, "conversations": [human, gpt, observation, gpt, ...]}.
Loss is computed on "gpt" turns only; "observation" (tool outputs) and "human" are masked by
LLaMA-Factory. The terminal answer_from_context call is the last gpt turn (no observation after).

Run: .venv/bin/python scripts/sft/build_sft_data.py [train|dev|both] [--max-est-tokens 12000]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from protocol import SYSTEM_PROMPT, assistant_message, first_user_message, tool_output_message  # noqa: E402

CHARS_PER_TOKEN = 3.5  # rough for English+JSON; manifest reports char counts too


def convert(traj: dict) -> dict:
    conv = [{"from": "human",
             "value": first_user_message(traj["initial_state"]["dataset_overview"], traj["question"])}]
    steps = traj["steps"]
    for i, s in enumerate(steps):
        tc = s["tool_call"]
        conv.append({"from": "gpt",
                     "value": assistant_message(s.get("think", ""), tc["tool"], tc["arguments"])})
        if i < len(steps) - 1:  # terminal call has no observation
            conv.append({"from": "observation", "value": tool_output_message(s["tool_output"])})
    return {"system": SYSTEM_PROMPT, "conversations": conv}


def est_tokens(rec: dict) -> int:
    chars = len(rec["system"]) + sum(len(m["value"]) for m in rec["conversations"])
    return int(chars / CHARS_PER_TOKEN)


def build(split: str, max_est_tokens: int) -> dict:
    src = os.path.join(ROOT, "data", "trajectories", f"spider_{split}.jsonl")
    out_dir = os.path.join(ROOT, "data", "sft")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"spider_v0_{split}.jsonl")

    kept = dropped = 0
    toks: list[int] = []
    len_hist = collections.Counter()
    tool_hist = collections.Counter()
    with open(src) as f, open(out_path, "w") as out:
        for line in f:
            traj = json.loads(line)
            rec = convert(traj)
            t = est_tokens(rec)
            if t > max_est_tokens:
                dropped += 1
                continue
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
            toks.append(t)
            len_hist[len(traj["steps"])] += 1
            for s in traj["steps"]:
                tool_hist[s["tool_call"]["tool"]] += 1

    toks.sort()
    pct = lambda p: toks[min(len(toks) - 1, int(p * len(toks)))] if toks else 0  # noqa: E731
    manifest = {
        "split": split, "kept": kept, "dropped_overlong": dropped,
        "max_est_tokens": max_est_tokens,
        "est_tokens": {"p50": pct(.5), "p90": pct(.9), "p95": pct(.95),
                       "max": toks[-1] if toks else 0,
                       "mean": int(statistics.mean(toks)) if toks else 0},
        "trajectory_length_hist": dict(sorted(len_hist.items())),
        "tool_hist": dict(tool_hist.most_common()),
        "output": os.path.relpath(out_path, ROOT),
        "note": "dev file is for eval-loss / inspection only — dev is the held-out eval set, never train on it",
    }
    json.dump(manifest, open(out_path.replace(".jsonl", ".manifest.json"), "w"), indent=2)
    return manifest


# LLaMA-Factory registration: merge this into its data/dataset_info.json
DATASET_INFO = {
    "spider_tools_v0": {
        "file_name": "spider_v0_train.jsonl",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system"},
        "tags": {"role_tag": "from", "content_tag": "value",
                 "user_tag": "human", "assistant_tag": "gpt", "observation_tag": "observation"},
    }
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", nargs="?", default="both", choices=["train", "dev", "both"])
    ap.add_argument("--max-est-tokens", type=int, default=12000)
    args = ap.parse_args()

    for split in (["train", "dev"] if args.split == "both" else [args.split]):
        m = build(split, args.max_est_tokens)
        print(f"{split}: kept {m['kept']}  dropped_overlong {m['dropped_overlong']}  "
              f"est_tokens p50/p95/max = {m['est_tokens']['p50']}/{m['est_tokens']['p95']}/{m['est_tokens']['max']}")
        print(f"  tools: {m['tool_hist']}")
        print(f"  -> {m['output']}")

    snip = os.path.join(ROOT, "data", "sft", "dataset_info.snippet.json")
    json.dump(DATASET_INFO, open(snip, "w"), indent=2)
    print(f"dataset_info snippet -> {os.path.relpath(snip, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
