#!/usr/bin/env python3
"""Splice a prior version's grounded `think` into new-schema trajectories at ZERO API cost.

The new schema may INJECT extra steps the source did not have (V2-ctx injects the read-only
perception steps describe_table / inspect_column / read_subtable). Those injected steps keep the
emitter's template think; every other (relational / memory / answer) step appears in the SAME ORDER
as in the source, so its filled think is reused by walking a pointer down the source steps. Where a
reused think names an old memory key (v1's `v3`), it is replaced with a neutral phrase.

Usage:
  # v1 -> v2a (equal length, no injection — pointer alignment degenerates to 1:1):
  .venv/bin/python src/sft/splice_think.py --split train
  # v2a -> v2-ctx (injected perception steps):
  .venv/bin/python src/sft/splice_think.py --split train \
      --source-suffix _v2_think --target-suffix _v2ctx --out-suffix _v2ctx_think
"""
from __future__ import annotations

import argparse
import json
import os
import re

PERCEPTION = {"describe_table", "inspect_column", "read_subtable"}   # injected; keep template think


def load_by_id(path: str) -> dict:
    out: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                out[t["trajectory_id"]] = t
    return out


def old_memory_keys(steps: list) -> list[str]:
    """v1 add_to_memory keys (e.g. 'v3') a reused think might name; replaced with a neutral phrase
    (no-op for a V2a source, which has no model-authored key)."""
    return [s["tool_call"]["arguments"].get("key") for s in steps
            if s["tool_call"]["tool"] == "add_to_memory" and s["tool_call"]["arguments"].get("key")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "dev"])
    ap.add_argument("--traj-dir", default="data/trajectories")
    ap.add_argument("--source-suffix", default="_think", help="filled-think source file suffix")
    ap.add_argument("--target-suffix", default="_v2", help="template-think target file suffix")
    ap.add_argument("--out-suffix", default="_v2_think", help="output file suffix")
    args = ap.parse_args()

    f = lambda suffix: os.path.join(args.traj_dir, f"spider_{args.split}{suffix}.jsonl")  # noqa: E731
    src = load_by_id(f(args.source_suffix))
    n = reused = subst = template = missing = 0

    with open(f(args.target_suffix), encoding="utf-8") as fin, open(f(args.out_suffix), "w", encoding="utf-8") as out:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            n += 1
            s_src = (src.get(t["trajectory_id"]) or {}).get("steps")
            if not s_src:
                missing += 1
                out.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
                continue
            oldkeys = old_memory_keys(s_src)
            j = 0                                          # pointer into the source's (filled) steps
            for st in t["steps"]:
                if st["tool_call"]["tool"] in PERCEPTION:  # injected step -> keep emitter template
                    template += 1
                    continue
                # align to the next source step of the same tool
                while j < len(s_src) and s_src[j]["tool_call"]["tool"] != st["tool_call"]["tool"]:
                    j += 1
                if j >= len(s_src):
                    template += 1
                    continue
                think = s_src[j].get("think", "")
                j += 1
                for k in oldkeys:
                    new = re.sub(rf"\b{re.escape(k)}\b", "the stored threshold", think)
                    if new != think:
                        subst += 1
                    think = new
                if think.strip():
                    st["think"] = think
                    reused += 1
                else:
                    template += 1
            out.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")

    print(f"[{args.split}] {n} trajectories -> {f(args.out_suffix)}")
    print(f"  steps: think reused {reused} (key-substituted {subst})  template-kept {template}  "
          f"| trajectories missing-in-source {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
