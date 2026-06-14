#!/usr/bin/env python3
"""Splice v1's grounded `think` into v2 trajectories at ZERO API cost.

v2 differs from v1 only by sidecar `references`/`produces`, semantic memory keys
(`v3` -> `min_min_dew_point_f`), and `answer.supporting_memory_ids`. Step COUNT and TOOL
SEQUENCE are unchanged, so v1's LLM-filled reasoning maps 1:1 by step index. Where a v1 think
NAMES the old memory key (e.g. "...store it as v3"), substitute the new semantic key so the
reasoning stays consistent with the renamed memory; otherwise the think is reused verbatim.
Steps with no usable v1 think keep the emitter's template think (no model call).

Usage:
  .venv/bin/python src/sft/splice_think.py --split train
  .venv/bin/python src/sft/splice_think.py --split dev
  in : data/trajectories/spider_{split}_think.jsonl  (v1, think-filled)
       data/trajectories/spider_{split}_v2.jsonl      (v2, template think)
  out: data/trajectories/spider_{split}_v2_think.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import re


def load_by_id(path: str) -> dict:
    out: dict = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                t = json.loads(line)
                out[t["trajectory_id"]] = t
    return out


def old_memory_keys(v1_steps: list) -> list[str]:
    """The v1 add_to_memory keys (e.g. 'v3') a v1 think might name. In V2a there is no model-authored
    key, so any such mention in a reused non-memory think is replaced with a neutral phrase."""
    return [s["tool_call"]["arguments"].get("key") for s in v1_steps
            if s["tool_call"]["tool"] == "add_to_memory" and s["tool_call"]["arguments"].get("key")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "dev"])
    ap.add_argument("--traj-dir", default="data/trajectories")
    args = ap.parse_args()

    v1_path = os.path.join(args.traj_dir, f"spider_{args.split}_think.jsonl")
    v2_path = os.path.join(args.traj_dir, f"spider_{args.split}_v2.jsonl")
    out_path = os.path.join(args.traj_dir, f"spider_{args.split}_v2_think.jsonl")

    v1 = load_by_id(v1_path)
    n = reused = subst_steps = template_steps = struct_mismatch = missing = 0

    with open(v2_path, encoding="utf-8") as f, open(out_path, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t2 = json.loads(line)
            n += 1
            t1 = v1.get(t2["trajectory_id"])
            if t1 is None:
                missing += 1
                out.write(json.dumps(t2, ensure_ascii=False, default=str) + "\n")
                continue
            s1, s2 = t1["steps"], t2["steps"]
            if len(s1) != len(s2):
                struct_mismatch += 1
                out.write(json.dumps(t2, ensure_ascii=False, default=str) + "\n")
                continue
            oldkeys = old_memory_keys(s1)
            for st1, st2 in zip(s1, s2):
                tool = st2["tool_call"]["tool"]
                if tool == "add_to_memory":
                    template_steps += 1            # V2a memory think is the emitter template (cite source, not value)
                    continue
                if st1["tool_call"]["tool"] != tool:
                    template_steps += 1            # tool drift -> keep v2 template think
                    continue
                think = st1.get("think", "")
                changed = False
                for k in oldkeys:
                    nt = re.sub(rf"\b{re.escape(k)}\b", "the stored threshold", think)
                    if nt != think:
                        changed = True
                    think = nt
                if think.strip():
                    st2["think"] = think
                    reused += 1
                    subst_steps += int(changed)
                else:
                    template_steps += 1            # empty v1 think -> keep template
            out.write(json.dumps(t2, ensure_ascii=False, default=str) + "\n")

    print(f"[{args.split}] {n} v2 trajectories -> {out_path}")
    print(f"  steps: think reused {reused}  (key-substituted {subst_steps})  "
          f"template-kept {template_steps}")
    print(f"  trajectories: missing-in-v1 {missing}  step-count-mismatch {struct_mismatch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
