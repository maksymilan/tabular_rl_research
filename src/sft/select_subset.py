#!/usr/bin/env python3
"""Select a small, high-quality SUBSET of verified train trajectories for the reflection/perception
data pipeline — instead of enriching all ~6.8k, we enrich <200 that PRESERVE the trajectory-length
distribution and maximise database coverage (≈ one per DB).

Why: the full set is overkill for the new (more expensive, LLM-in-the-loop) enrichment. A length-
stratified, DB-diverse subset keeps the difficulty spread of the original while cutting API cost and
training time by ~40x.

Outputs (under data/trajectories/):
  subset_<N>.jsonl        the selected skeleton trajectories (v2 relational backbone, no perception)
  subset_<N>.ids.json     {"subset": [...ids], "smoke": [...10 ids]} for downstream stages
The 10 smoke ids span the length range EVENLY (not all short), for the human-reviewed smoke test.

Usage:
  .venv/bin/python src/sft/select_subset.py [--n 180] [--per-db-cap 2] [--smoke 10]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "data", "trajectories", "spider_train_v2.jsonl")


def load_skeletons(path: str) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def length_targets(by_len: dict[int, list], n: int) -> dict[int, int]:
    """Proportional per-length quota that sums to ~n, with >=1 for every present length so the
    distribution's tail (rare long trajectories) is preserved, then trimmed/topped to hit n."""
    total = sum(len(v) for v in by_len.values())
    target = {L: max(1, round(n * len(v) / total)) for L, v in by_len.items()}
    # Reconcile the rounded sum to exactly n: add to / remove from the largest buckets first.
    order = sorted(by_len, key=lambda L: -len(by_len[L]))
    while sum(target.values()) > n:
        for L in reversed(order):  # trim smallest-population buckets but never below 1
            if sum(target.values()) <= n:
                break
            if target[L] > 1:
                target[L] -= 1
    while sum(target.values()) < n:
        for L in order:
            if sum(target.values()) >= n:
                break
            if target[L] < len(by_len[L]):
                target[L] += 1
    return target


def pick_db_diverse(cands: list[dict], k: int, used_db: collections.Counter, cap: int) -> list[dict]:
    """Pick k trajectories preferring databases used the fewest times so far (<= cap each)."""
    pool = sorted(cands, key=lambda t: (used_db[t["source"]["db_id"]], t["trajectory_id"]))
    chosen = []
    for t in pool:
        if len(chosen) >= k:
            break
        if used_db[t["source"]["db_id"]] < cap:
            chosen.append(t)
            used_db[t["source"]["db_id"]] += 1
    # If the cap blocked us from reaching k, relax and fill from the remainder.
    if len(chosen) < k:
        for t in pool:
            if len(chosen) >= k:
                break
            if t not in chosen:
                chosen.append(t)
                used_db[t["source"]["db_id"]] += 1
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=180, help="subset size (<200)")
    ap.add_argument("--per-db-cap", type=int, default=2)
    ap.add_argument("--smoke", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)

    skeletons = load_skeletons(SRC)
    by_len: dict[int, list] = collections.defaultdict(list)
    for t in skeletons:
        by_len[len(t["steps"])].append(t)

    target = length_targets(by_len, args.n)
    used_db: collections.Counter = collections.Counter()
    subset: list[dict] = []
    for L in sorted(by_len):
        cands = list(by_len[L])
        random.shuffle(cands)
        subset.extend(pick_db_diverse(cands, target[L], used_db, args.per_db_cap))

    # Smoke: EVENLY spread lengths (never all-short). Take distinct subset lengths, pick `smoke` of
    # them evenly across the range, one trajectory each (prefer a fresh DB).
    sub_by_len: dict[int, list] = collections.defaultdict(list)
    for t in subset:
        sub_by_len[len(t["steps"])].append(t)
    lengths = sorted(sub_by_len)
    if len(lengths) <= args.smoke:
        smoke_lengths = lengths
    else:
        idx = [round(i * (len(lengths) - 1) / (args.smoke - 1)) for i in range(args.smoke)]
        smoke_lengths = sorted({lengths[i] for i in idx})
    smoke = [sub_by_len[L][0] for L in smoke_lengths][: args.smoke]

    # Write artifacts.
    out_jsonl = os.path.join(ROOT, "data", "trajectories", f"subset_{args.n}.jsonl")
    out_ids = os.path.join(ROOT, "data", "trajectories", f"subset_{args.n}.ids.json")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for t in subset:
            f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
    json.dump(
        {"subset": [t["trajectory_id"] for t in subset], "smoke": [t["trajectory_id"] for t in smoke]},
        open(out_ids, "w", encoding="utf-8"), ensure_ascii=False, indent=2,
    )

    # Report: full vs subset length distribution (proportional preservation) + DB coverage.
    full_total, sub_total = len(skeletons), len(subset)
    print(f"selected {sub_total} / {full_total}  (DB coverage {len(used_db)} dbs, "
          f"max {max(used_db.values())}/db)")
    print(f"{'len':>4} {'full':>6} {'full%':>7} {'sub':>5} {'sub%':>7}")
    for L in sorted(by_len):
        fn, sn = len(by_len[L]), len(sub_by_len.get(L, []))
        print(f"{L:>4} {fn:>6} {100*fn/full_total:>6.1f}% {sn:>5} {100*sn/max(1,sub_total):>6.1f}%")
    print(f"\nsmoke {len(smoke)} (lengths {[len(t['steps']) for t in smoke]}):")
    for t in smoke:
        print(f"  {t['trajectory_id']:<22} len={len(t['steps']):<3} db={t['source']['db_id']}")
    print(f"\n-> {out_jsonl}\n-> {out_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
