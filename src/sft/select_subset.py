#!/usr/bin/env python3
"""Select a high-quality CURRENT-PROTOCOL skeleton subset for the perception/recovery data pipeline.

The output is still a skeleton trajectory file, not final SFT-ready clean data. For clean SFT, run
`enrich_traj.py` on the selected ids and train only on the resulting `quality_status == "ready"`
v3-enriched file.

Why: the full set is overkill for the new (more expensive, LLM-in-the-loop) enrichment. A length-
stratified, DB-diverse subset keeps the difficulty spread of the original while cutting API cost and
training time.

Outputs (under data/trajectories/):
  subset_<N>.jsonl        selected current-protocol skeleton trajectories (v3, no memory)
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
DEFAULT_SOURCE = os.path.join(ROOT, "data", "trajectories", "spider_train_v3.jsonl")
MEMORY_TOOLS = {"add_to_memory", "refine_memory"}


def has_memory_residue(traj: dict) -> bool:
    text = json.dumps(traj, ensure_ascii=False)
    if any(marker in text for marker in ("memory_id", "supporting_memory_ids", "mem_")):
        return True
    for step in traj.get("steps", []):
        if (step.get("tool_call") or {}).get("tool") in MEMORY_TOOLS:
            return True
    return False


def load_skeletons(path: str, *, allow_legacy_source: bool = False) -> list[dict]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if line:
                traj = json.loads(line)
                schema_version = str(traj.get("schema_version", ""))
                if not allow_legacy_source and not schema_version.startswith("v3"):
                    raise ValueError(
                        f"{path}:{line_no}: expected current v3 schema, got {schema_version!r}. "
                        "Regenerate with `src/harness/gen_trajectories.py train --tag=_v3` or pass "
                        "--allow-legacy-source only for migration/debugging."
                    )
                if not allow_legacy_source and has_memory_residue(traj):
                    raise ValueError(
                        f"{path}:{line_no}: memory residue found in skeleton source; do not use this "
                        "file for current no-memory SFT data."
                    )
                out.append(traj)
    return out


def load_excluded_ids(paths: list[str]) -> set[str]:
    excluded = set()
    for path in paths:
        if not path:
            continue
        with open(path, encoding="utf-8") as f:
            text = f.read()
            stripped = text.lstrip()
            if not stripped:
                continue
            if stripped[0] in "[{":
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = None
                if isinstance(payload, list):
                    excluded.update(str(item) for item in payload)
                    continue
                if isinstance(payload, dict):
                    if "trajectory_id" in payload:
                        excluded.add(payload["trajectory_id"])
                    for value in payload.values():
                        if isinstance(value, list):
                            excluded.update(str(item) for item in value)
                    continue
            for line in text.splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                if "trajectory_id" in item:
                    excluded.add(item["trajectory_id"])
    return excluded


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
    ap.add_argument("--out-prefix", default=None,
                    help="output prefix under data/trajectories; default subset_<N>")
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                    help="current-protocol skeleton JSONL source; default spider_train_v3.jsonl")
    ap.add_argument("--allow-legacy-source", action="store_true",
                    help="allow non-v3/memory-bearing sources for one-off migration/debugging only")
    ap.add_argument("--exclude-ids", action="append", default=[],
                    help="JSON/JSONL file containing trajectory ids to exclude; may be repeated")
    args = ap.parse_args()
    random.seed(args.seed)

    excluded = load_excluded_ids(args.exclude_ids)
    skeletons = [
        t for t in load_skeletons(args.source, allow_legacy_source=args.allow_legacy_source)
        if t["trajectory_id"] not in excluded
    ]
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
    out_prefix = args.out_prefix or f"subset_{args.n}"
    out_jsonl = os.path.join(ROOT, "data", "trajectories", f"{out_prefix}.jsonl")
    out_ids = os.path.join(ROOT, "data", "trajectories", f"{out_prefix}.ids.json")
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for t in subset:
            f.write(json.dumps(t, ensure_ascii=False, default=str) + "\n")
    json.dump(
        {"subset": [t["trajectory_id"] for t in subset], "smoke": [t["trajectory_id"] for t in smoke]},
        open(out_ids, "w", encoding="utf-8"), ensure_ascii=False, indent=2,
    )

    # Report: full vs subset length distribution (proportional preservation) + DB coverage.
    full_total, sub_total = len(skeletons), len(subset)
    print(f"source {os.path.relpath(args.source, ROOT)}")
    print(f"selected {sub_total} / {full_total}  (DB coverage {len(used_db)} dbs, "
          f"max {max(used_db.values())}/db)")
    if excluded:
        print(f"excluded {len(excluded)} trajectory ids")
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
