#!/usr/bin/env python3
"""Batch-generate execution-verified Spider trajectories (training data).

For each Spider (db_id, question, gold_sql): open the real SQLite DB, emit a trajectory
(compile -> run the tool chain -> verify against the gold SQL's result), and keep ONLY trajectories
that are execution-verified AND structurally legal. Failures are skipped and bucketed.

Run:   .venv/bin/python src/harness/gen_trajectories.py [N] [train|dev]
Output: data/trajectories/spider_<split>.jsonl   (one trajectory per line; /data is gitignored)
        src/harness/sample_trajectories/*.json   (a few committed samples for inspection)
        printed summary + data/trajectories/spider_<split>.manifest.json
"""
from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from compiler import CompileError          # noqa: E402
from emitter import emit, validate         # noqa: E402
from executor import Harness               # noqa: E402
from memory_semantics import MemoryGroundingError  # noqa: E402

SPIDER = os.path.join(ROOT, "data/spider_data")


def db_path(db_id: str) -> str:
    return os.path.join(SPIDER, "database", db_id, f"{db_id}.sqlite")


def main() -> int:
    n = None
    split = "train"
    tag = ""                      # output suffix: "" = v1 path (default), "_v2" = data-v2 path
    for a in sys.argv[1:]:
        if a.isdigit():
            n = int(a)
        elif a in ("train", "dev"):
            split = a
        elif a.startswith("--tag="):
            tag = a.split("=", 1)[1]
    src = os.path.join(SPIDER, "train_spider.json" if split == "train" else "dev.json")
    if not os.path.exists(src):
        print(f"missing {src} — download Spider DBs first (see src/harness/README.md)")
        return 1
    data = json.load(open(src))
    if n:
        data = data[:n]

    out_dir = os.path.join(ROOT, "data", "trajectories")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"spider_{split}{tag}.jsonl")
    sample_dir = os.path.join(HERE, "sample_trajectories")
    os.makedirs(sample_dir, exist_ok=True)

    stats = collections.Counter()
    by_len = collections.Counter()
    by_len_sample: dict[int, dict] = {}   # one representative trajectory per step-count
    kept = 0

    with open(out_path, "w") as out:
        for i, ex in enumerate(data):
            dbp = db_path(ex["db_id"])
            if not os.path.exists(dbp):
                stats["no_db"] += 1
                continue
            try:
                h = Harness(dbp)
                traj = emit(h, ex["question"], ex["query"], dataset="spider",
                            db_id=ex["db_id"], trajectory_id=f"spider_{split}_{i}")
            except CompileError:
                stats["compile_error"] += 1
                continue
            except MemoryGroundingError as e:
                stats[f"memory_reject_{e.code}"] += 1
                continue
            except Exception:
                stats["exec_error"] += 1
                continue
            if traj["label_status"] != "verified":
                stats["mismatch"] += 1
                continue
            if validate(traj):
                stats["illegal"] += 1
                continue
            out.write(json.dumps(traj, default=str) + "\n")
            kept += 1
            L = len(traj["steps"])
            by_len[L] += 1
            by_len_sample.setdefault(L, traj)   # keep the first trajectory seen at each length

    # write one readable sample per distinct trajectory length, for inspection
    bylen_dir = os.path.join(sample_dir, f"by_length{tag}")
    os.makedirs(bylen_dir, exist_ok=True)
    for L, t in sorted(by_len_sample.items()):
        json.dump(t, open(os.path.join(bylen_dir, f"len_{L:02d}.json"), "w"),
                  indent=2, default=str)
    samples = list(by_len_sample.values())

    total = len(data)
    manifest = {
        "split": split, "total_queries": total, "kept_verified_legal": kept,
        "keep_rate": round(kept / total, 4) if total else 0,
        "skipped": dict(stats), "trajectory_length_hist": dict(sorted(by_len.items())),
        "output": os.path.relpath(out_path, ROOT),
    }
    json.dump(manifest, open(out_path.replace(".jsonl", ".manifest.json"), "w"), indent=2)

    print(f"Spider {split}: {total} queries")
    print(f"  KEPT (verified + legal): {kept} ({100 * kept / total:.1f}%)")
    for k, v in stats.most_common():
        print(f"  skip {k:14}: {v}")
    print(f"  trajectory length histogram: {dict(sorted(by_len.items()))}")
    print(f"  -> {os.path.relpath(out_path, ROOT)}  (+ {len(samples)} samples in sample_trajectories/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
