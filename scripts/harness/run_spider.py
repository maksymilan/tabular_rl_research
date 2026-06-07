#!/usr/bin/env python3
"""Execution-verified evaluation on REAL Spider databases.

For each (db_id, question, gold_sql) in Spider's train_spider.json, open the matching SQLite DB
and round-trip: compile -> run the tool chain via the harness -> compare to executing the gold SQL
on the real database. This is *execution-verified* coverage (stronger than compile coverage in
run_all.py): it reveals real decomposition bugs (mismatches) and translation errors, not just parse
gaps.

Run:  .venv/bin/python scripts/harness/run_spider.py [N]      (N = sample size, default 1500)
"""
from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from executor import Harness
from verify import round_trip

SPIDER = os.path.join(ROOT, "data/spider_data")
DBROOT = os.path.join(SPIDER, "database")
TRAIN = os.path.join(SPIDER, "train_spider.json")


def db_path(db_id: str) -> str:
    return os.path.join(DBROOT, db_id, f"{db_id}.sqlite")


def bucket(status: str, info) -> str:
    if status == "mismatch":
        return "mismatch (wrong decomposition)"
    if status == "compile_error":
        return f"compile: {str(info).split(':')[0][:46]}"
    if status == "exec_error":
        return f"exec: {str(info).split(':')[0][:46]}"
    return status


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    data = json.load(open(TRAIN))[:limit]

    counts = collections.Counter()
    reasons = collections.Counter()
    examples: dict[str, dict] = {}

    for ex in data:
        db_id, sql = ex["db_id"], ex["query"]
        path = db_path(db_id)
        if not os.path.exists(path):
            counts["no_db"] += 1
            continue
        h = Harness(path)  # fresh per query (views accumulate within a run)
        status, info = round_trip(h, sql)
        counts[status] += 1
        if status != "ok":
            key = bucket(status, info)
            reasons[key] += 1
            examples.setdefault(key, {"sql": sql, "info": info})

    with_db = sum(counts.values()) - counts["no_db"]
    ok = counts["ok"]
    print(f"Spider train, {with_db} queries with a DB (of {len(data)} sampled)\n")
    print(f"  EXECUTION-VERIFIED: {ok}/{with_db} ({100.0 * ok / with_db:.1f}%)")
    for k in ("mismatch", "compile_error", "exec_error", "no_db"):
        if counts[k]:
            print(f"  {k:14}: {counts[k]}")
    print("\nTop failure buckets:")
    for key, n in reasons.most_common(12):
        ex = examples[key]["sql"]
        ex = (ex[:80] + "…") if len(ex) > 80 else ex
        print(f"  {n:4}  {key}")
        print(f"        e.g. {ex}")
    # one mismatch detail (most useful for debugging the decomposition)
    mm = next((v for k, v in examples.items() if k.startswith("mismatch")), None)
    if mm and isinstance(mm["info"], dict):
        print("\nExample mismatch detail:")
        print("  sql :", mm["sql"][:120])
        print("  plan:", mm["info"]["plan"])
        print("  got :", mm["info"]["got"], "\n  gold:", mm["info"]["gold"])


if __name__ == "__main__":
    main()
