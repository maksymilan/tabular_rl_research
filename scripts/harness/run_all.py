#!/usr/bin/env python3
"""Integration runner: runs every module's unit tests, the round-trip suite, and a Spider
compile-coverage probe, then writes a markdown test report (REPORT.md).

Run:  .venv/bin/python scripts/harness/run_all.py
"""
from __future__ import annotations

import collections
import datetime
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tests"))

from compiler import Compiler, CompileError  # noqa: E402

import test_executor          # noqa: E402
import test_plan              # noqa: E402
import test_compiler          # noqa: E402
import test_verify            # noqa: E402

SPIDER = os.path.join(ROOT, "data/spider/spider/train-00000-of-00001.parquet")


def run_unit_tests():
    """Run each module's run() -> list of (module, passed, failed, fails)."""
    results = []
    for mod in (test_executor, test_plan, test_compiler, test_verify):
        passed, failed, fails = mod.run()
        results.append((mod.run.__module__, passed, failed, fails))
    return results


def spider_queries(limit: int) -> list[str]:
    if not os.path.exists(SPIDER):
        return []
    out = subprocess.run(
        ["duckdb", "-json", "-c",
         f"SELECT query FROM read_parquet('{SPIDER}') LIMIT {limit}"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return []
    return [r["query"] for r in json.loads(out.stdout or "[]")]


def spider_compile_coverage(limit: int = 2000):
    """Compile (parse + decompose) real Spider SQL. NOTE: no DB execution here (the HF parquet
    ships no SQLite DBs), so this measures *compile* coverage, not verified correctness."""
    queries = spider_queries(limit)
    total = len(queries)
    ok = 0
    reasons = collections.Counter()
    examples: dict[str, str] = {}
    for q in queries:
        try:
            Compiler().compile(q)
            ok += 1
        except CompileError as exc:
            key = str(exc).split(":")[0][:48]
            reasons[key] += 1
            examples.setdefault(key, q)
    return {"total": total, "ok": ok, "reasons": reasons, "examples": examples}


def write_report(unit, cov, path):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    L = []
    L.append("# Harness + Compiler — Test Report")
    L.append(f"\nGenerated {now} · `python scripts/harness/run_all.py`\n")

    tot_p = sum(p for _, p, _, _ in unit)
    tot_f = sum(f for _, _, f, _ in unit)
    L.append(f"## Unit tests — {tot_p}/{tot_p + tot_f} passed\n")
    L.append("| module | passed | failed |")
    L.append("|---|---|---|")
    for mod, p, f, _ in unit:
        L.append(f"| {mod} | {p} | {f} |")
    fails = [(m, msg) for m, _, _, fl in unit for msg in fl]
    if fails:
        L.append("\n**Failures:**")
        for m, msg in fails:
            L.append(f"- `{m}`: {msg}")

    if cov["total"]:
        pct = 100.0 * cov["ok"] / cov["total"]
        L.append(f"\n## Spider compile coverage — {cov['ok']}/{cov['total']} ({pct:.1f}%)\n")
        L.append("Parse + decompose of real Spider gold SQL. *Compile coverage only* — the HF "
                 "parquet ships no SQLite DBs, so results are not execution-verified here.\n")
        L.append("Top unsupported-construct buckets:\n")
        L.append("| count | reason | example |")
        L.append("|---|---|---|")
        for reason, n in cov["reasons"].most_common(12):
            ex = cov["examples"][reason].replace("|", "\\|")
            ex = (ex[:70] + "…") if len(ex) > 70 else ex
            L.append(f"| {n} | {reason} | `{ex}` |")
    else:
        L.append("\n## Spider compile coverage — skipped (no parquet / duckdb)\n")

    L.append("\n## Notes\n")
    L.append("- Round-trip suite (`test_verify`) is the correctness backbone: each case is "
             "compiled, executed via the harness, and checked against gold SQL on a synthetic DB.")
    L.append("- Spider coverage measures how much of real-world SQL the compiler structurally "
             "handles; failure buckets are the prioritized backlog (joins-with-ambiguous-columns, "
             "subqueries, set ops, window, etc.).")
    L.append("- Pipeline: `compiler.py` (SQL→Plan) · `plan.py` (run) · `executor.py` (tools→SQL) · "
             "`verify.py` (round-trip). See `tool_design/final_tool_design.md`.")
    open(path, "w").write("\n".join(L) + "\n")


def main():
    unit = run_unit_tests()
    cov = spider_compile_coverage()
    report = os.path.join(HERE, "REPORT.md")
    write_report(unit, cov, report)

    tot_p = sum(p for _, p, _, _ in unit)
    tot_f = sum(f for _, _, f, _ in unit)
    print("UNIT TESTS")
    for mod, p, f, fails in unit:
        print(f"  {mod:28} {p:3} passed  {f:2} failed")
        for msg in fails:
            print(f"      FAIL {msg[:120]}")
    if cov["total"]:
        print(f"SPIDER compile coverage: {cov['ok']}/{cov['total']} "
              f"({100.0 * cov['ok'] / cov['total']:.1f}%)")
    print(f"report -> {os.path.relpath(report, ROOT)}")
    return 0 if tot_f == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
