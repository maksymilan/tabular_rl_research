#!/usr/bin/env python3
"""Round-trip verification: compile SQL -> plan -> execute via harness, compare to gold SQL.

The single source of correctness: a compiled tool chain is *faithful* iff its result set equals
executing the gold SQL on the same database. Statuses:
  ok            - results match (faithful trajectory)
  mismatch      - compiled but produced a different result (compiler bug / wrong decomposition)
  compile_error - construct outside compiler support (honest coverage gap)
  exec_error    - the plan failed to execute (harness/translation bug)
"""
from __future__ import annotations

from compiler import Compiler, CompileError
from plan import run_plan


def _norm(rows: list[tuple]) -> list[str]:
    return sorted(repr(tuple(r)) for r in rows)


def round_trip(harness, sql: str):
    """Return (status, info)."""
    try:
        plan = Compiler().compile(sql)
    except CompileError as exc:
        return "compile_error", str(exc)
    try:
        got = run_plan(harness, plan)
        gold = harness.gold(sql)
    except Exception as exc:  # translation/execution failure
        return "exec_error", f"{type(exc).__name__}: {exc}"
    if _norm(got) == _norm(gold):
        return "ok", plan
    return "mismatch", {
        "plan": [repr(s) for s in plan],
        "got": got[:5],
        "gold": gold[:5],
    }
