"""Shared test fixtures + a tiny dependency-free test harness (no pytest)."""
from __future__ import annotations

import os
import sys

# make sibling modules (executor, compiler, plan, verify) importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from executor import Harness  # noqa: E402


class T:
    """Minimal test collector: t.check(name, cond, detail)."""

    def __init__(self, module: str):
        self.module = module
        self.passed = 0
        self.failed = 0
        self.fails: list[str] = []

    def check(self, name: str, cond: bool, detail: str = "") -> None:
        if cond:
            self.passed += 1
        else:
            self.failed += 1
            self.fails.append(f"{name}: {detail}")

    def result(self) -> tuple[int, int, list[str]]:
        return self.passed, self.failed, self.fails


def employees_db() -> Harness:
    h = Harness(":memory:")
    h.conn.executescript(
        """
        CREATE TABLE employees(id INT, name TEXT, dept TEXT, salary INT, age INT);
        INSERT INTO employees VALUES
          (1,'A','eng',1200,30),(2,'B','eng',900,40),(3,'C','sales',1500,25),
          (4,'D','sales',1100,35),(5,'E','eng',2000,50),(6,'F','hr',800,28);
        CREATE TABLE depts(dept TEXT, location TEXT, budget INT);
        INSERT INTO depts VALUES ('eng','SF',5000),('sales','NY',3000),('hr','LA',1000);
        CREATE TABLE sales(item TEXT, price REAL, qty INT, dept TEXT);
        INSERT INTO sales VALUES ('x',10.0,3,'eng'),('y',20.0,1,'sales'),('z',5.0,4,'eng');
        """
    )
    h.register_sources()
    return h


def norm(rows) -> list[str]:
    return sorted(repr(tuple(r)) for r in rows)
