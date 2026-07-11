#!/usr/bin/env python3
"""Common dataset-task representation for harness/eval adapters.

This IR is intentionally smaller than a trajectory. It describes one executable task and the
database backend needed to run it. SQL-to-tool compilation may consume the IR for support checks,
but SFT data construction should not depend on compiler-derived trajectories by default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetTask:
    dataset: str
    split: str
    example_index: int
    example_id: str
    db_id: str
    question: str
    backend: str
    db_path: str | None = None
    gold_sql: str | None = None
    gold_sql_path: str | None = None
    gold_exec_results: tuple[str, ...] = ()
    external_knowledge: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def executable_locally(self) -> bool:
        return self.backend == "sqlite" and bool(self.db_path)

    @property
    def has_gold_sql(self) -> bool:
        return bool(self.gold_sql)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["query"] = self.gold_sql
        out["instance_id"] = self.example_id
        out["index"] = self.example_index
        out["gold_exec_results"] = list(self.gold_exec_results)
        return out
