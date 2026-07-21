#!/usr/bin/env python3
"""Adapter for the public BIRD 2024-06-27 development benchmark.

This module deliberately stays separate from ``bird_adapter.py``: BIRD train is used for
trajectory construction, while this released dev split is evaluation-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from dataset_ir import DatasetTask  # noqa: E402
from executor import Harness  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "bird" / "dev_20240627"


@dataclass(frozen=True)
class BirdDevExample:
    index: int
    question_id: int
    db_id: str
    question: str
    evidence: str
    sql: str
    difficulty: str

    @property
    def example_id(self) -> str:
        return f"bird_dev_{self.question_id:05d}"


def annotation_path(root: Path) -> Path:
    return root / "dev.json"


def sqlite_path(root: Path, db_id: str) -> Path:
    return root / "dev_databases" / db_id / f"{db_id}.sqlite"


def load_examples(root: Path) -> list[BirdDevExample]:
    with annotation_path(root).open() as f:
        raw = json.load(f)
    return [
        BirdDevExample(
            index=index,
            question_id=int(rec["question_id"]),
            db_id=rec["db_id"],
            question=rec["question"],
            evidence=rec.get("evidence", ""),
            sql=rec["SQL"],
            difficulty=rec.get("difficulty", "unknown"),
        )
        for index, rec in enumerate(raw)
    ]


def to_task(root: Path, ex: BirdDevExample) -> DatasetTask:
    return DatasetTask(
        dataset="bird-sql",
        split="dev",
        example_index=ex.index,
        example_id=ex.example_id,
        db_id=ex.db_id,
        question=ex.question,
        backend="sqlite",
        db_path=str(sqlite_path(root, ex.db_id)),
        gold_sql=ex.sql,
        external_knowledge=ex.evidence or None,
        metadata={
            "source": "bird-sql-dev-20240627",
            "difficulty": ex.difficulty,
            "question_id": ex.question_id,
        },
    )


def summarize(root: Path) -> dict:
    examples = load_examples(root)
    db_ids = sorted({ex.db_id for ex in examples})
    missing = [db_id for db_id in db_ids if not sqlite_path(root, db_id).is_file()]
    by_difficulty: dict[str, int] = {}
    for ex in examples:
        by_difficulty[ex.difficulty] = by_difficulty.get(ex.difficulty, 0) + 1
    return {
        "root": str(root),
        "split": "dev",
        "examples": len(examples),
        "databases": len(db_ids),
        "databases_missing_sqlite": missing,
        "by_difficulty": by_difficulty,
    }


def export_tasks(root: Path, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    examples = load_examples(root)
    with path.open("w") as f:
        for ex in examples:
            f.write(json.dumps(to_task(root, ex).to_json(), ensure_ascii=False) + "\n")
    return {"path": str(path), "records": len(examples)}


def run_gold(root: Path, limit: int) -> dict:
    examples = load_examples(root)
    if limit:
        examples = examples[:limit]
    passed = 0
    errors: list[dict[str, str]] = []
    for ex in examples:
        try:
            Harness(str(sqlite_path(root, ex.db_id))).gold(ex.sql)
            passed += 1
        except Exception as exc:  # noqa: BLE001
            errors.append({"example_id": ex.example_id, "error": f"{type(exc).__name__}: {exc}"})
    return {"checked": len(examples), "passed": passed, "errors": errors[:20]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--export", type=Path)
    parser.add_argument("--run-gold", type=int, metavar="N", help="execute N gold SQL queries; 0 means all")
    args = parser.parse_args()

    output = []
    if args.summary or not args.export and args.run_gold is None:
        output.append({"summary": summarize(args.root)})
    if args.export:
        output.append({"export": export_tasks(args.root, args.export)})
    if args.run_gold is not None:
        output.append({"run_gold": run_gold(args.root, args.run_gold)})
    print(json.dumps(output[0] if len(output) == 1 else output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
