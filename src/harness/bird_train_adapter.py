#!/usr/bin/env python3
"""BIRD-SQL training-set adapter for the local SQLite harness.

The filtered BIRD annotations and the official training databases are intentionally separate:
``train_filtered`` comes from Hugging Face while ``train.zip`` comes from BIRD.  This adapter
normalizes them into ``DatasetTask`` records and checks that every referenced SQLite database is
usable by the existing harness.

Examples:
  .venv/bin/python src/harness/bird_train_adapter.py --summary
  .venv/bin/python src/harness/bird_train_adapter.py --smoke-tools 0
  .venv/bin/python src/harness/bird_train_adapter.py --run-gold 100
  .venv/bin/python src/harness/bird_train_adapter.py --export data/eval_inputs/bird_train.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from dataset_ir import DatasetTask  # noqa: E402
from executor import Harness  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "bird"


@dataclass(frozen=True)
class BirdTrainExample:
    index: int
    db_id: str
    question: str
    evidence: str
    sql: str

    @property
    def example_id(self) -> str:
        return f"bird_train_{self.index:05d}"


def annotation_path(root: Path) -> Path:
    return root / "train_filtered" / "data" / "train-00000-of-00001.jsonl"


def databases_dir(root: Path) -> Path:
    """Support both the archive's top-level ``train/`` and a flattened layout."""
    for path in (root / "train_databases", root / "train" / "train_databases"):
        if path.is_dir():
            return path
    return root / "train_databases"


def load_examples(root: Path) -> list[BirdTrainExample]:
    path = annotation_path(root)
    out: list[BirdTrainExample] = []
    with path.open() as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            out.append(
                BirdTrainExample(
                    index=index,
                    db_id=rec["db_id"],
                    question=rec["question"],
                    evidence=rec.get("evidence", ""),
                    sql=rec["SQL"],
                )
            )
    return out


def sqlite_path(root: Path, db_id: str) -> Path | None:
    base = databases_dir(root)
    candidates = (
        base / db_id / f"{db_id}.sqlite",
        base / db_id / "sqlite" / f"{db_id}.sqlite",
        base / f"{db_id}.sqlite",
    )
    return next((path for path in candidates if path.is_file()), None)


def to_task(root: Path, ex: BirdTrainExample) -> DatasetTask:
    path = sqlite_path(root, ex.db_id)
    return DatasetTask(
        dataset="bird-sql",
        split="train",
        example_index=ex.index,
        example_id=ex.example_id,
        db_id=ex.db_id,
        question=ex.question,
        backend="sqlite",
        db_path=str(path) if path else None,
        gold_sql=ex.sql,
        external_knowledge=ex.evidence or None,
        metadata={"source": "bird23-train-filtered"},
    )


def summarize(root: Path) -> dict[str, Any]:
    examples = load_examples(root)
    db_ids = sorted({ex.db_id for ex in examples})
    found = [sqlite_path(root, db_id) for db_id in db_ids]
    existing = [path for path in found if path]
    return {
        "root": str(root),
        "annotation_path": str(annotation_path(root)),
        "examples": len(examples),
        "databases": len(db_ids),
        "databases_with_sqlite": len(existing),
        "databases_missing_sqlite": [db_id for db_id, path in zip(db_ids, found) if path is None][:50],
        "sqlite_bytes": sum(path.stat().st_size for path in existing),
        "by_database_examples": collections.Counter(ex.db_id for ex in examples).most_common(20),
    }


def smoke_tools(root: Path, limit: int) -> dict[str, Any]:
    db_ids = sorted({ex.db_id for ex in load_examples(root)})
    counts: collections.Counter[str] = collections.Counter()
    errors: list[dict[str, str]] = []
    checked = 0
    for db_id in db_ids:
        if limit and checked >= limit:
            break
        path = sqlite_path(root, db_id)
        if path is None:
            counts["missing_sqlite"] += 1
            errors.append({"db_id": db_id, "error": "SQLite database not found"})
            continue
        checked += 1
        try:
            harness = Harness(str(path))
            tables = harness.available_tables()
            if not tables:
                counts["empty_database"] += 1
                continue
            described = harness.describe_table(tables[: min(3, len(tables))])
            rows = harness.read_subtable(tables[0], limit=3)
            counts["ok"] += 1
            counts["tables_seen"] += len(tables)
            counts["sample_rows_seen"] += len(rows)
            if not described.get("tables"):
                counts["describe_empty"] += 1
        except Exception as exc:
            counts["error"] += 1
            errors.append({"db_id": db_id, "error": f"{type(exc).__name__}: {exc}"})
    return {"checked": checked, "counts": dict(counts), "errors": errors[:20]}


def run_gold(root: Path, limit: int) -> dict[str, Any]:
    counts: collections.Counter[str] = collections.Counter()
    errors: list[dict[str, str]] = []
    checked = 0
    for ex in load_examples(root):
        if limit and checked >= limit:
            break
        path = sqlite_path(root, ex.db_id)
        if path is None:
            counts["missing_sqlite"] += 1
            errors.append({"example_id": ex.example_id, "db_id": ex.db_id, "error": "SQLite database not found"})
            continue
        checked += 1
        try:
            rows = Harness(str(path)).gold(ex.sql)
            counts["ok"] += 1
            counts["rows_returned"] += len(rows)
        except Exception as exc:
            counts["exec_error"] += 1
            errors.append({"example_id": ex.example_id, "db_id": ex.db_id, "error": f"{type(exc).__name__}: {exc}"})
    return {"checked": checked, "counts": dict(counts), "errors": errors[:20]}


def export_tasks(root: Path, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = 0
    with path.open("w") as f:
        for ex in load_examples(root):
            f.write(json.dumps(to_task(root, ex).to_json(), ensure_ascii=False) + "\n")
            records += 1
    return {"path": str(path), "records": records}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--smoke-tools", type=int, metavar="N", help="run describe/read smoke on N databases; 0 means all")
    ap.add_argument("--run-gold", type=int, metavar="N", help="execute gold SQL for N training examples; 0 means all")
    ap.add_argument("--export", type=Path, help="write normalized BIRD training tasks as JSONL")
    args = ap.parse_args()

    outputs = []
    if args.summary or not any([
        args.smoke_tools is not None,
        args.run_gold is not None,
        args.export,
    ]):
        outputs.append({"summary": summarize(args.root)})
    if args.smoke_tools is not None:
        outputs.append({"smoke_tools": smoke_tools(args.root, args.smoke_tools)})
    if args.run_gold is not None:
        outputs.append({"run_gold": run_gold(args.root, args.run_gold)})
    if args.export:
        outputs.append({"export": export_tasks(args.root, args.export)})

    print(json.dumps(outputs[0] if len(outputs) == 1 else outputs, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
