#!/usr/bin/env python3
"""Spider 2.0-Lite adapter for the local SQLite subset.

Spider 2.0-Lite mixes BigQuery, Snowflake, and local SQLite tasks. The current harness executes
SQLite, so this adapter prepares the part we can run locally while keeping the full Lite metadata
visible for accounting.

Examples:
  .venv/bin/python src/harness/spider2_adapter.py --summary
  .venv/bin/python src/harness/spider2_adapter.py --run-local-gold 20
  .venv/bin/python src/harness/spider2_adapter.py --round-trip-local 20
  .venv/bin/python src/harness/spider2_adapter.py --export-local data/eval_inputs/spider2_lite_local.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from dataset_ir import DatasetTask  # noqa: E402
from executor import Harness  # noqa: E402
from verify import round_trip  # noqa: E402

DEFAULT_ROOT = ROOT / "data" / "spider2" / "Spider2"


@dataclass(frozen=True)
class Spider2LiteExample:
    index: int
    instance_id: str
    db: str
    question: str
    external_knowledge: Any

    @property
    def backend(self) -> str:
        if self.instance_id.startswith("local"):
            return "sqlite"
        if self.instance_id.startswith("sf"):
            return "snowflake"
        return "bigquery"


def lite_root(root: Path) -> Path:
    return root / "spider2-lite"


def load_examples(root: Path) -> list[Spider2LiteExample]:
    path = lite_root(root) / "spider2-lite.jsonl"
    out: list[Spider2LiteExample] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            rec = json.loads(line)
            out.append(
                Spider2LiteExample(
                    index=i,
                    instance_id=rec["instance_id"],
                    db=rec["db"],
                    question=rec["question"],
                    external_knowledge=rec.get("external_knowledge"),
                )
            )
    return out


def load_local_map(root: Path) -> dict[str, str]:
    path = lite_root(root) / "resource" / "databases" / "spider2-localdb" / "local-map.jsonl"
    if not path.exists():
        return {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    return {}


def sqlite_dir(root: Path) -> Path:
    return lite_root(root) / "resource" / "databases" / "spider2-localdb"


def sqlite_path(root: Path, ex: Spider2LiteExample, local_map: dict[str, str] | None = None) -> Path | None:
    if ex.backend != "sqlite":
        return None
    local_map = local_map or load_local_map(root)
    db_name = local_map.get(ex.instance_id, ex.db)
    candidates = [
        sqlite_dir(root) / f"{db_name}.sqlite",
        sqlite_dir(root) / f"{ex.db}.sqlite",
    ]
    lowered = {p.name.lower(): p for p in sqlite_dir(root).glob("*.sqlite")} if sqlite_dir(root).exists() else {}
    candidates.extend([lowered.get(f"{db_name}.sqlite".lower()), lowered.get(f"{ex.db}.sqlite".lower())])
    for p in candidates:
        if p and p.exists():
            return p
    return None


def gold_sql_path(root: Path, instance_id: str) -> Path:
    return lite_root(root) / "evaluation_suite" / "gold" / "sql" / f"{instance_id}.sql"


def gold_result_paths(root: Path, instance_id: str) -> list[Path]:
    folder = lite_root(root) / "evaluation_suite" / "gold" / "exec_result"
    return sorted(folder.glob(f"{instance_id}_*.csv"))


def read_gold_sql(root: Path, instance_id: str) -> str | None:
    path = gold_sql_path(root, instance_id)
    if not path.exists():
        return None
    return path.read_text().strip().rstrip(";")


def to_task(root: Path, ex: Spider2LiteExample, local_map: dict[str, str]) -> DatasetTask:
    db_path = sqlite_path(root, ex, local_map)
    sql_path = gold_sql_path(root, ex.instance_id)
    return DatasetTask(
        dataset="spider2-lite",
        split="lite",
        example_index=ex.index,
        example_id=ex.instance_id,
        db_id=ex.db,
        question=ex.question,
        backend=ex.backend,
        db_path=str(db_path) if db_path else None,
        gold_sql=read_gold_sql(root, ex.instance_id),
        gold_sql_path=str(sql_path) if sql_path.exists() else None,
        gold_exec_results=tuple(str(p) for p in gold_result_paths(root, ex.instance_id)),
        external_knowledge=ex.external_knowledge,
    )


def normalized_record(root: Path, ex: Spider2LiteExample, local_map: dict[str, str]) -> dict[str, Any]:
    return to_task(root, ex, local_map).to_json()


def summarize(root: Path) -> dict[str, Any]:
    examples = load_examples(root)
    local_map = load_local_map(root)
    by_backend = collections.Counter(ex.backend for ex in examples)
    by_db = collections.Counter(ex.db for ex in examples)
    local = [ex for ex in examples if ex.backend == "sqlite"]
    with_sqlite = sum(1 for ex in local if sqlite_path(root, ex, local_map))
    with_gold_sql = sum(1 for ex in examples if read_gold_sql(root, ex.instance_id))
    with_gold_csv = sum(1 for ex in examples if gold_result_paths(root, ex.instance_id))
    sqlite_files = sorted(sqlite_dir(root).glob("*.sqlite")) if sqlite_dir(root).exists() else []
    return {
        "root": str(root),
        "examples": len(examples),
        "by_backend": dict(sorted(by_backend.items())),
        "top_databases": by_db.most_common(20),
        "local_examples": len(local),
        "local_examples_with_sqlite": with_sqlite,
        "sqlite_files": len(sqlite_files),
        "gold_sql_examples": with_gold_sql,
        "gold_exec_result_examples": with_gold_csv,
    }


def run_local_gold(root: Path, limit: int) -> dict[str, Any]:
    examples = [ex for ex in load_examples(root) if ex.backend == "sqlite"]
    local_map = load_local_map(root)
    counts = collections.Counter()
    errors: list[dict[str, Any]] = []
    checked = 0
    for ex in examples:
        if limit and checked >= limit:
            break
        sql = read_gold_sql(root, ex.instance_id)
        path = sqlite_path(root, ex, local_map)
        if sql is None:
            counts["missing_gold_sql"] += 1
            continue
        if path is None:
            counts["missing_sqlite"] += 1
            continue
        checked += 1
        try:
            rows = Harness(str(path)).gold(sql)
        except Exception as exc:
            counts["exec_error"] += 1
            errors.append({"instance_id": ex.instance_id, "db": ex.db, "error": f"{type(exc).__name__}: {exc}"})
            continue
        counts["ok"] += 1
        counts["rows_returned"] += len(rows)
    return {"checked": checked, "counts": dict(counts), "errors": errors[:20]}


def smoke_local_tools(root: Path, limit: int) -> dict[str, Any]:
    files = sorted(sqlite_dir(root).glob("*.sqlite")) if sqlite_dir(root).exists() else []
    counts = collections.Counter()
    errors: list[dict[str, Any]] = []
    checked = 0
    for path in files:
        if limit and checked >= limit:
            break
        checked += 1
        try:
            h = Harness(str(path))
            tables = h.available_tables()
            if not tables:
                counts["empty_database"] += 1
                continue
            described = h.describe_table(tables[: min(3, len(tables))])
            first = tables[0]
            rows = h.read_subtable(first, limit=3)
            counts["ok"] += 1
            counts["tables_seen"] += len(tables)
            counts["sample_rows_seen"] += len(rows)
            if not described.get("tables"):
                counts["describe_empty"] += 1
        except Exception as exc:
            counts["error"] += 1
            errors.append({"db_path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    return {"checked": checked, "counts": dict(counts), "errors": errors[:20]}


def round_trip_local(root: Path, limit: int) -> dict[str, Any]:
    examples = [ex for ex in load_examples(root) if ex.backend == "sqlite"]
    local_map = load_local_map(root)
    counts = collections.Counter()
    reasons = collections.Counter()
    samples: dict[str, dict[str, Any]] = {}
    checked = 0
    for ex in examples:
        if limit and checked >= limit:
            break
        sql = read_gold_sql(root, ex.instance_id)
        path = sqlite_path(root, ex, local_map)
        if sql is None:
            counts["missing_gold_sql"] += 1
            continue
        if path is None:
            counts["missing_sqlite"] += 1
            continue
        checked += 1
        status, info = round_trip(Harness(str(path)), sql)
        counts[status] += 1
        if status != "ok":
            key = str(info).split(":")[0][:80]
            reasons[key] += 1
            samples.setdefault(key, {"instance_id": ex.instance_id, "db": ex.db, "sql": sql, "info": info})
    return {
        "checked": checked,
        "counts": dict(counts),
        "reasons": reasons.most_common(20),
        "samples": list(samples.values())[:10],
    }


def export_local(root: Path, path: Path) -> dict[str, Any]:
    local_map = load_local_map(root)
    examples = [ex for ex in load_examples(root) if ex.backend == "sqlite"]
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for ex in examples:
            rec = normalized_record(root, ex, local_map)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return {"path": str(path), "records": n}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--smoke-local-tools", type=int, metavar="N", help="run describe/read smoke on N local SQLite DB files; 0 means all")
    ap.add_argument("--run-local-gold", type=int, metavar="N", help="execute N local SQLite gold SQL files; 0 means all")
    ap.add_argument("--round-trip-local", type=int, metavar="N", help="compile+tool round-trip N local gold SQL files; 0 means all")
    ap.add_argument("--export-local", type=Path, help="write normalized local SQLite examples as JSONL")
    args = ap.parse_args()

    outputs = []
    if args.summary or not any([
        args.smoke_local_tools is not None,
        args.run_local_gold is not None,
        args.round_trip_local is not None,
        args.export_local,
    ]):
        outputs.append({"summary": summarize(args.root)})
    if args.smoke_local_tools is not None:
        outputs.append({"smoke_local_tools": smoke_local_tools(args.root, args.smoke_local_tools)})
    if args.run_local_gold is not None:
        outputs.append({"run_local_gold": run_local_gold(args.root, args.run_local_gold)})
    if args.round_trip_local is not None:
        outputs.append({"round_trip_local": round_trip_local(args.root, args.round_trip_local)})
    if args.export_local:
        outputs.append({"export_local": export_local(args.root, args.export_local)})

    print(json.dumps(outputs[0] if len(outputs) == 1 else outputs, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
