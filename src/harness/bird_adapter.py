#!/usr/bin/env python3
"""BIRD-SQL training-set adapter for the local SQLite harness.

The filtered BIRD annotations and the official training databases are intentionally separate:
``train_filtered`` comes from Hugging Face while ``train.zip`` comes from BIRD.  This adapter
normalizes them into ``DatasetTask`` records and checks that every referenced SQLite database is
usable by the existing harness.

Examples:
  .venv/bin/python src/harness/bird_adapter.py --summary
  .venv/bin/python src/harness/bird_adapter.py --smoke-tools 0
  .venv/bin/python src/harness/bird_adapter.py --run-gold 100
  .venv/bin/python src/harness/bird_adapter.py --round-trip 0
  .venv/bin/python src/harness/bird_adapter.py --export-compatible data/eval_inputs/bird_train_tool_compatible.jsonl
  .venv/bin/python src/harness/bird_adapter.py --export data/eval_inputs/bird_train.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))

from dataset_ir import DatasetTask  # noqa: E402
from executor import Harness  # noqa: E402
from verify import round_trip  # noqa: E402

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


def round_trip_train(root: Path, limit: int) -> dict[str, Any]:
    """Compile BIRD SQL into the current abstract tools and compare its result to gold SQL."""
    counts: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    samples: dict[str, dict[str, Any]] = {}
    checked = 0
    for ex in load_examples(root):
        if limit and checked >= limit:
            break
        path = sqlite_path(root, ex.db_id)
        if path is None:
            counts["missing_sqlite"] += 1
            continue
        checked += 1
        status, info = round_trip(Harness(str(path)), ex.sql)
        counts[status] += 1
        if status != "ok":
            reason = str(info).split(":", 1)[0][:160]
            reasons[reason] += 1
            samples.setdefault(reason, {
                "example_id": ex.example_id,
                "db_id": ex.db_id,
                "question": ex.question,
                "sql": ex.sql,
                "status": status,
                "info": str(info)[:2000],
            })
    return {
        "checked": checked,
        "counts": dict(counts),
        "reasons": reasons.most_common(30),
        "samples": list(samples.values())[:20],
    }


def _round_trip_example(root: Path, ex: BirdTrainExample, timeout_seconds: float) -> tuple[str, Any]:
    path = sqlite_path(root, ex.db_id)
    if path is None:
        return "missing_sqlite", "SQLite database not found"
    started = time.monotonic()
    harness: Harness | None = None
    try:
        harness = Harness(str(path))
        harness.conn.set_progress_handler(
            lambda: 1 if time.monotonic() - started > timeout_seconds else 0,
            10_000,
        )
        status, info = round_trip(harness, ex.sql)
        if (
            status == "exec_error"
            and time.monotonic() - started > timeout_seconds
            and "interrupted" in str(info).lower()
        ):
            return "timeout", f"exceeded {timeout_seconds:g}s: {info}"
        return status, info
    except sqlite3.OperationalError as exc:
        if time.monotonic() - started > timeout_seconds:
            return "timeout", f"exceeded {timeout_seconds:g}s: {exc}"
        return "adapter_error", f"OperationalError: {exc}"
    except Exception as exc:  # Database open/adapter failures should be reported, not abort the batch.
        return "adapter_error", f"{type(exc).__name__}: {exc}"
    finally:
        if harness is not None:
            harness.conn.set_progress_handler(None, 0)
            harness.conn.close()


def _round_trip_worker(root: Path, ex: BirdTrainExample, timeout_seconds: float, sender: Any) -> None:
    """Run one task in a disposable process; the parent enforces its wall-clock limit."""
    try:
        sender.send(_round_trip_example(root, ex, timeout_seconds))
    except BaseException as exc:
        try:
            sender.send(("adapter_error", f"worker {type(exc).__name__}: {exc}"))
        except (BrokenPipeError, OSError):
            pass
    finally:
        sender.close()


def isolated_round_trips(
    root: Path,
    examples: list[BirdTrainExample],
    workers: int,
    timeout_seconds: float,
) -> list[tuple[str, Any]]:
    """Bound each task even when SQLite is executing native code that ignores Python signals."""
    context = mp.get_context("spawn")
    outcomes: list[tuple[str, Any] | None] = [None] * len(examples)
    active: dict[int, tuple[Any, Any, float]] = {}
    next_index = 0

    while next_index < len(examples) or active:
        while next_index < len(examples) and len(active) < workers:
            receiver, sender = context.Pipe(duplex=False)
            proc = context.Process(
                target=_round_trip_worker,
                args=(root, examples[next_index], timeout_seconds, sender),
            )
            proc.start()
            sender.close()
            active[next_index] = (proc, receiver, time.monotonic())
            next_index += 1

        now = time.monotonic()
        for index, (proc, receiver, started) in list(active.items()):
            if receiver.poll():
                outcomes[index] = receiver.recv()
                proc.join()
            elif not proc.is_alive():
                proc.join()
                outcomes[index] = ("adapter_error", f"worker exited with code {proc.exitcode}")
            elif now - started > timeout_seconds:
                proc.terminate()
                proc.join()
                outcomes[index] = ("timeout", f"exceeded {timeout_seconds:g}s")
            else:
                continue
            receiver.close()
            del active[index]
        time.sleep(0.01)

    return [outcome if outcome is not None else ("adapter_error", "missing worker outcome") for outcome in outcomes]


def export_compatible_tasks(
    root: Path,
    path: Path,
    workers: int,
    timeout_seconds: float,
    limit: int = 0,
) -> dict[str, Any]:
    """Export only train tasks proven equivalent through the current tool compiler/harness."""
    path.parent.mkdir(parents=True, exist_ok=True)
    counts: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    samples: dict[str, dict[str, Any]] = {}
    exported = 0
    checked = 0
    examples = load_examples(root)
    if limit:
        examples = examples[:limit]
    outcomes = isolated_round_trips(root, examples, workers, timeout_seconds)
    with path.open("w") as f:
        for ex, (status, info) in zip(examples, outcomes):
            if status == "missing_sqlite":
                counts["missing_sqlite"] += 1
                continue
            checked += 1
            counts[status] += 1
            if status == "ok":
                record = to_task(root, ex).to_json()
                record["metadata"]["tool_round_trip"] = "verified"
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                exported += 1
                continue
            reason = str(info).split(":", 1)[0][:160]
            reasons[reason] += 1
            samples.setdefault(reason, {
                "example_id": ex.example_id,
                "db_id": ex.db_id,
                "question": ex.question,
                "sql": ex.sql,
                "status": status,
                "info": str(info)[:2000],
            })
    summary = {
        "source": str(annotation_path(root)),
        "checked": checked,
        "exported": exported,
        "timeout_seconds": timeout_seconds,
        "counts": dict(counts),
        "reasons": reasons.most_common(50),
        "samples": list(samples.values())[:30],
    }
    summary_path = path.with_suffix(path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    return {"path": str(path), "summary_path": str(summary_path), **summary}


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
    ap.add_argument("--round-trip", type=int, metavar="N", help="compile+tool round-trip N training examples; 0 means all")
    ap.add_argument("--export-compatible", type=Path,
                    help="write only train tasks verified by compile+tool round-trip, plus a summary JSON")
    ap.add_argument("--export", type=Path, help="write normalized BIRD training tasks as JSONL")
    ap.add_argument("--workers", type=int, default=1,
                    help="max disposable processes for --export-compatible")
    ap.add_argument("--timeout-seconds", type=float, default=30.0,
                    help="per-task SQLite execution limit for --export-compatible")
    ap.add_argument("--compatible-limit", type=int, default=0, metavar="N",
                    help="limit --export-compatible to N examples for a smoke test; 0 means all")
    args = ap.parse_args()

    outputs = []
    if args.summary or not any([
        args.smoke_tools is not None,
        args.run_gold is not None,
        args.round_trip is not None,
        args.export_compatible,
        args.export,
    ]):
        outputs.append({"summary": summarize(args.root)})
    if args.smoke_tools is not None:
        outputs.append({"smoke_tools": smoke_tools(args.root, args.smoke_tools)})
    if args.run_gold is not None:
        outputs.append({"run_gold": run_gold(args.root, args.run_gold)})
    if args.round_trip is not None:
        outputs.append({"round_trip": round_trip_train(args.root, args.round_trip)})
    if args.export_compatible:
        outputs.append({"export_compatible": export_compatible_tasks(
            args.root, args.export_compatible, args.workers, args.timeout_seconds, args.compatible_limit,
        )})
    if args.export:
        outputs.append({"export": export_tasks(args.root, args.export)})

    print(json.dumps(outputs[0] if len(outputs) == 1 else outputs, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
