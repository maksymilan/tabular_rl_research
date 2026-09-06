#!/usr/bin/env python3
"""Prepare paper-aligned Spider and Spider-variant evaluation inputs.

The official variant releases use slightly different JSON field names and database
layouts.  This adapter normalizes them to the ``rollout_passk.py`` DatasetTask
shape without changing questions or gold SQL.  It deliberately writes evaluation
inputs only; the generated manifest records source/database hashes so a later RL
run can bind to exactly the same cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
SPIDER = ROOT / "data" / "spider_data"
VARIANTS = ROOT / "data" / "spider_variants"
UPSTREAM = VARIANTS / "upstream"
OUT = VARIANTS / "eval"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    return digest.hexdigest()


def load_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"expected a JSON list: {path}")
    return value


def spider_db(
    root: Path,
    db_id: str,
    *,
    prefer_test: bool = False,
    fallback_root: Path | None = None,
) -> Path:
    roots = [root] + ([fallback_root] if fallback_root is not None else [])
    candidates: list[Path] = []
    for candidate_root in roots:
        regular = candidate_root / "database" / db_id / f"{db_id}.sqlite"
        test = candidate_root / "test_database" / db_id / f"{db_id}.sqlite"
        candidates.extend((test, regular) if prefer_test else (regular, test))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing Spider SQLite for {db_id}: {candidates}")


def normalize(
    *,
    dataset: str,
    split: str,
    source_rows: list[dict[str, Any]],
    question_key: str,
    db_root: Path,
    source_path: Path,
    prefer_test_db: bool = False,
    fallback_db_root: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    db_files: dict[str, dict[str, Any]] = {}
    for index, source in enumerate(source_rows):
        db_id = source.get("db_id")
        question = source.get(question_key)
        gold_sql = source.get("query")
        if not isinstance(db_id, str) or not isinstance(question, str) or not isinstance(gold_sql, str):
            raise ValueError(f"invalid {dataset} row {index}: expected db_id/question/query")
        db_path = spider_db(
            db_root,
            db_id,
            prefer_test=prefer_test_db,
            fallback_root=fallback_db_root,
        )
        db_files[db_id] = {
            "path": str(db_path),
            "sha256": sha256_file(db_path),
            "bytes": db_path.stat().st_size,
        }
        rows.append(
            {
                "dataset": dataset,
                "dataset_split": split,
                "example_index": index,
                "trajectory_id": f"{dataset.lower().replace('-', '_')}_{index:04d}",
                "db_id": db_id,
                "db_path": str(db_path),
                "question": question,
                "gold_sql": gold_sql,
            }
        )
    return rows, {
        "dataset": dataset,
        "split": split,
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "source_records": len(source_rows),
        "records": len(rows),
        "task_sha256": canonical_sha256(rows),
        "database_count": len(db_files),
        "databases": dict(sorted(db_files.items())),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return sha256_file(path)


def invalid_gold_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find source annotation errors without changing the published SQL."""
    invalid: list[dict[str, Any]] = []
    for row in rows:
        try:
            with sqlite3.connect(row["db_path"]) as connection:
                # Match the project Harness behavior for legacy Spider's lone
                # malformed UTF-8 byte sequence in ``wta_1``.
                connection.text_factory = lambda value: value.decode("utf-8", "replace")
                connection.execute(row["gold_sql"]).fetchall()
        except Exception as exc:  # noqa: BLE001 - persisted as an audit finding.
            invalid.append(
                {
                    "example_index": row["example_index"],
                    "db_id": row["db_id"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return invalid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not SPIDER.is_dir():
        raise SystemExit(f"missing local Spider release: {SPIDER}")
    jobs = [
        (
            "Spider-Test",
            "test",
            SPIDER / "test.json",
            "question",
            SPIDER,
            OUT / "spider_test.jsonl",
            True,
            None,
        ),
        (
            "Spider-DK",
            "dev",
            UPSTREAM / "Spider-DK" / "Spider-DK.json",
            "question",
            UPSTREAM / "Spider-DK",
            OUT / "spider_dk.jsonl",
            False,
            SPIDER,
        ),
        (
            "Spider-Syn",
            "dev",
            UPSTREAM / "Spider-Syn" / "Spider-Syn" / "dev.json",
            "SpiderSynQuestion",
            SPIDER,
            OUT / "spider_syn.jsonl",
            False,
            None,
        ),
        (
            "Spider-Realistic",
            "dev",
            UPSTREAM / "spider-realistic" / "spider-realistic.json",
            "question",
            SPIDER,
            OUT / "spider_realistic.jsonl",
            False,
            None,
        ),
    ]
    manifest: dict[str, Any] = {
        "schema_version": "spider-variant-eval-inputs-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "project_root": str(ROOT),
        "spider_root": str(SPIDER),
        "sources": {
            "spider_test": str(SPIDER / "test.json"),
            "spider_dk": "https://github.com/ygan/Spider-DK",
            "spider_syn": "https://github.com/ygan/Spider-Syn",
            "spider_realistic": "https://zenodo.org/records/5205322",
        },
        "datasets": {},
    }
    for dataset, split, source_path, question_key, db_root, output, prefer_test_db, fallback_db_root in jobs:
        if not source_path.is_file():
            raise SystemExit(f"missing source file: {source_path}")
        if output.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {output}; pass --overwrite")
        source_rows = load_json(source_path)
        rows, details = normalize(
            dataset=dataset,
            split=split,
            source_rows=source_rows,
            question_key=question_key,
            db_root=db_root,
            source_path=source_path,
            prefer_test_db=prefer_test_db,
            fallback_db_root=fallback_db_root,
        )
        invalid = invalid_gold_rows(rows)
        task_sha256 = write_jsonl(output, rows)
        scored_rows = [
            row for row in rows
            if row["example_index"] not in {item["example_index"] for item in invalid}
        ]
        scored_output = output.with_name(output.stem + "_scored.jsonl")
        scored_sha256 = write_jsonl(scored_output, scored_rows)
        details["output"] = str(output)
        details["output_sha256"] = task_sha256
        details["scored_output"] = str(scored_output)
        details["scored_output_sha256"] = scored_sha256
        details["scored_records"] = len(scored_rows)
        details["invalid_gold"] = invalid
        manifest["datasets"][dataset] = details
        print(
            f"{dataset}: {len(rows)} records ({len(scored_rows)} scored; "
            f"{len(invalid)} invalid gold), {len(details['databases'])} databases -> {output}"
        )

    manifest_path = OUT / "manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {manifest_path}; pass --overwrite")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
