#!/usr/bin/env python3
"""Incremental, auditable evaluation artifacts shared by baseline runners."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(obj) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ArtifactWriter:
    def __init__(self, result_dir: str, manifest: dict, resume: bool):
        self.path = Path(result_dir)
        self.path.mkdir(parents=True, exist_ok=True)
        self.all_path = self.path / "all.jsonl"
        self.success_path = self.path / "success.jsonl"
        self.failure_path = self.path / "failure.jsonl"
        self.success_cases = self.path / "success_cases"
        self.failure_cases = self.path / "failure_cases"
        self.manifest_path = self.path / "manifest.json"
        self.summary_path = self.path / "summary.json"
        self.completed: set[int] = set()
        self.success_cases.mkdir(exist_ok=True)
        self.failure_cases.mkdir(exist_ok=True)

        manifest = {**manifest, "config_sha256": canonical_hash(manifest)}
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("config_sha256") != manifest["config_sha256"]:
                raise ValueError(
                    f"existing manifest differs from this run: {self.manifest_path}"
                )
            if not resume:
                raise ValueError(f"result directory already exists; use --resume: {self.path}")
        else:
            manifest["created_at_utc"] = utc_now()
            self.manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        if resume and self.all_path.exists():
            with self.all_path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.completed.add(json.loads(line)["example_index"])

    def append(self, record: dict) -> None:
        record = {**record, "recorded_at_utc": utc_now()}
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        for path in (
            self.all_path,
            self.success_path if record["correct"] else self.failure_path,
        ):
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
        case_dir = self.success_cases if record["correct"] else self.failure_cases
        case_path = case_dir / f"q{record['example_index']:04d}.json"
        case_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        self.completed.add(record["example_index"])

    def summarize(self) -> dict:
        records = []
        if self.all_path.exists():
            with self.all_path.open(encoding="utf-8") as f:
                records = [json.loads(line) for line in f if line.strip()]
        total = len(records)
        failure_types = Counter(
            r.get("failure_type") or r.get("fail") or "wrong_answer"
            for r in records
            if not r.get("correct")
        )
        summary = {
            "total": total,
            "correct": sum(bool(r.get("correct")) for r in records),
            "failed": sum(not bool(r.get("correct")) for r in records),
            "accuracy": (
                sum(bool(r.get("correct")) for r in records) / total if total else 0.0
            ),
            "completed_example_indices": sorted(
                r["example_index"] for r in records
            ),
            "failure_types": dict(sorted(failure_types.items())),
            "updated_at_utc": utc_now(),
        }
        if any("legal" in r for r in records):
            summary["legal_answers"] = sum(bool(r.get("legal")) for r in records)
        if any("steps" in r for r in records):
            summary["average_steps"] = (
                sum(r.get("steps", 0) for r in records) / total if total else 0.0
            )
            summary["total_tool_errors"] = sum(r.get("errors", 0) for r in records)
        self.summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary
