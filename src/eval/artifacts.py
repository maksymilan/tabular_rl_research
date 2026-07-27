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
    def __init__(
        self,
        result_dir: str,
        manifest: dict,
        resume: bool,
        *,
        operational_resume_fields: set[str] | None = None,
        operational_resume_metadata: dict | None = None,
    ):
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
        self.fsync_every = int(os.environ.get("ARTIFACT_FSYNC_EVERY", "1"))
        self.append_count = 0
        self.success_cases.mkdir(exist_ok=True)
        self.failure_cases.mkdir(exist_ok=True)

        requested_manifest = dict(manifest)
        manifest = {**manifest, "config_sha256": canonical_hash(manifest)}
        if self.manifest_path.exists():
            existing = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if existing.get("config_sha256") != manifest["config_sha256"]:
                allowed = operational_resume_fields or set()
                existing_config = {
                    key: value
                    for key, value in existing.items()
                    if key not in {"config_sha256", "created_at_utc"}
                }
                differences = {
                    key: {
                        "previous": existing_config.get(key),
                        "requested": requested_manifest.get(key),
                    }
                    for key in sorted(set(existing_config) | set(requested_manifest))
                    if existing_config.get(key) != requested_manifest.get(key)
                }
                if (
                    not resume
                    or not differences
                    or not set(differences).issubset(allowed)
                ):
                    raise ValueError(
                        f"existing manifest differs from this run: {self.manifest_path}"
                    )
                completed_before_resume = 0
                if self.all_path.exists():
                    with self.all_path.open(encoding="utf-8") as handle:
                        completed_before_resume = sum(
                            1 for line in handle if line.strip()
                        )
                event = {
                    "recorded_at_utc": utc_now(),
                    "event": "operational_resume",
                    "manifest_config_sha256": existing.get("config_sha256"),
                    "requested_config_sha256": manifest.get("config_sha256"),
                    "allowed_fields": sorted(allowed),
                    "differences": differences,
                    "completed_before_resume": completed_before_resume,
                    "metadata": operational_resume_metadata or {},
                }
                event_path = self.path / "operational_resume_events.jsonl"
                with event_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
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
        self.append_count += 1
        for path in (
            self.all_path,
            self.success_path if record["correct"] else self.failure_path,
        ):
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
                if self.fsync_every > 0 and self.append_count % self.fsync_every == 0:
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
