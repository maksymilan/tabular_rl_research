#!/usr/bin/env python3
"""Incremental, auditable evaluation artifacts shared by baseline runners."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_hash(obj) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Best-effort durability for an atomic replace or newly created file."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, payload: str) -> None:
    """Write one complete derived artifact without exposing a torn destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.tmp-",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _append_fsync_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _record_line(record: dict) -> str:
    return json.dumps(record, ensure_ascii=False, default=str) + "\n"


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
        preexisting_material: list[Path] = []
        if self.path.exists():
            for child in self.path.iterdir():
                if child.is_dir() and not any(child.iterdir()):
                    continue
                preexisting_material.append(child)
        self.path.mkdir(parents=True, exist_ok=True)
        self.all_path = self.path / "all.jsonl"
        self.success_path = self.path / "success.jsonl"
        self.failure_path = self.path / "failure.jsonl"
        self.success_cases = self.path / "success_cases"
        self.failure_cases = self.path / "failure_cases"
        self.manifest_path = self.path / "manifest.json"
        self.summary_path = self.path / "summary.json"
        self.recovery_events_path = self.path / "artifact_recovery_events.jsonl"
        self.completed: set[int] = set()
        self.fsync_every = int(os.environ.get("ARTIFACT_FSYNC_EVERY", "1"))
        self.append_count = 0
        self.success_cases.mkdir(exist_ok=True)
        self.failure_cases.mkdir(exist_ok=True)

        requested_manifest = dict(manifest)
        manifest = {**manifest, "config_sha256": canonical_hash(manifest)}
        if not self.manifest_path.exists() and preexisting_material:
            names = ", ".join(sorted(path.name for path in preexisting_material))
            raise ValueError(
                f"result directory contains artifacts without a manifest: {names}"
            )
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
                    # Count only newline-terminated journal entries here.  The
                    # recovery pass below owns decoding and any torn-tail repair.
                    completed_before_resume = self.all_path.read_bytes().count(b"\n")
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
                _append_fsync_jsonl(event_path, event)
            if not resume:
                raise ValueError(f"result directory already exists; use --resume: {self.path}")
        else:
            manifest["created_at_utc"] = utc_now()
            _atomic_write_text(
                self.manifest_path,
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            )

        if resume:
            records = self._load_and_repair_journal()
            for record in records:
                self.completed.add(record["example_index"])
            self._reconcile_projections(records)

    @staticmethod
    def _validated_journal_record(value: object, *, location: str) -> dict:
        if not isinstance(value, dict):
            raise ValueError(f"journal record is not an object at {location}")
        example_index = value.get("example_index")
        if isinstance(example_index, bool) or not isinstance(example_index, int):
            raise ValueError(f"journal record has invalid example_index at {location}")
        if "correct" not in value:
            raise ValueError(f"journal record is missing correct at {location}")
        return value

    def _record_recovery_event(self, event: dict) -> None:
        _append_fsync_jsonl(
            self.recovery_events_path,
            {
                "recorded_at_utc": utc_now(),
                "schema_version": "artifact-recovery-v1",
                **event,
            },
        )

    def _load_and_repair_journal(self) -> list[dict]:
        """Load the authoritative all.jsonl, repairing only its final torn bytes."""

        if not self.all_path.exists():
            derived_paths = (self.success_path, self.failure_path)
            has_derived = any(
                path.exists() and path.stat().st_size for path in derived_paths
            )
            has_cases = any(self.success_cases.glob("q*.json")) or any(
                self.failure_cases.glob("q*.json")
            )
            if has_derived or has_cases:
                raise ValueError("derived artifacts exist without authoritative all.jsonl")
            return []

        payload = self.all_path.read_bytes()
        raw_lines = payload.splitlines(keepends=True)
        records: list[dict] = []
        seen: dict[int, dict] = {}
        repaired_tail = False
        for line_index, raw_line in enumerate(raw_lines):
            if not raw_line.strip():
                continue
            is_last = line_index == len(raw_lines) - 1
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if not is_last or payload.endswith(b"\n"):
                    raise ValueError(
                        f"malformed complete all.jsonl record at line {line_index + 1}"
                    ) from exc
                tail_offset = len(payload) - len(raw_line)
                bad_tail = payload[tail_offset:]
                with self.all_path.open("r+b") as handle:
                    handle.truncate(tail_offset)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._record_recovery_event({
                    "event": "truncate_torn_all_jsonl_tail",
                    "path": self.all_path.name,
                    "tail_offset": tail_offset,
                    "bad_tail_bytes": len(bad_tail),
                    "bad_tail_sha256": hashlib.sha256(bad_tail).hexdigest(),
                    "original_bytes": len(payload),
                    "repaired_bytes": tail_offset,
                })
                repaired_tail = True
                break
            record = self._validated_journal_record(
                value,
                location=f"all.jsonl:{line_index + 1}",
            )
            example_index = record["example_index"]
            if example_index in seen:
                conflict = seen[example_index] != record
                kind = "conflicting" if conflict else "duplicate"
                raise ValueError(
                    f"{kind} complete all.jsonl records for example_index {example_index}"
                )
            seen[example_index] = record
            records.append(record)

        if payload and not payload.endswith(b"\n") and not repaired_tail:
            # The JSON object is complete but its record delimiter was torn.
            # Seal that exact object before a later append can concatenate data.
            with self.all_path.open("ab") as handle:
                handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._record_recovery_event({
                "event": "seal_complete_all_jsonl_tail",
                "path": self.all_path.name,
                "tail_offset": len(payload),
                "bad_tail_bytes": 0,
                "bad_tail_sha256": hashlib.sha256(b"").hexdigest(),
                "original_bytes": len(payload),
                "repaired_bytes": len(payload) + 1,
            })
        return records

    @staticmethod
    def _load_projection_prefix(path: Path) -> tuple[list[dict], str | None]:
        if not path.exists():
            return [], "missing"
        payload = path.read_bytes()
        raw_lines = payload.splitlines(keepends=True)
        records: list[dict] = []
        for line_index, raw_line in enumerate(raw_lines):
            if not raw_line.strip():
                continue
            is_last = line_index == len(raw_lines) - 1
            try:
                value = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if is_last and not payload.endswith(b"\n"):
                    return records, "torn_tail"
                raise ValueError(
                    f"malformed complete projection record at {path.name}:{line_index + 1}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"projection record is not an object at {path.name}:{line_index + 1}"
                )
            records.append(value)
        if payload and not payload.endswith(b"\n"):
            return records, "missing_newline"
        return records, None

    def _reconcile_projections(self, records: list[dict]) -> None:
        expected_success = [record for record in records if record.get("correct")]
        expected_failure = [record for record in records if not record.get("correct")]
        recovery_reasons: list[str] = []

        for path, expected in (
            (self.success_path, expected_success),
            (self.failure_path, expected_failure),
        ):
            actual, tail_state = self._load_projection_prefix(path)
            if len(actual) > len(expected) or actual != expected[: len(actual)]:
                raise ValueError(
                    f"complete {path.name} content conflicts with authoritative all.jsonl"
                )
            if len(actual) < len(expected) or tail_state not in {None, "missing"}:
                recovery_reasons.append(
                    f"{path.name}:{tail_state or 'missing_tail'}:{len(actual)}/{len(expected)}"
                )
            elif tail_state == "missing" and expected:
                recovery_reasons.append(f"{path.name}:missing:0/{len(expected)}")

        expected_cases: dict[Path, dict] = {}
        for record in records:
            case_dir = self.success_cases if record.get("correct") else self.failure_cases
            expected_cases[case_dir / f"q{record['example_index']:04d}.json"] = record
        actual_case_paths = set(self.success_cases.glob("q*.json")) | set(
            self.failure_cases.glob("q*.json")
        )
        unexpected_cases = actual_case_paths - set(expected_cases)
        if unexpected_cases:
            names = ", ".join(
                sorted(str(path.relative_to(self.path)) for path in unexpected_cases)
            )
            raise ValueError(f"case artifacts conflict with authoritative all.jsonl: {names}")
        for path, expected in expected_cases.items():
            if not path.exists():
                recovery_reasons.append(f"{path.relative_to(self.path)}:missing")
                continue
            try:
                actual = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"complete case artifact is malformed: {path}") from exc
            if actual != expected:
                raise ValueError(
                    f"complete case artifact conflicts with authoritative all.jsonl: {path}"
                )

        if not recovery_reasons:
            return
        _atomic_write_text(
            self.success_path,
            "".join(_record_line(record) for record in expected_success),
        )
        _atomic_write_text(
            self.failure_path,
            "".join(_record_line(record) for record in expected_failure),
        )
        for path, record in expected_cases.items():
            _atomic_write_text(
                path,
                json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
            )
        self._record_recovery_event({
            "event": "rebuild_derived_artifacts_from_all_jsonl",
            "path": self.all_path.name,
            "record_count": len(records),
            "journal_sha256": hashlib.sha256(self.all_path.read_bytes()).hexdigest(),
            "reasons": sorted(recovery_reasons),
            "sft_eligible": False,
        })

    def append(self, record: dict) -> None:
        record = {**record, "recorded_at_utc": utc_now()}
        record = self._validated_journal_record(record, location="append")
        example_index = record["example_index"]
        if example_index in self.completed:
            raise ValueError(f"example_index {example_index} is already committed")
        line = _record_line(record)
        self.append_count += 1
        # all.jsonl is the authoritative write-ahead journal.  It is durable
        # before any rebuildable split/case projection is touched.
        with self.all_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.all_path.parent)
        self.completed.add(example_index)

        split_path = self.success_path if record["correct"] else self.failure_path
        with split_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            if self.fsync_every > 0 and self.append_count % self.fsync_every == 0:
                os.fsync(handle.fileno())
        case_dir = self.success_cases if record["correct"] else self.failure_cases
        case_path = case_dir / f"q{example_index:04d}.json"
        _atomic_write_text(
            case_path,
            json.dumps(record, ensure_ascii=False, indent=2, default=str) + "\n",
        )

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
        _atomic_write_text(
            self.summary_path,
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        )
        return summary
