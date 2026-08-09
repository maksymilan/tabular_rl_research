#!/usr/bin/env python3
"""Score released ``trustsql_eval`` conversations on BIRD dev.

The authors' evaluator writes one ``{id}.json`` file per question.  Each file is a
JSON list whose entries are distinguished by ``rollout_idx``.  This scorer treats
every rollout independently: it intentionally implements neither majority voting
nor pass@K selection.

Both predicted and reference SQL run against SQLite opened with ``mode=ro`` and
``PRAGMA query_only``.  A SQLite progress handler supplies a real VM deadline, so
timed-out queries are interrupted rather than continuing in a background thread.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "trust-sql-qwen3-8b-base-bird-dev-score-v1"
PROGRESS_HANDLER_STEPS = 10_000
_ANSWER_RE = re.compile(r"<answer\b[^>]*>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_FENCE_RE = re.compile(
    r"(?P<fence>```|''')[ \t]*(?:(?:sql|sqlite)[ \t]*)?"
    r"(?:\r?\n)(?P<body>.*?)(?P=fence)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(frozen=True)
class SqlExecution:
    """Internal result of one bounded, read-only SQLite query."""

    status: str
    elapsed_seconds: float
    rows: tuple[tuple[Any, ...], ...] | None = None
    columns: tuple[str, ...] = ()
    error: str | None = None

    def manifest_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "status": self.status,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        }
        if self.rows is not None:
            fields["row_count"] = len(self.rows)
            fields["column_count"] = len(self.columns)
            fields["sample_rows"] = [
                [_json_cell(cell) for cell in row] for row in self.rows[:5]
            ]
        if self.error is not None:
            fields["error"] = self.error
        return fields


@dataclass(frozen=True)
class ParsedAnswer:
    status: str
    sql: str | None = None
    message_source: str | None = None
    error: str | None = None


def _json_cell(value: Any) -> Any:
    """Represent every SQLite cell in standards-compliant JSON diagnostics."""

    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bird_set_equal(
    predicted: Iterable[Iterable[Any]], gold: Iterable[Iterable[Any]]
) -> bool:
    """BIRD EX: ignore row order and duplicate multiplicity, preserve column order."""

    return {tuple(row) for row in predicted} == {tuple(row) for row in gold}


def _strip_leading_sql_comments(sql: str) -> str:
    remaining = sql.lstrip()
    while True:
        if remaining.startswith("--"):
            newline = remaining.find("\n")
            if newline < 0:
                return ""
            remaining = remaining[newline + 1 :].lstrip()
            continue
        if remaining.startswith("/*"):
            end = remaining.find("*/", 2)
            if end < 0:
                return ""
            remaining = remaining[end + 2 :].lstrip()
            continue
        return remaining


def _query_candidate(text: str) -> str | None:
    candidate = text.strip()
    candidate = re.sub(r"^\s*SQL\s*:\s*", "", candidate, flags=re.IGNORECASE)
    visible = _strip_leading_sql_comments(candidate)
    if not re.match(r"^(?:SELECT|WITH)\b", visible, re.IGNORECASE):
        return None
    return candidate


def parse_last_assistant_answer(result: dict[str, Any]) -> ParsedAnswer:
    """Extract fenced or raw SQL from the last assistant ``<answer>`` block."""

    saw_message_list = False
    for field in ("conversation", "final_messages"):
        messages = result.get(field)
        if not isinstance(messages, list):
            continue
        saw_message_list = True
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            answers = list(_ANSWER_RE.finditer(content))
            if not answers:
                continue
            answer_body = answers[-1].group(1).strip()
            if not answer_body:
                return ParsedAnswer(
                    status="empty_answer", message_source=field, error="empty <answer> block"
                )
            fenced = list(_FENCE_RE.finditer(answer_body))
            for match in fenced:
                sql = _query_candidate(match.group("body"))
                if sql is not None:
                    return ParsedAnswer(status="ok", sql=sql, message_source=field)
            sql = _query_candidate(answer_body)
            if sql is not None:
                return ParsedAnswer(status="ok", sql=sql, message_source=field)
            return ParsedAnswer(
                status="no_sql_in_answer",
                message_source=field,
                error="last assistant <answer> contains no SELECT/WITH query",
            )
    if saw_message_list:
        return ParsedAnswer(
            status="missing_assistant_answer",
            error="no assistant message contains a complete <answer> block",
        )
    return ParsedAnswer(
        status="missing_message_list",
        error="rollout has neither a conversation nor final_messages list",
    )


def execute_read_only_sql(
    db_path: Path, sql: str, timeout_seconds: float
) -> SqlExecution:
    """Execute one query using a read-only URI and a cancellable SQLite deadline."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    started = time.monotonic()
    if not db_path.is_file():
        return SqlExecution(
            status="missing_database",
            elapsed_seconds=time.monotonic() - started,
            error=f"database file does not exist: {db_path}",
        )

    connection: sqlite3.Connection | None = None
    timed_out = False
    deadline = started + timeout_seconds

    def interrupt_at_deadline() -> int:
        nonlocal timed_out
        if time.monotonic() >= deadline:
            timed_out = True
            return 1
        return 0

    try:
        uri = db_path.resolve().as_uri() + "?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(interrupt_at_deadline, PROGRESS_HANDLER_STEPS)
        cursor = connection.execute(sql)
        columns = tuple(str(item[0]) for item in (cursor.description or ()))
        rows = tuple(tuple(row) for row in cursor.fetchall())
        return SqlExecution(
            status="ok",
            elapsed_seconds=time.monotonic() - started,
            rows=rows,
            columns=columns,
        )
    except sqlite3.Error as exc:
        return SqlExecution(
            status="timeout" if timed_out else "execution_error",
            elapsed_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:  # noqa: BLE001 - preserve unexpected runtime diagnostics.
        return SqlExecution(
            status="execution_error",
            elapsed_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if connection is not None:
            try:
                connection.set_progress_handler(None, 0)
            finally:
                connection.close()


def _gold_sql(example: dict[str, Any]) -> str | None:
    for key in ("SQL", "sql", "query", "gold_sql"):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _instance_id_candidates(example: dict[str, Any], dataset_index: int) -> list[str]:
    for key in ("id", "instance_id"):
        value = example.get(key)
        if value is not None and str(value):
            return [str(value)]
    question_id = example.get("question_id", dataset_index)
    raw = str(question_id)
    candidates = [raw]
    if not raw.startswith("dev_"):
        candidates.append(f"dev_{raw}")
    return candidates


def _read_rollout_file(path: Path) -> tuple[str, list[Any], str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid_json", [], f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, list):
        return "invalid_root", [], "official output root must be a JSON list"
    return "ok", payload, None


def _base_rollout_record(
    *, rollout_idx: int | None, source_position: int | None, expected: bool
) -> dict[str, Any]:
    return {
        "rollout_idx": rollout_idx,
        "source_position": source_position,
        "expected": expected,
        "status": None,
        "parse_status": "not_run",
        "prediction_execution": {"status": "not_run"},
        "correct": None,
    }


def _prepare_rollouts(
    payload: list[Any], expected_rollouts: int
) -> tuple[list[dict[str, Any]], list[int], list[int], list[int]]:
    prepared: list[dict[str, Any]] = []
    positions_by_idx: dict[int, list[int]] = {}
    invalid_positions: set[int] = set()
    expected_indices = set(range(expected_rollouts))

    for position, raw in enumerate(payload):
        if not isinstance(raw, dict):
            record = _base_rollout_record(
                rollout_idx=None, source_position=position, expected=False
            )
            record.update(
                status="invalid_rollout_record",
                parse_status="invalid_rollout_record",
                parse_error="rollout entry must be a JSON object",
            )
            prepared.append(record)
            invalid_positions.add(position)
            continue
        rollout_idx = raw.get("rollout_idx")
        if isinstance(rollout_idx, bool) or not isinstance(rollout_idx, int) or rollout_idx < 0:
            record = _base_rollout_record(
                rollout_idx=None, source_position=position, expected=False
            )
            record.update(
                status="invalid_rollout_idx",
                parse_status="invalid_rollout_idx",
                parse_error="rollout_idx must be a non-negative integer",
            )
            prepared.append(record)
            invalid_positions.add(position)
            continue
        positions_by_idx.setdefault(rollout_idx, []).append(position)
        record = _base_rollout_record(
            rollout_idx=rollout_idx,
            source_position=position,
            expected=rollout_idx in expected_indices,
        )
        record["terminated"] = raw.get("terminated")
        if "error" in raw:
            record["author_error"] = raw["error"]
        record["_source"] = raw
        prepared.append(record)

    duplicate_indices = sorted(idx for idx, positions in positions_by_idx.items() if len(positions) > 1)
    duplicate_positions = {
        position for idx in duplicate_indices for position in positions_by_idx[idx]
    }
    for record in prepared:
        position = record["source_position"]
        if position in duplicate_positions:
            record.update(
                status="duplicate_rollout_idx",
                parse_status="duplicate_rollout_idx",
                parse_error="rollout_idx appears more than once in the author output",
            )
            record.pop("_source", None)

    observed_indices = set(positions_by_idx)
    missing_indices = sorted(expected_indices - observed_indices)
    unexpected_indices = sorted(observed_indices - expected_indices)
    for rollout_idx in missing_indices:
        record = _base_rollout_record(
            rollout_idx=rollout_idx, source_position=None, expected=True
        )
        record.update(status="missing_rollout", parse_status="missing_rollout")
        prepared.append(record)

    prepared.sort(
        key=lambda record: (
            record["rollout_idx"] is None,
            record["rollout_idx"] if record["rollout_idx"] is not None else 0,
            record["source_position"] if record["source_position"] is not None else -1,
        )
    )
    return prepared, missing_indices, unexpected_indices, duplicate_indices


def _load_bird_dev(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("BIRD dev.json root must be a list")
    examples: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"BIRD dev item {index} must be an object")
        if not isinstance(item.get("db_id"), str) or not item["db_id"]:
            raise ValueError(f"BIRD dev item {index} has no db_id")
        examples.append(item)
    return examples


def _load_selected_source_indices(
    input_manifest: Path,
    *,
    bird_dev_json: Path,
    record_count: int,
) -> list[int]:
    try:
        payload = json.loads(input_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not read prepared-input manifest {input_manifest}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("prepared-input manifest root must be an object")

    source = payload.get("source")
    if not isinstance(source, dict):
        raise ValueError("prepared-input manifest must contain a source object")
    if source.get("record_count") != record_count:
        raise ValueError(
            "prepared-input manifest source.record_count does not match BIRD dev.json"
        )
    source_digest = source.get("bird_dev_sha256")
    if source_digest != sha256_file(bird_dev_json):
        raise ValueError(
            "prepared-input manifest source.bird_dev_sha256 does not match BIRD dev.json"
        )

    selection = payload.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("prepared-input manifest must contain a selection object")
    indices = selection.get("source_indices")
    if not isinstance(indices, list):
        raise ValueError(
            "prepared-input manifest selection.source_indices must be a list"
        )
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError("selection.source_indices must contain only integer indices")
    if len(indices) != len(set(indices)):
        raise ValueError("selection.source_indices must not contain duplicates")
    invalid = sorted(index for index in indices if not 0 <= index < record_count)
    if invalid:
        raise ValueError(
            f"selection.source_indices outside BIRD dev range 0..{record_count - 1}: "
            f"{invalid}"
        )
    return list(indices)


def score_bird_dev(
    *,
    official_outputs_dir: Path,
    bird_dev_json: Path,
    input_manifest: Path,
    dev_databases_root: Path,
    expected_rollouts: int = 1,
    sql_timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """Score the prepared manifest's BIRD cohort without mutating author outputs."""

    official_outputs_dir = Path(official_outputs_dir)
    bird_dev_json = Path(bird_dev_json)
    input_manifest = Path(input_manifest)
    dev_databases_root = Path(dev_databases_root)
    if expected_rollouts < 1:
        raise ValueError("expected_rollouts must be positive")
    if sql_timeout_seconds <= 0:
        raise ValueError("sql_timeout_seconds must be positive")
    if not official_outputs_dir.is_dir():
        raise ValueError(f"official output directory does not exist: {official_outputs_dir}")
    if not dev_databases_root.is_dir():
        raise ValueError(f"BIRD dev database root does not exist: {dev_databases_root}")

    examples = _load_bird_dev(bird_dev_json)
    selected_source_indices = _load_selected_source_indices(
        input_manifest,
        bird_dev_json=bird_dev_json,
        record_count=len(examples),
    )
    selected_id_stems = {
        candidate
        for dataset_index in selected_source_indices
        for candidate in _instance_id_candidates(examples[dataset_index], dataset_index)
    }
    all_json_outputs = sorted(official_outputs_dir.glob("*.json"))
    available_outputs = {
        path.stem: path for path in all_json_outputs if path.stem in selected_id_stems
    }
    ignored_non_task_json = [
        str(path) for path in all_json_outputs if path.stem not in selected_id_stems
    ]
    tasks: list[dict[str, Any]] = []

    for dataset_index in selected_source_indices:
        example = examples[dataset_index]
        id_candidates = _instance_id_candidates(example, dataset_index)
        matching_paths = [available_outputs[name] for name in id_candidates if name in available_outputs]
        db_id = example["db_id"]
        db_path = dev_databases_root / db_id / f"{db_id}.sqlite"
        task: dict[str, Any] = {
            "dataset_index": dataset_index,
            "question_id": example.get("question_id"),
            "db_id": db_id,
            "instance_id_candidates": id_candidates,
            "expected_rollouts": expected_rollouts,
            "database_path": str(db_path),
            "gold_execution": {"status": "not_run"},
        }

        if len(matching_paths) == 0:
            task["output_status"] = "missing_output_file"
            task["rollouts"] = []
            for rollout_idx in range(expected_rollouts):
                record = _base_rollout_record(
                    rollout_idx=rollout_idx, source_position=None, expected=True
                )
                record.update(status="missing_rollout", parse_status="missing_rollout")
                task["rollouts"].append(record)
            task["missing_rollout_indices"] = list(range(expected_rollouts))
            task["unexpected_rollout_indices"] = []
            task["duplicate_rollout_indices"] = []
            tasks.append(task)
            continue
        if len(matching_paths) > 1:
            task["output_status"] = "ambiguous_output_files"
            task["output_files"] = [str(path) for path in matching_paths]
            task["rollouts"] = []
            for rollout_idx in range(expected_rollouts):
                record = _base_rollout_record(
                    rollout_idx=rollout_idx, source_position=None, expected=True
                )
                record.update(status="missing_rollout", parse_status="missing_rollout")
                task["rollouts"].append(record)
            task["missing_rollout_indices"] = list(range(expected_rollouts))
            task["unexpected_rollout_indices"] = []
            task["duplicate_rollout_indices"] = []
            tasks.append(task)
            continue

        output_path = matching_paths[0]
        task["instance_id"] = output_path.stem
        task["output_file"] = str(output_path)
        task["output_sha256"] = sha256_file(output_path)
        output_status, payload, output_error = _read_rollout_file(output_path)
        task["output_status"] = output_status
        if output_error is not None:
            task["output_error"] = output_error
        rollouts, missing, unexpected, duplicates = _prepare_rollouts(
            payload, expected_rollouts
        )
        task["rollouts"] = rollouts
        task["missing_rollout_indices"] = missing
        task["unexpected_rollout_indices"] = unexpected
        task["duplicate_rollout_indices"] = duplicates

        parsed_records: list[dict[str, Any]] = []
        for record in rollouts:
            source = record.pop("_source", None)
            if source is None:
                continue
            parsed = parse_last_assistant_answer(source)
            record["parse_status"] = parsed.status
            if parsed.message_source is not None:
                record["message_source"] = parsed.message_source
            if parsed.error is not None:
                record["parse_error"] = parsed.error
            if parsed.sql is None:
                record["status"] = "parse_error"
                continue
            record["predicted_sql"] = parsed.sql
            parsed_records.append(record)

        gold_sql = _gold_sql(example)
        if not db_path.is_file():
            gold_execution = SqlExecution(
                status="missing_database",
                elapsed_seconds=0.0,
                error=f"database file does not exist: {db_path}",
            )
        elif gold_sql is None:
            gold_execution = SqlExecution(
                status="missing_gold_sql",
                elapsed_seconds=0.0,
                error="BIRD dev item contains no gold SQL",
            )
        else:
            gold_execution = execute_read_only_sql(
                db_path, gold_sql, sql_timeout_seconds
            )
        task["gold_execution"] = gold_execution.manifest_fields()

        if not parsed_records:
            tasks.append(task)
            continue

        for record in parsed_records:
            prediction = execute_read_only_sql(
                db_path, record["predicted_sql"], sql_timeout_seconds
            )
            record["prediction_execution"] = prediction.manifest_fields()
            if prediction.status == "timeout":
                record["status"] = "prediction_timeout"
                continue
            if prediction.status == "missing_database":
                record["status"] = "missing_database"
                continue
            if prediction.status != "ok":
                record["status"] = "prediction_execution_error"
                continue
            if gold_execution.status == "timeout":
                record["status"] = "gold_timeout"
                continue
            if gold_execution.status == "missing_database":
                record["status"] = "missing_database"
                continue
            if gold_execution.status == "missing_gold_sql":
                record["status"] = "missing_gold_sql"
                continue
            if gold_execution.status != "ok":
                record["status"] = "gold_execution_error"
                continue
            assert prediction.rows is not None and gold_execution.rows is not None
            record["correct"] = bird_set_equal(prediction.rows, gold_execution.rows)
            record["status"] = "correct" if record["correct"] else "wrong_result"

        tasks.append(task)

    rollout_records = [record for task in tasks for record in task["rollouts"]]
    observed_records = [record for record in rollout_records if record["source_position"] is not None]
    expected_records = [record for record in rollout_records if record["expected"]]
    independently_scored = [
        record for record in rollout_records if record["status"] in {"correct", "wrong_result"}
    ]
    correct_expected = sum(record["status"] == "correct" for record in expected_records)
    expected_slots = len(selected_source_indices) * expected_rollouts
    status_counts = Counter(str(record["status"]) for record in rollout_records)
    parse_counts = Counter(str(record["parse_status"]) for record in rollout_records)
    prediction_execution_counts = Counter(
        str(record["prediction_execution"]["status"]) for record in rollout_records
    )
    gold_counts = Counter(str(task["gold_execution"]["status"]) for task in tasks)
    output_counts = Counter(str(task["output_status"]) for task in tasks)

    summary = {
        "tasks": len(selected_source_indices),
        "expected_rollouts_per_task": expected_rollouts,
        "expected_rollout_slots": expected_slots,
        "observed_rollout_records": len(observed_records),
        "independently_scored_rollouts": len(independently_scored),
        "correct_expected_rollouts": correct_expected,
        "accuracy_over_expected_rollout_slots": (
            correct_expected / expected_slots if expected_slots else 0.0
        ),
        "rollout_status_counts": dict(sorted(status_counts.items())),
        "parse_status_counts": dict(sorted(parse_counts.items())),
        "prediction_execution_status_counts": dict(
            sorted(prediction_execution_counts.items())
        ),
        "gold_execution_status_counts": dict(sorted(gold_counts.items())),
        "output_status_counts": dict(sorted(output_counts.items())),
        "parse_errors": status_counts["parse_error"],
        "prediction_execution_errors": status_counts["prediction_execution_error"],
        "prediction_timeouts": status_counts["prediction_timeout"],
        "gold_execution_error_tasks": gold_counts["execution_error"],
        "gold_timeout_tasks": gold_counts["timeout"],
        "missing_output_files": output_counts["missing_output_file"],
        "missing_rollout_slots": status_counts["missing_rollout"],
        "missing_database_tasks": gold_counts["missing_database"],
        "missing_gold_sql_tasks": gold_counts["missing_gold_sql"],
        "duplicate_rollout_records": status_counts["duplicate_rollout_idx"],
        "unexpected_rollout_records": sum(
            record["source_position"] is not None and not record["expected"]
            for record in rollout_records
        ),
        "ignored_non_task_json_files": ignored_non_task_json,
    }
    if expected_rollouts == 1:
        summary["greedy_correct_tasks"] = correct_expected
        summary["greedy_accuracy"] = (
            correct_expected / len(selected_source_indices)
            if selected_source_indices
            else 0.0
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_outputs_modified": False,
        "inputs": {
            "official_outputs_dir": str(official_outputs_dir),
            "bird_dev_json": str(bird_dev_json),
            "bird_dev_sha256": sha256_file(bird_dev_json),
            "input_manifest": str(input_manifest),
            "input_manifest_sha256": sha256_file(input_manifest),
            "selected_source_indices": selected_source_indices,
            "dev_databases_root": str(dev_databases_root),
        },
        "evaluation": {
            "benchmark": "BIRD-dev",
            "mode": "greedy-k1" if expected_rollouts == 1 else "independent-rollouts",
            "denotation_metric": "bird-set",
            "denotation_definition": (
                "row order and duplicate-row multiplicity ignored; column order preserved"
            ),
            "rollout_aggregation": "none",
            "rollouts_scored_independently": True,
            "expected_rollouts_per_task": expected_rollouts,
            "sql_timeout_seconds": sql_timeout_seconds,
            "sqlite_access": "mode=ro + PRAGMA query_only + progress-handler deadline",
        },
        "summary": summary,
        "tasks": tasks,
    }


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--official-outputs-dir",
        "--results",
        dest="official_outputs_dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--bird-dev-json",
        "--bird-json",
        dest="bird_dev_json",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--input-manifest",
        required=True,
        type=Path,
        help="manifest emitted by prepare_bird_dev.py; selection.source_indices is authoritative",
    )
    parser.add_argument(
        "--dev-databases-root",
        "--database-root",
        dest="dev_databases_root",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output-manifest",
        "--output",
        dest="output_manifest",
        required=True,
        type=Path,
    )
    parser.add_argument("--expected-rollouts", type=int, default=1)
    parser.add_argument("--sql-timeout-seconds", type=float, default=10.0)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    manifest = score_bird_dev(
        official_outputs_dir=args.official_outputs_dir,
        bird_dev_json=args.bird_dev_json,
        input_manifest=args.input_manifest,
        dev_databases_root=args.dev_databases_root,
        expected_rollouts=args.expected_rollouts,
        sql_timeout_seconds=args.sql_timeout_seconds,
    )
    protected_task_paths = {
        Path(path).resolve()
        for task in manifest["tasks"]
        for path in (
            ([task["output_file"]] if "output_file" in task else [])
            + task.get("output_files", [])
        )
    }
    output_manifest = args.output_manifest.resolve()
    if output_manifest in protected_task_paths:
        raise ValueError("refusing to overwrite an official trustsql_eval task output file")
    if (
        output_manifest.parent == args.official_outputs_dir.resolve()
        and output_manifest.stem
        in {
            candidate
            for task in manifest["tasks"]
            for candidate in task["instance_id_candidates"]
        }
    ):
        raise ValueError("refusing to create a scorer manifest at a task-output filename")
    if output_manifest.is_file():
        try:
            existing = json.loads(output_manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "refusing to overwrite an existing non-scorer JSON file"
            ) from exc
        if not (
            isinstance(existing, dict)
            and existing.get("schema_version") == SCHEMA_VERSION
            and existing.get("source_outputs_modified") is False
        ):
            raise ValueError("refusing to overwrite an existing non-scorer JSON file")
    write_manifest(manifest, args.output_manifest)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
