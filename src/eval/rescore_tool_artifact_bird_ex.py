#!/usr/bin/env python3
"""Replay an atomic tool artifact under the exact BIRD EX terminal contract.

The historical tool scorer could accept a cited table after permuting same-width columns. BIRD EX
does not: it compares ``set(predicted_cursor.fetchall())`` with the corresponding gold set, so row
order and duplicate multiplicity are ignored while column position remains significant.

This audit replays only samples previously marked correct. A stricter contract cannot turn a
previously wrong sample into a correct one, so this is sufficient to recompute pass@k while avoiding
unnecessary replay of failed trajectories.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "sft"))

from denotation import bird_rows_equal  # noqa: E402
from executor import Harness  # noqa: E402
from passk import pass_at_from_flags  # noqa: E402
from rollout import (  # noqa: E402
    _evidence_table,
    execute_tool,
    load_tasks_json,
    new_ctx,
    overview,
    task_db_path,
    task_gold_sql,
)

TERMINAL_ANSWER_CONTRACT = "exact-cited-table-v1"


def _resolve_path(value: str, base: Path = ROOT) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _task_map(tasks_path: Path) -> dict[int, dict]:
    tasks = load_tasks_json(str(tasks_path))
    return {
        int(task.get("example_index", position)): task
        for position, task in enumerate(tasks)
    }


def _replay_exact(sample: dict, task: dict) -> dict:
    harness = Harness(task_db_path(task))
    context = new_ctx(overview(harness))
    created: set[str] = set()
    terminal_args = None
    replayed_actions = 0
    try:
        for position, turn in enumerate(sample.get("turns") or []):
            parsed = turn.get("parsed")
            if not isinstance(parsed, dict):
                continue
            tool = parsed.get("tool")
            arguments = parsed.get("arguments") or {}
            if tool == "answer_from_context":
                terminal_args = arguments
                break

            step_id = f"step_{int(turn.get('turn_index', position)) + 1}"
            recorded_error = bool(turn.get("execution_error"))
            try:
                _, table_name = execute_tool(
                    harness,
                    tool,
                    arguments,
                    context,
                    step_id,
                )
            except Exception:
                if recorded_error:
                    continue
                raise
            if recorded_error:
                raise RuntimeError(
                    f"{step_id} replay succeeded but the recorded action was rejected"
                )
            replayed_actions += 1
            if table_name:
                created.add(table_name)

        if terminal_args is None:
            raise RuntimeError("previously correct sample has no replayable terminal action")
        evidence_table = _evidence_table(terminal_args.get("evidence"))
        if not evidence_table or (
            evidence_table not in created and evidence_table not in harness.views
        ):
            raise RuntimeError(f"terminal evidence table is unavailable: {evidence_table!r}")

        predicted_rows = harness.rows(evidence_table)
        gold_sql = task_gold_sql(task)
        if not gold_sql:
            raise RuntimeError("task has no gold SQL")
        gold_rows = harness.gold(gold_sql)
        return {
            "official_correct": bird_rows_equal(predicted_rows, gold_rows),
            "evidence_table": evidence_table,
            "predicted_row_count": len(predicted_rows),
            "gold_row_count": len(gold_rows),
            "predicted_sample": [list(row) for row in predicted_rows[:5]],
            "gold_sample": [list(row) for row in gold_rows[:5]],
            "replayed_actions": replayed_actions,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - audit records the exact replay failure.
        return {
            "official_correct": None,
            "evidence_table": None,
            "predicted_row_count": None,
            "gold_row_count": None,
            "predicted_sample": [],
            "gold_sample": [],
            "replayed_actions": replayed_actions,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        harness.conn.close()


def _pass_k_from_manifest(manifest: dict) -> tuple[int, ...]:
    values = manifest.get("pass_k")
    if isinstance(values, list) and values:
        return tuple(int(value) for value in values)
    return (1,)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--tasks-json", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    artifact_dir = args.artifact_dir.resolve()
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("tool_scheme") != "atomic":
        parser.error("this audit currently supports atomic tool artifacts only")
    if manifest.get("denotation_comparison") != "bird-set":
        parser.error("source artifact must use denotation_comparison='bird-set'")

    tasks_path = args.tasks_json
    if tasks_path is None:
        dataset = manifest.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            parser.error("manifest has no dataset path; pass --tasks-json")
        tasks_path = _resolve_path(dataset)
    tasks = _task_map(tasks_path.resolve())
    records = _load_jsonl(artifact_dir / "all.jsonl")
    pass_k = _pass_k_from_manifest(manifest)

    output_records = []
    replay_errors = []
    revoked_samples = []
    replay_pool = ThreadPoolExecutor(max_workers=args.workers)
    replay_futures = {}
    for record_position, record in enumerate(records):
        task = tasks.get(int(record["example_index"]))
        if task is None:
            continue
        samples = record.get("samples")
        if not isinstance(samples, list):
            samples = [record]
        for sample_position, sample in enumerate(samples):
            if sample.get("correct"):
                replay_futures[(record_position, sample_position)] = replay_pool.submit(
                    _replay_exact,
                    sample,
                    task,
                )

    for record_position, record in enumerate(records):
        example_index = int(record["example_index"])
        task = tasks.get(example_index)
        if task is None:
            replay_errors.append(
                {"example_index": example_index, "error": "task not found"}
            )
            continue

        samples = record.get("samples")
        if not isinstance(samples, list):
            samples = [record]
        official_flags = []
        sample_audits = []
        for sample_position, sample in enumerate(samples):
            legacy_correct = bool(sample.get("correct"))
            if legacy_correct:
                audit = replay_futures[(record_position, sample_position)].result()
                if audit["error"]:
                    replay_errors.append(
                        {
                            "example_index": example_index,
                            "sample_index": sample.get("sample_index", sample_position),
                            "error": audit["error"],
                        }
                    )
                    official_correct = None
                else:
                    official_correct = bool(audit["official_correct"])
                    if not official_correct:
                        revoked_samples.append(
                            {
                                "example_index": example_index,
                                "sample_index": sample.get("sample_index", sample_position),
                            }
                        )
            else:
                audit = None
                official_correct = False

            official_flags.append(official_correct)
            sample_audits.append(
                {
                    "sample_index": sample.get("sample_index", sample_position),
                    "legacy_correct": legacy_correct,
                    "official_correct": official_correct,
                    "audit": audit,
                }
            )

        known_flags = [bool(flag) if flag is not None else False for flag in official_flags]
        output_records.append(
            {
                "example_index": example_index,
                "db_id": record.get("db_id"),
                "legacy_pass_at": record.get("pass_at") or {"1": bool(record.get("correct"))},
                "official_pass_at": pass_at_from_flags(known_flags, pass_k),
                "samples": sample_audits,
            }
        )
    replay_pool.shutdown(wait=True)

    total = len(records)
    legacy_counts = {
        str(k): sum(
            bool((record.get("pass_at") or {"1": record.get("correct")}).get(str(k)))
            for record in records
        )
        for k in pass_k
    }
    official_counts = {
        str(k): sum(bool(record["official_pass_at"][str(k)]) for record in output_records)
        for k in pass_k
    }
    summary = {
        "source_artifact": str(artifact_dir),
        "source_manifest_terminal_answer_contract": manifest.get(
            "terminal_answer_contract"
        ),
        "denotation_comparison": "bird-set",
        "terminal_answer_contract": TERMINAL_ANSWER_CONTRACT,
        "tasks_json": str(tasks_path.resolve()),
        "total": total,
        "audited_records": len(output_records),
        "workers": args.workers,
        "replay_error_count": len(replay_errors),
        "revoked_sample_count": len(revoked_samples),
        "pass_at": {
            str(k): {
                "legacy_correct": legacy_counts[str(k)],
                "official_correct": official_counts[str(k)],
                "total": total,
                "official_rate": official_counts[str(k)] / total if total else 0.0,
            }
            for k in pass_k
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as target:
        for record in output_records:
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "replay_errors.json").write_text(
        json.dumps(replay_errors, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "revoked_samples.json").write_text(
        json.dumps(revoked_samples, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 2 if replay_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
