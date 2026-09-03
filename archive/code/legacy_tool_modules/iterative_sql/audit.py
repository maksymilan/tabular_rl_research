#!/usr/bin/env python3
"""Fresh replay and structural audit for iterative-SQL diagnostic episodes.

The emitted audit deliberately excludes SQL text, reasoning, gold rows, and verifier payloads.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from denotation import compare_denotations  # noqa: E402
from executor import Harness  # noqa: E402
from rollout import task_db_path, task_gold_sql  # noqa: E402
from text2sql import execute_predicted_sql  # noqa: E402
from tool_modules.iterative_sql.protocol import (  # noqa: E402
    ITERATIVE_SQL_INTERFACE,
    ITERATIVE_SQL_INTERFACE_V3,
    ITERATIVE_SQL_INTERFACE_V4,
    ITERATIVE_SQL_INTERFACE_V5,
    ITERATIVE_SQL_INTERFACES,
    ITERATIVE_SQL_PROTOCOL_VERSION,
    ITERATIVE_SQL_PROTOCOL_VERSION_V3,
    ITERATIVE_SQL_PROTOCOL_VERSION_V4,
    ITERATIVE_SQL_PROTOCOL_VERSION_V5,
    ITERATIVE_SQL_PROTOCOL_VERSIONS,
    iterative_sql_tool_schema_hash,
    parse_iterative_sql_action,
)
from tool_modules.sql_common.runner import (  # noqa: E402
    canonical_sql_text,
    enrich_direct_sql_preview,
    execute_preview,
)


FORBIDDEN_MODEL_INPUT_KEYS = {
    "gold_sql",
    "gold_exec_results",
    "gold_sql_path",
    "gold_sample",
    "gold_row_count",
}


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def nested_keys(value) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            keys.add(str(key))
            keys.update(nested_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.update(nested_keys(child))
    return keys


def audit_record(
    record: dict,
    task: dict,
    *,
    max_steps: int,
    preview_rows: int,
    execution_timeout_seconds: float,
    denotation_comparison: str,
    expected_protocol_version: str,
    expected_interface: str,
) -> dict:
    reasons: list[str] = []
    if record.get("tool_scheme") != "iterative-sql":
        reasons.append("tool_scheme_mismatch")
    if record.get("protocol_version") != expected_protocol_version:
        reasons.append("protocol_version_mismatch")
    if record.get("interface") != expected_interface:
        reasons.append("interface_mismatch")
    turns = record.get("turns") or []
    if len(turns) > max_steps:
        reasons.append("max_steps_exceeded")

    h = Harness(task_db_path(task))
    h.conn.execute("PRAGMA query_only = ON")
    inspected_sql: set[str] = set()
    replayed_tool_outputs = 0
    parsed_actions = 0
    execute_calls = 0
    submit_calls = 0
    final_sql: str | None = None
    try:
        for turn_index, turn in enumerate(turns):
            if nested_keys(turn.get("model_input")) & FORBIDDEN_MODEL_INPUT_KEYS:
                reasons.append("hidden_key_in_model_input")
            parsed = turn.get("parsed")
            if not parsed:
                continue
            parsed_actions += 1
            try:
                think, tool, arguments = parse_iterative_sql_action(
                    turn.get("model_output", "")
                )
            except Exception:
                reasons.append("strict_reparse_failed")
                continue
            if (
                think != parsed.get("think")
                or tool != parsed.get("tool")
                or arguments != parsed.get("arguments")
            ):
                reasons.append("parsed_action_drift")
                continue
            if turn.get("error_event"):
                later_inputs = [
                    item.get("model_input")
                    for item in turns[turn_index + 1:turn_index + 2]
                ]
                if later_inputs and not any(
                    "LAST SQL ERROR" in json.dumps(value, ensure_ascii=False)
                    and str(turn["error_event"].get("error_type"))
                    in json.dumps(value, ensure_ascii=False)
                    for value in later_inputs
                ):
                    reasons.append("structured_error_not_visible_next_turn")
                continue

            sql = arguments["sql"]
            if tool == "execute_sql":
                execute_calls += 1
                expected = execute_preview(
                    h,
                    sql,
                    limit=preview_rows,
                    timeout_seconds=execution_timeout_seconds,
                    strict_schema_pragmas=True,
                )
                expected = enrich_direct_sql_preview(
                    expected,
                    include_query_shape_audit=(
                        expected_protocol_version in {
                            ITERATIVE_SQL_PROTOCOL_VERSION_V4,
                            ITERATIVE_SQL_PROTOCOL_VERSION_V5,
                            ITERATIVE_SQL_PROTOCOL_VERSION,
                        }
                    ),
                    include_data_dependency_audit=(
                        expected_protocol_version in {
                            ITERATIVE_SQL_PROTOCOL_VERSION_V5,
                            ITERATIVE_SQL_PROTOCOL_VERSION,
                        }
                    ),
                )
                if expected != turn.get("tool_output"):
                    reasons.append("sql_preview_replay_mismatch")
                else:
                    replayed_tool_outputs += 1
                inspected_sql.add(canonical_sql_text(sql))
                continue

            submit_calls += 1
            final_sql = sql
            if turn_index != len(turns) - 1:
                reasons.append("submit_not_last_turn")
            if canonical_sql_text(sql) not in inspected_sql:
                reasons.append("submit_without_prior_execute")

        if record.get("legal") and submit_calls != 1:
            reasons.append("legal_episode_submit_count")
        if record.get("correct") and not record.get("legal"):
            reasons.append("correct_without_legal_terminal")

        replay_correct = False
        if final_sql is not None:
            predicted = execute_predicted_sql(h, final_sql, execution_timeout_seconds)
            gold = h.gold(task_gold_sql(task))
            replay_correct = compare_denotations(
                predicted,
                gold,
                denotation_comparison,
            )
        if replay_correct != bool(record.get("correct")):
            reasons.append("terminal_replay_outcome_mismatch")
    except Exception:
        replay_correct = False
        reasons.append("replay_exception")
    finally:
        h.conn.close()

    structural_pass = not reasons
    retained = bool(record.get("correct")) and replay_correct and structural_pass
    return {
        "example_index": record.get("example_index"),
        "recorded_correct": bool(record.get("correct")),
        "recorded_legal": bool(record.get("legal")),
        "replay_correct": replay_correct,
        "structural_pass": structural_pass,
        "retained_diagnostic_success": retained,
        "parsed_actions": parsed_actions,
        "replayed_tool_outputs": replayed_tool_outputs,
        "execute_calls": execute_calls,
        "submit_calls": submit_calls,
        "failure_reasons": sorted(set(reasons)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    manifest = json.loads((result_dir / "manifest.json").read_text(encoding="utf-8"))
    protocol_version = manifest.get("protocol_version")
    interface = manifest.get("interface")
    if protocol_version not in ITERATIVE_SQL_PROTOCOL_VERSIONS:
        raise ValueError(f"unsupported iterative-SQL protocol {protocol_version!r}")
    if interface not in ITERATIVE_SQL_INTERFACES:
        raise ValueError(f"unsupported iterative-SQL interface {interface!r}")
    expected_pairs = {
        ITERATIVE_SQL_PROTOCOL_VERSION_V3: ITERATIVE_SQL_INTERFACE_V3,
        ITERATIVE_SQL_PROTOCOL_VERSION_V4: ITERATIVE_SQL_INTERFACE_V4,
        ITERATIVE_SQL_PROTOCOL_VERSION_V5: ITERATIVE_SQL_INTERFACE_V5,
        ITERATIVE_SQL_PROTOCOL_VERSION: ITERATIVE_SQL_INTERFACE,
    }
    if expected_pairs[protocol_version] != interface:
        raise ValueError(
            f"protocol/interface mismatch: {protocol_version!r} with {interface!r}"
        )

    records = load_jsonl(result_dir / "all.jsonl")
    tasks = {
        int(item["example_index"]): item
        for item in load_jsonl(Path(args.tasks_json))
    }
    audited = [
        audit_record(
            record,
            tasks[int(record["example_index"])],
            max_steps=int(manifest["max_steps"]),
            preview_rows=int(manifest["preview_rows"]),
            execution_timeout_seconds=float(manifest["execution_timeout_seconds"]),
            denotation_comparison=manifest["denotation_comparison"],
            expected_protocol_version=protocol_version,
            expected_interface=interface,
        )
        for record in records
    ]
    summary = {
        "schema_version": "iterative-sql-audit-v1",
        "tool_scheme": manifest.get("tool_scheme"),
        "protocol_version": protocol_version,
        "protocol_hash": manifest.get("protocol_hash"),
        "public_tool_schema_sha256": iterative_sql_tool_schema_hash(),
        "denotation_comparison": manifest.get("denotation_comparison"),
        "training_admission": "diagnostic_only",
        "total": len(audited),
        "recorded_correct": sum(item["recorded_correct"] for item in audited),
        "replay_correct": sum(item["replay_correct"] for item in audited),
        "structural_pass": sum(item["structural_pass"] for item in audited),
        "retained_diagnostic_success": sum(
            item["retained_diagnostic_success"] for item in audited
        ),
        "records": audited,
    }
    Path(args.out).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(
        {key: value for key, value in summary.items() if key != "records"},
        ensure_ascii=False,
        indent=2,
    ))
    return 0 if summary["retained_diagnostic_success"] == summary["recorded_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
