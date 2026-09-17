"""Read-only fixed-suffix action deletion replay for Atomic v26 rollouts.

This module deliberately reports a structural estimand only: whether the *recorded
suffix* still executes and reaches the same terminal result after one action is
omitted.  It is not a policy counterfactual and must not be used as a reward label.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any


@dataclass(frozen=True)
class CounterfactualCase:
    trajectory_id: str
    skip_turn_index: int
    skip_tool: str
    original_correct: bool
    label: str
    original_failure_type: str | None
    branch_correct: bool
    branch_failure_type: str | None
    baseline_fidelity: bool
    baseline_mismatch: str | None = None
    first_new_error: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "skip_turn_index": self.skip_turn_index,
            "skip_tool": self.skip_tool,
            "original_correct": self.original_correct,
            "label": self.label,
            "original_failure_type": self.original_failure_type,
            "branch_correct": self.branch_correct,
            "branch_failure_type": self.branch_failure_type,
            "baseline_fidelity": self.baseline_fidelity,
            "baseline_mismatch": self.baseline_mismatch,
            "first_new_error": self.first_new_error,
        }


def _freeze(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    return value


def _turn_signature(turn: dict[str, Any]) -> tuple:
    parsed = turn.get("parsed") or {}
    output = turn.get("tool_output") or {}
    error = turn.get("execution_error_type")
    if error:
        return ("error", error, parsed.get("tool"))
    # These fields are the semantic public output contract.  Reasoning and raw
    # error text are intentionally excluded from the fidelity gate.
    output_signature = tuple(
        (key, _freeze(output.get(key)))
        for key in ("table", "kind", "row_count", "columns", "rows", "tables", "value", "scalar")
        if key in output
    )
    return ("success", parsed.get("tool"), output_signature, _freeze(turn.get("pred_sample")))


def _env_from_record(record: dict[str, Any], db_path: str):
    # Runtime selection is explicit in the caller; never silently import an archive.
    from rl.runtime.tool_environment_v26 import ToolUseEnv
    if not Path(db_path).is_file():
        raise FileNotFoundError(db_path)
    example = {
        "db_id": record["db_id"],
        "question": record["question"],
        "gold_sql": record.get("gold_sql"),
        "db_path": db_path,
    }
    env = ToolUseEnv(
        example,
        max_steps=max(40, len(record.get("turns") or []) + 2),
        max_errors_per_type=3,
        error_feedback_version=record.get("error_feedback_version", "legacy"),
        tool_execution_timeout_seconds=float(record.get("tool_execution_timeout_seconds") or 10.0),
    )
    # Temp relation caches are allowed; persistent databases must never be mutated.
    write_operations = {getattr(sqlite3, name) for name in (
        "SQLITE_INSERT", "SQLITE_UPDATE", "SQLITE_DELETE", "SQLITE_CREATE_INDEX",
        "SQLITE_CREATE_TABLE", "SQLITE_CREATE_TRIGGER", "SQLITE_CREATE_VIEW",
        "SQLITE_DROP_INDEX", "SQLITE_DROP_TABLE", "SQLITE_DROP_TRIGGER",
        "SQLITE_DROP_VIEW", "SQLITE_ALTER_TABLE", "SQLITE_REINDEX", "SQLITE_ANALYZE",
    )}
    env.harness.conn.set_authorizer(
        lambda action, arg1, arg2, database, trigger: sqlite3.SQLITE_DENY
        if (action in write_operations and database != "temp") or action in (
            sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH,
        ) else sqlite3.SQLITE_OK
    )
    return env


def _run(record: dict[str, Any], db_path: str, skip_turn_index: int | None) -> tuple[dict, str | None]:
    """Replay one record, retaining original step and handle identity.

    ``ToolUseEnv.apply_model_output`` increments its logical step before execution.
    Setting ``steps`` to the original predecessor preserves ``step_N`` value_ref
    identities.  Setting ``Harness._n`` before a table-producing survivor preserves
    the original table handle.  A skipped producer is therefore never accidentally
    aliased to a later output.  This is a diagnostic-only identity pin, not a change
    to the production environment.
    """
    env = _env_from_record(record, db_path)
    turns = record.get("turns") or []
    try:
        for index, source_turn in enumerate(turns):
            if index == skip_turn_index:
                continue
            if env.done:
                break
            raw = source_turn.get("model_output")
            if not isinstance(raw, str):
                return env.record(), "missing_raw_model_output"
            parsed = source_turn.get("parsed") or {}
            output = source_turn.get("tool_output") or {}
            # Stable original step id, including for an action after an omitted turn.
            env.steps = index
            recorded_table = output.get("table")
            if isinstance(recorded_table, str):
                try:
                    env.harness._n = int(recorded_table.rsplit("_", 1)[1]) - 1
                except (ValueError, IndexError):
                    return env.record(), "unrecognized_recorded_handle"
            env.apply_model_output(raw)
            actual_turn = env.turns[-1]
            if skip_turn_index is None and _turn_signature(actual_turn) != _turn_signature(source_turn):
                return env.record(), f"turn_{index}_fidelity_mismatch"
        result = env.record()
        last_call = (env.turns[-1].get("parsed") or {}) if env.turns else {}
        if last_call.get("tool") == "answer_from_context":
            evidence = (last_call.get("arguments") or {}).get("evidence") or {}
            table = evidence.get("table") if isinstance(evidence, dict) else evidence
            result["replay_terminal_evidence_exists"] = table in env.harness.views
        return result, None
    finally:
        env.close()


def _branch_label(original: dict, branch: dict, *, had_new_error: bool) -> str:
    if had_new_error:
        return "new_execution_error"
    if branch.get("replay_terminal_evidence_exists") is False:
        return "missing_terminal_evidence"
    if "replay_terminal_evidence_exists" not in branch:
        return "unknown_no_terminal"
    original_correct = bool(original.get("correct"))
    branch_correct = bool(branch.get("correct"))
    if original_correct and branch_correct:
        return "executable_same_result"
    if original_correct and not branch_correct:
        return "executable_changed_result"
    if not original_correct and not branch_correct:
        return "still_wrong"
    return "wrong_to_correct"


def audit_trajectory(record: dict[str, Any], db_path: str) -> list[CounterfactualCase]:
    """Baseline-gated skip audit for every nonterminal parsed tool action."""
    baseline, mismatch = _run(record, db_path, None)
    original_turns = record.get("turns") or []
    if mismatch is None:
        expected = (bool(record.get("correct")), bool(record.get("legal")), record.get("failure_type"))
        actual = (bool(baseline.get("correct")), bool(baseline.get("legal")), baseline.get("failure_type"))
        if expected != actual:
            mismatch = f"final_fidelity_mismatch:{expected!r}!={actual!r}"
    if mismatch is not None:
        incomplete = mismatch.startswith("missing_raw_model_output")
        return [CounterfactualCase(
            trajectory_id=str(record.get("trajectory_id")), skip_turn_index=-1,
            skip_tool="<incomplete>", original_correct=bool(record.get("correct")),
            label="unknown_incomplete" if incomplete else "baseline_mismatch",
            original_failure_type=record.get("failure_type"), branch_correct=bool(baseline.get("correct")),
            branch_failure_type=baseline.get("failure_type"), baseline_fidelity=False,
            baseline_mismatch=mismatch,
        )]
    cases: list[CounterfactualCase] = []
    for index, turn in enumerate(original_turns[:-1]):
        parsed = turn.get("parsed") or {}
        tool = parsed.get("tool")
        if not isinstance(tool, str) or tool == "answer_from_context":
            continue
        branch, branch_mismatch = _run(record, db_path, index)
        original_errors = {(e.get("action_index"), e.get("error_type")) for e in record.get("error_events") or []}
        new_errors = [e for e in branch.get("error_events") or []
                      if (e.get("action_index"), e.get("error_type")) not in original_errors]
        had_new_error = branch_mismatch is not None or bool(new_errors)
        label = (
            "unknown_incomplete" if branch_mismatch and branch_mismatch.startswith("missing_raw_model_output")
            else "baseline_mismatch" if branch_mismatch
            else _branch_label(record, branch, had_new_error=had_new_error)
        )
        cases.append(CounterfactualCase(
            trajectory_id=str(record.get("trajectory_id")), skip_turn_index=index,
            skip_tool=tool, original_correct=bool(record.get("correct")), label=label,
            original_failure_type=record.get("failure_type"),
            branch_correct=bool(branch.get("correct")), branch_failure_type=branch.get("failure_type"),
            baseline_fidelity=branch_mismatch is None, baseline_mismatch=branch_mismatch,
            first_new_error={key: new_errors[0].get(key) for key in (
                "action_index", "error_type", "error_code", "attempted_tool",
            )} if new_errors else None,
        ))
    return cases


def summarize_cases(cases: list[CounterfactualCase]) -> dict[str, Any]:
    labels = Counter(case.label for case in cases)
    return {
        "cases": len(cases),
        "labels": dict(sorted(labels.items())),
        "by_original_correct": {
            "correct": sum(case.original_correct for case in cases),
            "wrong": sum(not case.original_correct for case in cases),
        },
    }
