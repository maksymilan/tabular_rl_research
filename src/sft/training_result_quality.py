"""Training-only empty-result policy for causal database-tool trajectories.

Empty intermediate results remain in the executed trajectory and in later model-visible history,
but the action that produced an empty row relation is not a positive SFT target.  A trajectory
whose terminal evidence relation is empty is excluded wholesale.  Scalar answers such as COUNT=0
remain eligible because their evidence relation contains one row whose value is zero.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


EMPTY_RESULT_POLICY_VERSION = "causal-empty-result-target-filter-v1"
EMPTY_RESULT_TARGET_REASON = "empty_table_result"
EMPTY_TERMINAL_REASON = "empty_terminal_evidence"


def is_empty_row_result(output: Any) -> bool:
    """Return true only for a tool result that explicitly reports zero result rows."""
    return (
        isinstance(output, Mapping)
        and not isinstance(output.get("row_count"), bool)
        and output.get("row_count") == 0
    )


def empty_result_target_reason(step: Mapping[str, Any]) -> str | None:
    if is_empty_row_result(step.get("tool_output")):
        return EMPTY_RESULT_TARGET_REASON
    return None


def terminal_evidence_row_count(step: Mapping[str, Any]) -> int | None:
    call = step.get("tool_call")
    if not isinstance(call, Mapping) or call.get("tool") != "answer_from_context":
        return None
    arguments = call.get("arguments")
    evidence = arguments.get("evidence") if isinstance(arguments, Mapping) else None
    table = evidence.get("table") if isinstance(evidence, Mapping) else None
    state = step.get("environment_state_before")
    tables = state.get("tables") if isinstance(state, Mapping) else None
    table_state = tables.get(table) if isinstance(tables, Mapping) and isinstance(table, str) else None
    row_count = table_state.get("row_count") if isinstance(table_state, Mapping) else None
    if isinstance(row_count, bool) or not isinstance(row_count, int):
        return None
    return row_count


def trajectory_has_empty_terminal_evidence(
    steps: Sequence[Mapping[str, Any]],
) -> bool:
    return any(terminal_evidence_row_count(step) == 0 for step in steps)


def apply_empty_result_target_annotation(step: dict[str, Any]) -> None:
    """Mark an empty-result action context-only without deleting its causal observation."""
    reason = empty_result_target_reason(step)
    if reason is None:
        return
    step["sft_target_eligible"] = False
    step["sft_target_exclusion_reason"] = reason


def training_quality_summary(steps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    empty_steps = [
        str(step.get("step_id"))
        for step in steps
        if empty_result_target_reason(step) is not None
    ]
    terminal_counts = [
        terminal_evidence_row_count(step)
        for step in steps
        if isinstance(step.get("tool_call"), Mapping)
        and step["tool_call"].get("tool") == "answer_from_context"
    ]
    terminal_row_count = terminal_counts[-1] if terminal_counts else None
    empty_terminal = terminal_row_count == 0
    return {
        "empty_result_policy": EMPTY_RESULT_POLICY_VERSION,
        "empty_result_context_only_step_ids": empty_steps,
        "empty_result_context_only_steps": len(empty_steps),
        "has_terminal_evidence": bool(terminal_counts),
        "terminal_evidence_row_count": terminal_row_count,
        "empty_terminal_evidence": empty_terminal,
        "trajectory_sft_eligible_under_empty_result_policy": not empty_terminal,
    }
