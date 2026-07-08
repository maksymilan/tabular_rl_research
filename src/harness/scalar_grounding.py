#!/usr/bin/env python3
"""Harness scalar-reference grounding (V2b; supersedes the V2a `memory_semantics` scalar memory).

THE TRUST BOUNDARY. A predicate's `value_ref` names the trajectory STEP that produced a scalar; the
model never writes the value itself, and there is no separate `add_to_memory` step. The harness
extracts the real scalar from that step's verified output, with strict validation, so a `value_ref`
is always trustworthy and forge-proof. The SFT emitter and the live rollout both call
`extract_scalar` on the same `history`, so a `value_ref` resolves to the same value offline and
online.

`history` is `{step_id -> {"tool", "arguments", "output", "references"}}` for already-executed,
successful steps (the emitter's running step list / the rollout tool history). `output` is the
step's tool_output dict (legacy `aggregate -> {"result_sample": [[v]], ...}`; a table -> a
metadata handle that carries `rows`/`columns` only when it is a 1x1 scalar-shaped result).

Only an unambiguous scalar source is accepted: a scalar tool result, or a one-row one-column table,
non-NULL. Everything else is rejected (never guessed), so a `value_ref` can never resolve to an
ambiguous or fabricated value.
"""
from __future__ import annotations


class ScalarGroundingError(Exception):
    """The cited source step cannot be grounded as a trustworthy scalar (rejected, never guessed)."""
    def __init__(self, reason: str, code: str = "non_scalar_source") -> None:
        super().__init__(reason)
        self.code = code            # machine-readable bucket for the manifest / RL reject accounting


def extract_scalar(history: dict, source_step_id: str):
    """Return the real scalar produced at `source_step_id`, or raise ScalarGroundingError. Strict: a
    scalar tool result or a 1x1 table, non-NULL. Rejects missing / 0-row / multi-row / multi-column /
    NULL so a `value_ref` can never resolve to a guessed or ambiguous value."""
    step = history.get(source_step_id)
    if step is None:
        raise ScalarGroundingError(f"source step {source_step_id!r} not in history", "missing_source")
    out = step.get("output", {})
    if "result_sample" in out:                       # scalar tool (aggregate)
        rs = out["result_sample"]
        if out.get("row_count", len(rs)) != 1 or len(rs) != 1 or len(rs[0]) != 1:
            raise ScalarGroundingError("scalar tool did not yield exactly one value", "non_scalar_source")
        value = rs[0][0]
    elif "rows" in out and "columns" in out:         # table source: must be exactly 1x1
        rows, cols = out["rows"], out["columns"]
        if out.get("row_count", len(rows)) != 1:
            raise ScalarGroundingError(f"table source has {out.get('row_count')} rows, not 1", "multi_row_source")
        if len(cols) != 1:
            raise ScalarGroundingError(f"table source has {len(cols)} columns, not 1", "multi_col_source")
        if len(rows) != 1:
            raise ScalarGroundingError("table source preview is not a single row", "non_scalar_source")
        value = rows[0][0]
    else:
        raise ScalarGroundingError("source output is not scalar-extractable", "non_scalar_source")
    if value is None:
        raise ScalarGroundingError("source scalar is NULL", "null_source")
    return value


def ground_scalar_reference(history: dict, source_step_id: str):
    """Neutral public API: the validated scalar that a `value_ref` resolves to. Raises
    ScalarGroundingError on any non-scalar / NULL / missing source. This is the single place a
    `value_ref`'s value is produced (offline emitter + online rollout both call it)."""
    return extract_scalar(history, source_step_id)


if __name__ == "__main__":   # unit demo: ground an aggregate scalar; reject a multi-row table source
    history = {
        "step_1": {"tool": "condition_filter",
                   "arguments": {"table": "weather", "conditions": {"column": "zip_code", "op": "=", "value": 94107}},
                   "output": {"columns": ["min_dew_point_f"], "rows": [[10], [5]], "row_count": 2},
                   "references": [{"type": "data", "source": "weather", "role": "table"}]},
        "step_2": {"tool": "aggregate",
                   "arguments": {"table": "filter_001", "column": "min_dew_point_f", "op": "min"},
                   "output": {"result_sample": [[5]], "row_count": 1},
                   "references": [{"type": "data", "step": "step_1", "role": "table"}]},
    }
    print("step_2 scalar:", ground_scalar_reference(history, "step_2"))
    for bad in ("step_1",):    # multi-row source must be rejected
        try:
            extract_scalar(history, bad)
        except ScalarGroundingError as e:
            print(f"rejected {bad}: [{e.code}] {e}")
