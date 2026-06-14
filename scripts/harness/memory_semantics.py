#!/usr/bin/env python3
"""Harness-authored memory grounding (V2a: scalar `derived_value` only).

THE TRUST BOUNDARY. The model never authors a memory value or its provenance; it only cites a
`source_step_id`. The harness independently (1) extracts the real scalar from that step's verified
output, (2) builds a structured `derivation` from the normalized operation graph, and (3) renders a
deterministic semantic key + content. SFT emitter and live rollout call THESE SAME functions, so
grounded memory means the same thing offline and online.

`history` is `{step_id -> {"tool", "arguments", "output", "references"}}` for already-executed,
successful steps (the emitter's running step list / the rollout tool history). `output` is the
step's tool_output dict (aggregate -> {"result_sample": [[v]], ...}; a table -> {"columns","rows",
"row_count",...}).

V2a accepts ONLY an unambiguous scalar source: a scalar tool result, or a one-row one-column table,
non-NULL. Everything else is rejected (never guessed) so a `derived_value` is always trustworthy.
"""
from __future__ import annotations

import re


class MemoryGroundingError(Exception):
    """The cited source cannot be grounded as a trustworthy scalar (rejected, never guessed)."""
    def __init__(self, reason: str, code: str = "non_scalar_source") -> None:
        super().__init__(reason)
        self.code = code            # machine-readable bucket for the manifest


def _ref_step_ids(step: dict) -> list[str]:
    return [r["step"] for r in step.get("references", []) if "step" in r]


def _source_table(step: dict, history: dict) -> str:
    """Nearest base source table feeding `step` (walk references back to a {'source': ...})."""
    seen, frontier = set(), [step]
    while frontier:
        s = frontier.pop()
        for r in s.get("references", []):
            if "source" in r:
                return r["source"]
            if "step" in r and r["step"] not in seen and r["step"] in history:
                seen.add(r["step"])
                frontier.append(history[r["step"]])
    return "the table"


def _ancestors(history: dict, step_id: str) -> list[str]:
    """`step_id` plus all step ancestors reachable through references, in step order."""
    seen, frontier = set(), [step_id]
    while frontier:
        x = frontier.pop()
        if x in seen or x not in history:
            continue
        seen.add(x)
        frontier.extend(_ref_step_ids(history[x]))
    return sorted(seen, key=lambda s: int(s.split("_")[1]) if s.split("_")[-1].isdigit() else 0)


def _render_filter(filter_step: dict) -> str:
    """Compact readable rendering of a condition_filter's predicate (for content only)."""
    def render(c):
        if isinstance(c, list):
            return " and ".join(render(x) for x in c)
        if not isinstance(c, dict):
            return str(c)
        for k in ("and", "or"):
            if k in c:
                return (" " + k + " ").join(render(x) for x in c[k])
        if "not" in c:
            return "not (" + render(c["not"]) + ")"
        col, op = c.get("column", "?"), c.get("op", "=")
        if "value_ref" in c:
            return f"{col} {op} <stored value>"
        if "in_table" in c:
            return f"{col} in <subset>"
        if "values" in c:
            return f"{col} in ({', '.join(map(str, c['values']))})"
        if "column_value" in c:
            return f"{col} {op} {c['column_value']}"
        return f"{col} {op} {c.get('value', '?')}"
    return render(filter_step.get("arguments", {}).get("conditions"))


def extract_scalar(history: dict, source_step_id: str):
    """Return the real scalar at `source_step_id`, or raise MemoryGroundingError. Strict: a scalar
    tool result or a 1x1 table, non-NULL. Rejects 0-row / multi-row / multi-column / NULL."""
    step = history.get(source_step_id)
    if step is None:
        raise MemoryGroundingError(f"source step {source_step_id!r} not in history", "missing_source")
    out = step.get("output", {})
    if "result_sample" in out:                       # scalar tool (aggregate)
        rs = out["result_sample"]
        if out.get("row_count", len(rs)) != 1 or len(rs) != 1 or len(rs[0]) != 1:
            raise MemoryGroundingError("scalar tool did not yield exactly one value", "non_scalar_source")
        value = rs[0][0]
    elif "rows" in out and "columns" in out:         # table source: must be exactly 1x1
        rows, cols = out["rows"], out["columns"]
        if out.get("row_count", len(rows)) != 1:
            raise MemoryGroundingError(f"table source has {out.get('row_count')} rows, not 1", "multi_row_source")
        if len(cols) != 1:
            raise MemoryGroundingError(f"table source has {len(cols)} columns, not 1", "multi_col_source")
        if len(rows) != 1:
            raise MemoryGroundingError("table source preview is not a single row", "non_scalar_source")
        value = rows[0][0]
    else:
        raise MemoryGroundingError("source output is not scalar-extractable", "non_scalar_source")
    if value is None:
        raise MemoryGroundingError("source scalar is NULL", "null_source")
    return value


def build_derivation(history: dict, source_step_id: str) -> dict:
    """Structured derivation classified from the operation graph (the source of truth for key/content
    and RL). Covers the common Spider scalar shapes; unknown graphs fall back to `operation_result`
    with a real selected column (never the opaque `value_subquery`)."""
    src = history[source_step_id]
    tool = src["tool"]
    args = src.get("arguments", {})
    parents = [history[s] for s in _ref_step_ids(src) if s in history]
    p = parents[0] if parents else None
    table = _source_table(src, history)
    supporting = _ancestors(history, source_step_id)

    if tool == "aggregate":
        op, col = args.get("op"), args.get("column")
        if p and p["tool"] == "condition_filter":
            return {"type": "filtered_aggregate", "op": op, "column": col,
                    "filter": _render_filter(p), "source_table": table, "supporting_step_ids": supporting}
        if p and p["tool"] == "derive_column":
            return {"type": "derived_aggregate", "op": op, "column": col,
                    "source_table": table, "supporting_step_ids": supporting}
        return {"type": "aggregate_stat", "op": op, "column": col,
                "source_table": table, "supporting_step_ids": supporting}

    # a single-cell lookup materialized as a 1x1 table (project / extreme_value_select chains)
    selected = _selected_column(src)
    if p and p["tool"] == "extreme_value_select":
        order = (p.get("arguments", {}).get("order_by") or ["?"])[0]
        direction = "min" if order.upper().endswith(" DESC") is False and "DESC" not in order.upper() else "max"
        direction = "max" if "DESC" in order.upper() else "min"
        selector = order.split()[0]
        gp = [history[s] for s in _ref_step_ids(p) if s in history]
        grouped = bool(gp and gp[0]["tool"] == "group_aggregate")
        return {"type": ("group_argmax" if grouped else "argmax_lookup") if direction == "max"
                else ("group_argmin" if grouped else "argmin_lookup"),
                "selected_column": selected, "selector": selector, "direction": direction,
                "source_table": table, "supporting_step_ids": supporting}
    if p and p["tool"] == "condition_filter":
        return {"type": "filtered_lookup", "selected_column": selected,
                "filter": _render_filter(p), "source_table": table, "supporting_step_ids": supporting}

    return {"type": "operation_result", "selected_column": selected, "source_table": table,
            "operations": [{"step_id": s, "tool": history[s]["tool"]} for s in supporting],
            "supporting_step_ids": supporting}


def _selected_column(step: dict) -> str:
    args = step.get("arguments", {})
    exprs = args.get("expressions") or args.get("return_columns")
    if exprs:
        return str(exprs[0]).split(" AS ")[-1].strip()
    return args.get("column", "value")


def build_memory_key(derivation: dict) -> str:
    """Deterministic semantic key from the derivation (identity is the memory_id, not this key)."""
    t = derivation.get("type")
    if t in ("aggregate_stat", "filtered_aggregate", "derived_aggregate"):
        base = f"{derivation.get('op')}_{derivation.get('column')}"
    elif t in ("argmax_lookup", "argmin_lookup", "group_argmax", "group_argmin"):
        base = f"{derivation.get('selected_column')}_at_{derivation.get('direction')}_{derivation.get('selector')}"
    elif t == "filtered_lookup":
        base = f"{derivation.get('selected_column')}_lookup"
    else:
        base = f"{derivation.get('selected_column', 'value')}_result"
    return re.sub(r"\W+", "_", base).strip("_").lower() or "scalar"


def render_memory_content(derivation: dict) -> str:
    """Deterministic NL rendering of `derivation` (a view of the structured object, not the truth)."""
    t = derivation.get("type")
    if t == "aggregate_stat":
        return f"{derivation['op']}({derivation['column']}) over {derivation['source_table']}"
    if t == "filtered_aggregate":
        return f"{derivation['op']}({derivation['column']}) over rows where {derivation['filter']}"
    if t == "derived_aggregate":
        return f"{derivation['op']}({derivation['column']}) over derived rows of {derivation['source_table']}"
    if t in ("argmax_lookup", "argmin_lookup", "group_argmax", "group_argmin"):
        scope = "each group of " if t.startswith("group_") else "the row with "
        return (f"{derivation['selected_column']} of {scope}{derivation['direction']} "
                f"{derivation['selector']} in {derivation['source_table']}")
    if t == "filtered_lookup":
        return f"{derivation['selected_column']} of the row where {derivation['filter']}"
    return f"{derivation.get('selected_column', 'value')} from {derivation['source_table']}"


def ground_derived_value(history: dict, source_step_id: str) -> dict:
    """Full grounding: extract scalar + build derivation + key + content. Raises on non-scalar."""
    value = extract_scalar(history, source_step_id)
    derivation = build_derivation(history, source_step_id)
    return {
        "memory_id": f"mem_{source_step_id}",
        "key": build_memory_key(derivation),
        "value": value,
        "authority": "harness_grounded",
        "content": render_memory_content(derivation),
        "source_step_id": source_step_id,
        "derivation": derivation,
    }


if __name__ == "__main__":  # unit demo: ground a filtered_aggregate from a synthetic history
    history = {
        "step_1": {"tool": "condition_filter",
                   "arguments": {"table": "weather", "conditions": {"column": "zip_code", "op": "=", "value": 94107}},
                   "output": {"columns": ["min_dew_point_f"], "rows": [[10], [5]], "row_count": 2},
                   "references": [{"source": "weather", "as": "table"}]},
        "step_2": {"tool": "aggregate",
                   "arguments": {"table": "filter_001", "column": "min_dew_point_f", "op": "min"},
                   "output": {"result_sample": [[5]], "row_count": 1},
                   "references": [{"step": "step_1", "as": "table"}]},
    }
    import json
    print(json.dumps(ground_derived_value(history, "step_2"), ensure_ascii=False, indent=2))
    for bad in ("step_1",):   # multi-row source must be rejected
        try:
            extract_scalar(history, bad)
        except MemoryGroundingError as e:
            print(f"rejected {bad}: [{e.code}] {e}")
