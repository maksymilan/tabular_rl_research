#!/usr/bin/env python3
"""Shared provenance: harness-authored reference edges (V2b unified schema).

Every trajectory step carries a `references` array of harness-derived edges — the model never
authors them. Both the offline emitter and the online rollout build edges through `build_references`,
over the SAME model-facing argument shape (a table ref is a real handle or a source name; a
`value_ref` is the producing step_id), so SFT and eval can never drift.

Edge schema (one dict per edge):
  {"type": "data"|"value"|"grounding", "step"|"source": id, "role": ..., "target": {...}}
  - data:   a table input        role=table|in_table   target={"handle":h} | {"table":t}
  - value:  a scalar threshold   role=value_ref        target={"column":c}
  - grounding (V2c, not emitted this round): a perception step grounding a column/literal/evidence.
`target` is a structured object so the process reward reads fields directly and never parses a
string. (Cross-table column lineage — rendering `T2__x` back to its source `table.column` — lands
with the grounding edges in V2c; this round a `value` edge's column is just the prefix-stripped name.)

`resolve_step(ref) -> step_id | None`: maps a model-facing reference (a produced table handle, or an
already-known step_id for a value_ref) to its trajectory step_id; returns None for a base source
table. The emitter and rollout each supply their own resolver over the same arg shape.
"""
from __future__ import annotations

from plan import TABLE_REF_ARGS


def _base_col(col):
    """Strip a join prefix: 'T2__Time_of_day' -> 'Time_of_day' (full lineage回溯 is a V2c concern)."""
    return col.split("__", 1)[-1] if isinstance(col, str) and "__" in col else col


def _cond_value_refs(cond):
    """Yield (kind, ref, column) for each value_ref / in_table found anywhere in a condition tree."""
    if isinstance(cond, list):
        for c in cond:
            yield from _cond_value_refs(c)
        return
    if not isinstance(cond, dict):
        return
    for k in ("and", "or"):
        if k in cond:
            for c in cond[k]:
                yield from _cond_value_refs(c)
            return
    if "not" in cond:
        yield from _cond_value_refs(cond["not"])
        return
    col = cond.get("column")
    if "value_ref" in cond:
        yield ("value_ref", cond["value_ref"], col)
    if "in_table" in cond:
        yield ("in_table", cond["in_table"], col)


def build_references(tool: str, args: dict, resolve_step) -> list[dict]:
    """Harness-authored consumption edges for one action, over the model-facing arg shape. The
    emitter and rollout call this with their own `resolve_step`, so the edge schema is built in ONE
    place and can never drift between SFT and eval."""
    refs: list[dict] = []
    for key in TABLE_REF_ARGS.get(tool, []):
        v = args.get(key)
        if v is None:
            continue
        # An N-way join carries its inputs as a list under `tables`; a data edge per input.
        for item in (v if isinstance(v, list) else [v]):
            sid = resolve_step(item)
            if sid:
                refs.append({"type": "data", "step": sid, "role": "table", "target": {"handle": item}})
            else:
                refs.append({"type": "data", "source": item, "role": "table", "target": {"table": item}})
    if tool == "condition_filter":
        for kind, ref, col in _cond_value_refs(args.get("conditions")):
            sid = resolve_step(ref)
            if not sid:
                continue
            if kind == "value_ref":
                refs.append({"type": "value", "step": sid, "role": "value_ref",
                             "target": {"column": _base_col(col)}})
            else:  # an IN-subquery's set membership is a data input
                refs.append({"type": "data", "step": sid, "role": "in_table",
                             "target": {"handle": ref}})
    return refs


def backward_slice(traj: dict, reference_type=("data", "value")) -> set[str]:
    """Walk `references` backward from the terminal answer to the set of step_ids it depends on.
    `reference_type` selects which edge types count (default = the answer-computation slice: data +
    value). Grounding edges are excluded by default so the data slice keeps its 'which computation
    steps produced the answer' meaning; include 'grounding' for the perception-aware slice."""
    types = {reference_type} if isinstance(reference_type, str) else set(reference_type)
    by_id = {s["step_id"]: s for s in traj.get("steps", [])}
    if not traj.get("steps"):
        return set()

    def edges(step):
        return [r["step"] for r in step.get("references", [])
                if "step" in r and r.get("type", "data") in types]

    frontier = edges(traj["steps"][-1])
    seen: set[str] = set()
    while frontier:
        x = frontier.pop()
        if x in seen or x not in by_id:
            continue
        seen.add(x)
        frontier.extend(edges(by_id[x]))
    return seen


if __name__ == "__main__":   # unit demo: data + value edges over the model-facing arg shape
    history_ids = {"step_1", "step_2"}
    handle_to_step = {"filter_002": "step_2"}

    def resolve(ref):
        if ref in handle_to_step:
            return handle_to_step[ref]
        if ref in history_ids:            # already a step_id (a value_ref)
            return ref
        return None

    refs = build_references("condition_filter", {
        "table": "filter_002",
        "conditions": {"column": "T2__age", "op": ">", "value_ref": "step_1"},
    }, resolve)
    import json
    print(json.dumps(refs, ensure_ascii=False, indent=2))
