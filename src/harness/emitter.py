#!/usr/bin/env python3
"""Trajectory emitter + validator (final_tool_design.md).

`emit(harness, question, gold_sql)` compiles the SQL to a Plan, executes it while capturing each
step's real tool output, verifies the final result against the gold SQL, and packages a training
trajectory (ReAct steps + a terminal `answer_from_context`). `validate(traj)` checks structural
legality (step 4 of the build): tool names, step shape, terminal answer, citation integrity, and
that the trajectory was execution-verified.

V2b — unified provenance, no memory step: every step carries harness-authored `references` (typed
data/value edges with a structured `target`, built by `provenance.build_references`) and a `produces`
descriptor. A predicate's `value_ref` cites the producing step directly (there is no `add_to_memory`).
`backward_slice(traj)` walks the data+value references from the answer to recover exactly the steps it
depends on. `validate()` = legality + reference-integrity gate. The model never emits
`references`/`produces` — they are harness-derived sidecars.

V2c-plan — bounded context (so large DBs fit): the opening overview is a lazy CATALOG (table names +
row_counts + FK relations, NO columns); table-producing steps emit metadata-only handles (no inlined
rows; a 1x1 scalar result keeps its one cell). The raw emitter produces the verified relational
backbone only. Perception steps (`describe_table`, `inspect_column`, `read_subtable`) are inserted by
the external-model enrichment pass so they can follow the actual reasoning context instead of a
mechanical template.
"""
from __future__ import annotations

from compiler import Compiler
from environment_state import EnvironmentState
from plan import TABLE_REF_ARGS, resolve_cond
from provenance import backward_slice, build_references

SCHEMA_VERSION = "v3"  # V2b memory removal + unified references; lazy catalog, metadata-only outputs

# every tool the model may call (table-producing + reading/scalar + perception + memory/terminal)
TOOLS = set(TABLE_REF_ARGS) | {
    "aggregate", "extreme_value_select", "read_subtable", "describe_table",
    "inspect_column", "plan", "answer_from_context",
}

THINK = {
    "condition_filter": "Filter to the rows that satisfy the WHERE conditions.",
    "join_tables": "Join the two tables on their key to combine the needed columns.",
    "group_aggregate": "Group the rows and compute the requested aggregates.",
    "derive_column": "Derive the computed column needed by the question.",
    "extreme_value_select": "Order the rows and keep the top ones (the extreme values).",
    "project": "Project the output columns the question asks for.",
    "set_op": "Combine the two row sets with the set operation.",
    "aggregate": "Compute the scalar aggregate that answers the question.",
    "plan": "Create or update the task plan so the next tool calls follow explicit subgoals.",
}


def _catalog(h) -> dict:
    """Lazy opening overview: table names + row counts + FK relations ONLY (no columns). The model
    acquires a table's schema on demand via `describe_table`, so the opening context is bounded by
    table COUNT, not by total column count — the property that makes huge DBs fit."""
    tables = [{"table_name": name,
               "num_rows": h.conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]}
              for (name,) in h.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    relations = []
    for (name,) in h.conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        for r in h.conn.execute(f'PRAGMA foreign_key_list("{name}")'):
            relations.append({"from": f"{name}.{r[3]}", "to": f"{r[2]}.{r[4]}"})
    return {"tables": tables, "relations": relations}


def _norm(rows):
    return sorted(repr(tuple(r)) for r in rows)


def _rewrite_value_ref(cond, planid_to_stepid: dict):
    """Display-side: rewrite a predicate's `value_ref` from the internal plan-step id to the
    model-facing trajectory step_id (execution still threads by the plan id)."""
    if isinstance(cond, list):
        return [_rewrite_value_ref(c, planid_to_stepid) for c in cond]
    if not isinstance(cond, dict):
        return cond
    for k in ("and", "or"):
        if k in cond:
            return {k: [_rewrite_value_ref(x, planid_to_stepid) for x in cond[k]]}
    if "not" in cond:
        return {"not": _rewrite_value_ref(cond["not"], planid_to_stepid)}
    out = dict(cond)
    if out.get("value_ref") in planid_to_stepid:
        out["value_ref"] = planid_to_stepid[out["value_ref"]]
    return out



def emit(h, question: str, gold_sql: str, *, dataset: str = "", db_id: str = "",
         trajectory_id: str = "traj") -> dict:
    plan = Compiler(h.schema()).compile(gold_sql)
    id_to_table: dict[str, str] = {}        # plan step id -> real harness view name (for EXECUTION)
    planid_to_stepid: dict[str, str] = {}   # plan step id -> trajectory step id (value_ref rewrite)
    handle_to_stepid: dict[str, str] = {}   # real view name -> trajectory step id (for REFERENCES)
    history: dict[str, dict] = {}           # trajectory step_id -> {tool, arguments, output, references}
    values: dict = {}
    steps: list[dict] = []
    final = None
    last_result_stepid: str | None = None   # the step whose output IS the running result (table/scalar)
    env_state = EnvironmentState(_catalog(h))

    def emit_step(tool, display, references, produces, tool_output, think, *, plan_id=None) -> str:
        sid = f"step_{len(steps) + 1}"
        if plan_id is not None:
            planid_to_stepid[plan_id] = sid
        history[sid] = {"tool": tool, "arguments": display, "output": tool_output, "references": references}
        if tool == "plan":
            env_state.apply_plan_ops(display.get("ops"), sid)
        elif tool != "answer_from_context":
            env_state.apply_tool_result(tool, display, tool_output, sid)
        steps.append({"step_id": sid, "think": think,
                      "tool_call": {"tool": tool, "arguments": display},
                      "references": references, "produces": produces, "tool_output": tool_output,
                      "environment_state": env_state.snapshot()})
        return sid

    def resolve_step(ref):
        """Map a model-facing reference (a produced view name, or a value_ref's step_id) to its
        trajectory step_id; None for a base source table. Same arg shape the rollout resolver uses."""
        if ref in handle_to_stepid:
            return handle_to_stepid[ref]
        if ref in history:                  # already a trajectory step_id (a value_ref)
            return ref
        return None

    # --- main plan loop: metadata-only table outputs; perception is handled by enrichment ---
    for step in plan:
        args = dict(step.args)
        for key in TABLE_REF_ARGS.get(step.tool, []):
            if args.get(key) in id_to_table:
                args[key] = id_to_table[args[key]]   # display/exec args: table refs are real view names

        if step.tool == "condition_filter":
            disp_conds = _rewrite_value_ref(resolve_cond(args.get("conditions"), id_to_table), planid_to_stepid)
            display = {**args, "conditions": disp_conds}
            exec_args = {**args, "conditions": resolve_cond(args.get("conditions"), id_to_table, values)}
        else:
            display = exec_args = args

        references = build_references(step.tool, display, resolve_step)
        out = getattr(h, step.tool)(**exec_args)
        if isinstance(out, dict) and "table_name" in out:
            id_to_table[step.id] = out["table_name"]
            final = ("table", out["table_name"])
            tool_output = {"table": out["table_name"], "kind": out["kind"],     # METADATA ONLY — no rows
                           "columns": out["columns"], "row_count": out["row_count"]}
            if out["row_count"] == 1 and len(out["columns"]) == 1:    # scalar-shaped result: keep the 1 cell
                tool_output["rows"] = [list(r) for r in h.rows(out["table_name"])]
            last_result_stepid = emit_step(step.tool, display, references,
                                           {"kind": "table", "handle": out["table_name"]},
                                           tool_output, THINK.get(step.tool, ""), plan_id=step.id)
            handle_to_stepid[out["table_name"]] = last_result_stepid
        else:
            if step.tool == "aggregate":
                values[step.id] = out
            final = ("value", out)
            rows = out if isinstance(out, list) else [(out,)]
            tool_output = {"result_sample": [list(r) if isinstance(r, tuple) else r for r in rows[:5]],
                           "row_count": len(rows)}
            last_result_stepid = emit_step(step.tool, display, references, {"kind": "scalar"},
                                           tool_output, THINK.get(step.tool, ""), plan_id=step.id)

    kind, payload = final
    result = h.rows(payload) if kind == "table" else (payload if isinstance(payload, list) else [(payload,)])
    verified = _norm(result) == _norm(h.gold(gold_sql))
    answer_table = payload if kind == "table" else None

    answer_refs = ([{"type": "data", "step": last_result_stepid, "role": "table",
                     "target": {"handle": answer_table}}] if last_result_stepid else [])
    emit_step("answer_from_context",
              {"answer": result[:50], "evidence": {"table": answer_table},
               "reason": "Derived by the verified tool chain."},
              answer_refs, {"kind": "answer"}, {"final_answer": result[:50]},
              "Answer the question from the final evidence rows.")

    return {
        "trajectory_id": trajectory_id,
        "schema_version": SCHEMA_VERSION,
        "source": {"dataset": dataset, "db_id": db_id, "gold_sql": gold_sql},
        "question": question,
        "label_status": "verified" if verified else "mismatch",
        "initial_state": {"dataset_overview": _catalog(h)},
        "steps": steps,
        "final_answer": result[:50],
    }


def validate(traj: dict) -> list[str]:
    """Structural + reference-integrity legality check. Returns a list of problems ([] = legal)."""
    errs: list[str] = []
    for k in ("trajectory_id", "question", "initial_state", "steps", "label_status", "final_answer"):
        if k not in traj:
            errs.append(f"missing top-level field: {k}")
    if traj.get("label_status") != "verified":
        errs.append(f"not execution-verified (label_status={traj.get('label_status')!r})")

    steps = traj.get("steps", [])
    if not steps:
        errs.append("no steps")
    sources = {t["table_name"] for t in
               traj.get("initial_state", {}).get("dataset_overview", {}).get("tables", [])}
    sources_lc = {x.lower() for x in sources}   # SQL identifiers are case-insensitive
    created = set()
    seen_step_ids: set[str] = set()
    for s in steps:
        sid = s.get("step_id", "?")
        for k in ("step_id", "tool_call", "tool_output", "references", "produces"):
            if k not in s:
                errs.append(f"{sid}: missing {k}")
        tc = s.get("tool_call", {})
        if tc.get("tool") not in TOOLS:
            errs.append(f"{sid}: unknown tool {tc.get('tool')!r}")
        if "arguments" not in tc:
            errs.append(f"{sid}: tool_call missing arguments")
        # reference integrity: every {"step": X} must point to an ALREADY-SEEN (earlier) step;
        # every {"source": T} must be a real source table.
        for r in s.get("references", []):
            if "step" in r and r["step"] not in seen_step_ids:
                errs.append(f"{sid}: references unknown/forward step {r['step']!r}")
            if "source" in r and r["source"].lower() not in sources_lc:
                errs.append(f"{sid}: references unknown source table {r['source']!r}")
        tbl = s.get("tool_output", {}).get("table")
        if tbl:
            created.add(tbl)
        seen_step_ids.add(sid)

    if steps and steps[-1]["tool_call"]["tool"] != "answer_from_context":
        errs.append("final step must be answer_from_context")

    if steps:
        ans = steps[-1]["tool_call"]["arguments"]
        ev = (ans.get("evidence") or {}).get("table")
        if ev is not None and ev not in created | sources:
            errs.append(f"answer cites unknown table {ev!r}")
        # the answer must be backward-reachable to at least one step that produced evidence
        if len(steps) > 1 and not backward_slice(traj):
            errs.append("answer has no resolvable provenance (empty backward slice)")
    return errs
