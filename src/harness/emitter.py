#!/usr/bin/env python3
"""Trajectory emitter + validator (final_tool_design.md).

`emit(harness, question, gold_sql)` compiles the SQL to a Plan, executes it while capturing each
step's real tool output, verifies the final result against the gold SQL, and packages a training
trajectory (ReAct steps + a terminal `answer_from_context`). `validate(traj)` checks structural
legality (step 4 of the build): tool names, step shape, terminal answer, citation integrity, and
that the trajectory was execution-verified.

V2a — provenance + grounded memory: every step carries harness-authored `references` (the trajectory
step_ids / source-table names it consumed) and a `produces` descriptor. `add_to_memory` is grounded
by `memory_semantics` (the model emits only `{type, source_step_id}`; the harness owns
value/key/derivation). `backward_slice(traj)` walks `references` from the answer to recover exactly the
steps it depends on. `validate()` = legality + reference-integrity gate. The model never emits
`references`/`produces` — they are harness-derived sidecars.

V2-ctx — bounded context (so large DBs fit): the opening overview is a lazy CATALOG (table names +
row_counts + FK relations, NO columns); table-producing steps emit metadata-only handles (no inlined
rows; a 1x1 scalar result keeps its one cell); the emitter injects read-only resident perception —
`describe_table` the touched tables, `inspect_column` before a string-literal filter, `read_subtable`
the evidence before answering.
"""
from __future__ import annotations

from compiler import Compiler
from memory_semantics import ground_derived_value
from plan import TABLE_REF_ARGS, resolve_cond

SCHEMA_VERSION = "v2-ctx"  # + lazy catalog, metadata-only outputs, injected resident perception

# every tool the model may call (table-producing + reading/scalar + perception + memory/terminal)
TOOLS = set(TABLE_REF_ARGS) | {
    "aggregate", "extreme_value_select", "read_subtable", "describe_table",
    "inspect_column", "add_to_memory", "refine_memory", "answer_from_context",
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
    "add_to_memory": "Register the scalar computed by the cited source step as a value a later step "
                     "can reference (the harness grounds the actual number and its derivation).",
    "describe_table": "Read the columns and keys of the tables this question needs before operating.",
    "inspect_column": "Check the actual values in this column so the filter literal is grounded.",
    "read_subtable": "Read the evidence rows so the answer is grounded in real data.",
}

# how many evidence rows to read before answering (bounded — replaces per-step row inlining)
READ_EVIDENCE_LIMIT = 20


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


def _cond_refs(cond):
    """Yield ('value_ref', key) / ('in_table', plan_step_id) found anywhere in a condition tree."""
    if isinstance(cond, list):
        for c in cond:
            yield from _cond_refs(c)
        return
    if not isinstance(cond, dict):
        return
    for k in ("and", "or"):
        if k in cond:
            for c in cond[k]:
                yield from _cond_refs(c)
            return
    if "not" in cond:
        yield from _cond_refs(cond["not"])
        return
    if "value_ref" in cond:
        yield ("value_ref", cond["value_ref"])
    if "in_table" in cond:
        yield ("in_table", cond["in_table"])


def _references(step, planid_to_stepid: dict) -> list[dict]:
    """Explicit consumption edges (in trajectory step_ids / source-table names) of one plan step."""
    refs: list[dict] = []
    for key in TABLE_REF_ARGS.get(step.tool, []):
        raw = step.args.get(key)
        if raw in planid_to_stepid:
            refs.append({"step": planid_to_stepid[raw], "as": key})
        elif raw is not None:
            refs.append({"source": raw, "as": key})           # a base (source) table
    if step.tool == "condition_filter":
        for kind, val in _cond_refs(step.args.get("conditions")):
            if val in planid_to_stepid:                        # value_ref -> the add_to_memory step;
                refs.append({"step": planid_to_stepid[val], "via": kind})  # in_table -> the subset step
    if step.tool == "add_to_memory":
        src = step.args.get("source")
        if src in planid_to_stepid:
            refs.append({"step": planid_to_stepid[src], "via": "source"})
    return refs


def _rewrite_value_ref(cond, memid_map: dict):
    """Display-side: rewrite a predicate's `value_ref` from the internal add_to_memory plan id to the
    model-facing stable `memory_id` (execution still threads by the plan id)."""
    if isinstance(cond, list):
        return [_rewrite_value_ref(c, memid_map) for c in cond]
    if not isinstance(cond, dict):
        return cond
    for k in ("and", "or"):
        if k in cond:
            return {k: [_rewrite_value_ref(x, memid_map) for x in cond[k]]}
    if "not" in cond:
        return {"not": _rewrite_value_ref(cond["not"], memid_map)}
    out = dict(cond)
    if out.get("value_ref") in memid_map:
        out["value_ref"] = memid_map[out["value_ref"]]
    return out


def _source_tables(plan) -> list[str]:
    """Distinct base (source) tables the plan touches, in first-seen order — for the injected
    `describe_table` step (acquire only the RELEVANT schemas, not the whole DB)."""
    step_ids = {s.id for s in plan}
    seen, out = set(), []
    for s in plan:
        for key in TABLE_REF_ARGS.get(s.tool, []):
            v = s.args.get(key)
            if isinstance(v, str) and v not in step_ids and v not in seen:
                seen.add(v)
                out.append(v)
    return out


def _string_literal_columns(cond):
    """Columns compared to a STRING literal (=/contains/like) — where grounding via inspect_column
    matters (conservative injection: numeric filters are not injected)."""
    if isinstance(cond, list):
        for c in cond:
            yield from _string_literal_columns(c)
        return
    if not isinstance(cond, dict):
        return
    for k in ("and", "or"):
        if k in cond:
            for c in cond[k]:
                yield from _string_literal_columns(c)
            return
    if "not" in cond:
        yield from _string_literal_columns(cond["not"])
        return
    if cond.get("op") in ("=", "contains", "like") and isinstance(cond.get("value"), str):
        yield cond["column"]


def emit(h, question: str, gold_sql: str, *, dataset: str = "", db_id: str = "",
         trajectory_id: str = "traj") -> dict:
    plan = Compiler(h.schema()).compile(gold_sql)
    id_to_table: dict[str, str] = {}        # plan step id -> real harness view name (for EXECUTION)
    planid_to_stepid: dict[str, str] = {}   # plan step id -> trajectory step id (for REFERENCES)
    memid_map: dict[str, str] = {}          # add_to_memory plan id -> memory_id (value_ref rewrite)
    memory_ids: list[str] = []              # all grounded memory_ids (for answer.supporting_memory_ids)
    history: dict[str, dict] = {}           # trajectory step_id -> {tool, arguments, output, references}
    values: dict = {}
    steps: list[dict] = []
    final = None
    last_result_stepid: str | None = None   # the step whose output IS the running result (table/scalar)

    def emit_step(tool, display, references, produces, tool_output, think, *, plan_id=None) -> str:
        sid = f"step_{len(steps) + 1}"
        if plan_id is not None:
            planid_to_stepid[plan_id] = sid
        history[sid] = {"tool": tool, "arguments": display, "output": tool_output, "references": references}
        steps.append({"step_id": sid, "think": think,
                      "tool_call": {"tool": tool, "arguments": display},
                      "references": references, "produces": produces, "tool_output": tool_output})
        return sid

    # --- inject: describe ONLY the source tables this question needs (acquire relevant schemas) ---
    src_tables = _source_tables(plan)
    if src_tables:
        try:
            emit_step("describe_table", {"tables": src_tables},
                      [{"source": t, "as": "table"} for t in src_tables],
                      {"kind": "schema"}, h.describe_table(src_tables), THINK["describe_table"])
        except Exception:
            pass

    # --- main plan loop: metadata-only table outputs; inspect_column injected before literal filters ---
    for step in plan:
        references = _references(step, planid_to_stepid)
        args = dict(step.args)
        for key in TABLE_REF_ARGS.get(step.tool, []):
            if args.get(key) in id_to_table:
                args[key] = id_to_table[args[key]]

        if step.tool == "add_to_memory":
            # V2a trust boundary: the model only cites source_step_id; the harness grounds the value.
            source_step_id = planid_to_stepid[step.args["source"]]
            grounded = ground_derived_value(history, source_step_id)
            memid_map[step.id] = grounded["memory_id"]
            memory_ids.append(grounded["memory_id"])
            values[step.id] = grounded["value"]
            emit_step("add_to_memory", {"type": "derived_value", "source_step_id": source_step_id},
                      references, {"kind": "memory", "memory_id": grounded["memory_id"], "key": grounded["key"]},
                      {"memory": grounded}, THINK["add_to_memory"], plan_id=step.id)
            continue

        if step.tool == "condition_filter":
            tbl = args["table"]
            tbl_ref = ([{"step": planid_to_stepid[step.args["table"]]}]
                       if step.args["table"] in planid_to_stepid else [{"source": step.args["table"]}])
            for col in dict.fromkeys(_string_literal_columns(step.args.get("conditions"))):
                try:                                      # best-effort grounding; never drop the trajectory
                    ic = h.inspect_column(tbl, col)
                except Exception:
                    continue
                emit_step("inspect_column", {"table": tbl, "column": col}, tbl_ref,
                          {"kind": "column_domain"}, ic, THINK["inspect_column"])
            disp_conds = _rewrite_value_ref(resolve_cond(args.get("conditions"), id_to_table), memid_map)
            display = {**args, "conditions": disp_conds}
            exec_args = {**args, "conditions": resolve_cond(args.get("conditions"), id_to_table, values)}
        else:
            display = exec_args = args

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

    # --- inject: read the evidence rows once before answering (the ONE place rows enter context) ---
    if answer_table is not None and last_result_stepid is not None:
        try:
            ev = h.read_subtable(answer_table, limit=READ_EVIDENCE_LIMIT)
            emit_step("read_subtable", {"table": answer_table, "limit": READ_EVIDENCE_LIMIT},
                      [{"step": last_result_stepid}], {"kind": "rows"},
                      {"table": answer_table, "rows": [list(r) for r in ev], "row_count": len(result)},
                      THINK["read_subtable"])
        except Exception:
            pass

    answer_refs = [{"step": last_result_stepid}] if last_result_stepid else []
    emit_step("answer_from_context",
              {"answer": result[:50], "evidence": {"table": answer_table},
               "supporting_memory_ids": memory_ids, "reason": "Derived by the verified tool chain."},
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


def backward_slice(traj: dict) -> set[str]:
    """Walk `references` backward from the terminal answer to recover the set of step_ids the answer
    depends on (the provenance slice). Steps NOT in the slice are provably off the answer's path."""
    by_id = {s["step_id"]: s for s in traj.get("steps", [])}
    if not traj.get("steps"):
        return set()
    frontier = [r["step"] for r in traj["steps"][-1].get("references", []) if "step" in r]
    seen: set[str] = set()
    while frontier:
        x = frontier.pop()
        if x in seen or x not in by_id:
            continue
        seen.add(x)
        for r in by_id[x].get("references", []):
            if "step" in r:
                frontier.append(r["step"])
    return seen


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
    memkeys: set[str] = set()
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
        if tc.get("tool") == "add_to_memory":
            memkeys.add(s.get("produces", {}).get("memory_id"))   # harness-assigned identity
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
        for mk in ans.get("supporting_memory_ids", []) or []:
            if mk not in memkeys:
                errs.append(f"answer cites unknown memory key {mk!r}")
        # the answer must be backward-reachable to at least one step that produced evidence
        if len(steps) > 1 and not backward_slice(traj):
            errs.append("answer has no resolvable provenance (empty backward slice)")
    return errs
