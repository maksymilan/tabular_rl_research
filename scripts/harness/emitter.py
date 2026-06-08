#!/usr/bin/env python3
"""Trajectory emitter + validator (final_tool_design.md).

`emit(harness, question, gold_sql)` compiles the SQL to a Plan, executes it while capturing each
step's real tool output, verifies the final result against the gold SQL, and packages a training
trajectory (ReAct steps + a terminal `answer_from_context`). `validate(traj)` checks structural
legality (step 4 of the build): tool names, step shape, terminal answer, citation integrity, and
that the trajectory was execution-verified.
"""
from __future__ import annotations

from compiler import Compiler
from plan import TABLE_REF_ARGS

# every tool the model may call (table-producing + reading/scalar + memory/terminal)
TOOLS = set(TABLE_REF_ARGS) | {
    "aggregate", "extreme_value_select", "read_subtable",
    "inspect_column", "add_to_memory", "refine_memory", "answer_from_context",
}

THINK = {
    "condition_filter": "Filter to the rows that satisfy the WHERE conditions.",
    "join_tables": "Join the two tables on their key to combine the needed columns.",
    "group_aggregate": "Group the rows and compute the requested aggregates.",
    "derive_column": "Derive the computed column needed by the question.",
    "order_limit": "Order the rows and keep the requested top ones.",
    "project": "Project the output columns the question asks for.",
    "set_op": "Combine the two row sets with the set operation.",
    "aggregate": "Compute the scalar aggregate that answers the question.",
}


def _overview(h) -> dict:
    tables = []
    for (name,) in h.conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        cols = [{"name": r[1], "type": (r[2] or "text").lower()}
                for r in h.conn.execute(f'PRAGMA table_info("{name}")')]
        n = h.conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        tables.append({"table_name": name, "num_rows": n, "columns": cols})
    return {"tables": tables}


def _norm(rows):
    return sorted(repr(tuple(r)) for r in rows)


def emit(h, question: str, gold_sql: str, *, dataset: str = "", db_id: str = "",
         trajectory_id: str = "traj") -> dict:
    plan = Compiler(h.schema()).compile(gold_sql)
    id_to_table: dict[str, str] = {}
    steps: list[dict] = []
    final = None

    for i, step in enumerate(plan, 1):
        args = dict(step.args)
        for key in TABLE_REF_ARGS.get(step.tool, []):
            if args.get(key) in id_to_table:
                args[key] = id_to_table[args[key]]
        out = getattr(h, step.tool)(**args)
        if isinstance(out, dict) and "table_name" in out:
            id_to_table[step.id] = out["table_name"]
            final = ("table", out["table_name"])
            tool_output = {"created_table": out}
        else:
            final = ("value", out)
            rows = out if isinstance(out, list) else [(out,)]
            tool_output = {"result_sample": rows[:5], "row_count": len(rows)}
        steps.append({
            "step_id": f"step_{i}",
            "think": THINK.get(step.tool, ""),
            "tool_call": {"tool": step.tool, "arguments": args},
            "tool_output": tool_output,
        })

    kind, payload = final
    result = h.rows(payload) if kind == "table" else (payload if isinstance(payload, list) else [(payload,)])
    verified = _norm(result) == _norm(h.gold(gold_sql))
    answer_table = payload if kind == "table" else None

    steps.append({
        "step_id": f"step_{len(steps) + 1}",
        "think": "Answer the question from the final evidence rows.",
        "tool_call": {"tool": "answer_from_context", "arguments": {
            "answer": result[:50],
            "evidence": {"table": answer_table},
            "reason": "Derived by the verified tool chain.",
        }},
        "tool_output": {"final_answer": result[:50]},
    })

    return {
        "trajectory_id": trajectory_id,
        "source": {"dataset": dataset, "db_id": db_id, "gold_sql": gold_sql},
        "question": question,
        "label_status": "verified" if verified else "mismatch",
        "initial_state": {"dataset_overview": _overview(h)},
        "steps": steps,
        "final_answer": result[:50],
    }


def validate(traj: dict) -> list[str]:
    """Structural legality check. Returns a list of problems ([] = legal)."""
    errs: list[str] = []
    for k in ("trajectory_id", "question", "initial_state", "steps", "label_status", "final_answer"):
        if k not in traj:
            errs.append(f"missing top-level field: {k}")
    if traj.get("label_status") != "verified":
        errs.append(f"not execution-verified (label_status={traj.get('label_status')!r})")

    steps = traj.get("steps", [])
    if not steps:
        errs.append("no steps")
    created = set()
    for s in steps:
        for k in ("step_id", "tool_call", "tool_output"):
            if k not in s:
                errs.append(f"{s.get('step_id', '?')}: missing {k}")
        tc = s.get("tool_call", {})
        if tc.get("tool") not in TOOLS:
            errs.append(f"{s.get('step_id', '?')}: unknown tool {tc.get('tool')!r}")
        if "arguments" not in tc:
            errs.append(f"{s.get('step_id', '?')}: tool_call missing arguments")
        ct = s.get("tool_output", {}).get("created_table")
        if ct:
            created.add(ct["table_name"])

    if steps and steps[-1]["tool_call"]["tool"] != "answer_from_context":
        errs.append("final step must be answer_from_context")

    sources = {t["table_name"] for t in traj.get("initial_state", {}).get("dataset_overview", {}).get("tables", [])}
    if steps:
        ev = steps[-1]["tool_call"]["arguments"].get("evidence", {}).get("table")
        if ev is not None and ev not in created | sources:
            errs.append(f"answer cites unknown table {ev!r}")
    return errs
