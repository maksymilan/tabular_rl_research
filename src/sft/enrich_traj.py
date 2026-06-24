#!/usr/bin/env python3
"""LLM-in-the-loop trajectory enrichment with the executor as ground-truth verifier.

Pipeline (per verified v2 skeleton trajectory):
  1. ask the external LLM to ANNOTATE where to insert load-bearing perception (describe_table /
     inspect_column / read_subtable) and first-person reasoning — turning the mechanical relational
     backbone into a "survey globally -> zoom locally -> confirm the unknowns -> execute" trajectory;
  2. splice the annotations into the (fixed-order) relational backbone;
  3. REPLAY the spliced sequence through the harness (`execute_tool`) so every observation is real
     and harness-owned, and the final answer is re-scored against the gold SQL  (L1);
  4. check that each inserted perception is actually CONSUMED downstream — not a decorative read
     (L2, the gate that execution alone cannot enforce because reads never change the result);
  5. check that correction steps are logically plausible and written as the model's own reasoning
     (first person, no "the model might..." narration, no guessing a nonexistent column after the
     table schema has already ruled it out);
  6. on any L1/L2/style failure, feed the SPECIFIC reason back and regenerate (<=3 attempts); if it never
     validates, fall back to the clean skeleton so a failed enrichment never blocks.

The LLM authors only PLACEMENT + reasons (think); the harness owns observations, step ids and
provenance — the same trust boundary as V2a. This is the smoke harness for human review; run it on
the 10 length-spread smoke ids first.

Usage:
  .venv/bin/python src/sft/enrich_traj.py --ids data/trajectories/subset_180.ids.json --which smoke \
      [--model deepseek-v4-pro] [--out data/trajectories/smoke_enriched.jsonl]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "harness"))
sys.path.insert(0, os.path.join(ROOT, "src", "eval"))
sys.path.insert(0, HERE)

from executor import Harness                                              # noqa: E402
from plan import resolve_cond                                             # noqa: E402
from scalar_grounding import extract_scalar                               # noqa: E402
from protocol import ProtocolError, TOOLS, rows_equal                     # noqa: E402
from fill_think import call, load_api                                     # noqa: E402

SPIDER = os.path.join(ROOT, "data", "spider_data")
PERCEPTION = ("describe_table", "inspect_column", "read_subtable")
EXECUTABLE_TOOLS = TOOLS
STR_OPS = ("=", "==", "contains", "like", "in")
RATIONALE_FIELDS = ("question_cue", "observed_evidence", "decision", "supports_next_step")
MAX_REJECTED_CANDIDATES = 5
BANNED_NARRATION = (
    "the model", "a model", "the agent", "the assistant", "might guess",
    "would guess", "without first checking",
)
FIRST_PERSON = re.compile(r"\b(I|my|me|I'll|I'm|I will|I need|I see|I should)\b", re.I)
CAUSAL_MARKER = re.compile(
    r"\b(because|so|therefore|to|in order to|asks?|requested|requires?|supports?|"
    r"before|after|ground|confirm|verify|match|identify|produces?|output|final|"
    r"meaning|result|returned|answers?|fulfills?|derived|computed)\b",
    re.I,
)
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SQL_WORDS = {
    "as", "asc", "desc", "and", "or", "not", "in", "is", "null", "count", "distinct",
    "sum", "avg", "mean", "min", "max", "case", "when", "then", "else", "end",
}
GENERIC_SCHEMA_TERMS = {
    "id", "ids", "name", "names", "code", "codes", "date", "dates", "time", "times",
    "type", "types", "status", "statuses", "number", "numbers", "count", "counts",
    "phone", "phones", "email", "emails", "address", "addresses", "description",
    "descriptions", "detail", "details",
}


def db_path(db_id: str) -> str:
    return os.path.join(SPIDER, "database", db_id, f"{db_id}.sqlite")


def new_ctx() -> dict:
    return {"history": {}, "handle_to_step": {}}


def _cond_refs(cond) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []

    def walk(c):
        if not isinstance(c, dict):
            return
        for key in ("and", "or", "conditions"):
            for child in c.get(key, []):
                walk(child)
        if "not" in c:
            walk(c["not"])
        if isinstance(c.get("value_ref"), str):
            refs.append(("value_ref", c["value_ref"]))
        if isinstance(c.get("in_table"), str):
            refs.append(("in_table", c["in_table"]))

    walk(cond)
    return refs


def execute_tool(h: Harness, tool: str, args: dict, ctx: dict, step_id: str):
    if tool not in EXECUTABLE_TOOLS or tool == "answer_from_context":
        raise ProtocolError(f"tool {tool!r} not executable here")

    if tool in PERCEPTION:
        out = getattr(h, tool)(**args)
        output = out if isinstance(out, dict) else {"rows": [list(r) for r in out], "row_count": len(out)}
        ctx["history"][step_id] = {"tool": tool, "arguments": args, "output": output}
        return output, None

    exec_args = dict(args)
    if tool == "condition_filter":
        values = {ref: extract_scalar(ctx["history"], ref)
                  for kind, ref in _cond_refs(exec_args.get("conditions")) if kind == "value_ref"}
        exec_args["conditions"] = resolve_cond(exec_args.get("conditions"), {}, values)
    out = getattr(h, tool)(**exec_args)
    if isinstance(out, dict) and "table_name" in out:
        output = {"table": out["table_name"], "kind": out["kind"],
                  "columns": out["columns"], "row_count": out["row_count"]}
        if out["row_count"] == 1 and len(out["columns"]) == 1:
            output["rows"] = [list(r) for r in h.rows(out["table_name"])]
        ctx["handle_to_step"][out["table_name"]] = step_id
        created = out["table_name"]
    else:
        rows = out if isinstance(out, list) else [(out,)]
        output = {"result_sample": [list(r) for r in rows[:5]], "row_count": len(rows)}
        created = None
    ctx["history"][step_id] = {"tool": tool, "arguments": args, "output": output}
    return output, created


def score(h: Harness, gold_sql: str, answer_args: dict, created: set) -> tuple[bool, list, list]:
    gold = h.gold(gold_sql)
    ev = (answer_args.get("evidence") or {}).get("table")
    pred = None
    if ev and ev in created:
        try:
            pred = h.rows(ev)
        except Exception:
            pred = None
    if pred is None:
        pred = answer_args.get("answer") or []
    try:
        ok = rows_equal(pred, gold)
    except Exception:
        ok = False
    return ok, pred[:5], gold[:5]


# ---------------------------------------------------------------- context for the LLM ------------
def _base_col(col) -> str:
    """Strip a join prefix: 'T5__dept_name' -> 'dept_name' (post-join filters carry prefixes)."""
    return col.split("__", 1)[-1] if isinstance(col, str) and "__" in col else col


def string_filter_cols(steps: list[dict]) -> set[str]:
    """Base column names (prefix-stripped) the backbone filters on with a STRING literal — the values
    the model must confirm with inspect_column before filtering."""
    out: set[str] = set()

    def walk(cond):
        if not isinstance(cond, dict):
            return
        for key in ("and", "or"):
            for c in cond.get(key, []):
                walk(c)
        if "not" in cond:
            walk(cond["not"])
        if cond.get("op") in STR_OPS and isinstance(cond.get("value"), str):
            out.add(_base_col(cond.get("column", "")))

    for s in steps:
        if s["tool_call"]["tool"] == "condition_filter":
            walk(s["tool_call"]["arguments"].get("conditions"))
    return {c for c in out if c}


def condition_columns(cond) -> set[str]:
    out: set[str] = set()

    def walk(c):
        if not isinstance(c, dict):
            return
        for key in ("and", "or"):
            for child in c.get(key, []):
                walk(child)
        if "not" in c:
            walk(c["not"])
        if isinstance(c.get("column"), str):
            out.add(_base_col(c["column"]))
        if isinstance(c.get("column_value"), str):
            out.add(_base_col(c["column_value"]))

    walk(cond)
    return out


def expression_columns(expr: str) -> set[str]:
    cols = set()
    for tok in IDENT.findall(str(expr)):
        if tok.lower() not in SQL_WORDS:
            cols.add(_base_col(tok))
    return cols


def expected_semantic_terms(tool: str, args: dict) -> set[str]:
    """Column/table names that should be semantically justified in the generated think."""
    terms: set[str] = set()
    for key in ("table", "left", "right"):
        if isinstance(args.get(key), str) and not _is_derived_table(args[key]):
            terms.add(args[key])
    if tool == "condition_filter":
        terms |= condition_columns(args.get("conditions"))
    elif tool == "project":
        for expr in args.get("expressions", []):
            terms |= expression_columns(expr)
    elif tool == "aggregate":
        if args.get("column") != "*":
            terms.add(_base_col(args.get("column", "")))
        terms.add(str(args.get("op", "")))
    elif tool == "group_aggregate":
        terms |= {_base_col(c) for c in args.get("group_by", [])}
        for agg in args.get("aggregations", []):
            if agg.get("column") != "*":
                terms.add(_base_col(agg.get("column", "")))
            if agg.get("as"):
                terms.add(_base_col(agg["as"]))
    elif tool == "extreme_value_select":
        for item in args.get("order_by", []):
            terms |= expression_columns(str(item).replace(" DESC", "").replace(" ASC", ""))
        for item in args.get("return_columns") or []:
            terms |= expression_columns(item)
    elif tool == "join_tables":
        for item in args.get("on", []):
            terms.add(_base_col(item.get("left", "")))
            terms.add(_base_col(item.get("right", "")))
    elif tool == "describe_table":
        terms |= set(args.get("tables", []))
    elif tool in {"inspect_column", "read_subtable"}:
        table = args.get("table")
        if table and not _is_derived_table(table):
            terms.add(_base_col(table))
        if args.get("column"):
            terms.add(_base_col(args["column"]))
    return {t for t in terms if t and t != "*"}


def expected_column_terms(tool: str, args: dict) -> set[str]:
    """Exact column identifiers that must appear verbatim in the generated think.

    Table names and natural-language aliases are useful context, but they cannot replace the actual
    schema identifiers. For example, a filter on `T2__Time_of_day` must say `Time_of_day`; "broadcast
    time" is readable but too ambiguous for this training signal.
    """
    cols: set[str] = set()
    if tool == "condition_filter":
        cols |= condition_columns(args.get("conditions"))
    elif tool == "project":
        for expr in args.get("expressions", []):
            cols |= expression_columns(expr)
    elif tool == "aggregate":
        if args.get("column") != "*":
            cols.add(_base_col(args.get("column", "")))
    elif tool == "group_aggregate":
        cols |= {_base_col(c) for c in args.get("group_by", [])}
        for agg in args.get("aggregations", []):
            if agg.get("column") != "*":
                cols.add(_base_col(agg.get("column", "")))
            if agg.get("as"):
                cols.add(_base_col(agg["as"]))
    elif tool == "extreme_value_select":
        for item in args.get("order_by", []):
            cols |= expression_columns(str(item).replace(" DESC", "").replace(" ASC", ""))
        for item in args.get("return_columns") or []:
            cols |= expression_columns(item)
    elif tool == "join_tables":
        for item in args.get("on", []):
            cols.add(_base_col(item.get("left", "")))
            cols.add(_base_col(item.get("right", "")))
    elif tool == "inspect_column" and args.get("column"):
        cols.add(_base_col(args["column"]))
    return {c for c in cols if c and c != "*"}


def column_domains(h: Harness, overview: dict, base_cols: set[str]) -> dict[str, list]:
    """Real distinct values for each string-filter column, located on its SOURCE table (the backbone
    may filter it post-join under a prefix), so the LLM's reasons/literals are grounded."""
    dom: dict[str, list] = {}
    for col in base_cols:
        src = None
        for (table,) in h.conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
            for row in h.conn.execute(f'PRAGMA table_info("{table}")'):
                if row[1].lower() == col.lower():
                    src = table
                    break
            if src:
                break
        if not src:
            continue
        try:
            out, _ = execute_tool(h, "inspect_column", {"table": src, "column": col}, new_ctx(), "probe")
            vals = [v[0] if isinstance(v, (list, tuple)) else v
                    for v in (out.get("frequent_values") or out.get("values") or [])]
            dom[col] = vals[:10]
        except Exception:
            pass
    return dom


def schema_snapshot(h: Harness) -> list[dict]:
    """Full DB schema for the annotator only. The trajectory model still starts from the lazy
    catalog and must acquire these columns through describe_table observations."""
    tables = [name for (name,) in h.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    return h.describe_table(tables)["tables"]


def catalog_snapshot(h: Harness) -> dict:
    """Lazy model-visible opening state: table names, row counts and FK relations only."""
    tables = []
    relations = []
    for (name,) in h.conn.execute("SELECT name FROM sqlite_master WHERE type='table'"):
        n = h.conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
        tables.append({"table_name": name, "num_rows": n})
        for r in h.conn.execute(f'PRAGMA foreign_key_list("{name}")'):
            relations.append({"from": f"{name}.{r[3]}", "to": f"{r[2]}.{r[4]}"})
    return {"tables": tables, "relations": relations}


def skeleton_view(traj: dict) -> list[dict]:
    """Compact backbone description handed to the LLM: index, tool, args, produced handle."""
    view = []
    for i, s in enumerate(traj["steps"]):
        produces = s.get("produces") or {}
        view.append({
            "i": i,
            "tool": s["tool_call"]["tool"],
            "arguments": s["tool_call"]["arguments"],
            "produces": produces.get("handle") or produces.get("table"),
        })
    return view


# ---------------------------------------------------------------- LLM prompt ---------------------
SYS = (
    "Role and task:\n"
    "- You are simulating the TABLE-TOOL AGENT being trained, not an outside annotator. Write the "
    "assistant's `<think>` as the agent's own first-person reasoning at that exact moment: 'I need...', "
    "'I see...', 'I should...'. Never write third-person narration such as 'the model might...'.\n"
    "- Your goal is to create a state-faithful tool-use trajectory that teaches four behaviors: notice "
    "missing information, choose an observation tool, update the usable state from the observation, "
    "then justify the next relational action. The verified relational backbone is fixed; by default "
    "you may only insert read-only perception steps and rewrite reasoning.\n\n"
    "Visible-state boundary:\n"
    "- At inference the model initially sees ONLY the lazy catalog: table names, row counts, and "
    "foreign-key relations. It has NO column names and NO cell values.\n"
    "- A think may use only the question, the lazy catalog, previous tool calls, and previous tool "
    "observations. Authoring-only schema, domains, and backbone are construction aids; they must not "
    "appear in reasoning before the simulated agent has observed them.\n"
    "- Before a `describe_table` result is observed, reason with question concepts and table names "
    "only, e.g. 'order status' or 'customer identity', not schema identifiers like "
    "`order_status_code`. After schema is observed, later action thinks MUST use exact column names.\n"
    "- The trajectory should read like normal table work: survey global structure first, zoom into "
    "local values or intermediate rows, then execute the action whose preconditions are now known.\n\n"
    "Reasoning quality requirement:\n"
    "- Every inserted perception and rewritten backbone action must explain the semantic bridge from "
    "the QUESTION to the exact table/column/tool choice. Do not say only 'I project the output "
    "columns'. Say why these columns are the requested fields and what later step this information "
    "supports.\n"
    "- Mention exact column names when choosing them, e.g. `Song_Name` because the question asks for "
    "song names, `Song_release_year` because it asks for release years, `Age` because it asks for "
    "the youngest singer.\n"
    "- Exact means the literal schema/tool identifier with underscores and prefixes stripped when "
    "needed. If the tool uses `T2__Time_of_day`, your think must explicitly say `Time_of_day`; "
    "phrases like 'broadcast time' or 'time column' are NOT acceptable substitutes.\n"
    "- A good `describe_table` reason is logical without leaking schema: explain which question "
    "concept points to which candidate table(s), what information is missing, and which future "
    "operation the schema will support.\n\n"
    "Counterexamples for state-aware reasoning:\n"
    "- BAD before describe_table: 'I need Customer_Orders before filtering on `order_status_code`.' "
    "This leaks a column name that is not visible yet.\n"
    "- GOOD before describe_table: 'I need Customer_Orders because the question is about orders and "
    "cancellation status, but I still need the schema to discover which exact column represents "
    "that status before filtering.'\n"
    "- BAD before describe_table: 'I describe Broadcast to use `Time_of_day` for Morning.' This "
    "already assumes the schema identifier.\n"
    "- GOOD before describe_table: 'I describe Broadcast because the question depends on broadcast "
    "time-of-day, and I need the schema to find the exact time/status column before grounding the "
    "Morning literal.'\n"
    "- GOOD after describe_table: 'Now that the schema shows `Time_of_day`, I inspect that column "
    "to verify the literal Morning before filtering.' Exact column names are required AFTER the "
    "observation exposes them.\n\n"
    "Perception tools:\n"
    "- describe_table {\"tables\":[...]} : acquire columns/types/keys. REQUIRED before the first time "
    "the trajectory operates on a table, since the catalog carries no columns.\n"
    "- inspect_column {\"table\":t,\"column\":c} : see a column's real values. REQUIRED before a filter "
    "comparing that column to a STRING literal, so the literal is grounded, not guessed. Use the "
    "SOURCE table + base column name even if the backbone filters it post-join under a prefix.\n"
    "- read_subtable {\"table\":handle} : inspect an intermediate result's rows to confirm a filter/"
    "join produced what the next step assumes (e.g. non-empty) before building on it.\n\n"
    "Rules:\n"
    "1. Insert exactly the perception needed so every action's required info is already acquired; do "
    "not insert a read whose information is already known or never used.\n"
    "2. Prefer a read_subtable to CONFIRM a filter/join's rows whenever the next step's correctness "
    "depends on that intermediate (e.g. before an aggregate or the final answer).\n"
    "3. All `think` strings must be first person. Bad: \"The model might guess "
    "acting\". Good: \"I have not checked the schema yet, so I first need to inspect the table instead "
    "of guessing a column name.\"\n"
    "4. Every action think must include every exact column identifier used by that action. Do not "
    "paraphrase schema names.\n"
    "5. Output ONLY a JSON object with two keys: `rewrites` and `insertions`.\n"
    "6. Every rewrite/insertion must include a structured `rationale` object with EXACTLY these "
    "non-empty fields: `question_cue` (the phrase/need in the user question), `observed_evidence` "
    "(what has been or will be learned from schema/value/rows), `decision` (why this exact "
    "table/column/tool is selected), and `supports_next_step` (which later action this enables). "
    "The natural-language `think` must be standalone and must summarize this rationale; the "
    "`rationale` is for validation, not a replacement for `think`.\n"
    "`rewrites` must contain one entry for EVERY backbone step: "
    "{\"step\": <backbone index>, \"think\": <first-person semantic reason>, "
    "\"rationale\": {\"question_cue\": ..., \"observed_evidence\": ..., "
    "\"decision\": ..., \"supports_next_step\": ...}}.\n"
    "`insertions` contains optional perception steps. Perception element: "
    "{\"after\": <backbone step index, -1 = before "
    "the first step>, \"tool\": <describe_table|inspect_column|read_subtable>, \"arguments\": {...}, "
    "\"think\": <first-person sentence>, \"rationale\": {...}}."
)

CORRECTION_SYS = (
    "\n\nCORRECTION MODE ONLY:\n"
    "You may add a FEW realistic SELF-CORRECTION moments so the model learns to recover from "
    "mistakes. An error element attempts a WRONG variant of the next backbone action only when the "
    "wrong action is justified by the current uncertainty. For example: if I have not inspected a "
    "text column yet, I may try an ungrounded literal and get 0 rows; if two tables/columns are "
    "semantically ambiguous, I may probe one and then recover. Do NOT guess a nonexistent column "
    "after describe_table has already shown the real columns. Do NOT add an error merely for drama.\n"
    "Error element schema: "
    "{\"after\": <index>, \"tool\": \"error\", "
    "\"wrong\": {\"tool\": <relational tool>, \"arguments\": {...the WRONG variant...}}, "
    "\"think\": <first-person reason for trying it>, \"rationale\": {...}, "
    "\"recovery_think\": <first-person sentence: "
    "how the error/empty result changes my next action>}."
)


def build_prompt(traj: dict, domains: dict, allow_errors: bool = True) -> list[dict]:
    ov = traj["initial_state"]["dataset_overview"]
    h = Harness(db_path(traj["source"]["db_id"]))
    system = SYS + (CORRECTION_SYS if allow_errors else "")
    if not allow_errors:
        system += (
            "\n\nCURRENT RUN MODE: OBSERVATION ONLY. Do NOT add correction paths and do NOT output "
            "any insertion with tool='error'. Insertions may contain only describe_table, "
            "inspect_column, and read_subtable. The goal is to teach information acquisition before "
            "action; correction trajectories will be generated in a separate dataset."
        )
    user = (
        f"QUESTION: {traj['question']}\n\n"
        f"MODEL'S INITIAL CATALOG (what the model sees at turn 0; no columns):\n"
        f"{json.dumps(ov, ensure_ascii=False)}\n\n"
        f"AUTHORING-ONLY FULL SCHEMA (use this to write grounded column-choice reasoning; "
        f"the model must still call describe_table before it can rely on these columns):\n"
        f"{json.dumps(schema_snapshot(h), ensure_ascii=False)}\n\n"
        f"STRING-FILTER COLUMN DOMAINS (what inspect_column would show):\n"
        f"{json.dumps(domains, ensure_ascii=False)}\n\n"
        f"RELATIONAL BACKBONE (fixed order; rewrite every think and insert perception around these):\n"
        f"{json.dumps(skeleton_view(traj), ensure_ascii=False)}\n\n"
        "Return the JSON object now."
    )
    if domains.get("_feedback"):
        user += f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {domains['_feedback']}\nFix and re-output."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


OBS_SYS = (
    "Role: you are simulating the table-tool agent's first observation decision from the catalog-only "
    "state, not annotating as a third party. Write first-person reasoning for what I should inspect "
    "before taking any relational action.\n\n"
    "The agent currently sees ONLY the lazy catalog: table names, row counts, and foreign-key "
    "relations. It does NOT know any column names or cell values yet. Your job is to decide which "
    "source tables to inspect with describe_table first.\n\n"
    "Important: do not mention exact column names, because they are not visible yet. Explain the "
    "logic using question concepts, table names, and catalog relations only. A good reason says: "
    "which question concept suggests this table, what information is missing, and what kind of "
    "future operation the schema will support.\n\n"
    "Output ONLY JSON: {\"insertions\": [ ... ]}. Each insertion must be a describe_table step with "
    "`after:-1`, arguments {\"tables\":[...]}, first-person `think`, and `rationale` with exactly "
    "these non-empty fields: question_cue, observed_evidence, decision, supports_next_step.\n\n"
    "Exploratory observations are allowed: if the question asks for an output entity such as "
    "'all info of students', it is reasonable to describe the entity table even if a later minimal "
    "gold plan might not consume it directly. Still, also include the tables needed to resolve the "
    "question conditions, because later actions may only use schemas that have actually been "
    "observed.\n\n"
    "Bad: 'I describe Customer_Orders before filtering on order_status_code.'\n"
    "Good: 'I describe Customer_Orders because the question is about cancelled orders, but I do not "
    "yet know which exact column represents order status; the schema will let me choose the precise "
    "filter field later.'"
)


def build_observation_prompt(traj: dict, feedback: str = "") -> list[dict]:
    ov = traj["initial_state"]["dataset_overview"]
    user = (
        f"QUESTION: {traj['question']}\n\n"
        f"MODEL'S INITIAL CATALOG (the only information available now; no columns):\n"
        f"{json.dumps(ov, ensure_ascii=False)}\n\n"
        "Choose the initial describe_table observation(s) now."
    )
    if feedback:
        user += f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {feedback}\nFix and re-output."
    return [{"role": "system", "content": OBS_SYS}, {"role": "user", "content": user}]


def observed_schema_outputs(traj: dict, describe_insertions: list[dict]) -> list[dict]:
    h = Harness(db_path(traj["source"]["db_id"]))
    outs = []
    for ins in describe_insertions:
        if ins.get("tool") != "describe_table":
            continue
        try:
            outs.append({"arguments": ins.get("arguments", {}),
                         "output": h.describe_table(ins.get("arguments", {}).get("tables", []))})
        except Exception as e:  # noqa: BLE001
            outs.append({"arguments": ins.get("arguments", {}), "error": f"{type(e).__name__}: {e}"})
    return outs


def build_staged_action_prompt(traj: dict, domains: dict, describe_insertions: list[dict]) -> list[dict]:
    user = (
        f"QUESTION: {traj['question']}\n\n"
        f"MODEL'S INITIAL CATALOG:\n"
        f"{json.dumps(traj['initial_state']['dataset_overview'], ensure_ascii=False)}\n\n"
        f"OBSERVED SCHEMA FROM PRIOR describe_table STEPS:\n"
        f"{json.dumps(observed_schema_outputs(traj, describe_insertions), ensure_ascii=False)}\n\n"
        f"STRING-FILTER COLUMN DOMAINS (available only after the relevant schema column is observed "
        f"and inspect_column is called):\n{json.dumps(domains, ensure_ascii=False)}\n\n"
        f"RELATIONAL BACKBONE (fixed order; rewrite every think and insert only local observations "
        f"around these actions):\n{json.dumps(skeleton_view(traj), ensure_ascii=False)}\n\n"
        "Return JSON with `rewrites` for every backbone step and optional `insertions`. In this staged "
        "second pass, do NOT output describe_table insertions; only inspect_column/read_subtable are "
        "allowed as new observations."
    )
    system = SYS + (
        "\n\nCURRENT RUN MODE: STAGED ACTION PASS. Initial describe_table observations have already "
        "been chosen from the catalog. Do NOT add more describe_table steps. You may use exact column "
        "names only for tables whose schema appears in OBSERVED SCHEMA; otherwise the trajectory will "
        "be rejected as relying on unobserved schema."
    )
    if domains.get("_feedback"):
        user += f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {domains['_feedback']}\nFix and re-output."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


REWRITE_SYS = (
    "You rewrite the <think> text of a verified table-tool trajectory. Output ONLY a JSON array of "
    "exactly {n} first-person strings, one for each backbone step in order. Each string must explain "
    "the semantic bridge from the user's question to the exact table/column/tool choice. Mention the "
    "key table and column names used by that step. The trajectory already has separate observations "
    "for global schema; your job is to make the ACTION reasoning understandable to a human reader."
)


def build_rewrite_prompt(traj: dict) -> list[dict]:
    h = Harness(db_path(traj["source"]["db_id"]))
    view = skeleton_view(traj)
    user = (
        f"QUESTION: {traj['question']}\n\n"
        f"FULL SCHEMA FOR AUTHORING:\n{json.dumps(schema_snapshot(h), ensure_ascii=False)}\n\n"
        f"BACKBONE STEPS TO REWRITE:\n{json.dumps(view, ensure_ascii=False)}\n\n"
        f"Return exactly {len(view)} first-person reasoning strings as JSON."
    )
    return [{"role": "system", "content": REWRITE_SYS.format(n=len(view))},
            {"role": "user", "content": user}]


def parse_rewrite_array(text: str, n: int) -> dict[int, str]:
    arr_match = re.search(r"\[.*\]", text, re.S)
    if not arr_match:
        return {}
    try:
        arr = json.loads(arr_match.group(0))
    except json.JSONDecodeError:
        return {}
    if not isinstance(arr, list) or len(arr) != n:
        return {}
    if not all(isinstance(x, str) and x.strip() for x in arr):
        return {}
    return {i: x.strip() for i, x in enumerate(arr)}


def parse_annotations(text: str) -> tuple[list[dict], dict[int, dict]]:
    obj_match = re.search(r"\{.*\}", text, re.S)
    arr_match = re.search(r"\[.*\]", text, re.S)
    payload = None
    if obj_match:
        try:
            payload = json.loads(obj_match.group(0))
        except json.JSONDecodeError:
            payload = None
    if payload is None and arr_match:
        try:
            payload = json.loads(arr_match.group(0))
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        return [], {}

    if isinstance(payload, dict):
        arr = payload.get("insertions", [])
        raw_rewrites = payload.get("rewrites", [])
    elif isinstance(payload, list):  # backward-compatible with older prompt attempts
        arr = payload
        raw_rewrites = []
    else:
        return [], {}

    rewrites: dict[int, dict] = {}
    if isinstance(raw_rewrites, dict):
        for k, v in raw_rewrites.items():
            if str(k).lstrip("-").isdigit():
                if isinstance(v, str):
                    rewrites[int(k)] = {"think": v}
                elif isinstance(v, dict) and isinstance(v.get("think"), str):
                    rewrites[int(k)] = {"think": v["think"], "rationale": v.get("rationale")}
    elif isinstance(raw_rewrites, list):
        for item in raw_rewrites:
            if isinstance(item, dict) and isinstance(item.get("think"), str):
                rewrites[int(item.get("step", -999))] = {
                    "think": item["think"],
                    "rationale": item.get("rationale"),
                }

    out = []
    for a in arr if isinstance(arr, list) else []:
        if not isinstance(a, dict) or "after" not in a:
            continue
        if "reason" in a and "think" not in a:
            a["think"] = a["reason"]
        if "recovery" in a and "recovery_think" not in a:
            a["recovery_think"] = a["recovery"]
        if a.get("tool") in PERCEPTION:
            out.append(a)
        elif a.get("tool") == "error" and isinstance(a.get("wrong"), dict):
            out.append(a)
    return out, rewrites


def deterministic_perception_insertions(traj: dict) -> list[dict]:
    """Small, stable observation scaffold for smoke tests: first acquire the source-table schemas,
    then read the final evidence table before answering when there is one."""
    source_tables: list[str] = []
    for s in traj["steps"]:
        args = s["tool_call"]["arguments"]
        for key in ("table", "left", "right"):
            value = args.get(key)
            if isinstance(value, str) and not _is_derived_table(value) and value not in source_tables:
                source_tables.append(value)
    out: list[dict] = []
    if source_tables:
        table_list = ", ".join(source_tables)
        out.append({
            "after": -1,
            "tool": "describe_table",
            "arguments": {"tables": source_tables},
            "think": (
                f"I first need the global schema of {table_list} so I can map the question "
                "to the exact columns before taking actions."
            ),
            "rationale": {
                "question_cue": traj.get("question", ""),
                "observed_evidence": "source table columns, types, and keys are not available in the initial catalog",
                "decision": f"describe_table on {table_list} is needed before operating on these source tables",
                "supports_next_step": "the following relational actions that reference these tables and columns",
            },
        })
    for i, s in enumerate(traj["steps"][:-1]):
        produced = (s.get("produces") or {}).get("handle")
        next_tool = traj["steps"][i + 1]["tool_call"]["tool"]
        if produced and next_tool == "answer_from_context":
            out.append({
                "after": i,
                "tool": "read_subtable",
                "arguments": {"table": produced, "limit": 20},
                "think": f"I read {produced} locally to verify the rows before answering.",
                "rationale": {
                    "question_cue": traj.get("question", ""),
                    "observed_evidence": f"{produced} is the final derived table produced by the previous action",
                    "decision": f"read_subtable on {produced} shows the rows that will be used as evidence",
                    "supports_next_step": "answer_from_context",
                },
            })
    return out


def parse_insertions(text: str) -> list[dict]:
    """Backward-compatible helper for older callers/tests."""
    return parse_annotations(text)[0]


def parse_observation_insertions(text: str) -> list[dict]:
    """Parse the catalog-only observation pass.

    Some providers follow the semantic instruction but omit the literal `tool` key or use
    `type:"describe_table"`. For this first pass, a JSON insertion with `arguments.tables` is
    unambiguously a describe_table observation, so normalize it instead of throwing away a good
    reasoning sample.
    """
    ann, _ = parse_annotations(text)
    out = [a for a in ann if a.get("tool") == "describe_table"]
    if out:
        return out
    obj_match = re.search(r"\{.*\}", text, re.S)
    if not obj_match:
        return []
    try:
        payload = json.loads(obj_match.group(0))
    except json.JSONDecodeError:
        return []
    arr = payload.get("insertions", []) if isinstance(payload, dict) else []
    for item in arr if isinstance(arr, list) else []:
        if not isinstance(item, dict):
            continue
        args = item.get("arguments") or {}
        if not args and item.get("tables"):
            args = {"tables": item.get("tables")}
        if item.get("tool") == "describe_table" or item.get("type") == "describe_table" or args.get("tables"):
            norm = dict(item)
            norm["tool"] = "describe_table"
            norm["arguments"] = args
            norm.setdefault("after", -1)
            out.append(norm)
    return out


def _strip_answer_memory_args(tool: str, args: dict) -> dict:
    """V2b cleanup: answer_from_context no longer carries supporting_memory_ids."""
    out = copy.deepcopy(args)
    if tool == "answer_from_context":
        out.pop("supporting_memory_ids", None)
    return out


def _rewrite_value_refs(cond, legacy_refs: dict[str, str]):
    if isinstance(cond, list):
        for item in cond:
            _rewrite_value_refs(item, legacy_refs)
        return
    if not isinstance(cond, dict):
        return
    for key in ("and", "or", "conditions"):
        for child in cond.get(key, []):
            _rewrite_value_refs(child, legacy_refs)
    if "not" in cond:
        _rewrite_value_refs(cond["not"], legacy_refs)
    ref = cond.get("value_ref")
    seen = set()
    while isinstance(ref, str) and ref in legacy_refs and ref not in seen:
        seen.add(ref)
        ref = legacy_refs[ref]
    if ref is not None:
        cond["value_ref"] = ref


def normalize_legacy_memory(traj: dict) -> dict:
    """Migrate old V2a/V2-ctx skeletons to the current V2b no-memory protocol.

    Old smoke/subset files may still contain:
    - explicit `add_to_memory` steps;
    - predicates whose `value_ref` points to that memory step or a `mem_*` id;
    - `answer_from_context.supporting_memory_ids`.

    The current protocol removed memory. A scalar predicate should reference the producing step
    directly, and final answers should cite only `evidence`.
    """
    out = copy.deepcopy(traj)
    legacy_refs: dict[str, str] = {}
    kept: list[dict] = []
    for step in out.get("steps", []):
        call = step.get("tool_call") or {}
        tool = call.get("tool")
        args = call.get("arguments") or {}
        if tool == "add_to_memory":
            src = args.get("source_step_id") or args.get("source")
            if isinstance(src, str):
                legacy_refs[step.get("step_id", "")] = src
                legacy_refs[f"mem_{src}"] = src
                produced = step.get("produces") or {}
                output_mem = (step.get("tool_output") or {}).get("memory") or {}
                for key in (produced.get("memory_id"), produced.get("id"),
                            output_mem.get("memory_id"), output_mem.get("id")):
                    if isinstance(key, str):
                        legacy_refs[key] = src
            continue
        new_step = copy.deepcopy(step)
        new_args = _strip_answer_memory_args(tool, args)
        if tool == "condition_filter":
            _rewrite_value_refs(new_args.get("conditions"), legacy_refs)
        new_step["tool_call"]["arguments"] = new_args
        kept.append(new_step)
    out["steps"] = kept
    out["schema_version"] = "v3-input-normalized"
    return out


# ---------------------------------------------------------------- splice + replay (L1) -----------
def _remap_cond(cond, m: dict):
    if not isinstance(cond, dict):
        return
    for key in ("and", "or", "conditions"):
        for c in cond.get(key, []):
            _remap_cond(c, m)
    if "not" in cond:
        _remap_cond(cond["not"], m)
    if "value_ref" in cond:                      # value_ref now cites a producing step_id directly
        cond["value_ref"] = m.get(cond["value_ref"], cond["value_ref"])


def _remap_step_refs(tool: str, args: dict, m: dict):
    """Rewrite step-id references after perception insertion shifts the numbering: a predicate's
    value_ref now cites the producing step_id directly."""
    if tool == "condition_filter":
        _remap_cond(args.get("conditions"), m)


def spliced_sequence(traj: dict, insertions: list[dict],
                     rewrites: dict[int, dict] | None = None) -> list[dict]:
    """Interleave inserted perception into the fixed backbone, then remap step-id references to the
    new (shifted) numbering. Each element: {tool, arguments, reason, backbone?}."""
    by_after: dict[int, list] = {}
    for a in insertions:
        by_after.setdefault(int(a["after"]), []).append(a)
    seq: list[dict] = []
    old_to_new: dict[str, str] = {}
    pending_recovery: list[str] = []
    rewrites = rewrites or {}

    def emit_backbone(i: int, s):
        rewrite = rewrites.get(i, {})
        think = rewrite.get("think", s.get("think", "")) if isinstance(rewrite, dict) else str(rewrite)
        rationale = rewrite.get("rationale") if isinstance(rewrite, dict) else None
        if pending_recovery:
            think = " ".join(pending_recovery) + " " + think
            pending_recovery.clear()
        tool = s["tool_call"]["tool"]
        args = _strip_answer_memory_args(tool, s["tool_call"]["arguments"])
        seq.append({"tool": tool, "arguments": args,
                    "reason": think, "rationale": rationale, "backbone": True})
        old_to_new[s["step_id"]] = f"step_{len(seq)}"

    def emit_ann(a):
        if a.get("tool") == "error":
            w = a["wrong"]
            seq.append({"tool": w.get("tool"), "arguments": w.get("arguments", {}),
                        "reason": a.get("think", ""), "rationale": a.get("rationale"),
                        "error_attempt": True})
            if a.get("recovery_think"):
                pending_recovery.append(a["recovery_think"])
        else:
            seq.append({"tool": a["tool"], "arguments": a.get("arguments", {}),
                        "reason": a.get("think", ""), "rationale": a.get("rationale"),
                        "observation_role": a.get("observation_role"),
                        "exploratory_tables": a.get("exploratory_tables", [])})

    for a in by_after.get(-1, []):
        emit_ann(a)
    for i, s in enumerate(traj["steps"]):
        emit_backbone(i, s)
        for a in by_after.get(i, []):
            emit_ann(a)
    for step in seq:
        if step.get("backbone"):
            _remap_step_refs(step["tool"], step["arguments"], old_to_new)
    return seq


def replay_validate(traj: dict, seq: list[dict]) -> tuple[bool, str, list[dict]]:
    """Replay the spliced sequence through the harness, attaching the REAL observation to each step.
    Returns (l1_ok, error_message, enriched_steps). L1 = every step executes + final answer == gold."""
    h = Harness(db_path(traj["source"]["db_id"]))
    ctx = new_ctx()
    created: set[str] = set()
    enriched: list[dict] = []
    for n, step in enumerate(seq, 1):
        sid = f"step_{n}"
        tool, args = step["tool"], step["arguments"]
        if step.get("error_attempt"):
            # Execute the WRONG action on a throwaway ctx, then undo its views + counter so the main
            # chain's handles stay stable. It must really error or return 0 rows to be teachable.
            tmp = copy.deepcopy(ctx)
            n0, keys0 = h._n, set(h.views)
            ok_err, obs = False, None
            try:
                out, _ = execute_tool(h, tool, args, tmp, sid)
                rc = out.get("row_count")
                if rc == 0:
                    ok_err, obs = True, ("empty", out)
                else:
                    obs = ("unexpected", out)
            except Exception as e:  # noqa: BLE001
                ok_err, obs = True, ("error", {"error": f"{type(e).__name__}: {e}"})
            for k in set(h.views) - keys0:
                del h.views[k]
            h._n = n0
            if not ok_err:
                return False, f"error attempt step {n} ({tool}) returned rows instead of failing/emptying", enriched
            enriched.append({"step_id": sid, "think": step["reason"], "rationale": step.get("rationale"),
                             "error_attempt": True,
                             "tool_call": {"tool": tool, "arguments": args},
                             "tool_status": obs[0], "tool_output": obs[1]})
            continue
        if tool == "answer_from_context":
            args = _strip_answer_memory_args(tool, args)
            enriched.append({"step_id": sid, "think": step["reason"], "rationale": step.get("rationale"),
                             "tool_call": {"tool": tool, "arguments": args}})
            correct, _, _ = score(h, traj["source"]["gold_sql"], args, created)
            if not correct:
                return False, "final answer no longer matches gold after enrichment", enriched
            return True, "", enriched
        try:
            out, tname = execute_tool(h, tool, args, ctx, sid)
        except Exception as e:  # noqa: BLE001
            return False, f"step {n} {tool} failed to execute: {type(e).__name__}: {e}", enriched
        if tname:
            created.add(tname)
        item = {"step_id": sid, "think": step["reason"], "rationale": step.get("rationale"),
                "tool_call": {"tool": tool, "arguments": args},
                "tool_output": out, "perception": tool in PERCEPTION}
        if step.get("observation_role"):
            item["observation_role"] = step["observation_role"]
        if step.get("exploratory_tables"):
            item["exploratory_tables"] = step["exploratory_tables"]
        enriched.append(item)
    return False, "trajectory has no answer_from_context", enriched


# ---------------------------------------------------------------- load-bearing check (L2) --------
def check_load_bearing(traj: dict, insertions: list[dict]) -> tuple[bool, list[str]]:
    """Each inserted read must be CONSUMED downstream, else it is ritual. inspect_column matches (by
    base column name) a later string filter; read_subtable reads a produced/referenced table.

    describe_table is different: the model may make exploratory observations from the catalog
    before it knows the exact minimal gold path. Those observations are allowed and tagged instead
    of rejected, while later action validation still requires every source-table action to have
    observed schema before use.
    """
    steps = traj["steps"]
    str_cols = {c.lower() for c in string_filter_cols(steps)}
    used_cols: set[str] = set()
    for s in steps:
        call = s.get("tool_call") or {}
        used_cols |= {c.lower() for c in expected_column_terms(
            call.get("tool"), call.get("arguments") or {}
        )}
    referenced: set[str] = set()
    handles: set[str] = set()
    for s in steps:
        a = s["tool_call"]["arguments"]
        for k in ("table", "left", "right"):
            if isinstance(a.get(k), str):
                referenced.add(a[k].lower())
            produces = s.get("produces") or {}
            h = produces.get("handle") or produces.get("table")
            if h:
                handles.add(h.lower())
    issues: list[str] = []
    for a in insertions:
        tool, args = a["tool"], a.get("arguments", {})
        if tool == "inspect_column":
            raw_col = args.get("column", "")
            if not isinstance(raw_col, str):   # malformed inspect (column not a single name) — let replay reject it
                continue
            col = _base_col(raw_col).lower()
            if col not in str_cols and col not in used_cols:
                issues.append(f"inspect_column {args.get('column')} matches no later string filter (ritual)")
        elif tool == "read_subtable":
            tbl = args.get("table")
            tbl_key = tbl.lower() if isinstance(tbl, str) else tbl
            if tbl_key not in handles and tbl_key not in referenced:
                issues.append(f"read_subtable {tbl} reads no produced/backbone table")
        elif tool == "describe_table":
            tabs = {t.lower() for t in (args.get("tables") or [])}
            load_bearing = tabs & referenced
            exploratory = tabs - referenced
            if load_bearing and exploratory:
                a["observation_role"] = "mixed_observation"
            elif load_bearing:
                a["observation_role"] = "required_observation"
            else:
                a["observation_role"] = "exploratory_observation"
            a["exploratory_tables"] = sorted(exploratory)
    return (not issues), issues


def _narration_issue(text: str, label: str) -> str | None:
    """Genuine quality red line (HARD gate): empty reasoning, or third-person / meta narration the
    SFT model must never imitate (e.g. 'the model might guess', 'the agent ...'). First-person
    phrasing is a separate SOFT preference — a clear imperative think ('Join X to get Y') is fine
    even without an explicit 'I', so it never gates."""
    if not text or not text.strip():
        return f"{label} is empty"
    lowered = text.lower()
    for phrase in BANNED_NARRATION:
        if phrase in lowered:
            return f"{label} uses third-person / meta narration: {phrase!r}"
    return None


def _is_generated_alias(identifier: str) -> bool:
    return bool(re.match(r"^(count|sum|mean|min|max|avg)_\d+$", str(identifier).lower()))


def _rationale_text(rationale) -> str:
    if not isinstance(rationale, dict):
        return ""
    return " ".join(str(rationale.get(k, "")) for k in RATIONALE_FIELDS)


def _rationale_quality_issues(rationale, label: str) -> list[str]:
    if not isinstance(rationale, dict):
        return [f"{label} is missing structured rationale"]
    issues: list[str] = []
    for field in RATIONALE_FIELDS:
        value = rationale.get(field)
        if isinstance(value, list):
            ok = any(str(x).strip() for x in value)
        else:
            ok = bool(str(value or "").strip())
        if not ok:
            issues.append(f"{label}.rationale.{field} is empty")
    return issues


def _mentioned_identifier(text: str, identifier: str) -> bool:
    if not identifier:
        return False
    pattern = r"(?<![A-Za-z0-9_])" + re.escape(identifier.lower()) + r"(?![A-Za-z0-9_])"
    return re.search(pattern, text.lower()) is not None


def _catalog_visible_identifiers(overview: dict | None) -> set[str]:
    """Identifiers visible in the opening catalog through FK relations.

    The lazy catalog intentionally hides full schemas, but it does expose FK endpoints such as
    `broadcast.Program_ID -> program.Program_ID`. Mentioning those exact key columns before
    describe_table is therefore not a leak.
    """
    visible: set[str] = set()
    if not isinstance(overview, dict):
        return visible
    for rel in overview.get("relations", []):
        if not isinstance(rel, dict):
            continue
        for side in ("from", "to"):
            value = str(rel.get(side, "")).strip()
            if not value or "." not in value:
                continue
            table, column = value.split(".", 1)
            visible.add(value.lower())
            visible.add(column.lower())
            visible.add(f"{table.lower()}.{column.lower()}")
    return visible


def _identifier_words(identifier: str) -> set[str]:
    words = re.findall(r"[A-Za-z]+", identifier.lower())
    return {w for word in words for w in (word, word.rstrip("s")) if w}


def _is_natural_question_concept(identifier: str, question: str) -> bool:
    """Whether an identifier is better treated as a natural question concept than a hidden column.

    Examples: `Name` in a question asking for names, `customer_id` in a question asking for customer
    ids. This prevents rejecting reasonable pre-schema language like "the customer id field" while
    still rejecting specific hidden identifiers like `order_status_code` when the question only says
    "Cancelled".
    """
    words = _identifier_words(identifier)
    if not words:
        return False
    if len(words) == 1 and next(iter(words)) in GENERIC_SCHEMA_TERMS:
        return True
    qwords = _identifier_words(question)
    meaningful = {w for w in words if w not in {"id", "code", "type", "name", "date"}}
    if meaningful and meaningful <= qwords:
        return True
    return len(words) <= 2 and bool(words & qwords) and bool(words & GENERIC_SCHEMA_TERMS)


def _schema_leak_issues(step: dict, question: str, label: str,
                        overview: dict | None = None) -> list[str]:
    """A describe_table rationale cannot use columns revealed by that same describe call.

    The annotator sees the full schema, but the trained model should not. It may say "I need the
    order-status column if it exists", but not the exact schema identifier `order_status_code`
    before the observation has returned it.
    """
    if step.get("tool_call", {}).get("tool") != "describe_table":
        return []
    text = f"{step.get('think', '')} {_rationale_text(step.get('rationale'))}".lower()
    question_low = question.lower()
    catalog_visible = _catalog_visible_identifiers(overview)
    leaked: list[str] = []
    for table in (step.get("tool_output") or {}).get("tables", []):
        for col in table.get("columns", []):
            name = str(col.get("name", ""))
            if not name or _mentioned_identifier(question_low, name):
                continue
            table_name = str(table.get("table_name", ""))
            full = f"{table_name}.{name}".lower()
            if name.lower() in catalog_visible or full in catalog_visible:
                continue
            if _is_natural_question_concept(name, question):
                continue
            if _mentioned_identifier(text, name):
                leaked.append(f"{table.get('table_name')}.{name}")
    if leaked:
        return [
            f"{label} leaks exact schema identifier(s) before describe_table observation: "
            f"{sorted(leaked)[:8]}"
        ]
    return []


def _described_columns(enriched_steps: list[dict], upto: int) -> dict[str, set[str]]:
    described: dict[str, set[str]] = {}
    for s in enriched_steps[:upto]:
        if s.get("tool_call", {}).get("tool") != "describe_table":
            continue
        out = s.get("tool_output") or {}
        for table in out.get("tables", []):
            described[table["table_name"].lower()] = {c["name"].lower() for c in table.get("columns", [])}
    return described


def _observed_value_domains(enriched_steps: list[dict], upto: int) -> set[tuple[str, str]]:
    seen: set[tuple[str, str]] = set()
    for s in enriched_steps[:upto]:
        if s.get("tool_call", {}).get("tool") != "inspect_column":
            continue
        args = s.get("tool_call", {}).get("arguments", {})
        if args.get("table") and args.get("column"):
            seen.add((args["table"].lower(), _base_col(args["column"]).lower()))
    return seen


def _is_derived_table(name: str) -> bool:
    return bool(re.match(r"^(filter|project|join|group|top|set|setop|derive)_\d{3}$", str(name)))


def _table_refs_for_action(tool: str, args: dict) -> list[str]:
    keys = {
        "condition_filter": ["table"],
        "project": ["table"],
        "group_aggregate": ["table"],
        "aggregate": ["table"],
        "extreme_value_select": ["table"],
        "set_op": ["left", "right"],
        "join_tables": ["left", "right"],
    }.get(tool, [])
    return [args[k] for k in keys if isinstance(args.get(k), str)]


def quality_check(enriched_steps: list[dict], question: str = "",
                  overview: dict | None = None) -> tuple[list[str], list[str]]:
    """Split validation into (hard_issues, soft_issues).

    HARD = structural / executional problems that genuinely break the enrichment for SFT and MUST
    reject (these reflect the catalog-only agent's real preconditions):
      - operating on a source table before describe_table exposed its columns;
      - filtering a text column before inspect_column grounded its value domain;
      - a correction attempt that guesses a column the schema already ruled out.
    SOFT = stylistic preferences recorded for review but NOT a reject gate. Over-strict text rules
    (first-person phrasing, exact-column mentions, rationale completeness, table/column
    justification) were rejecting large amounts of usable data, so they no longer gate. The
    describe-step schema-leak rule is dropped entirely: a describe step naming the column it is about
    to fetch is intent, not a leak.
    """
    hard: list[str] = []
    soft: list[str] = []
    for i, step in enumerate(enriched_steps):
        think = step.get("think", "")
        tool_call = step.get("tool_call") or {}
        tool = tool_call.get("tool")
        args = tool_call.get("arguments") or {}

        # ---- hard: genuine quality red lines (empty reasoning / third-person narration) ----
        narration = _narration_issue(think, f"step {i + 1} think")
        if narration:
            hard.append(narration)
        # ---- soft: stylistic / readability (recorded, never a reject) ----
        if think.strip() and not FIRST_PERSON.search(think):
            soft.append(f"step {i + 1} think is not written as first-person task reasoning")
        soft.extend(_rationale_quality_issues(step.get("rationale"), f"step {i + 1}"))
        terms = expected_semantic_terms(tool, args)
        column_terms = expected_column_terms(tool, args)
        rationale_text = _rationale_text(step.get("rationale"))
        low = think.lower()
        combined_low = f"{think} {rationale_text}".lower()
        missing_columns = [
            term for term in sorted(column_terms)
            if term.lower() not in low and not _is_generated_alias(term)
        ]
        if missing_columns:
            soft.append(f"step {i + 1} think does not name exact column(s) {missing_columns}")
        elif terms and not any(term.lower() in combined_low for term in terms):
            soft.append(f"step {i + 1} think does not justify its concrete table/column choice")

        # ---- hard: structural preconditions of a catalog-only agent ----
        described = _described_columns(enriched_steps, i)
        for table in _table_refs_for_action(tool, args):
            if not _is_derived_table(table) and table.lower() not in described:
                hard.append(
                    f"step {i + 1} operates on source table {table!r} before describe_table exposed "
                    f"its columns"
                )
        if tool == "condition_filter":
            domains = _observed_value_domains(enriched_steps, i)
            for col in string_filter_cols([{"tool_call": {"tool": "condition_filter",
                                                          "arguments": args}}]):
                source_table = args.get("table")
                if isinstance(source_table, str) and not _is_derived_table(source_table):
                    if (source_table.lower(), col.lower()) not in domains:
                        hard.append(
                            f"step {i + 1} filters text column {source_table}.{col} before "
                            f"inspect_column grounded its value domain"
                        )
        if not step.get("error_attempt"):
            continue
        columns_seen = described
        table = args.get("table")
        column = args.get("column")
        if isinstance(table, str) and isinstance(column, str):
            known = columns_seen.get(table.lower())
            if known is not None and column.lower() not in known:
                hard.append(
                    f"step {i + 1} guesses missing column {table}.{column} after describe_table "
                    f"already showed the schema"
                )
        cond = args.get("conditions")
        if isinstance(table, str) and table.lower() in columns_seen and isinstance(cond, dict):
            for col in string_filter_cols([{"tool_call": {"tool": "condition_filter",
                                                          "arguments": {"conditions": cond}}}]):
                if col.lower() not in columns_seen[table.lower()]:
                    hard.append(
                        f"step {i + 1} filters on missing column {table}.{col} after schema was known"
                    )
    return hard, soft


# ---------------------------------------------------------------- per-trajectory loop ------------
def call_retry(base: str, key: str, model: str, messages: list[dict], tries: int = 4,
               timeout: int = 180):
    """LLM call with backoff — the aggregator drops connections under load ('Remote end closed')."""
    last = None
    for i in range(tries):
        try:
            return call(base, key, model, messages, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise last


def _remember_rejection(rejections: list[dict], *, attempt: int, stage: str, issues: str,
                        candidate_steps: list[dict] | None = None, **meta) -> None:
    """Keep the failed candidate trajectory for human audit.

    `annotation_history` explains what the external model returned and why validation failed;
    `rejected_candidates` preserves the actual spliced/replayed trajectory that was rejected, so we
    can inspect whether the model was wrong or the validator/prompt is too strict.
    """
    item = {"attempt": attempt, "stage": stage, "issues": issues}
    item.update({k: v for k, v in meta.items() if v is not None})
    if candidate_steps is not None:
        item["candidate_steps"] = candidate_steps
    rejections.append(item)
    if len(rejections) > MAX_REJECTED_CANDIDATES:
        del rejections[0]


def enrich_one(traj: dict, base: str, key: str, model: str, max_attempts: int = 3,
               api_timeout: int = 180, api_retries: int = 4,
               allow_errors: bool = True) -> dict:
    h = Harness(db_path(traj["source"]["db_id"]))
    overview = traj["initial_state"]["dataset_overview"]
    domains = column_domains(h, overview, string_filter_cols(traj["steps"]))
    history: list[dict] = []
    rejections: list[dict] = []
    best = None  # perception-only is the reliable floor; errors are added ONLY when they validate,
    feedback = ""  # so the error pass can never drag a trajectory below its perception enrichment.
    for attempt in range(1, max_attempts + 1):
        ctx_domains = dict(domains)
        if feedback:
            ctx_domains["_feedback"] = feedback
        try:
            text, usage = call_retry(
                base, key, model, build_prompt(traj, ctx_domains, allow_errors=allow_errors),
                tries=api_retries, timeout=api_timeout,
            )
        except Exception as e:  # noqa: BLE001
            history.append({"attempt": attempt, "error": f"api: {e}"})
            continue
        ann, rewrites = parse_annotations(text)
        missing_rewrites = [
            i for i in range(len(traj["steps"]))
            if i not in rewrites or not str(rewrites.get(i, {}).get("think", "")).strip()
        ]
        if missing_rewrites:
            feedback = (
                "missing semantic first-person rewrites for backbone steps "
                f"{missing_rewrites}; rewrite every original step and explain exact table/column choices"
            )
            history.append({"attempt": attempt, "perc": 0, "errs": 0,
                            "perception_ok": False, "issues": feedback,
                            "model_output": text, "usage": usage})
            _remember_rejection(
                rejections, attempt=attempt, stage="rewrite_parse", issues=feedback,
                model_output=text, usage=usage,
            )
            continue
        perc = [a for a in ann if a.get("tool") in PERCEPTION]
        errs = [a for a in ann if a.get("tool") == "error"]
        if errs and not allow_errors:
            feedback = (
                "this generation run is perception_only; do not output error/correction insertions"
            )
            history.append({"attempt": attempt, "perc": len(perc), "errs": len(errs),
                            "perception_ok": False, "issues": feedback,
                            "model_output": text, "usage": usage})
            _remember_rejection(
                rejections, attempt=attempt, stage="mode_guard", issues=feedback,
                model_output=text, usage=usage, n_perception=len(perc), n_error=len(errs),
            )
            continue
        l2_ok, l2_issues = check_load_bearing(traj, perc)
        l1p_ok, l1p_err, enr_p = replay_validate(traj, spliced_sequence(traj, perc, rewrites))
        q_hard, q_soft = quality_check(
            enr_p, traj.get("question", ""), traj.get("initial_state", {}).get("dataset_overview")
        )
        if not (l1p_ok and l2_ok and not q_hard):
            feedback = "; ".join(([l1p_err] if not l1p_ok else []) + l2_issues + q_hard) or "no valid perception"
            history.append({"attempt": attempt, "perc": len(perc), "errs": len(errs),
                            "perception_ok": False, "issues": feedback, "soft_issues": q_soft,
                            "model_output": text, "usage": usage})
            _remember_rejection(
                rejections, attempt=attempt, stage="perception_validation", issues=feedback,
                candidate_steps=enr_p, model_output=text, usage=usage, l1_ok=l1p_ok,
                l2_ok=l2_ok, q_ok=not q_hard, n_perception=len(perc), n_error=len(errs),
            )
            continue
        best = {"steps": enr_p, "mode": "perception_only", "n_perception": len(perc), "n_error": 0}
        if not errs:
            history.append({"attempt": attempt, "perc": len(perc), "errs": 0, "ok": True,
                            "model_output": text, "usage": usage})
            break
        l1f_ok, l1f_err, enr_f = replay_validate(traj, spliced_sequence(traj, perc + errs, rewrites))
        qf_hard, qf_soft = quality_check(
            enr_f, traj.get("question", ""), traj.get("initial_state", {}).get("dataset_overview")
        )
        if l1f_ok and not qf_hard:
            best = {"steps": enr_f, "mode": "full", "n_perception": len(perc), "n_error": len(errs)}
            history.append({"attempt": attempt, "perc": len(perc), "errs": len(errs), "ok": True,
                            "model_output": text, "usage": usage})
            break
        feedback = "perception is correct; fix ONLY the error attempts: " + "; ".join(
            ([l1f_err] if not l1f_ok else []) + qf_hard
        )
        history.append({"attempt": attempt, "perc": len(perc), "errs": len(errs),
                        "perception_ok": True, "errors_ok": False, "issues": feedback,
                        "model_output": text, "usage": usage})
        _remember_rejection(
            rejections, attempt=attempt, stage="correction_validation", issues=feedback,
            candidate_steps=enr_f, model_output=text, usage=usage, l1_ok=l1f_ok, q_ok=not qf_hard,
            n_perception=len(perc), n_error=len(errs),
        )
    if best:
        out = copy.deepcopy(traj)
        out["schema_version"] = "v3-enriched"
        out["label_status"] = "verified"
        out["initial_state"]["dataset_overview"] = catalog_snapshot(h)
        out["steps"] = best["steps"]
        out["enrichment"] = {"status": "enriched", "mode": best["mode"],
                             "generator": {"model": model,
                                           "mode_requested": "full" if allow_errors else "perception_only"},
                             "n_perception": best["n_perception"], "n_error": best["n_error"],
                             "rejected_candidates": rejections,
                             "annotation_history": history}
        return out
    _, _, base_steps = replay_validate(traj, spliced_sequence(traj, []))
    out = copy.deepcopy(traj)
    out["schema_version"] = "v3-enriched"
    out["label_status"] = "verified"
    out["initial_state"]["dataset_overview"] = catalog_snapshot(h)
    out["steps"] = base_steps
    out["enrichment"] = {"status": "fallback_skeleton", "mode": "skeleton",
                         "generator": {"model": model,
                                       "mode_requested": "full" if allow_errors else "perception_only"},
                         "n_perception": 0, "n_error": 0,
                         "rejected_candidates": rejections,
                         "annotation_history": history}
    return out


def enrich_one_staged(traj: dict, base: str, key: str, model: str, max_attempts: int = 3,
                      api_timeout: int = 180, api_retries: int = 2) -> dict:
    """Two-pass perception enrichment.

    Pass 1 sees only question + lazy catalog and chooses initial describe_table observations. Pass 2
    sees the real schemas returned by those observations plus the fixed backbone, then writes action
    reasoning and local inspect/read observations. This tests whether observation choice can be made
    from model-visible state instead of from author-only schema knowledge.
    """
    h = Harness(db_path(traj["source"]["db_id"]))
    overview = traj["initial_state"]["dataset_overview"]
    domains = column_domains(h, overview, string_filter_cols(traj["steps"]))
    history: list[dict] = []
    rejections: list[dict] = []
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        try:
            obs_text, obs_usage = call_retry(
                base, key, model, build_observation_prompt(traj, feedback),
                tries=api_retries, timeout=api_timeout,
            )
        except Exception as e:  # noqa: BLE001
            history.append({"attempt": attempt, "stage": "observation", "error": f"api: {e}"})
            continue

        obs_ann = parse_observation_insertions(obs_text)
        describes = []
        for a in obs_ann:
            if a.get("tool") != "describe_table":
                continue
            b = copy.deepcopy(a)
            b["after"] = -1
            describes.append(b)
        if not describes:
            feedback = "stage 1 produced no describe_table observation; choose at least one candidate table from the catalog"
            history.append({"attempt": attempt, "stage": "observation", "ok": False,
                            "issues": feedback, "model_output": obs_text, "usage": obs_usage})
            _remember_rejection(
                rejections, attempt=attempt, stage="observation", issues=feedback,
                model_output=obs_text, usage=obs_usage,
            )
            continue

        ctx_domains = dict(domains)
        if feedback:
            ctx_domains["_feedback"] = feedback
        try:
            act_text, act_usage = call_retry(
                base, key, model, build_staged_action_prompt(traj, ctx_domains, describes),
                tries=api_retries, timeout=api_timeout,
            )
        except Exception as e:  # noqa: BLE001
            history.append({"attempt": attempt, "stage": "action", "error": f"api: {e}",
                            "observation_model_output": obs_text, "observation_usage": obs_usage})
            continue

        act_ann, rewrites = parse_annotations(act_text)
        missing_rewrites = [
            i for i in range(len(traj["steps"]))
            if i not in rewrites or not str(rewrites.get(i, {}).get("think", "")).strip()
        ]
        if missing_rewrites:
            feedback = (
                "stage 2 missing semantic first-person rewrites for backbone steps "
                f"{missing_rewrites}; rewrite every original step"
            )
            history.append({"attempt": attempt, "stage": "action", "ok": False,
                            "issues": feedback, "observation_model_output": obs_text,
                            "observation_usage": obs_usage, "model_output": act_text,
                            "usage": act_usage})
            _remember_rejection(
                rejections, attempt=attempt, stage="action", issues=feedback,
                observation_model_output=obs_text, model_output=act_text,
                observation_usage=obs_usage, usage=act_usage,
            )
            continue

        local_perc = [a for a in act_ann if a.get("tool") in {"inspect_column", "read_subtable"}]
        errs = [a for a in act_ann if a.get("tool") == "error"]
        if errs:
            feedback = "staged_perception is observation-only; do not output error/correction insertions"
            history.append({"attempt": attempt, "stage": "action", "ok": False,
                            "issues": feedback, "observation_model_output": obs_text,
                            "observation_usage": obs_usage, "model_output": act_text,
                            "usage": act_usage})
            _remember_rejection(
                rejections, attempt=attempt, stage="action", issues=feedback,
                observation_model_output=obs_text, model_output=act_text,
                observation_usage=obs_usage, usage=act_usage,
                stage1_describes=len(describes), stage2_perception=len(local_perc),
            )
            continue

        perc = describes + local_perc
        l2_ok, l2_issues = check_load_bearing(traj, perc)
        l1_ok, l1_err, steps = replay_validate(traj, spliced_sequence(traj, perc, rewrites))
        q_hard, q_soft = quality_check(
            steps, traj.get("question", ""), traj.get("initial_state", {}).get("dataset_overview")
        )
        if l1_ok and l2_ok and not q_hard:
            out = copy.deepcopy(traj)
            out["schema_version"] = "v3-enriched"
            out["label_status"] = "verified"
            out["initial_state"]["dataset_overview"] = catalog_snapshot(h)
            out["steps"] = steps
            out["enrichment"] = {"status": "enriched", "mode": "staged_perception",
                                 "generator": {"model": model,
                                               "mode_requested": "staged_perception"},
                                 "n_perception": len(perc), "n_error": 0,
                                 "rejected_candidates": rejections,
                                 "annotation_history": history + [{
                                     "attempt": attempt, "ok": True, "soft_issues": q_soft,
                                     "observation_model_output": obs_text,
                                     "observation_usage": obs_usage,
                                     "model_output": act_text, "usage": act_usage,
                                     "stage1_describes": len(describes),
                                     "stage2_perception": len(local_perc),
                                 }]}
            return out

        feedback = "; ".join(([l1_err] if not l1_ok else []) +
                             ([] if l2_ok else l2_issues) +
                             q_hard) or "staged enrichment failed validation"
        history.append({"attempt": attempt, "ok": False, "issues": feedback, "soft_issues": q_soft,
                        "observation_model_output": obs_text, "observation_usage": obs_usage,
                        "model_output": act_text, "usage": act_usage,
                        "stage1_describes": len(describes), "stage2_perception": len(local_perc)})
        _remember_rejection(
            rejections, attempt=attempt, stage="validation", issues=feedback,
            candidate_steps=steps, observation_model_output=obs_text, model_output=act_text,
            observation_usage=obs_usage, usage=act_usage, l1_ok=l1_ok, l2_ok=l2_ok, q_ok=not q_hard,
            stage1_describes=len(describes), stage2_perception=len(local_perc),
        )

    _, _, base_steps = replay_validate(traj, spliced_sequence(traj, []))
    out = copy.deepcopy(traj)
    out["schema_version"] = "v3-enriched"
    out["label_status"] = "verified"
    out["initial_state"]["dataset_overview"] = catalog_snapshot(h)
    out["steps"] = base_steps
    out["enrichment"] = {"status": "fallback_skeleton", "mode": "skeleton",
                         "generator": {"model": model, "mode_requested": "staged_perception"},
                         "n_perception": 0, "n_error": 0,
                         "rejected_candidates": rejections,
                         "annotation_history": history}
    return out


def enrich_one_rewrite_only(traj: dict, base: str, key: str, model: str,
                            api_timeout: int = 180, api_retries: int = 1) -> dict:
    """Lightweight smoke path: deterministic global/local observation scaffold, external LLM only
    rewrites semantic reasoning for backbone actions."""
    h = Harness(db_path(traj["source"]["db_id"]))
    history: list[dict] = []
    rejections: list[dict] = []
    try:
        text, usage = call_retry(
            base, key, model, build_rewrite_prompt(traj),
            tries=api_retries, timeout=api_timeout,
        )
        rewrites = parse_rewrite_array(text, len(traj["steps"]))
    except Exception as e:  # noqa: BLE001
        rewrites = {}
        history.append({"attempt": 1, "error": f"api: {type(e).__name__}: {e}"})
    if not rewrites:
        _, _, base_steps = replay_validate(traj, spliced_sequence(traj, []))
        out = copy.deepcopy(traj)
        out["schema_version"] = "v3-enriched"
        out["label_status"] = "verified"
        out["initial_state"]["dataset_overview"] = catalog_snapshot(h)
        out["steps"] = base_steps
        out["enrichment"] = {"status": "fallback_skeleton", "mode": "skeleton",
                             "generator": {"model": model,
                                           "mode_requested": "semantic_rewrite"},
                             "n_perception": 0, "n_error": 0,
                             "rejected_candidates": rejections,
                             "annotation_history": history}
        return out

    insertions = deterministic_perception_insertions(traj)
    ok, err, steps = replay_validate(traj, spliced_sequence(traj, insertions, rewrites))
    qhard, qsoft = quality_check(
        steps, traj.get("question", ""), traj.get("initial_state", {}).get("dataset_overview")
    )
    out = copy.deepcopy(traj)
    out["schema_version"] = "v3-enriched"
    out["label_status"] = "verified"
    out["initial_state"]["dataset_overview"] = catalog_snapshot(h)
    if ok and not qhard:
        out["steps"] = steps
        out["enrichment"] = {"status": "enriched", "mode": "semantic_rewrite",
                             "generator": {"model": model,
                                           "mode_requested": "semantic_rewrite"},
                             "n_perception": len(insertions), "n_error": 0,
                             "rejected_candidates": rejections,
                             "annotation_history": [{"attempt": 1, "ok": True,
                                                      "insertions": len(insertions),
                                                      "model_output": text, "usage": usage}]}
    else:
        issues = "; ".join(([err] if not ok else []) + qhard)
        _remember_rejection(
            rejections, attempt=1, stage="semantic_rewrite_validation", issues=issues,
            candidate_steps=steps, model_output=text, usage=usage, ok=ok, q_ok=not qhard,
            n_perception=len(insertions), n_error=0,
        )
        _, _, base_steps = replay_validate(traj, spliced_sequence(traj, []))
        out["steps"] = base_steps
        out["enrichment"] = {"status": "fallback_skeleton", "mode": "skeleton",
                             "generator": {"model": model,
                                           "mode_requested": "semantic_rewrite"},
                             "n_perception": 0, "n_error": 0,
                             "rejected_candidates": rejections,
                             "annotation_history": [{"attempt": 1, "ok": False,
                                                      "issues": issues,
                                                      "model_output": text, "usage": usage}]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=os.path.join(ROOT, "data", "trajectories", "subset_180.ids.json"))
    ap.add_argument("--which", default="smoke", choices=["smoke", "subset", "probe"])
    ap.add_argument("--subset-file", default=os.path.join(ROOT, "data", "trajectories", "subset_180.jsonl"))
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "trajectories", "smoke_enriched.jsonl"))
    ap.add_argument("--mode", choices=["full", "perception_only", "semantic_rewrite", "staged_perception"],
                    default="full",
                    help=("full asks the LLM for rewrites+perception+optional corrections; "
                          "perception_only forbids correction/error insertions; semantic_rewrite "
                          "uses a deterministic observation scaffold and asks only for semantic thinks; "
                          "staged_perception first chooses describe_table from catalog-only state"))
    ap.add_argument("--limit", type=int, default=0,
                    help="optional cap for quick smoke generation")
    ap.add_argument("--api-timeout", type=int, default=180,
                    help="seconds per external LLM request before retry/fallback")
    ap.add_argument("--api-retries", type=int, default=4,
                    help="external LLM retries per annotation attempt")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="annotation attempts per trajectory after validation feedback")
    args = ap.parse_args()

    key, base = load_api()
    wanted = set(json.load(open(args.ids))[args.which])
    trajs = [normalize_legacy_memory(json.loads(l)) for l in open(args.subset_file) if l.strip()]
    trajs = [t for t in trajs if t["trajectory_id"] in wanted]
    if args.limit:
        trajs = trajs[: args.limit]
    print(f"enriching {len(trajs)} ({args.which}) with {args.model}\n")

    results = []
    with open(args.out, "w", encoding="utf-8") as f:
        for t in trajs:
            try:
                if args.mode == "semantic_rewrite":
                    r = enrich_one_rewrite_only(
                        t, base, key, args.model,
                        api_timeout=args.api_timeout, api_retries=args.api_retries,
                    )
                elif args.mode == "staged_perception":
                    r = enrich_one_staged(
                        t, base, key, args.model,
                        max_attempts=args.max_attempts,
                        api_timeout=args.api_timeout, api_retries=args.api_retries,
                    )
                else:
                    r = enrich_one(
                        t, base, key, args.model,
                        max_attempts=args.max_attempts,
                        api_timeout=args.api_timeout, api_retries=args.api_retries,
                        allow_errors=args.mode == "full",
                    )
            except Exception as e:  # noqa: BLE001 — one bad trajectory must never abort the whole batch
                print(f"  {t['trajectory_id']:<20} ERROR: {type(e).__name__}: {e}")
                continue
            results.append(r)
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            f.flush()
            meta = r["enrichment"]
            print(f"  {r['trajectory_id']:<20} len={len(r['steps']):<3} +perc={meta['n_perception']:<2} "
                  f"+err={meta['n_error']:<2} {meta['mode']:<16} db={r['source']['db_id']}")
    enr = sum(1 for r in results if r["enrichment"]["status"] == "enriched")
    full = sum(1 for r in results if r["enrichment"].get("mode") == "full")
    print(f"\nenriched {enr}/{len(results)} (with errors {full})  fallback {len(results)-enr}\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
