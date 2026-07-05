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
  6. on any L1/L2/style failure, feed the SPECIFIC reason back and regenerate (<=10 attempts); if it never
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
import collections
import copy
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "harness"))
sys.path.insert(0, os.path.join(ROOT, "src", "eval"))
sys.path.insert(0, HERE)

from executor import Harness                                              # noqa: E402
from environment_state import EnvironmentState                            # noqa: E402
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
    "the model", "the agent", "the assistant", "might guess",
    "would guess", "without first checking",
)
FIRST_PERSON = re.compile(r"\b(I|my|me|I'll|I'm|I will|I need|I see|I should)\b", re.I)
CONFIRM_WORDS = re.compile(r"\b(confirm(?:ed|s)?|verif(?:y|ied|ies)|exists?|present)\b", re.I)
STALE_OBSERVATION_WORDING = re.compile(
    r"\b(i\s+)?(should|need to|must|have to|will)\s+(first\s+)?(inspect|read|check|verify)\b|"
    r"\blet me\s+(inspect|read|check|verify)\b|"
    r"\bbefore\s+\w+ing\b[^.]{0,140}\b(i\s+)?(should|need to|must|have to)\s+"
    r"(first\s+)?(inspect|read|check|verify)\b",
    re.I,
)
FINAL_ANSWER_LEAK = re.compile(
    r"\b(verified|expected|gold|correct)\s+answer\b|"
    r"\banswer\s+(matches|is)\s+the\s+(expected|gold|correct)\b",
    re.I,
)
TEMPLATE_THINK_PATTERNS = (
    r"^I have the needed schema and prior observation context, so I now apply\b",
    r"^The current intermediate table already contains the requested fields\. I now project\b",
    r"^I now compute\b",
    r"^I now group\b",
    r"^I now sort/select\b",
    r"^I now join\b",
    r"^I now apply\b",
    r"^I now call\b",
    r"\bbecause the prior observations and intermediate results provide the information this step needs\b",
    r"\bto keep the rows required by the question\b",
    r"\bso the output matches the columns asked for in the question\b",
    r"\bbecause the question requires this scalar result from the already prepared table\b",
    r"\bbecause the question requires deduplicated or grouped evidence\b",
    r"\bbecause the question asks for an ordered or extreme result\b",
    r"\bso the columns needed by the question are in one intermediate table\b",
    r"\bbecause the question requires combining two derived result sets\b",
    r"\bI read `[^`]+` locally to verify the rows before answering\b",
)
TOOL_ACTION_CUES = {
    "condition_filter": (r"\bfilter\b", r"\bwhere\b", r"\bcondition_filter\b", r"\bkeep only\b"),
    "project": (
        r"\bproject\b",
        r"\bselect\b",
        r"\bextract\b",
        r"\bkeep just\b",
        r"\bkeep only (?:the|these|those)?\s*(?:fields|columns)\b",
    ),
    "join_tables": (r"\bjoin\b", r"\bconnect\b", r"\blink\b"),
    "group_aggregate": (r"\bgroup\b", r"\bdeduplicate\b", r"\bgroup_aggregate\b"),
    "aggregate": (r"\baggregate\b", r"\bcompute\b", r"\bcount\b", r"\bsum\b", r"\baverage\b", r"\bminimum\b", r"\bmaximum\b"),
    "extreme_value_select": (r"\bsort\b", r"\border\b", r"\btop\b", r"\bhighest\b", r"\blowest\b", r"\bsmallest\b", r"\blargest\b"),
    "set_op": (r"\bintersect\b", r"\bintersection\b", r"\bunion\b", r"\bexcept\b", r"\bset\b"),
    "answer_from_context": (r"\banswer\b", r"\bevidence\b", r"\bfinal\b"),
}
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


def review_path_for(path: str) -> str:
    if path.endswith(".jsonl") and not path.endswith("_review.jsonl"):
        return path[:-6] + "_review.jsonl"
    return path


def new_ctx() -> dict:
    return {"history": {}, "handle_to_step": {}, "environment": EnvironmentState()}


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

    if tool == "plan":
        output = ctx.setdefault("environment", EnvironmentState()).apply_plan_ops(
            args.get("ops"), step_id, ctx["history"]
        )
        ctx["history"][step_id] = {"tool": tool, "arguments": args, "output": output}
        return output, None

    if tool in PERCEPTION:
        out = getattr(h, tool)(**args)
        output = out if isinstance(out, dict) else {"rows": [list(r) for r in out], "row_count": len(out)}
        ctx["history"][step_id] = {"tool": tool, "arguments": args, "output": output}
        ctx.setdefault("environment", EnvironmentState()).apply_tool_result(tool, args, output, step_id)
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
    ctx.setdefault("environment", EnvironmentState()).apply_tool_result(tool, args, output, step_id)
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


def _join_on_edges(args: dict) -> list[dict]:
    """Flatten a join_tables `on` into one list of {left,right} edges, across the N-way form
    (on = a list of per-fold edge-lists) and the legacy 2-table form (on = a flat edge-list)."""
    edges: list[dict] = []
    for entry in args.get("on", []) or []:
        if isinstance(entry, list):
            edges.extend(e for e in entry if isinstance(e, dict))
        elif isinstance(entry, dict):
            edges.append(entry)
    return edges


def _join_table_refs(args: dict) -> list[str]:
    """Input table refs of a join_tables call, across the N-way form (`tables`) and the legacy
    2-table form (`left`/`right`)."""
    if isinstance(args.get("tables"), list):
        return [t for t in args["tables"] if isinstance(t, str)]
    return [args[k] for k in ("left", "right") if isinstance(args.get(k), str)]


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
            out.update(expression_columns(c["column"]))
        if isinstance(c.get("column_value"), str):
            out.update(expression_columns(c["column_value"]))

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
        for item in _join_on_edges(args):
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
        for item in _join_on_edges(args):
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
    "- Prefer concrete, observation-grounded thoughts over stock templates. Instead of a generic "
    "'I now apply/compute/group/join...' sentence, name the observed table/column/result and the "
    "question phrase that makes this operation useful.\n"
    "- Mention exact column names when choosing them, e.g. `Song_Name` because the question asks for "
    "song names, `Song_release_year` because it asks for release years, `Age` because it asks for "
    "the youngest singer.\n"
    "- Exact means the literal schema/tool identifier with underscores and prefixes stripped when "
    "needed. If the tool uses `T2__Time_of_day`, your think must explicitly say `Time_of_day`; "
    "phrases like 'broadcast time' or 'time column' are NOT acceptable substitutes.\n"
    "- A good `describe_table` reason is logical without leaking schema: explain which question "
    "concept points to which candidate table(s), what information is missing, and which future "
    "operation the schema will support.\n"
    "- Temporal consistency matters: if a prior observation already happened, the next action must "
    "use past-tense grounded wording such as 'I have inspected X, so I can filter now'. Do NOT write "
    "'I should inspect/read first' inside a non-observation action; insert the observation before the "
    "action instead.\n"
    "- The final answer think should cite the evidence table or scalar result. Do NOT say 'the "
    "verified answer is', 'the expected answer is', or otherwise speak like an annotator revealing "
    "gold labels.\n\n"
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
    "observation exposes them.\n"
    "- BAD action after an inspect step: 'I should inspect the status values before filtering.' This "
    "is stale because the inspect step already happened.\n"
    "- GOOD action after an inspect step: 'I have inspected the status values and confirmed the "
    "literal, so I can now filter on the exact status column.'\n"
    "- OK but weak action: 'I now compute count over * from filter_001 because the question requires "
    "this scalar result.'\n"
    "- BETTER action: '`filter_001` is already restricted to the qualifying rows, and the question "
    "asks how many such rows there are, so counting `*` on that table gives the requested scalar.'\n"
    "- OK but weak join: 'I now join students and enrollment on student_id.'\n"
    "- BETTER join: 'The question needs student attributes together with enrollment records, and "
    "`student_id` is the observed key connecting those two tables.'\n"
    "- BAD final answer: 'The verified answer is X.'\n"
    "- GOOD final answer: 'The final evidence table contains the requested rows, so I answer from "
    "that table.'\n\n"
    "Perception tools:\n"
    "- describe_table {\"tables\":[...]} : acquire columns/types/keys. REQUIRED before the first time "
    "the trajectory operates on a table, since the catalog carries no columns. Put all source tables "
    "you want to inspect at the same point into ONE describe_table call; do not emit one "
    "describe_table step per table.\n"
    "- inspect_column {\"table\":t,\"column\":c} : see a column's real values. REQUIRED before a filter "
    "comparing that column to a STRING literal, so the literal is grounded, not guessed. Use the "
    "SOURCE table + base column name even if the backbone filters it post-join under a prefix. "
    "GROUNDING HONESTY: only claim what the output returned. If frequent_values LISTS the literal, you "
    "may say you saw it. If the output is truncated (truncated=true) and does NOT list it, do NOT say "
    "you confirmed/verified the value exists — say you filter on the value the question asks for and the "
    "condition_filter row count is what validates it. Never fabricate a confirmation the output did not give.\n"
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
    "CRITICAL — no exact schema identifiers: NEVER write an exact column name here, not even as a "
    "guess or example. Do NOT write 'a column like temporary_acting', 'e.g. COMMISSION_PCT', or 'the "
    "order_status_code field'. Those are hidden schema names the agent cannot know until "
    "describe_table returns them — naming them (in `think` OR in any `rationale` field) is a hard "
    "error, even hedged with 'like', 'such as', or 'e.g.'. Refer to columns ONLY by natural question "
    "concepts: 'the acting-status column', 'a commission field', 'the order date'. Table names and "
    "foreign-key columns shown in the catalog relations are fine; everything else must stay a natural "
    "concept. A good reason says: which question concept suggests this table, what information is "
    "missing, and what kind of future operation the schema will support.\n\n"
    "Output ONLY JSON: {\"insertions\": [ ... ]}. Each insertion must be a describe_table step with "
    "`after:-1`, arguments {\"tables\":[...]}, first-person `think`, and `rationale` with exactly "
    "these non-empty fields: question_cue, observed_evidence, decision, supports_next_step. Every "
    "rationale field must also obey the no-exact-identifier rule above. If several tables should be "
    "described initially, include all of them in that single insertion's `tables` list; do NOT output "
    "multiple separate describe_table insertions for individual tables.\n\n"
    "Exploratory observations are allowed: if the question asks for an output entity such as "
    "'all info of students', it is reasonable to describe the entity table even if a later minimal "
    "gold plan might not consume it directly. Still, also include the tables needed to resolve the "
    "question conditions, because later actions may only use schemas that have actually been "
    "observed.\n\n"
    "Bad: 'I inspect management to see if it has a column like temporary_acting.'\n"
    "Bad: 'I describe employees to find the commission column, e.g. COMMISSION_PCT.'\n"
    "Good: 'I describe management because the question asks about acting statuses, but I do not yet "
    "know the exact status column; observing the schema will reveal it.'\n"
    "Good: 'I describe employees because the question is about employees with a commission; the exact "
    "commission and department columns stay unknown until I observe the schema.'"
)


def build_observation_prompt(traj: dict, feedback: str = "") -> list[dict]:
    # the agent's true opening state is the LAZY catalog (table names + row counts + FK relations,
    # NO columns). The input trajectory's stored overview still carries full columns, so deriving the
    # lazy catalog here is what actually hides schema from stage 1 — feeding the stored overview was
    # the real source of "schema leak" in describe-step thinks.
    ov = catalog_snapshot(Harness(db_path(traj["source"]["db_id"])))
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
        if isinstance(ins.get("tool_output"), dict):
            outs.append({"arguments": ins.get("arguments", {}), "output": ins["tool_output"]})
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
        f"{json.dumps(catalog_snapshot(Harness(db_path(traj['source']['db_id']))), ensure_ascii=False)}\n\n"
        f"OBSERVED SCHEMA FROM PRIOR describe_table STEPS:\n"
        f"{json.dumps(observed_schema_outputs(traj, describe_insertions), ensure_ascii=False)}\n\n"
        f"STRING-FILTER COLUMN DOMAINS (available only after the relevant schema column is observed "
        f"and inspect_column is called):\n{json.dumps(domains, ensure_ascii=False)}\n\n"
        f"RELATIONAL BACKBONE (fixed order; rewrite every think and insert only local observations "
        f"around these actions):\n{json.dumps(skeleton_view(traj), ensure_ascii=False)}\n\n"
        "Return JSON with `rewrites` for every backbone step and optional `insertions`. In this staged "
        "second pass, do NOT output describe_table insertions. If the backbone already contains a "
        "describe_table step, treat that schema observation as already present in the trajectory. "
        "Only inspect_column/read_subtable are allowed as new observations."
    )
    system = SYS + (
        "\n\nCURRENT RUN MODE: STAGED ACTION PASS. The schema observations listed above have already "
        "been chosen from the catalog or already exist in the backbone trajectory. Do NOT add more "
        "describe_table steps. You may use exact column "
        "names only for tables whose schema appears in OBSERVED SCHEMA; otherwise the trajectory will "
        "be rejected as relying on unobserved schema."
    )
    if domains.get("_feedback"):
        user += f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {domains['_feedback']}\nFix and re-output."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


STATEFUL_STEP_SYS = (
    "Role: you are the table-tool agent at the current turn. You are not an outside annotator and "
    "you must not describe the whole future trajectory. You see only the question, the lazy catalog, "
    "the previous tool calls/observations, and ONE target tool call that the verified backbone wants "
    "to execute now.\n\n"
    "Your job for this turn:\n"
    "1. Decide whether the current visible state is sufficient to justify the target tool call.\n"
    "2. If it is sufficient, write the target step's first-person `<think>` explaining why this exact "
    "tool, table, column, handle, or predicate is now justified by the visible state.\n"
    "3. If there is a gap, insert the smallest necessary read-only observation BEFORE the target "
    "step. Use only describe_table, inspect_column, or read_subtable. The observation must bridge a "
    "specific missing precondition for the target action, not be decorative.\n"
    "4. After any inserted observation, still write the target step's first-person reason as if the "
    "agent has now seen that observation.\n\n"
    "State boundary:\n"
    "- Before a describe_table observation, do not name hidden exact columns. Use natural concepts "
    "from the question instead.\n"
    "- After a schema observation is in PREVIOUS TOOL HISTORY, target action thinks must name the "
    "exact columns used by the target tool call.\n"
    "- If the target filters a string literal and the column's value domain has not been inspected, "
    "insert inspect_column first.\n"
    "- If the target depends on a derived handle's rows and those rows are not visible, insert "
    "read_subtable first when that row-level evidence is needed.\n"
    "- Before answer_from_context, if the answer contains concrete row values from a table handle "
    "and those rows are not already visible in PREVIOUS TOOL HISTORY, insert read_subtable on that "
    "evidence table first. A table handle plus row_count is not enough to justify exact answer "
    "values. Set read_subtable.limit high enough to cover the answer rows when the table row_count is "
    "known (up to the answer cap of 50); do not rely on the default limit for a multi-row answer.\n"
    "- Existing previous observations count. Do not repeat an observation already present in the "
    "history.\n\n"
    "OBSERVATION TOOL ARGUMENTS — use these EXACT argument keys. The harness does NOT tolerate other "
    "key names; a wrong key fails the turn:\n"
    "- describe_table: {\"tables\": [\"TableA\", \"TableB\"]}  — the key is `tables` and its value is a "
    "LIST of table-name strings. Do NOT use `table`, `table_name`, or a bare string.\n"
    "- inspect_column: {\"table\": \"TableA\", \"column\": \"ColumnX\"}  — shows a column's value domain "
    "(distinct count, most-frequent values, whether it is truncated).\n"
    "- read_subtable: {\"table\": \"TableA\", \"columns\": [\"ColumnX\"], \"limit\": 20}  — `columns` "
    "(a list) and `limit` (an int) are optional; `table` is required.\n\n"
    "GROUNDING HONESTY — the <think> must only claim what the observation actually returned:\n"
    "- If inspect_column's frequent_values LISTS the literal you filter on, you may say you saw it there.\n"
    "- If the output is truncated (truncated=true) and does NOT list that literal, DO NOT say you "
    "confirmed / verified that the value exists — the observation did not show it. Instead reason that "
    "you filter on the value the question asks for and the condition_filter result (its row count) is "
    "what validates whether the value matched. Never fabricate a confirmation the output did not give.\n\n"
    "Output ONLY JSON with this schema:\n"
    "{\n"
    "  \"insertions\": [\n"
    "    {\"tool\": \"describe_table|inspect_column|read_subtable\", \"arguments\": {...}, "
    "\"think\": \"first-person reason\", \"rationale\": {\"question_cue\":..., "
    "\"observed_evidence\":..., \"decision\":..., \"supports_next_step\":...}}\n"
    "  ],\n"
    "  \"target\": {\"think\": \"first-person reason for executing the given target call now\", "
    "\"rationale\": {\"question_cue\":..., \"observed_evidence\":..., \"decision\":..., "
    "\"supports_next_step\":...}}\n"
    "}\n\n"
    "Rules:\n"
    "- Never change the target tool call or its arguments.\n"
    "- Do not add correction/error steps in this mode.\n"
    "- Each think must read like current-moment reasoning from observations to decision, not a "
    "post-hoc explanation of a known full path. A plain 'I now apply/compute/join' sentence is "
    "acceptable only if it is tied to the concrete observed state; prefer: '`filter_003` already "
    "contains the qualifying rows, so counting `*` answers the how-many question.'\n"
    "- Every rationale field must be non-empty."
)


def _history_for_prompt(accepted_steps: list[dict], max_steps: int = 14) -> list[dict]:
    """Compact visible transcript for the stateful per-step prompt."""
    items = []
    for step in accepted_steps[-max_steps:]:
        call = step.get("tool_call") or {}
        out = step.get("tool_output")
        if isinstance(out, dict):
            visible_out = copy.deepcopy(out)
            if "rows" in visible_out and isinstance(visible_out["rows"], list):
                visible_out["rows"] = visible_out["rows"][:5]
            if "result_sample" in visible_out and isinstance(visible_out["result_sample"], list):
                visible_out["result_sample"] = visible_out["result_sample"][:5]
        else:
            visible_out = out
        items.append({
            "step_id": step.get("step_id"),
            "tool": call.get("tool"),
            "arguments": call.get("arguments"),
            "output": visible_out,
        })
    return items


def build_stateful_step_prompt(
    traj: dict,
    accepted_steps: list[dict],
    target_index: int,
    target_step: dict,
    target_args: dict,
    feedback: str = "",
) -> list[dict]:
    h = Harness(db_path(traj["source"]["db_id"]))
    call = target_step["tool_call"]
    user = (
        f"QUESTION:\n{traj['question']}\n\n"
        f"INITIAL LAZY CATALOG:\n"
        f"{json.dumps(catalog_snapshot(h), ensure_ascii=False)}\n\n"
        f"PREVIOUS TOOL HISTORY VISIBLE TO THE AGENT:\n"
        f"{json.dumps(_history_for_prompt(accepted_steps), ensure_ascii=False)}\n\n"
        f"CURRENT BACKBONE STEP INDEX: {target_index}\n"
        f"TARGET TOOL CALL TO JUSTIFY NOW (do not change this call):\n"
        f"{json.dumps({'tool': call['tool'], 'arguments': target_args}, ensure_ascii=False)}\n\n"
        "Return the JSON object for this single turn."
    )
    if feedback:
        user += f"\n\nPREVIOUS ATTEMPT FAILED VALIDATION: {feedback}\nFix this single turn only."
    return [{"role": "system", "content": STATEFUL_STEP_SYS}, {"role": "user", "content": user}]


def parse_stateful_step(text: str) -> tuple[list[dict], dict]:
    obj_match = re.search(r"\{.*\}", text, re.S)
    if not obj_match:
        return [], {}
    try:
        payload = json.loads(obj_match.group(0))
    except json.JSONDecodeError:
        return [], {}
    if not isinstance(payload, dict):
        return [], {}
    insertions = []
    for item in payload.get("insertions", []) if isinstance(payload.get("insertions"), list) else []:
        if isinstance(item, dict) and item.get("tool") in PERCEPTION:
            insertions.append(item)
    target = payload.get("target") if isinstance(payload.get("target"), dict) else {}
    return insertions, target


def _seen_perception_keys(accepted_steps: list[dict]) -> set[tuple]:
    seen = set()
    for step in accepted_steps:
        call = step.get("tool_call") or {}
        key = _canonical_perception_key(call.get("tool"), call.get("arguments") or {})
        if key:
            seen.add(key)
    return seen


def _dedupe_against_state(insertions: list[dict], accepted_steps: list[dict]) -> list[dict]:
    seen = _seen_perception_keys(accepted_steps)
    out = []
    for ins in insertions:
        key = _canonical_perception_key(ins.get("tool"), ins.get("arguments") or {})
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(ins)
    return out


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
                "think": (
                    f"The answer will cite `{produced}`, so I inspect its rows to ground the final "
                    "response in visible evidence."
                ),
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


def _canonical_perception_key(tool: str, args: dict) -> tuple | None:
    if tool not in PERCEPTION or not isinstance(args, dict):
        return None
    if tool == "describe_table":
        tables = tuple(sorted(str(t) for t in (args.get("tables") or [])))
        return (tool, tables)
    if tool == "inspect_column":
        return (tool, str(args.get("table", "")), _base_col(str(args.get("column", ""))).lower())
    if tool == "read_subtable":
        columns = args.get("columns")
        if isinstance(columns, list):
            columns = tuple(columns)
        return (tool, str(args.get("table", "")), args.get("limit"), columns)
    return (tool, json.dumps(args, sort_keys=True, ensure_ascii=False, default=str))


def backbone_describe_observations(traj: dict) -> list[dict]:
    """Existing v3 skeletons already contain initial describe_table steps.

    The enrichment prompt needs those schemas as observed context, but they must not be re-inserted
    as extra perception steps. Returning the actual backbone observation keeps the staged action pass
    state-faithful without creating duplicate reads.
    """
    out: list[dict] = []
    for s in traj.get("steps", []):
        tc = s.get("tool_call") or {}
        if tc.get("tool") != "describe_table":
            continue
        out.append({
            "tool": "describe_table",
            "arguments": tc.get("arguments", {}),
            "tool_output": s.get("tool_output", {}),
        })
    return out


def dedupe_perception_insertions(traj: dict, insertions: list[dict]) -> list[dict]:
    """Drop perception insertions that duplicate observations already present in the skeleton.

    Current v3 skeletons are not pure relational backbones: the emitter has already injected
    describe_table/inspect_column/read_subtable observations. Without this guard, the external LLM
    can add a second identical describe_table at the start, teaching a bad repeated-read habit.
    """
    seen = set()
    for s in traj.get("steps", []):
        tc = s.get("tool_call") or {}
        key = _canonical_perception_key(tc.get("tool"), tc.get("arguments") or {})
        if key:
            seen.add(key)

    out = []
    for ins in insertions:
        key = _canonical_perception_key(ins.get("tool"), ins.get("arguments") or {})
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        out.append(ins)
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

    def describe_tables(args: dict) -> list[str]:
        tables = args.get("tables") if isinstance(args, dict) else []
        if isinstance(tables, str):
            tables = [tables]
        return [str(t) for t in tables or [] if str(t)]

    def append_seq_step(step: dict) -> int:
        """Append one logical step, merging consecutive describe_table calls into one survey turn.

        describe_table already accepts a list of tables. Keeping one table per turn teaches a noisy
        ritual and bloats trajectories; consecutive schema-survey calls should be represented as one
        observation over all relevant tables.
        """
        if step.get("tool") == "describe_table" and seq and seq[-1].get("tool") == "describe_table":
            prev = seq[-1]
            seen = set()
            merged: list[str] = []
            for table in describe_tables(prev.get("arguments") or {}) + describe_tables(step.get("arguments") or {}):
                key = table.lower()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(table)
            prev["arguments"] = {"tables": merged}
            prev["reason"] = _perception_repair_text("describe_table", prev["arguments"], traj.get("question", ""))
            prev["rationale"] = _perception_repair_rationale("describe_table", prev["arguments"], traj.get("question", ""))
            if step.get("backbone"):
                prev["backbone"] = True
            return len(seq)
        seq.append(step)
        return len(seq)

    def emit_backbone(i: int, s):
        rewrite = rewrites.get(i, {})
        think = rewrite.get("think", s.get("think", "")) if isinstance(rewrite, dict) else str(rewrite)
        rationale = rewrite.get("rationale") if isinstance(rewrite, dict) else None
        if pending_recovery:
            think = " ".join(pending_recovery) + " " + think
            pending_recovery.clear()
        tool = s["tool_call"]["tool"]
        args = _strip_answer_memory_args(tool, s["tool_call"]["arguments"])
        new_index = append_seq_step({"tool": tool, "arguments": args,
                                     "reason": think, "rationale": rationale, "backbone": True})
        old_to_new[s["step_id"]] = f"step_{new_index}"

    def emit_ann(a):
        if a.get("tool") == "error":
            w = a["wrong"]
            append_seq_step({"tool": w.get("tool"), "arguments": w.get("arguments", {}),
                             "reason": a.get("think", ""), "rationale": a.get("rationale"),
                             "error_attempt": True})
            if a.get("recovery_think"):
                pending_recovery.append(a["recovery_think"])
        else:
            append_seq_step({"tool": a["tool"], "arguments": a.get("arguments", {}),
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


def _templated_think_style_issues(text: str, label: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    issues = []
    for pattern in TEMPLATE_THINK_PATTERNS:
        if re.search(pattern, text.strip(), re.I):
            issues.append(f"{label} may be overly templated; prefer a more observation-specific reason")
            break
    return issues


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
    # a single-word column (no underscore compound) is a natural concept the agent can reasonably
    # name before observing the schema (age, country, party, beds); real pre-describe leaks are
    # multi-word exact identifiers (order_status_code, temporary_acting) that keep their underscores.
    if "_" not in identifier and not re.search(r"[a-z][A-Z]", identifier):
        return True
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
    catalog_tables = {
        str(table.get("table_name", "")).lower()
        for table in (overview or {}).get("tables", [])
        if isinstance(table, dict)
    }
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
            if name.lower() in catalog_tables:
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


def _grounded_columns(enriched_steps: list[dict], upto: int,
                      overview: dict | None = None) -> dict[str, set[str]]:
    """Source-table columns the agent has legitimately seen before step `upto`, from ANY grounding
    source — not just describe_table. describe_table is one path; the model also grounds via:
    - the initial catalog's `relations` (foreign keys expose the join-key columns of each table), and
    - inspect_column (the inspected column's existence + value domain).
    Keyed by lowercased table name -> set of lowercased base column names. Used to decide whether an
    action touches a column it has never observed (a real grounding gap) versus one already exposed by
    the catalog FK graph or an inspect (which the old describe_table-only check mis-flagged)."""
    grounded: dict[str, set[str]] = {t: set(cols) for t, cols in _described_columns(enriched_steps, upto).items()}

    def add(table: str, col: str) -> None:
        if table and col:
            grounded.setdefault(table.lower(), set()).add(_base_col(col).lower())

    for rel in (overview or {}).get("relations", []) or []:
        for side in ("from", "to"):
            ref = rel.get(side, "")
            if isinstance(ref, str) and "." in ref:
                table, col = ref.split(".", 1)
                add(table, col)
    for s in enriched_steps[:upto]:
        call = s.get("tool_call") or {}
        if call.get("tool") == "inspect_column":
            a = call.get("arguments") or {}
            if isinstance(a.get("table"), str) and isinstance(a.get("column"), str):
                add(a["table"], a["column"])
    return grounded


def _source_columns_used(tool: str, args: dict, table: str) -> list[str]:
    """Base columns of source `table` that this action actually reads. For a join, only the `on` keys
    on the side that is `table` (the accumulated left operand may be a derived handle). For a
    single-table op, every column term belongs to that one source table."""
    if tool == "join_tables":
        cols: list[str] = []
        tables = args.get("tables")
        if isinstance(tables, list):                        # N-way form: on[k] joins tables[k+1]
            for k, edges in enumerate(args.get("on") or []):
                for e in (edges or []):
                    if k + 1 < len(tables) and tables[k + 1] == table and e.get("right"):
                        cols.append(_base_col(e["right"]))   # right key belongs to the newly-joined table
                    if k == 0 and tables and tables[0] == table and e.get("left"):
                        cols.append(_base_col(e["left"]))    # first fold's left key belongs to tables[0]
            return [c for c in cols if c]
        for pair in args.get("on", []) or []:               # legacy 2-table form
            if args.get("right") == table and pair.get("right"):
                cols.append(_base_col(pair["right"]))
            if args.get("left") == table and pair.get("left"):
                cols.append(_base_col(pair["left"]))
        return [c for c in cols if c]
    if tool == "group_aggregate":
        cols = [_base_col(c) for c in args.get("group_by", [])]
        for agg in args.get("aggregations", []):
            col = agg.get("column")
            if col and col != "*":
                cols.append(_base_col(col))
        return [c for c in cols if c]
    return [_base_col(c) for c in expected_column_terms(tool, args) if _base_col(c)]


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
    if tool == "join_tables" and isinstance(args.get("tables"), list):   # N-way join
        return [t for t in args["tables"] if isinstance(t, str)]
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


def _string_literal_pairs(cond):
    """Yield (base_column, literal) for each string-literal =/contains/like filter predicate."""
    if isinstance(cond, list):
        for c in cond:
            yield from _string_literal_pairs(c)
    elif isinstance(cond, dict):
        for k in ("and", "or"):
            for c in cond.get(k, []):
                yield from _string_literal_pairs(c)
        if "not" in cond:
            yield from _string_literal_pairs(cond["not"])
        if cond.get("op") in STR_OPS and isinstance(cond.get("value"), str):
            yield (_base_col(cond.get("column", "")), cond["value"])


def _unsupported_confirmation_issues(step, enriched_steps, upto, label):
    """REPAIRABLE: a think that claims an observation CONFIRMED a literal, when that literal is not in
    the observed frequent_values (especially under truncation), overstates what the observation
    actually proved. The fix is to downgrade the wording, not to invent evidence."""
    tc = step.get("tool_call") or {}
    if tc.get("tool") != "condition_filter":
        return []
    text = f"{step.get('think', '')} {_rationale_text(step.get('rationale'))}"
    if not CONFIRM_WORDS.search(text):
        return []
    observed: dict[str, tuple[set, bool]] = {}
    for s in enriched_steps[:upto]:
        if (s.get("tool_call") or {}).get("tool") != "inspect_column":
            continue
        a = s["tool_call"]["arguments"]
        o = s.get("tool_output") or {}
        observed[_base_col(a.get("column", "")).lower()] = (
            {str(v).lower() for v in (o.get("frequent_values") or [])}, bool(o.get("truncated")))
    issues: list[str] = []
    for col, lit in _string_literal_pairs(tc.get("arguments", {}).get("conditions")):
        vt = observed.get(col.lower())
        if vt and lit.lower() not in vt[0]:
            issues.append(
                f"{label} claims an observation confirmed literal {lit!r} on {col}, but "
                f"inspect_column did not show it (truncated={vt[1]})"
            )
    return issues


def _stale_observation_wording_issues(step: dict, label: str) -> list[str]:
    """REPAIRABLE: non-observation actions should not say they still need to inspect/read first.

    Once an action is rendered, all required observations should already be in the prior transcript.
    Phrases like "I should inspect first" inside a condition_filter/project/answer step teach the
    wrong temporal policy even when the tool path itself is executable.
    """
    tool = (step.get("tool_call") or {}).get("tool")
    if tool in PERCEPTION:
        return []
    text = f"{step.get('think', '')} {_rationale_text(step.get('rationale'))}"
    if STALE_OBSERVATION_WORDING.search(text):
        return [
            f"{label} uses stale observation wording inside {tool}: the observation should already "
            "have happened before this action"
        ]
    return []


def _final_answer_leak_issues(step: dict, label: str) -> list[str]:
    """REPAIRABLE: final thoughts should cite evidence, not talk like an annotator revealing gold."""
    tool = (step.get("tool_call") or {}).get("tool")
    if tool != "answer_from_context":
        return []
    text = f"{step.get('think', '')} {_rationale_text(step.get('rationale'))}"
    if FINAL_ANSWER_LEAK.search(text):
        return [
            f"{label} final think uses annotator/gold-answer wording; cite the evidence table/result "
            "instead of saying the verified/expected answer"
        ]
    return []


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def _tool_action_mismatch_issues(step: dict, label: str) -> list[str]:
    """REPAIRABLE: the action verb in think should match the actual tool call.

    This catches long-chain shifted rewrites like a `project` step whose think says "I filter ...",
    or a `set_op` step whose think describes a condition_filter. It is intentionally conservative:
    if the current tool's own cue is present, we do not flag incidental mentions of prior/future work.
    """
    tool = (step.get("tool_call") or {}).get("tool")
    if tool in PERCEPTION or tool not in TOOL_ACTION_CUES:
        return []
    # The think string is the text the model learns to emit before the tool call; do not let a
    # correct cue hidden in the structured rationale mask a shifted/wrong action sentence.
    text = str(step.get("think", ""))
    if tool == "condition_filter" and re.search(r"\bjoin(?:ing)?\s+\w+|join\s+\w+\s+with\b", text, re.I):
        return [
            f"{label} think appears to describe `join_tables` while the actual tool is `{tool}`"
        ]
    if _matches_any(text, TOOL_ACTION_CUES[tool]):
        return []
    for other, cues in TOOL_ACTION_CUES.items():
        if other == "answer_from_context":
            terminal_answer = re.search(
                r"\banswer_from_context\b|\bfinal answer\b|"
                r"\b(?:can|will|now|directly|confidently|ready to)\s+"
                r"(?:answer|return|provide|give)\b",
                text,
                re.I,
            )
            if not terminal_answer:
                continue
        if other != tool and _matches_any(text, cues):
            return [
                f"{label} think appears to describe `{other}` while the actual tool is `{tool}`"
            ]
    return []


def _missing_action_column_issues(step: dict, label: str) -> list[str]:
    """REPAIRABLE for non-observation actions: exact columns used by the action should be visible in
    the think text itself, not only buried in structured rationale."""
    tool = (step.get("tool_call") or {}).get("tool")
    if tool in PERCEPTION:
        return []
    args = (step.get("tool_call") or {}).get("arguments") or {}
    low = str(step.get("think", "")).lower()
    missing = [
        term for term in sorted(expected_column_terms(tool, args))
        if term.lower() not in low and not _is_generated_alias(term)
    ]
    if missing:
        return [f"{label} think does not name exact column(s) {missing}"]
    return []


def _missing_action_reference_issues(step: dict, label: str) -> list[str]:
    """REPAIRABLE: important table handles/source tables in the actual call should be visible in the
    think text. This catches shifted join explanations that mention the previous join target while
    the current call joins a different table."""
    tool = (step.get("tool_call") or {}).get("tool")
    if tool in PERCEPTION:
        return []
    args = (step.get("tool_call") or {}).get("arguments") or {}
    needed: list[str] = []
    if tool in {"condition_filter", "project", "aggregate", "group_aggregate", "extreme_value_select"}:
        if isinstance(args.get("table"), str):
            needed.append(args["table"])
    elif tool == "join_tables":
        # Require the join's input tables named. Prevents shifted explanations that describe a
        # different join. Covers both the N-way `tables` list and the legacy left/right form.
        needed.extend(_join_table_refs(args))
    elif tool == "set_op":
        for key in ("left", "right"):
            if isinstance(args.get(key), str):
                needed.append(args[key])
    elif tool == "answer_from_context":
        evidence = (args.get("evidence") or {}).get("table")
        if isinstance(evidence, str):
            needed.append(evidence)
    low = str(step.get("think", "")).lower()
    missing = [ref for ref in needed if ref.lower() not in low]
    if missing:
        return [f"{label} think does not name table/reference(s) {missing} used by the action"]
    return []


def quality_check(enriched_steps: list[dict], question: str = "",
                  overview: dict | None = None) -> tuple[list[str], list[str], list[str]]:
    """Return (hard, repairable, style).

    HARD = structural / executional red lines that make the trajectory unusable for SFT (reject /
    fallback): operating on a source table before describe_table; filtering a text column before
    inspect_column grounded its domain; a correction guessing a ruled-out column; empty or
    third-person reasoning.
    REPAIRABLE = state-visibility / honesty violations fixable by rewriting the think WITHOUT
    touching the (correct) tool path, and that MUST be fixed before SFT export: naming an exact
    hidden schema identifier before describe_table observed it (schema leak); claiming an observation
    confirmed a literal the observation did not actually show (unsupported confirmation); stale
    "I should inspect/read first" wording inside a non-observation action; annotator-style final
    answer leakage; tool-action mismatch.
    STYLE = soft readability preferences, recorded but never blocking: non-first-person phrasing;
    missing exact column after schema; weak semantic bridge; incomplete rationale.
    """
    hard: list[str] = []
    repairable: list[str] = []
    style: list[str] = []
    for i, step in enumerate(enriched_steps):
        think = step.get("think", "")
        tool_call = step.get("tool_call") or {}
        tool = tool_call.get("tool")
        args = tool_call.get("arguments") or {}
        label = f"step {i + 1}"

        # ---- hard: genuine quality red lines (empty reasoning / third-person narration) ----
        narration = _narration_issue(think, f"{label} think")
        if narration:
            hard.append(narration)

        # ---- repairable: state-visibility / honesty (fixable by rewriting the think) ----
        repairable.extend(_schema_leak_issues(step, question, label, overview))
        repairable.extend(_unsupported_confirmation_issues(step, enriched_steps, i, label))
        repairable.extend(_stale_observation_wording_issues(step, label))
        repairable.extend(_final_answer_leak_issues(step, label))
        repairable.extend(_tool_action_mismatch_issues(step, label))
        repairable.extend(_missing_action_column_issues(step, label))
        repairable.extend(_missing_action_reference_issues(step, label))

        # ---- style: readability preferences (recorded, never a reject) ----
        if think.strip() and not FIRST_PERSON.search(think):
            style.append(f"{label} think is not written as first-person task reasoning")
        style.extend(_templated_think_style_issues(think, f"{label} think"))
        style.extend(_rationale_quality_issues(step.get("rationale"), label))
        terms = expected_semantic_terms(tool, args)
        column_terms = expected_column_terms(tool, args)
        rationale_text = _rationale_text(step.get("rationale"))
        low = think.lower()
        combined_low = f"{think} {rationale_text}".lower()
        missing_columns = [
            term for term in sorted(column_terms)
            if term.lower() not in low and not _is_generated_alias(term)
        ]
        if missing_columns and tool in PERCEPTION:
            style.append(f"{label} think does not name exact column(s) {missing_columns}")
        elif not missing_columns and terms and not any(term.lower() in combined_low for term in terms):
            style.append(f"{label} think does not justify its concrete table/column choice")

        # ---- hard: structural preconditions of a catalog-only agent ----
        # A source-table column is grounded by ANY of describe_table, the catalog FK graph (join
        # keys), or inspect_column — and derived-table columns are re-exposed by each producing step's
        # output. So flag only columns this action reads from a SOURCE table that no such source has
        # exposed yet — not merely "this table was never describe_table'd" (which mis-flagged legal
        # catalog-grounded joins and inspect-grounded filters).
        grounded = _grounded_columns(enriched_steps, i, overview)
        for table in _table_refs_for_action(tool, args):
            if _is_derived_table(table):
                continue
            ungrounded = sorted(
                c for c in _source_columns_used(tool, args, table)
                if c.lower() not in grounded.get(table.lower(), set())
            )
            if ungrounded:
                hard.append(
                    f"{label} reads column(s) {ungrounded} of source table {table!r} before any "
                    f"observation (describe_table / inspect_column) or the catalog grounded them"
                )
        if tool == "condition_filter":
            domains = _observed_value_domains(enriched_steps, i)
            for col in string_filter_cols([{"tool_call": {"tool": "condition_filter",
                                                          "arguments": args}}]):
                source_table = args.get("table")
                if isinstance(source_table, str) and not _is_derived_table(source_table):
                    if (source_table.lower(), col.lower()) not in domains:
                        hard.append(
                            f"{label} filters text column {source_table}.{col} before "
                            f"inspect_column grounded its value domain"
                        )
        if not step.get("error_attempt"):
            continue
        # This branch is specifically "after a describe_table exposed the schema, did the retry still
        # guess a column that schema does not have" — so it uses the describe_table-only view, not the
        # broader grounded set (catalog FK / inspect do not 'show the full schema').
        columns_seen = _described_columns(enriched_steps, i)
        table = args.get("table")
        column = args.get("column")
        if isinstance(table, str) and isinstance(column, str):
            known = columns_seen.get(table.lower())
            if known is not None and column.lower() not in known:
                hard.append(
                    f"{label} guesses missing column {table}.{column} after describe_table "
                    f"already showed the schema"
                )
        cond = args.get("conditions")
        if isinstance(table, str) and table.lower() in columns_seen and isinstance(cond, dict):
            for col in string_filter_cols([{"tool_call": {"tool": "condition_filter",
                                                          "arguments": {"conditions": cond}}}]):
                if col.lower() not in columns_seen[table.lower()]:
                    hard.append(
                        f"{label} filters on missing column {table}.{col} after schema was known"
                    )
    return hard, repairable, style


def _compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _action_repair_text(tool: str, args: dict, question: str) -> str:
    """Deterministic repair for temporal/annotator wording.

    This intentionally repairs only the narration; it never changes arguments or observations.
    """
    if tool == "condition_filter":
        return (
            f"The observed state supports filtering `{args.get('table')}` with "
            f"`{_compact_json(args.get('conditions'))}`; this narrows the candidate rows toward "
            "the condition expressed in the question."
        )
    if tool == "project":
        return (
            f"I project `{args.get('table')}` to keep just "
            f"`{_compact_json(args.get('expressions', []))}`, since those fields are the useful "
            "columns for the next result."
        )
    if tool == "aggregate":
        return (
            f"The question asks for a scalar summary, so `{args.get('op')}` over "
            f"`{args.get('column')}` from `{args.get('table')}` is the needed computation."
        )
    if tool == "group_aggregate":
        return (
            f"The comparison needs grouped evidence from `{args.get('table')}`: group by "
            f"`{_compact_json(args.get('group_by', []))}` and compute "
            f"`{_compact_json(args.get('aggregations', []))}`."
        )
    if tool == "extreme_value_select":
        return (
            f"The ordering criterion `{_compact_json(args.get('order_by', []))}` on "
            f"`{args.get('table')}` identifies the extreme or ordered rows requested."
        )
    if tool == "join_tables":
        refs = _join_table_refs(args)
        keys = sorted({_base_col(e.get("left", "")) for e in _join_on_edges(args)}
                      | {_base_col(e.get("right", "")) for e in _join_on_edges(args)} - {""})
        return (
            f"The question needs information spread across {', '.join('`%s`' % r for r in refs)}, "
            f"so the key(s) {', '.join('`%s`' % k for k in keys)} connect those rows into one table."
        )
    if tool == "set_op":
        return (
            f"The two derived sets `{args.get('left')}` and `{args.get('right')}` represent the "
            f"branches of the question, and `{args.get('op')}` combines them in the required way."
        )
    if tool == "answer_from_context":
        evidence = (args.get("evidence") or {}).get("table")
        if evidence:
            return (
                f"The evidence table `{evidence}` contains the rows requested by the question, so I "
                "answer from that table."
            )
        return (
            "The previous tool result contains the scalar requested by the question, so I answer "
            "from that result."
        )
    return (
        f"The visible state points to `{tool}` with `{_compact_json(args)}` as the next operation "
        "needed for the question."
    )


def _action_repair_rationale(tool: str, args: dict, question: str) -> dict:
    if tool == "answer_from_context":
        evidence = (args.get("evidence") or {}).get("table")
        return {
            "question_cue": question,
            "observed_evidence": (
                f"the evidence table `{evidence}`" if evidence else
                "the previous scalar/result-producing tool output"
            ),
            "decision": "answer_from_context uses the already computed evidence without adding new claims",
            "supports_next_step": "final answer",
        }
    return {
        "question_cue": question,
        "observed_evidence": "prior schema/value observations and intermediate tool outputs are already available",
        "decision": f"use `{tool}` with arguments `{_compact_json(args)}`",
        "supports_next_step": "the next tool call or final answer",
    }


def _perception_repair_text(tool: str, args: dict, question: str) -> str:
    if tool == "describe_table":
        tables = args.get("tables") or []
        return (
            f"I describe `{_compact_json(tables)}` because the question requires information from "
            "these table concepts, and I need the schema observation before naming exact columns or "
            "choosing relational actions."
        )
    if tool == "inspect_column":
        return (
            f"I inspect `{args.get('table')}.{args.get('column')}` because this column is needed by "
            "a later predicate or decision, and I need its observed values before relying on it."
        )
    if tool == "read_subtable":
        return (
            f"I read `{args.get('table')}` because the next step depends on the rows in this "
            "intermediate result, and I need to ground that action in the actual output."
        )
    return (
        f"I call `{tool}` with arguments `{_compact_json(args)}` to acquire information needed by "
        "the following action."
    )


def _perception_repair_rationale(tool: str, args: dict, question: str) -> dict:
    if tool == "describe_table":
        tables = args.get("tables") or []
        return {
            "question_cue": question,
            "observed_evidence": "the initial catalog has table names and relations but no full column schema",
            "decision": (
                f"describe_table on `{_compact_json(tables)}` is needed before exact columns can be "
                "named or used"
            ),
            "supports_next_step": "later relational actions over the observed source table schema",
        }
    if tool == "inspect_column":
        return {
            "question_cue": question,
            "observed_evidence": f"values from `{args.get('table')}.{args.get('column')}`",
            "decision": "inspect_column grounds the column's observed value domain before a later action uses it",
            "supports_next_step": "the later filter, comparison, or answer step that depends on this column",
        }
    if tool == "read_subtable":
        return {
            "question_cue": question,
            "observed_evidence": f"rows from intermediate table `{args.get('table')}`",
            "decision": "read_subtable grounds the next action in the actual intermediate result",
            "supports_next_step": "the next dependent relational action or final answer",
        }
    return {
        "question_cue": question,
        "observed_evidence": "the requested observation output",
        "decision": f"use `{tool}` with arguments `{_compact_json(args)}`",
        "supports_next_step": "the following tool call",
    }


def _normalize_agent_voice(text: str) -> str:
    """Prefer the trained agent's singular first-person voice without rewriting the content."""
    if not isinstance(text, str):
        return text
    repl = [
        (r"\bWe need to\b", "I need to"),
        (r"\bwe need to\b", "I need to"),
        (r"\bwe first need\b", "I first need"),
        (r"\bwe need\b", "I need"),
        (r"\bWe should\b", "I should"),
        (r"\bwe should\b", "I should"),
        (r"\bWe can\b", "I can"),
        (r"\bwe can\b", "I can"),
        (r"\bWe have\b", "I have"),
        (r"\bwe have\b", "I have"),
        (r"\bWe now have\b", "I now have"),
        (r"\bwe now have\b", "I now have"),
        (r"\bSince we\b", "Since I"),
        (r"\bsince we\b", "since I"),
    ]
    out = text
    for pat, rep in repl:
        out = re.sub(pat, rep, out)
    return out


def repair_reasoning(traj: dict) -> tuple[dict, int]:
    """Repair only think/rationale text for deterministic repairable issues.

    Tool calls, tool outputs, final answers, step ids, and provenance are intentionally untouched.
    """
    out = copy.deepcopy(traj)
    repaired = 0
    question = out.get("question", "")
    overview = out.get("initial_state", {}).get("dataset_overview")
    steps = out.get("steps", [])
    for index, step in enumerate(steps):
        tool = (step.get("tool_call") or {}).get("tool")
        args = (step.get("tool_call") or {}).get("arguments") or {}
        before = (step.get("think"), step.get("rationale"))
        if isinstance(step.get("think"), str):
            step["think"] = _normalize_agent_voice(step["think"])
        text = f"{step.get('think', '')} {_rationale_text(step.get('rationale'))}"

        schema_leaks = _schema_leak_issues(step, question, "step", overview)
        unsupported = _unsupported_confirmation_issues(step, steps, index, "step")
        stale = bool(STALE_OBSERVATION_WORDING.search(text))
        final_leak = bool(FINAL_ANSWER_LEAK.search(text))
        mismatch = _tool_action_mismatch_issues(step, "step")
        missing_columns = _missing_action_column_issues(step, "step")
        missing_refs = _missing_action_reference_issues(step, "step")

        if tool in PERCEPTION and schema_leaks:
            step["think"] = _perception_repair_text(tool, args, question)
            step["rationale"] = _perception_repair_rationale(tool, args, question)
        elif tool == "answer_from_context" and (stale or final_leak or mismatch):
            step["think"] = _action_repair_text(tool, args, question)
            step["rationale"] = _action_repair_rationale(tool, args, question)
        elif tool not in PERCEPTION and (stale or unsupported or mismatch or missing_columns or missing_refs):
            step["think"] = _action_repair_text(tool, args, question)
            step["rationale"] = _action_repair_rationale(tool, args, question)

        if (step.get("think"), step.get("rationale")) != before:
            repaired += 1

    hard, repairable, style = quality_check(
        out.get("steps", []), out.get("question", ""), out.get("initial_state", {}).get("dataset_overview")
    )
    enr = out.setdefault("enrichment", {})
    enr["quality_status"] = "reject" if hard else ("repairable" if repairable else "ready")
    enr["repairable_issues"] = repairable
    enr["soft_issues"] = style
    repair_meta = enr.setdefault("repair_history", [])
    repair_meta.append({
        "method": "deterministic_think_repair",
        "changed_steps": repaired,
        "remaining_hard_issues": hard,
        "remaining_repairable_issues": repairable,
    })
    return out, repaired


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


QUALITY_AUDIT_SYS = (
    "You are the final quality reviewer for SFT data of a table-tool reasoning agent. "
    "Judge whether one trajectory should be kept for training. You are not rewriting the "
    "trajectory now; if it is not directly usable, return concrete revision requirements.\n\n"
    "The agent starts from a lazy catalog: table names, row counts, and foreign-key relations. "
    "It learns through tool calls and tool observations. The tool calls, the relational decomposition, "
    "and the FINAL ANSWER are ALREADY execution-verified equivalent to the gold query by the harness — "
    "so do NOT judge whether a join/filter/aggregate/intersect is the 'right' method or whether the "
    "answer is correct (both are guaranteed). Judge ONLY whether the reasoning and observations teach "
    "faithful, non-hallucinated behavior: grounding, honest use of observations, no leaked hidden "
    "state, and non-formulaic reasoning.\n\n"
    "The data may arrive either before or after the plan-insertion pass. Always review perception + "
    "reasoning (describe_table / inspect_column / read_subtable observations and each step's "
    "<think>). If plan tool steps are present, also review the plan subgoals/status/evidence. If plan "
    "tool steps are absent, do NOT reject solely for missing plan; a later pass may add it.\n\n"
    "A deterministic checker has already flagged CANDIDATE STRUCTURAL ISSUES in the user message. Treat "
    "them as leads, not verdicts: confirm the real ones (fold them into revision_requests), discard "
    "false positives, and still catch anything semantic the checker cannot see.\n\n"
    "KEEP when:\n"
    "- Each <think> explains the current decision from the visible question/catalog/history, not "
    "from hidden gold answers.\n"
    "- Schema identifiers, literal values, intermediate rows, and final values are mentioned only "
    "after they are visible in previous observations or tool outputs.\n"
    "- Observation calls are useful but not ritualistic: describe_table can inspect several relevant "
    "tables at once; inspect_column grounds a string literal or ambiguous value; read_subtable checks "
    "rows only when row evidence matters.\n"
    "- The reasoning may be concise, but it should be specific to this question and this state.\n\n"
    "REPAIR when the tool sequence and observations are usable, but local wording should be fixed "
    "before SFT, for example stale wording ('I should inspect' after inspection already happened), "
    "third-person/meta narration, overly generic boilerplate, or a missing explanation for why a "
    "chosen column/table answers the question. Repair means the same tool calls and outputs can stay.\n\n"
    "REJECT when the trajectory would teach the model a wrong state boundary or unreliable behavior: "
    "it relies on hidden schema/value/row/final-answer information, hallucinates what an observation "
    "showed (e.g. claims a value was 'confirmed' when the cited inspect_column was truncated and did "
    "not list it), skips a necessary observation before a risky string/value decision, has a tool/think "
    "mismatch that cannot be fixed by wording alone, ignores important tool feedback, or is dominated "
    "by templated reasoning that would make the dataset mode-collapse.\n\n"
    "PLAN QUALITY (stage 2): the plan is control state, and its evidence must be harness-grounded. "
    "KEEP plans whose subgoals map to real backbone operations and whose status/evidence reflect "
    "what actually executed. REPAIR minor plan wording. REJECT plan HALLUCINATION (a goal asserting "
    "a value, row, or outcome no cited evidence step produced) and OVER-PLANNING (a subgoal per "
    "trivial step, ritualistic updates that restate the prior state with no real progress, or more "
    "plan calls than actual work). A good plan is a few meaningful subgoals updated when state truly changes.\n\n"
    "BAD PATTERNS to actively catch (repair if local, reject if pervasive):\n"
    "- formulaic reasoning: many steps sharing 'I now apply/compute/project X because the question asks...';\n"
    "- ritual observation: an inspect/read that no later step consumes, or describe/inspect narrated every step;\n"
    "- fabricated grounding: 'I confirmed/verified X' when the observation did not actually show X;\n"
    "- ritual planning: a plan update after nearly every step, or subgoals that just echo the tool name.\n\n"
    "Do not reject merely because the style differs from your favorite phrasing. Prefer preserving "
    "natural, varied reasoning when it is faithful.\n\n"
    "Output ONLY JSON with this schema:\n"
    "{\n"
    "  \"decision\": \"keep|repair|reject\",\n"
    "  \"confidence\": 0.0,\n"
    "  \"summary\": \"one sentence\",\n"
    "  \"strengths\": [\"...\"],\n"
    "  \"risks\": [\"...\"],\n"
    "  \"revision_requests\": [\n"
    "    {\"step_id\": \"step_3\", \"severity\": \"minor|major|fatal\", "
    "\"requirement\": \"what must change\", \"reason\": \"why\"}\n"
    "  ]\n"
    "}\n"
    "For keep, revision_requests should usually be empty. For BOTH repair AND reject, you MUST return "
    "revision_requests — one per real problem, each naming the exact step_id and a concrete, actionable "
    "requirement a later generator can apply to fix that step's think or plan (repair keeps the same tool "
    "calls/outputs; reject may need a regenerated observation/plan). Never return an empty "
    "revision_requests for repair or reject."
)


def _clip_text(value, limit: int = 1200) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"... <truncated {len(text) - limit} chars>"


def _compact_audit_output(output, tool: str | None = None):
    if not isinstance(output, dict):
        return output
    out = copy.deepcopy(output)
    if isinstance(out.get("rows"), list):
        if tool == "read_subtable":
            out["rows"] = out["rows"][:50]
        else:
            out["rows"] = out["rows"][:5]
    for key in ("result_sample",):
        if isinstance(out.get(key), list):
            out[key] = out[key][:5]
    if isinstance(out.get("tables"), list):
        compact_tables = []
        for table in out["tables"]:
            if not isinstance(table, dict):
                compact_tables.append(table)
                continue
            t = dict(table)
            if isinstance(t.get("columns"), list):
                t["columns"] = t["columns"][:40]
            compact_tables.append(t)
        out["tables"] = compact_tables
    return out


def _compact_trajectory_for_quality(traj: dict) -> dict:
    steps = []
    for i, step in enumerate(traj.get("steps", []), 1):
        call = step.get("tool_call") or {}
        item = {
            "index": i,
            "step_id": step.get("step_id"),
            "think": step.get("think", ""),
            "rationale": step.get("rationale"),
            "tool": call.get("tool"),
            "arguments": call.get("arguments"),
            "output": _compact_audit_output(step.get("tool_output"), call.get("tool")),
        }
        if step.get("perception"):
            item["perception"] = True
        steps.append(item)
    enrichment = traj.get("enrichment") or {}
    programmatic = enrichment.get("programmatic_quality")
    candidate_issues = {}
    if isinstance(programmatic, dict):
        candidate_issues = {
            "status": programmatic.get("status"),
            "hard_issues": programmatic.get("hard_issues") or [],
            "repairable_issues": programmatic.get("repairable_issues") or [],
            "soft_issues": programmatic.get("soft_issues") or [],
        }
    contains_plan = any((s.get("tool_call") or {}).get("tool") == "plan" for s in traj.get("steps", []))
    return {
        "trajectory_id": traj.get("trajectory_id"),
        "db_id": (traj.get("source") or {}).get("db_id"),
        "question": traj.get("question"),
        "contains_plan_tool": contains_plan,
        "initial_catalog": (traj.get("initial_state") or {}).get("dataset_overview"),
        "candidate_structural_issues": candidate_issues,
        "steps": steps,
        "final_answer": traj.get("final_answer"),
    }


def _step_ref_from_issue(text: str) -> str:
    m = re.search(r"\bstep[ _](\d+)\b", str(text))
    return f"step_{m.group(1)}" if m else "trajectory"


def _plan_structural_issues(traj: dict) -> list[dict]:
    """Deterministic structural checks specific to the plan tool: dangling references and an
    over-planning signal. Semantic judgements (hallucinated evidence, ritualistic goals) are left to
    the LLM auditor — these are only the mechanically decidable candidates."""
    steps = traj.get("steps", [])
    out: list[dict] = []
    plan_idx = [i for i, s in enumerate(steps) if (s.get("tool_call") or {}).get("tool") == "plan"]
    work = len(steps) - len(plan_idx)
    if work and len(plan_idx) > max(2, work):
        out.append({"step_id": "plan", "severity": "minor",
                    "issue": f"possible over-planning: {len(plan_idx)} plan calls vs {work} tool/work steps"})
    seen: set = set()
    for s in steps:
        tc = s.get("tool_call") or {}
        if tc.get("tool") == "plan":
            for op in (tc.get("arguments") or {}).get("ops", []) or []:
                ev = op.get("evidence", op.get("evidence_step_id"))
                if isinstance(ev, str) and ev and ev not in seen:
                    out.append({"step_id": s.get("step_id"), "severity": "major",
                                "issue": f"plan op {op.get('id')!r} cites evidence {ev!r} "
                                         f"which is not an earlier executed step"})
        seen.add(s.get("step_id"))
    return out


def structural_diagnosis(traj: dict) -> list[dict]:
    """Deterministic structural problem diagnosis handed to the LLM auditor as CANDIDATES to verify.
    Execution/rules decide the structural part (grounding gaps, truncation-fabrication, stale wording,
    tool/think mismatch, plan step-ref validity, over-planning signals); the LLM makes the semantic
    call and writes repair requirements. Programmatic verdicts never decide retention on their own."""
    hard, repairable, _style = quality_check(
        traj.get("steps", []), traj.get("question", ""),
        traj.get("initial_state", {}).get("dataset_overview"),
    )
    out: list[dict] = []
    for severity, issues in (("major", hard), ("minor", repairable)):
        for text in issues:
            out.append({"step_id": _step_ref_from_issue(text), "severity": severity, "issue": text})
    out.extend(_plan_structural_issues(traj))
    return out


def build_quality_audit_prompt(traj: dict) -> list[dict]:
    payload = _compact_trajectory_for_quality(traj)
    diagnosis = structural_diagnosis(traj)
    diag_block = (
        "CANDIDATE STRUCTURAL ISSUES (found by a deterministic checker — VERIFY each against the "
        "trajectory; confirm real ones in your revision_requests, ignore false positives, and add any "
        "the checker missed):\n" + json.dumps(diagnosis, ensure_ascii=False)
        if diagnosis else
        "CANDIDATE STRUCTURAL ISSUES: none flagged by the deterministic checker (still judge semantics "
        "and plan quality yourself)."
    )
    user = (
        "Review this trajectory for SFT data quality.\n\n"
        f"TRAJECTORY:\n{_clip_text(payload, 24000)}\n\n"
        f"{diag_block}\n\n"
        "Return the JSON review now."
    )
    return [{"role": "system", "content": QUALITY_AUDIT_SYS}, {"role": "user", "content": user}]


def _json_object_from_text(text: str) -> dict:
    obj_match = re.search(r"\{.*\}", text, re.S)
    if not obj_match:
        return {}
    try:
        obj = json.loads(obj_match.group(0))
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


def parse_quality_audit(text: str) -> dict:
    obj = _json_object_from_text(text)
    decision = str(obj.get("decision", "")).strip().lower()
    aliases = {
        "ready": "keep",
        "accept": "keep",
        "accepted": "keep",
        "use": "keep",
        "usable": "keep",
        "needs_repair": "repair",
        "revise": "repair",
        "fix": "repair",
        "drop": "reject",
        "discard": "reject",
        "not usable": "reject",
    }
    decision = aliases.get(decision, decision)
    if decision not in {"keep", "repair", "reject"}:
        decision = "reject"
        obj.setdefault("summary", "quality audit did not return a valid decision")
    obj["decision"] = decision
    reqs = obj.get("revision_requests")
    if not isinstance(reqs, list):
        obj["revision_requests"] = []
    else:
        obj["revision_requests"] = [r for r in reqs if isinstance(r, dict)]
    for key in ("strengths", "risks"):
        if not isinstance(obj.get(key), list):
            obj[key] = []
    try:
        obj["confidence"] = float(obj.get("confidence", 0.0))
    except (TypeError, ValueError):
        obj["confidence"] = 0.0
    obj["summary"] = str(obj.get("summary", "")).strip()
    return obj


def _audit_issue_texts(audit: dict) -> list[str]:
    out = []
    for req in audit.get("revision_requests") or []:
        step = req.get("step_id") or req.get("step") or "trajectory"
        requirement = str(req.get("requirement", "")).strip()
        reason = str(req.get("reason", "")).strip()
        if requirement and reason:
            out.append(f"{step}: {requirement} ({reason})")
        elif requirement:
            out.append(f"{step}: {requirement}")
    if not out and audit.get("summary"):
        out.append(audit["summary"])
    return out


def apply_quality_review(
    traj: dict,
    *,
    mode: str,
    base: str | None = None,
    key: str | None = None,
    model: str | None = None,
    api_timeout: int = 180,
    api_retries: int = 4,
) -> dict:
    """Attach final quality status.

    Programmatic checks are kept as diagnostics, but in LLM mode they no longer decide whether a
    trajectory is retained. The external audit is the final status source.
    """
    out = traj
    enr = out.setdefault("enrichment", {})
    prog_hard, prog_rep, prog_style = quality_check(
        out.get("steps", []), out.get("question", ""),
        out.get("initial_state", {}).get("dataset_overview"),
    )
    programmatic = {
        "status": "reject" if prog_hard else ("repairable" if prog_rep else "ready"),
        "hard_issues": prog_hard,
        "repairable_issues": prog_rep,
        "soft_issues": prog_style,
    }
    enr["programmatic_quality"] = programmatic

    if mode == "none":
        enr["quality_status_source"] = "none"
        enr.setdefault("quality_status", "unreviewed")
        return out

    if mode == "programmatic":
        enr["quality_status_source"] = "programmatic"
        enr["quality_status"] = programmatic["status"]
        enr["hard_issues"] = prog_hard
        enr["repairable_issues"] = prog_rep
        enr["soft_issues"] = prog_style
        return out

    if enr.get("status") != "enriched":
        audit = {
            "decision": "reject",
            "confidence": 1.0,
            "summary": "trajectory generation fell back to skeleton, so it is not usable for enriched SFT",
            "strengths": [],
            "risks": ["fallback_skeleton"],
            "revision_requests": [{
                "step_id": "trajectory",
                "severity": "fatal",
                "requirement": "regenerate the trajectory with successful enrichment",
                "reason": "fallback skeleton has not passed external quality review",
            }],
            "status": "skipped_fallback",
        }
    else:
        if not (base and key and model):
            audit = {
                "decision": "reject",
                "confidence": 0.0,
                "summary": "external quality audit was requested but API credentials/model were unavailable",
                "strengths": [],
                "risks": ["audit_not_run"],
                "revision_requests": [{
                    "step_id": "trajectory",
                    "severity": "fatal",
                    "requirement": "run external quality audit before using this record",
                    "reason": "no audit result is present",
                }],
                "status": "api_missing",
            }
        else:
            try:
                text, usage = call_retry(
                    base, key, model, build_quality_audit_prompt(out),
                    tries=api_retries, timeout=api_timeout,
                )
                audit = parse_quality_audit(text)
                audit["status"] = "ok"
                audit["model_output"] = text
                audit["usage"] = usage
            except Exception as e:  # noqa: BLE001
                audit = {
                    "decision": "reject",
                    "confidence": 0.0,
                    "summary": f"external quality audit failed: {type(e).__name__}: {e}",
                    "strengths": [],
                    "risks": ["audit_api_error"],
                    "revision_requests": [{
                        "step_id": "trajectory",
                        "severity": "fatal",
                        "requirement": "rerun external quality audit",
                        "reason": f"{type(e).__name__}: {e}",
                    }],
                    "status": "api_error",
                }

    status_map = {"keep": "ready", "repair": "repairable", "reject": "reject"}
    final_status = status_map.get(audit.get("decision"), "reject")
    enr["quality_status_source"] = "llm"
    enr["quality_status"] = final_status
    enr["quality_audit"] = audit
    enr["quality_reviewer"] = {"model": model, "mode": "llm_quality_audit"}
    issues = _audit_issue_texts(audit)
    enr["hard_issues"] = issues if final_status == "reject" else []
    enr["repairable_issues"] = issues if final_status == "repairable" else []
    enr["soft_issues"] = audit.get("risks", []) if final_status == "ready" else []
    return out


REPAIR_SYS = (
    "You revise ONLY the <think> wording of specific steps in a verified table-tool trajectory so it "
    "satisfies a reviewer's requirements. HARD RULES: never change any tool call, arguments, tool "
    "output, plan ops, step order, or the final answer — only the first-person <think> of the named "
    "steps. Keep it concise and faithful: describe the decision from what the visible observations "
    "ACTUALLY showed; never claim a value/row/schema was confirmed if the cited observation did not "
    "show it (if it was truncated, say the value was not shown and proceed accordingly); name the exact "
    "table handle and columns the step uses.\n\n"
    "Output ONLY JSON: {\"revisions\": [{\"step_id\": \"step_4\", \"think\": \"revised first-person "
    "think\"}]}. Include one entry per step named in the requirements; omit steps you do not change."
)


def build_repair_prompt(traj: dict, revision_requests: list[dict]) -> list[dict]:
    payload = _compact_trajectory_for_quality(traj)
    reqs = [{"step_id": r.get("step_id"), "requirement": r.get("requirement"), "reason": r.get("reason")}
            for r in revision_requests if isinstance(r, dict)]
    user = (
        f"TRAJECTORY:\n{_clip_text(payload, 24000)}\n\n"
        f"REVISION REQUIREMENTS:\n{json.dumps(reqs, ensure_ascii=False)}\n\n"
        "Return the JSON revisions now."
    )
    return [{"role": "system", "content": REPAIR_SYS}, {"role": "user", "content": user}]


def apply_think_revisions(traj: dict, revisions: list[dict]) -> int:
    """Apply LLM think rewrites in place — think text ONLY; tool calls/outputs/plan ops are untouched."""
    by_id = {r.get("step_id"): r.get("think") for r in revisions
             if isinstance(r, dict) and isinstance(r.get("think"), str) and r["think"].strip()}
    changed = 0
    for step in traj.get("steps", []):
        new = by_id.get(step.get("step_id"))
        if new and new.strip() != str(step.get("think", "")).strip():
            step["think"] = new.strip()
            changed += 1
    return changed


def review_and_repair(
    traj: dict, *, mode: str, base: str | None, key: str | None, model: str | None,
    max_repair: int = 3, api_timeout: int = 180, api_retries: int = 4,
) -> dict:
    """Audit -> if repair/reject with actionable requests, let the model rewrite the flagged steps'
    think and re-audit, up to `max_repair` rounds. The LLM audit stays the final gate; repair only
    edits think wording (never tools/outputs), so a verified trajectory stays verified."""
    traj = apply_quality_review(traj, mode=mode, base=base, key=key, model=model,
                                api_timeout=api_timeout, api_retries=api_retries)
    if mode != "llm" or not (base and key and model):
        return traj
    rank = {"ready": 2, "repairable": 1, "reject": 0}

    def _rank(t: dict) -> int:
        return rank.get((t.get("enrichment") or {}).get("quality_status"), -1)

    # Monotonic: keep the best-status version ever seen (including the pre-repair one). A repair round
    # that the noisy judge re-scores worse can never drag the record below where it already was.
    best = copy.deepcopy(traj)
    history: list[dict] = []
    for attempt in range(1, max_repair + 1):
        enr = traj.get("enrichment") or {}
        if enr.get("quality_status") == "ready":
            break
        audit = enr.get("quality_audit") or {}
        reqs = audit.get("revision_requests") or []
        if audit.get("status") != "ok" or not reqs:
            break                                   # skeleton / api error / nothing actionable
        before = enr.get("quality_status")
        try:
            text, _usage = call_retry(base, key, model, build_repair_prompt(traj, reqs),
                                      tries=api_retries, timeout=api_timeout)
            revisions = _json_object_from_text(text).get("revisions") or []
            n = apply_think_revisions(traj, revisions if isinstance(revisions, list) else [])
        except Exception as e:  # noqa: BLE001
            history.append({"attempt": attempt, "before": before, "error": f"{type(e).__name__}: {e}"})
            break
        if not n:
            history.append({"attempt": attempt, "before": before, "revised_steps": 0})
            break                                   # model changed nothing -> stop, avoid looping
        traj = apply_quality_review(traj, mode=mode, base=base, key=key, model=model,
                                    api_timeout=api_timeout, api_retries=api_retries)
        after = (traj.get("enrichment") or {}).get("quality_status")
        history.append({"attempt": attempt, "before": before, "revised_steps": n, "after": after})
        # >= (not >): on a tie, prefer the REPAIRED version — it addressed the flagged issues; only a
        # strictly-worse re-score is discarded. This keeps a fabrication fix even when residual minor
        # issues hold the coarse status at 'repairable'.
        if _rank(traj) >= _rank(best):
            best = copy.deepcopy(traj)
        if _rank(best) >= rank["repairable"]:
            break                                   # reached a usable version -> stop before noise regresses it
    best.setdefault("enrichment", {})["repair_history"] = history
    return best


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
        perc = dedupe_perception_insertions(traj, [a for a in ann if a.get("tool") in PERCEPTION])
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
        q_hard, q_rep, q_style = quality_check(
            enr_p, traj.get("question", ""), catalog_snapshot(h)
        )
        if not (l1p_ok and l2_ok):
            feedback = "; ".join(([l1p_err] if not l1p_ok else []) + l2_issues) or "no valid perception"
            history.append({"attempt": attempt, "perc": len(perc), "errs": len(errs),
                            "perception_ok": False, "issues": feedback,
                            "repairable_issues": q_rep, "soft_issues": q_style,
                            "model_output": text, "usage": usage})
            _remember_rejection(
                rejections, attempt=attempt, stage="perception_validation", issues=feedback,
                candidate_steps=enr_p, model_output=text, usage=usage, l1_ok=l1p_ok,
                l2_ok=l2_ok, programmatic_q_ok=not q_hard, n_perception=len(perc), n_error=len(errs),
            )
            continue
        best = {"steps": enr_p, "mode": "perception_only", "n_perception": len(perc), "n_error": 0,
                "repairable": q_rep, "style": q_style}
        if not errs:
            history.append({"attempt": attempt, "perc": len(perc), "errs": 0, "ok": True,
                            "model_output": text, "usage": usage})
            break
        l1f_ok, l1f_err, enr_f = replay_validate(traj, spliced_sequence(traj, perc + errs, rewrites))
        qf_hard, qf_rep, qf_style = quality_check(
            enr_f, traj.get("question", ""), catalog_snapshot(h)
        )
        if l1f_ok:
            best = {"steps": enr_f, "mode": "full", "n_perception": len(perc), "n_error": len(errs),
                    "repairable": qf_rep, "style": qf_style}
            history.append({"attempt": attempt, "perc": len(perc), "errs": len(errs), "ok": True,
                            "model_output": text, "usage": usage})
            break
        feedback = "perception is correct; fix ONLY the error attempts: " + "; ".join(
            ([l1f_err] if not l1f_ok else [])
        )
        history.append({"attempt": attempt, "perc": len(perc), "errs": len(errs),
                        "perception_ok": True, "errors_ok": False, "issues": feedback,
                        "model_output": text, "usage": usage})
        _remember_rejection(
            rejections, attempt=attempt, stage="correction_validation", issues=feedback,
            candidate_steps=enr_f, model_output=text, usage=usage, l1_ok=l1f_ok,
            programmatic_q_ok=not qf_hard,
            n_perception=len(perc), n_error=len(errs),
        )
    if best:
        out = copy.deepcopy(traj)
        out["schema_version"] = "v3-enriched"
        out["label_status"] = "verified"
        out["initial_state"]["dataset_overview"] = catalog_snapshot(h)
        out["steps"] = best["steps"]
        out["enrichment"] = {"status": "enriched", "mode": best["mode"],
                             "quality_status": "repairable" if best.get("repairable") else "ready",
                             "repairable_issues": best.get("repairable", []),
                             "soft_issues": best.get("style", []),
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
                         "quality_status": "reject",
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
    existing_describes = backbone_describe_observations(traj)
    history: list[dict] = []
    rejections: list[dict] = []
    feedback = ""
    for attempt in range(1, max_attempts + 1):
        obs_text, obs_usage = "", {}
        describes: list[dict] = []
        if not existing_describes:
            try:
                obs_text, obs_usage = call_retry(
                    base, key, model, build_observation_prompt(traj, feedback),
                    tries=api_retries, timeout=api_timeout,
                )
            except Exception as e:  # noqa: BLE001
                history.append({"attempt": attempt, "stage": "observation", "error": f"api: {e}"})
                continue

            obs_ann = parse_observation_insertions(obs_text)
            for a in obs_ann:
                if a.get("tool") != "describe_table":
                    continue
                b = copy.deepcopy(a)
                b["after"] = -1
                describes.append(b)
            describes = dedupe_perception_insertions(traj, describes)

        schema_observations = existing_describes + describes
        if not schema_observations:
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
                base, key, model, build_staged_action_prompt(traj, ctx_domains, schema_observations),
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

        local_perc = dedupe_perception_insertions(
            traj, [a for a in act_ann if a.get("tool") in {"inspect_column", "read_subtable"}]
        )
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
        q_hard, q_rep, q_style = quality_check(
            steps, traj.get("question", ""), catalog_snapshot(h)
        )
        if l1_ok and l2_ok:
            out = copy.deepcopy(traj)
            out["schema_version"] = "v3-enriched"
            out["label_status"] = "verified"
            out["initial_state"]["dataset_overview"] = catalog_snapshot(h)
            out["steps"] = steps
            out["enrichment"] = {"status": "enriched", "mode": "staged_perception",
                                 "quality_status": "repairable" if q_rep else "ready",
                                 "repairable_issues": q_rep, "soft_issues": q_style,
                                 "generator": {"model": model,
                                               "mode_requested": "staged_perception"},
                                 "n_perception": len(perc), "n_error": 0,
                                 "rejected_candidates": rejections,
                                 "annotation_history": history + [{
                                     "attempt": attempt, "ok": True,
                                     "repairable_issues": q_rep, "soft_issues": q_style,
                                     "observation_model_output": obs_text,
                                     "observation_usage": obs_usage,
                                     "model_output": act_text, "usage": act_usage,
                                     "stage1_describes": len(describes),
                                     "backbone_describes": len(existing_describes),
                                     "stage2_perception": len(local_perc),
                                 }]}
            return out

        feedback = "; ".join(([l1_err] if not l1_ok else []) +
                             ([] if l2_ok else l2_issues)) or "staged enrichment failed validation"
        history.append({"attempt": attempt, "ok": False, "issues": feedback,
                        "repairable_issues": q_rep, "soft_issues": q_style,
                        "observation_model_output": obs_text, "observation_usage": obs_usage,
                        "model_output": act_text, "usage": act_usage,
                        "stage1_describes": len(describes),
                        "backbone_describes": len(existing_describes),
                        "stage2_perception": len(local_perc)})
        _remember_rejection(
            rejections, attempt=attempt, stage="validation", issues=feedback,
            candidate_steps=steps, observation_model_output=obs_text, model_output=act_text,
            observation_usage=obs_usage, usage=act_usage, l1_ok=l1_ok, l2_ok=l2_ok,
            programmatic_q_ok=not q_hard,
            stage1_describes=len(describes), backbone_describes=len(existing_describes),
            stage2_perception=len(local_perc),
        )

    _, _, base_steps = replay_validate(traj, spliced_sequence(traj, []))
    out = copy.deepcopy(traj)
    out["schema_version"] = "v3-enriched"
    out["label_status"] = "verified"
    out["initial_state"]["dataset_overview"] = catalog_snapshot(h)
    out["steps"] = base_steps
    out["enrichment"] = {"status": "fallback_skeleton", "mode": "skeleton",
                         "quality_status": "reject",
                         "generator": {"model": model, "mode_requested": "staged_perception"},
                         "n_perception": 0, "n_error": 0,
                         "rejected_candidates": rejections,
                         "annotation_history": history}
    return out


def _replay_accepted_prefix(traj: dict, accepted_steps: list[dict]):
    h = Harness(db_path(traj["source"]["db_id"]))
    ctx = new_ctx()
    created: set[str] = set()
    for step in accepted_steps:
        call = step.get("tool_call") or {}
        tool = call.get("tool")
        args = call.get("arguments") or {}
        if tool == "answer_from_context":
            break
        out, tname = execute_tool(h, tool, args, ctx, step.get("step_id", "step_0"))
        if tname:
            created.add(tname)
        # Keep the replayed output in sync with the accepted transcript. This matters when a prior
        # accepted step was created from a provider response but the harness owns the actual output.
        step["tool_output"] = out
    return h, ctx, created


def enrich_one_stateful_step(traj: dict, base: str, key: str, model: str, max_attempts: int = 3,
                             api_timeout: int = 180, api_retries: int = 4) -> dict:
    """State-conditioned enrichment.

    The external LLM sees one target backbone step at a time plus the actual previous tool history.
    It may insert observation steps before that target if the visible state is insufficient, then it
    writes the target's reasoning. This keeps generation closer to a real tool-call transcript than
    whole-trajectory post-hoc rewriting.
    """
    h0 = Harness(db_path(traj["source"]["db_id"]))
    accepted_steps: list[dict] = []
    old_to_new: dict[str, str] = {}
    annotation_history: list[dict] = []
    rejections: list[dict] = []
    added_perception = 0

    for target_index, target_step in enumerate(traj["steps"]):
        call = target_step["tool_call"]
        target_tool = call["tool"]
        target_args = _strip_answer_memory_args(target_tool, copy.deepcopy(call["arguments"]))
        _remap_step_refs(target_tool, target_args, old_to_new)
        feedback = ""
        accepted_this_turn = None

        for attempt in range(1, max_attempts + 1):
            try:
                text, usage = call_retry(
                    base, key, model,
                    build_stateful_step_prompt(
                        traj, accepted_steps, target_index, target_step, target_args, feedback
                    ),
                    tries=api_retries, timeout=api_timeout,
                )
            except Exception as e:  # noqa: BLE001
                annotation_history.append({
                    "target_index": target_index, "attempt": attempt,
                    "stage": "api", "error": f"{type(e).__name__}: {e}",
                })
                continue

            insertions, target = parse_stateful_step(text)
            insertions = _dedupe_against_state(insertions, accepted_steps)
            if target_tool in PERCEPTION and insertions:
                feedback = (
                    "the target step is already an observation; do not insert another observation "
                    "before it"
                )
                annotation_history.append({
                    "target_index": target_index, "attempt": attempt, "ok": False,
                    "issues": feedback, "model_output": text, "usage": usage,
                })
                _remember_rejection(
                    rejections, attempt=attempt, stage="stateful_parse",
                    issues=f"target {target_index}: {feedback}",
                    model_output=text, usage=usage,
                )
                continue
            target_think = str(target.get("think", "")).strip()
            if not target_think:
                feedback = "missing target.think for the current target tool call"
                annotation_history.append({
                    "target_index": target_index, "attempt": attempt, "ok": False,
                    "issues": feedback, "model_output": text, "usage": usage,
                })
                _remember_rejection(
                    rejections, attempt=attempt, stage="stateful_parse",
                    issues=f"target {target_index}: {feedback}",
                    model_output=text, usage=usage,
                )
                continue

            try:
                h, ctx, created = _replay_accepted_prefix(traj, copy.deepcopy(accepted_steps))
                candidate_steps = copy.deepcopy(accepted_steps)

                turn_steps: list[dict] = []
                for ins in insertions:
                    sid = f"step_{len(candidate_steps) + len(turn_steps) + 1}"
                    out, _ = execute_tool(h, ins["tool"], ins.get("arguments", {}), ctx, sid)
                    turn_steps.append({
                        "step_id": sid,
                        "think": str(ins.get("think", "")).strip(),
                        "rationale": ins.get("rationale"),
                        "tool_call": {"tool": ins["tool"], "arguments": ins.get("arguments", {})},
                        "tool_output": out,
                        "perception": True,
                    })

                sid = f"step_{len(candidate_steps) + len(turn_steps) + 1}"
                target_record = {
                    "step_id": sid,
                    "think": target_think,
                    "rationale": target.get("rationale"),
                    "tool_call": {"tool": target_tool, "arguments": target_args},
                }
                if target_tool == "answer_from_context":
                    correct, pred, gold = score(h, traj["source"]["gold_sql"], target_args, created)
                    if not correct:
                        raise ValueError(
                            f"final answer no longer matches gold; pred={pred[:3]} gold={gold[:3]}"
                        )
                    target_record["tool_output"] = {"final_answer": target_args.get("answer")}
                else:
                    out, tname = execute_tool(h, target_tool, target_args, ctx, sid)
                    if tname:
                        created.add(tname)
                    target_record["tool_output"] = out
                    if target_tool in PERCEPTION:
                        target_record["perception"] = True

                turn_steps.append(target_record)
            except Exception as e:  # noqa: BLE001
                feedback = f"turn execution failed: {type(e).__name__}: {e}"
                annotation_history.append({
                    "target_index": target_index, "attempt": attempt, "ok": False,
                    "issues": feedback, "model_output": text, "usage": usage,
                    "n_insertions": len(insertions),
                })
                _remember_rejection(
                    rejections, attempt=attempt, stage="stateful_execution",
                    issues=f"target {target_index}: {feedback}",
                    candidate_steps=(accepted_steps + locals().get("turn_steps", [])),
                    model_output=text, usage=usage,
                )
                continue

            accepted_this_turn = turn_steps
            annotation_history.append({
                "target_index": target_index, "attempt": attempt, "ok": True,
                "model_output": text, "usage": usage,
                "n_insertions": len(insertions),
            })
            break

        if accepted_this_turn is None:
            _, _, base_steps = replay_validate(traj, spliced_sequence(traj, []))
            out = copy.deepcopy(traj)
            out["schema_version"] = "v3-enriched"
            out["label_status"] = "verified"
            out["initial_state"]["dataset_overview"] = catalog_snapshot(h0)
            out["steps"] = base_steps
            out["enrichment"] = {
                "status": "fallback_skeleton",
                "mode": "skeleton",
                "quality_status": "reject",
                "generator": {"model": model, "mode_requested": "stateful_step"},
                "n_perception": 0,
                "n_error": 0,
                "rejected_candidates": rejections,
                "annotation_history": annotation_history,
            }
            return out

        old_to_new[target_step["step_id"]] = accepted_this_turn[-1]["step_id"]
        added_perception += sum(
            1 for s in accepted_this_turn
            if (s.get("tool_call") or {}).get("tool") in PERCEPTION
            and (s.get("tool_call") or {}).get("tool") != target_tool
        )
        accepted_steps.extend(accepted_this_turn)

    q_hard, q_rep, q_style = quality_check(
        accepted_steps, traj.get("question", ""), catalog_snapshot(h0)
    )
    out = copy.deepcopy(traj)
    out["schema_version"] = "v3-enriched"
    out["label_status"] = "verified"
    out["initial_state"]["dataset_overview"] = catalog_snapshot(h0)
    out["steps"] = accepted_steps
    out["enrichment"] = {
        "status": "enriched",
        "mode": "stateful_step",
        "quality_status": "reject" if q_hard else ("repairable" if q_rep else "ready"),
        "repairable_issues": q_rep,
        "soft_issues": q_style,
        "hard_issues": q_hard,
        "generator": {"model": model, "mode_requested": "stateful_step"},
        "n_perception": added_perception,
        "n_error": 0,
        "rejected_candidates": rejections,
        "annotation_history": annotation_history,
    }
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
                             "quality_status": "reject",
                             "generator": {"model": model,
                                           "mode_requested": "semantic_rewrite"},
                             "n_perception": 0, "n_error": 0,
                             "rejected_candidates": rejections,
                             "annotation_history": history}
        return out

    insertions = dedupe_perception_insertions(traj, deterministic_perception_insertions(traj))
    ok, err, steps = replay_validate(traj, spliced_sequence(traj, insertions, rewrites))
    qhard, qrep, qstyle = quality_check(
        steps, traj.get("question", ""), catalog_snapshot(h)
    )
    out = copy.deepcopy(traj)
    out["schema_version"] = "v3-enriched"
    out["label_status"] = "verified"
    out["initial_state"]["dataset_overview"] = catalog_snapshot(h)
    if ok:
        out["steps"] = steps
        out["enrichment"] = {"status": "enriched", "mode": "semantic_rewrite",
                             "quality_status": "repairable" if qrep else "ready",
                             "repairable_issues": qrep, "soft_issues": qstyle,
                             "generator": {"model": model,
                                           "mode_requested": "semantic_rewrite"},
                             "n_perception": len(insertions), "n_error": 0,
                             "rejected_candidates": rejections,
                             "annotation_history": [{"attempt": 1, "ok": True,
                                                      "insertions": len(insertions),
                                                      "model_output": text, "usage": usage}]}
    else:
        issues = "; ".join(([err] if not ok else []))
        _remember_rejection(
            rejections, attempt=1, stage="semantic_rewrite_validation", issues=issues,
            candidate_steps=steps, model_output=text, usage=usage, ok=ok,
            programmatic_q_ok=not qhard,
            n_perception=len(insertions), n_error=0,
        )
        _, _, base_steps = replay_validate(traj, spliced_sequence(traj, []))
        out["steps"] = base_steps
        out["enrichment"] = {"status": "fallback_skeleton", "mode": "skeleton",
                             "quality_status": "reject",
                             "generator": {"model": model,
                                           "mode_requested": "semantic_rewrite"},
                             "n_perception": 0, "n_error": 0,
                             "rejected_candidates": rejections,
                             "annotation_history": [{"attempt": 1, "ok": False,
                                                      "issues": issues,
                                                      "model_output": text, "usage": usage}]}
    return out


def _issue_bucket(issue: str) -> str:
    if "leaks exact schema identifier" in issue:
        return "schema_leak"
    if "confirmed literal" in issue or "did not show it" in issue:
        return "unsupported_confirmation"
    if "stale observation wording" in issue:
        return "stale_observation_wording"
    if "final think uses annotator" in issue:
        return "final_answer_leak"
    if "actual tool is" in issue:
        return "tool_action_mismatch"
    if "does not name exact column" in issue:
        return "missing_action_column"
    if "does not name table/reference" in issue:
        return "missing_action_reference"
    if "before describe_table" in issue:
        return "unobserved_schema_use"
    if "before inspect_column" in issue:
        return "unobserved_literal_filter"
    if "third-person" in issue:
        return "third_person"
    if "empty" in issue:
        return "empty_think"
    return issue.split(":", 1)[0][:80]


def _enrichment_debug_summary(enr: dict) -> dict | None:
    history = enr.get("annotation_history") or []
    rejected = enr.get("rejected_candidates") or []
    if not history and not rejected:
        return None
    last = history[-1] if history else {}
    last_rejected = rejected[-1] if rejected else {}

    def length_of(key: str) -> int:
        value = last.get(key)
        if value is None:
            value = last_rejected.get(key)
        return len(value) if isinstance(value, str) else 0

    return {
        "annotation_attempts": len(history),
        "rejected_candidates": len(rejected),
        "latest_stage": last.get("stage") or last_rejected.get("stage"),
        "latest_issues": last.get("issues") or last_rejected.get("issues"),
        "latest_model_output_chars": length_of("model_output"),
        "latest_observation_model_output_chars": length_of("observation_model_output"),
        "raw_model_output_locations": [
            "enrichment.annotation_history[*].model_output",
            "enrichment.annotation_history[*].observation_model_output",
            "enrichment.rejected_candidates[*].model_output",
            "enrichment.rejected_candidates[*].observation_model_output",
        ],
    }


def quality_manifest(results: list[dict], *, out_path: str, args: argparse.Namespace) -> dict:
    status_counts = collections.Counter()
    stored_status_counts = collections.Counter()
    mode_counts = collections.Counter()
    generator_counts = collections.Counter()
    quality_source_counts = collections.Counter()
    issue_counts = collections.Counter()
    hard_issue_counts = collections.Counter()
    audit_decision_counts = collections.Counter()
    records = []
    for r in results:
        enr = r.get("enrichment") or {}
        generator = enr.get("generator") or {}
        source = enr.get("quality_status_source", "stored")
        if source == "programmatic" or (
            source == "stored" and getattr(args, "quality_mode", "programmatic") == "programmatic"
        ):
            hard, issues, soft = quality_check(
                r.get("steps", []), r.get("question", ""),
                r.get("initial_state", {}).get("dataset_overview"),
            )
            q = "reject" if hard else ("repairable" if issues else "ready")
        else:
            q = enr.get("quality_status", "missing")
            hard = enr.get("hard_issues", []) or []
            issues = enr.get("repairable_issues", []) or []
            soft = enr.get("soft_issues", []) or []
        stored_status_counts[enr.get("quality_status", "missing")] += 1
        status_counts[q] += 1
        mode_counts[enr.get("mode", "unknown")] += 1
        generator_counts[generator.get("model", "unknown")] += 1
        quality_source_counts[source] += 1
        for issue in issues:
            issue_counts[_issue_bucket(issue)] += 1
        for issue in hard:
            hard_issue_counts[_issue_bucket(issue)] += 1
        audit = enr.get("quality_audit") or {}
        if audit:
            audit_decision_counts[audit.get("decision", "unknown")] += 1
        tools = [s.get("tool_call", {}).get("tool") for s in r.get("steps", [])]
        record = {
            "trajectory_id": r.get("trajectory_id"),
            "question": r.get("question"),
            "db_id": (r.get("source") or {}).get("db_id"),
            "status": enr.get("status"),
            "stored_quality_status": enr.get("quality_status"),
            "quality_status": q,
            "mode": enr.get("mode"),
            "generator_model": generator.get("model"),
            "quality_status_source": source,
            "quality_audit_decision": audit.get("decision"),
            "quality_audit_summary": audit.get("summary"),
            "quality_audit_confidence": audit.get("confidence"),
            "n_steps": len(r.get("steps", [])),
            "n_perception": enr.get("n_perception", 0),
            "n_error": enr.get("n_error", 0),
            "tools": tools,
            "hard_issues": hard,
            "repairable_issues": issues,
            "soft_issues": soft,
            "programmatic_quality": enr.get("programmatic_quality"),
            "recommended_action": (
                "use_for_sft" if q == "ready" else
                "repair_then_recheck" if q == "repairable" else
                "drop_or_manual_review"
            ),
        }
        if q != "ready":
            debug = _enrichment_debug_summary(enr)
            if debug:
                record["debug"] = debug
        records.append(record)
    def manifest_scalar(counter: collections.Counter, fallback: str | None = None) -> str | None:
        keys = [key for key, count in counter.items() if count and key not in (None, "unknown")]
        if len(keys) == 1:
            return str(keys[0])
        if len(keys) > 1:
            return "mixed"
        return fallback

    return {
        "output": out_path,
        "model": manifest_scalar(generator_counts, args.model),
        "mode": manifest_scalar(mode_counts, args.mode),
        "which": "audit" if getattr(args, "audit_file", None) else args.which,
        "source_file": getattr(args, "audit_file", None) or args.subset_file,
        "subset_file": None if getattr(args, "audit_file", None) else args.subset_file,
        "n": len(results),
        "quality_status_counts": dict(status_counts),
        "stored_quality_status_counts": dict(stored_status_counts),
        "mode_counts": dict(mode_counts),
        "generator_model_counts": dict(generator_counts),
        "quality_status_source_counts": dict(quality_source_counts),
        "audit_decision_counts": dict(audit_decision_counts),
        "hard_issue_counts": dict(hard_issue_counts),
        "repairable_issue_counts": dict(issue_counts),
        "records": records,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=os.path.join(ROOT, "data", "trajectories", "subset_180.ids.json"))
    ap.add_argument("--which", default="smoke", choices=["smoke", "subset", "probe"])
    ap.add_argument("--subset-file", default=os.path.join(ROOT, "data", "trajectories", "subset_180.jsonl"))
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "trajectories", "smoke_enriched.jsonl"))
    ap.add_argument("--mode", choices=[
        "full", "perception_only", "semantic_rewrite", "staged_perception", "stateful_step"
    ],
                    default="full",
                    help=("full asks the LLM for rewrites+perception+optional corrections; "
                          "perception_only forbids correction/error insertions; semantic_rewrite "
                          "uses a deterministic observation scaffold and asks only for semantic thinks; "
                          "staged_perception first chooses describe_table from catalog-only state; "
                          "stateful_step prompts one target tool call at a time from current history"))
    ap.add_argument("--limit", type=int, default=0,
                    help="optional cap for quick smoke generation")
    ap.add_argument("--select-longest", action="store_true",
                    help="sort selected trajectories by descending step count before applying --limit")
    ap.add_argument("--api-timeout", type=int, default=180,
                    help="seconds per external LLM request before retry/fallback")
    ap.add_argument("--api-retries", type=int, default=4,
                    help="external LLM retries per annotation attempt")
    ap.add_argument("--max-attempts", type=int, default=10,
                    help="annotation attempts per trajectory after validation feedback")
    ap.add_argument("--workers", type=int, default=16,
                    help="parallel trajectory enrichment workers; external API is assumed to tolerate concurrency")
    ap.add_argument("--quality-mode", choices=["llm", "programmatic", "none"], default="llm",
                    help=("final quality decision source. llm uses an external model reviewer; "
                          "programmatic preserves the old rule-based status; none records no final review"))
    ap.add_argument("--quality-model", default=None,
                    help="external model used for quality review; default is --model")
    ap.add_argument("--max-repair", type=int, default=3,
                    help="max LLM think-repair rounds after a repair/reject audit (llm mode; 0 = off)")
    ap.add_argument("--quality-manifest", default=None,
                    help="path for per-trajectory quality manifest; default is OUT.quality_manifest.json")
    ap.add_argument("--audit-file", default=None,
                    help="recompute quality review/manifest for an existing enriched jsonl")
    ap.add_argument("--audit-out", default=None,
                    help="optional jsonl path for --audit-file after attaching refreshed quality review")
    ap.add_argument("--repair-file", default=None,
                    help="repair think/rationale text in an existing enriched jsonl without calling the API")
    ap.add_argument("--repair-out", default=None,
                    help="output jsonl for --repair-file; default is REPAIR_FILE.repaired.jsonl")
    ap.add_argument("--no-auto-repair", action="store_true",
                    help="disable deterministic text-only repair during new generation")
    args = ap.parse_args()

    if args.audit_file:
        with open(args.audit_file, encoding="utf-8") as f:
            source = [json.loads(line) for line in f if line.strip()]
        audit_model = args.quality_model or args.model
        api_key = api_base = None
        if args.quality_mode == "llm":
            api_key, api_base = load_api()
        results = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    review_and_repair,
                    traj,
                    mode=args.quality_mode,
                    base=api_base,
                    key=api_key,
                    model=audit_model,
                    max_repair=args.max_repair,
                    api_timeout=args.api_timeout,
                    api_retries=args.api_retries,
                ): traj
                for traj in source
            }
            for future in as_completed(futures):
                results.append(future.result())
        audit_out = args.audit_out
        if audit_out:
            with open(audit_out, "w", encoding="utf-8") as f:
                for traj in results:
                    f.write(json.dumps(traj, ensure_ascii=False, default=str) + "\n")
            review_out = review_path_for(audit_out)
            if review_out != audit_out:
                with open(audit_out, encoding="utf-8") as src, open(review_out, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
        q_manifest_path = args.quality_manifest or (args.audit_file + ".quality_manifest.json")
        manifest = quality_manifest(results, out_path=(audit_out or args.audit_file), args=args)
        with open(q_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
            f.write("\n")
        print(
            f"audited {len(results)} trajectories"
            f"\nquality {manifest['quality_status_counts']}"
            f"\nquality_source {manifest['quality_status_source_counts']}"
            f"\nissues {manifest['repairable_issue_counts']}"
            + (f"\n-> {audit_out}" if audit_out else "")
            + f"\n-> {q_manifest_path}"
        )
        return 0

    if args.repair_file:
        with open(args.repair_file, encoding="utf-8") as f:
            source = [json.loads(line) for line in f if line.strip()]
        repair_key = repair_base = None
        if args.quality_mode == "llm":
            repair_key, repair_base = load_api()
        results = []
        changed = 0
        for traj in source:
            repaired, n_changed = repair_reasoning(traj)
            repaired = review_and_repair(
                repaired,
                mode=args.quality_mode,
                base=repair_base,
                key=repair_key,
                model=args.quality_model or args.model,
                max_repair=args.max_repair,
                api_timeout=args.api_timeout,
                api_retries=args.api_retries,
            )
            results.append(repaired)
            changed += n_changed
        repair_out = args.repair_out or (args.repair_file + ".repaired.jsonl")
        with open(repair_out, "w", encoding="utf-8") as f:
            for traj in results:
                f.write(json.dumps(traj, ensure_ascii=False, default=str) + "\n")
        q_manifest_path = args.quality_manifest or (repair_out + ".quality_manifest.json")
        manifest = quality_manifest(results, out_path=repair_out, args=args)
        manifest["repair"] = {
            "input": args.repair_file,
            "changed_steps": changed,
            "method": "deterministic_think_repair",
        }
        with open(q_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
            f.write("\n")
        print(
            f"repaired {len(results)} trajectories; changed_steps={changed}"
            f"\nquality {manifest['quality_status_counts']}"
            f"\nissues {manifest['repairable_issue_counts']}"
            f"\n-> {repair_out}"
            f"\n-> {q_manifest_path}"
        )
        return 0

    key, base = load_api()
    audit_model = args.quality_model or args.model
    wanted = set(json.load(open(args.ids))[args.which])
    trajs = [normalize_legacy_memory(json.loads(l)) for l in open(args.subset_file) if l.strip()]
    trajs = [t for t in trajs if t["trajectory_id"] in wanted]
    if args.select_longest:
        trajs = sorted(trajs, key=lambda t: (-len(t.get("steps", [])), t.get("trajectory_id", "")))
    if args.limit:
        trajs = trajs[: args.limit]
    print(f"enriching {len(trajs)} ({args.which}) with {args.model}\n")

    def enrich_item(t: dict) -> tuple[dict | None, int, str | None]:
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
            elif args.mode == "stateful_step":
                r = enrich_one_stateful_step(
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
            return None, 0, f"{t['trajectory_id']:<20} ERROR: {type(e).__name__}: {e}"
        n_repaired = 0
        if not args.no_auto_repair:
            r, n_repaired = repair_reasoning(r)
        r = review_and_repair(
            r,
            mode=args.quality_mode,
            base=base,
            key=key,
            model=audit_model,
            max_repair=args.max_repair,
            api_timeout=args.api_timeout,
            api_retries=args.api_retries,
        )
        return r, n_repaired, None

    results = []
    done = 0
    with open(args.out, "w", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {pool.submit(enrich_item, t): t for t in trajs}
            for future in as_completed(futures):
                t = futures[future]
                r, n_repaired, error = future.result()
                done += 1
                if error:
                    print(f"  [{done}/{len(trajs)}] {error}")
                    continue
                assert r is not None
                results.append(r)
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
                f.flush()
                meta = r["enrichment"]
                print(f"  [{done}/{len(trajs)}] {r['trajectory_id']:<20} len={len(r['steps']):<3} "
                      f"+perc={meta['n_perception']:<2} +err={meta['n_error']:<2} "
                      f"{meta['mode']:<16} quality={meta.get('quality_status', 'unknown'):<10} "
                      f"repair={n_repaired:<2} db={r['source']['db_id']}")
    enr = sum(1 for r in results if r["enrichment"]["status"] == "enriched")
    full = sum(1 for r in results if r["enrichment"].get("mode") == "full")
    q_manifest_path = args.quality_manifest or (args.out + ".quality_manifest.json")
    manifest = quality_manifest(results, out_path=args.out, args=args)
    with open(q_manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
        f.write("\n")
    review_out = review_path_for(args.out)
    if review_out != args.out:
        with open(args.out, encoding="utf-8") as src, open(review_out, "w", encoding="utf-8") as dst:
            dst.write(src.read())
    print(
        f"\nenriched {enr}/{len(results)} (with errors {full})  fallback {len(results)-enr}"
        f"\nquality {manifest['quality_status_counts']}"
        f"\n-> {args.out}"
        f"\n-> {review_out}"
        f"\n-> {q_manifest_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
