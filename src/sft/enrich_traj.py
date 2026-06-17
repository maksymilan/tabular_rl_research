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
from rollout import db_path, execute_tool, new_ctx, score                 # noqa: E402
from fill_think import call, load_api                                     # noqa: E402

PERCEPTION = ("describe_table", "inspect_column", "read_subtable")
STR_OPS = ("=", "==", "contains", "like", "in")
BANNED_NARRATION = (
    "the model", "a model", "the agent", "the assistant", "might guess",
    "would guess", "without first checking",
)
FIRST_PERSON = re.compile(r"\b(I|my|me|I'll|I'm|I will|I need|I see|I should)\b", re.I)
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SQL_WORDS = {
    "as", "asc", "desc", "and", "or", "not", "in", "is", "null", "count", "distinct",
    "sum", "avg", "mean", "min", "max", "case", "when", "then", "else", "end",
}


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
    "You enrich a VERIFIED relational tool-trajectory so it teaches a model HOW to acquire the "
    "information each action needs BEFORE taking that action. You are writing the assistant's "
    "<think> content, so every reason must be FIRST PERSON from inside the task: say 'I need...', "
    "'I see...', 'I should...'. Never write third-person narration such as 'the model might...'.\n"
    "CRITICAL: at inference the model starts with ONLY the catalog — table names, row counts and "
    "foreign-key relations, but NO column names and NO cell values. So it cannot operate on a table "
    "until it has learned that table's columns, and cannot filter on a text value until it has "
    "confirmed that value. You INSERT the read-only perception steps that satisfy each backbone "
    "action's information preconditions (you never change the relational backbone), in order: survey "
    "globally -> acquire the columns of the tables you will touch -> confirm any literal/intermediate "
    "the next action depends on -> execute. Give a one-sentence first-person think for each.\n\n"
    "Reasoning quality requirement:\n"
    "- Every rewritten backbone think must explain the semantic bridge from the QUESTION to the "
    "exact table/column choice. Do not say only 'I project the output columns'. Say why these "
    "columns are the requested fields.\n"
    "- Mention exact column names when choosing them, e.g. `Song_Name` because the question asks for "
    "song names, `Song_release_year` because it asks for release years, `Age` because it asks for "
    "the youngest singer.\n"
    "- The trajectory should look like a human reading a table: first get global structure "
    "(candidate tables and columns), then local evidence (values or intermediate rows), then act.\n\n"
    "Perception tools:\n"
    "- describe_table {\"tables\":[...]} : acquire columns/types/keys. REQUIRED before the first time "
    "the trajectory operates on a table, since the catalog carries no columns.\n"
    "- inspect_column {\"table\":t,\"column\":c} : see a column's real values. REQUIRED before a filter "
    "comparing that column to a STRING literal, so the literal is grounded, not guessed. Use the "
    "SOURCE table + base column name even if the backbone filters it post-join under a prefix.\n"
    "- read_subtable {\"table\":handle} : inspect an intermediate result's rows to confirm a filter/"
    "join produced what the next step assumes (e.g. non-empty) before building on it.\n\n"
    "You may also add a FEW realistic SELF-CORRECTION moments so the model learns to recover from "
    "mistakes (the verified backbone never shows this). An error element attempts a WRONG variant of "
    "the next backbone action only when the wrong action is justified by the current uncertainty. "
    "For example: if I have not inspected a text column yet, I may try an ungrounded literal and get "
    "0 rows; if two tables/columns are semantically ambiguous, I may probe one and then recover. "
    "Do NOT guess a nonexistent column after describe_table has already shown the real columns. "
    "Do NOT add an error merely for drama.\n\n"
    "Rules:\n"
    "1. Insert exactly the perception needed so every action's required info is already acquired; do "
    "not insert a read whose information is already known or never used.\n"
    "2. Prefer a read_subtable to CONFIRM a filter/join's rows whenever the next step's correctness "
    "depends on that intermediate (e.g. before an aggregate or the final answer).\n"
    "3. Add error elements only where a mistake is genuinely plausible from the observations seen so "
    "far; 0-2 per trajectory, not every step.\n"
    "4. All `think` and `recovery_think` strings must be first person. Bad: \"The model might guess "
    "acting\". Good: \"I have not checked the schema yet, so I first need to inspect the table instead "
    "of guessing a column name.\"\n"
    "5. Output ONLY a JSON object with two keys: `rewrites` and `insertions`.\n"
    "`rewrites` must contain one entry for EVERY backbone step: "
    "{\"step\": <backbone index>, \"think\": <first-person semantic reason>}.\n"
    "`insertions` contains optional perception/correction steps. Perception element: "
    "{\"after\": <backbone step index, -1 = before "
    "the first step>, \"tool\": <describe_table|inspect_column|read_subtable>, \"arguments\": {...}, "
    "\"think\": <first-person sentence>}. Error element: {\"after\": <index>, \"tool\": \"error\", "
    "\"wrong\": {\"tool\": <relational tool>, \"arguments\": {...the WRONG variant...}}, "
    "\"think\": <first-person reason for trying it>, \"recovery_think\": <first-person sentence: "
    "how the error/empty result changes my next action>}."
)


def build_prompt(traj: dict, domains: dict) -> list[dict]:
    ov = traj["initial_state"]["dataset_overview"]
    h = Harness(db_path(traj["source"]["db_id"]))
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
    return [{"role": "system", "content": SYS}, {"role": "user", "content": user}]


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


def parse_annotations(text: str) -> tuple[list[dict], dict[int, str]]:
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

    rewrites: dict[int, str] = {}
    if isinstance(raw_rewrites, dict):
        for k, v in raw_rewrites.items():
            if str(k).lstrip("-").isdigit() and isinstance(v, str):
                rewrites[int(k)] = v
    elif isinstance(raw_rewrites, list):
        for item in raw_rewrites:
            if isinstance(item, dict) and isinstance(item.get("think"), str):
                rewrites[int(item.get("step", -999))] = item["think"]

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
            })
    return out


def parse_insertions(text: str) -> list[dict]:
    """Backward-compatible helper for older callers/tests."""
    return parse_annotations(text)[0]


# ---------------------------------------------------------------- splice + replay (L1) -----------
def _remap_mem(memid, m: dict):
    if isinstance(memid, str) and memid.startswith("mem_"):
        return "mem_" + m.get(memid[4:], memid[4:])
    return memid


def _remap_cond(cond, m: dict):
    if not isinstance(cond, dict):
        return
    for c in cond.get("conditions", []):
        _remap_cond(c, m)
    if "value_ref" in cond:
        cond["value_ref"] = _remap_mem(cond["value_ref"], m)


def _remap_step_refs(tool: str, args: dict, m: dict):
    """Rewrite step-id references after perception insertion shifts the numbering: add_to_memory's
    source_step_id, condition_filter value_refs, and the answer's supporting_memory_ids."""
    if tool == "add_to_memory" and "source_step_id" in args:
        args["source_step_id"] = m.get(args["source_step_id"], args["source_step_id"])
    if tool == "condition_filter":
        _remap_cond(args.get("conditions"), m)
    if tool == "answer_from_context" and isinstance(args.get("supporting_memory_ids"), list):
        args["supporting_memory_ids"] = [_remap_mem(x, m) for x in args["supporting_memory_ids"]]


def spliced_sequence(traj: dict, insertions: list[dict],
                     rewrites: dict[int, str] | None = None) -> list[dict]:
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
        think = rewrites.get(i, s.get("think", ""))
        if pending_recovery:
            think = " ".join(pending_recovery) + " " + think
            pending_recovery.clear()
        seq.append({"tool": s["tool_call"]["tool"], "arguments": copy.deepcopy(s["tool_call"]["arguments"]),
                    "reason": think, "backbone": True})
        old_to_new[s["step_id"]] = f"step_{len(seq)}"

    def emit_ann(a):
        if a.get("tool") == "error":
            w = a["wrong"]
            seq.append({"tool": w.get("tool"), "arguments": w.get("arguments", {}),
                        "reason": a.get("think", ""), "error_attempt": True})
            if a.get("recovery_think"):
                pending_recovery.append(a["recovery_think"])
        else:
            seq.append({"tool": a["tool"], "arguments": a.get("arguments", {}), "reason": a.get("think", "")})

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
            enriched.append({"step_id": sid, "think": step["reason"], "error_attempt": True,
                             "tool_call": {"tool": tool, "arguments": args},
                             "tool_status": obs[0], "tool_output": obs[1]})
            continue
        if tool == "answer_from_context":
            enriched.append({"step_id": sid, "think": step["reason"], "tool_call": {"tool": tool, "arguments": args}})
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
        enriched.append({"step_id": sid, "think": step["reason"],
                         "tool_call": {"tool": tool, "arguments": args},
                         "tool_output": out, "perception": tool in PERCEPTION})
    return False, "trajectory has no answer_from_context", enriched


# ---------------------------------------------------------------- load-bearing check (L2) --------
def check_load_bearing(traj: dict, insertions: list[dict]) -> tuple[bool, list[str]]:
    """Each inserted read must be CONSUMED downstream, else it is ritual. inspect_column matches (by
    base column name) a later string filter; read_subtable reads a produced/referenced table;
    describe_table describes a table the backbone references (including join sources left/right)."""
    steps = traj["steps"]
    str_cols = {c.lower() for c in string_filter_cols(steps)}
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
            if _base_col(args.get("column", "")).lower() not in str_cols:
                issues.append(f"inspect_column {args.get('column')} matches no later string filter (ritual)")
        elif tool == "read_subtable":
            tbl = args.get("table")
            tbl_key = tbl.lower() if isinstance(tbl, str) else tbl
            if tbl_key not in handles and tbl_key not in referenced:
                issues.append(f"read_subtable {tbl} reads no produced/backbone table")
        elif tool == "describe_table":
            tabs = {t.lower() for t in (args.get("tables") or [])}
            if not (tabs & referenced):
                issues.append(f"describe_table {sorted(tabs)} describes no table the backbone uses")
    return (not issues), issues


def _text_quality_issue(text: str, label: str, require_first_person: bool) -> str | None:
    lowered = text.lower()
    for phrase in BANNED_NARRATION:
        if phrase in lowered:
            return f"{label} uses third-person / meta narration: {phrase!r}"
    if require_first_person and not FIRST_PERSON.search(text):
        return f"{label} is not written as first-person task reasoning"
    return None


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


def quality_check(enriched_steps: list[dict]) -> tuple[bool, list[str]]:
    """Style + plausibility gate. Execution correctness alone is not enough for SFT: the generated
    turns must read like the model's own reasoning, and correction attempts must be consistent with
    what the model had already observed."""
    issues: list[str] = []
    for i, step in enumerate(enriched_steps):
        think = step.get("think", "")
        tool_call = step.get("tool_call") or {}
        tool = tool_call.get("tool")
        args = tool_call.get("arguments") or {}
        issue = _text_quality_issue(think, f"step {i + 1} think", require_first_person=True)
        if issue:
            issues.append(issue)
        terms = expected_semantic_terms(tool, args)
        low = think.lower()
        if terms and not any(term.lower() in low for term in terms):
            issues.append(
                f"step {i + 1} think does not justify its concrete table/column choice; "
                f"expected one of {sorted(terms)[:8]}"
            )

        described = _described_columns(enriched_steps, i)
        for table in _table_refs_for_action(tool, args):
            if not _is_derived_table(table) and table.lower() not in described:
                issues.append(
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
                        issues.append(
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
                issues.append(
                    f"step {i + 1} guesses missing column {table}.{column} after describe_table "
                    f"already showed the schema"
                )
        cond = args.get("conditions")
        if isinstance(table, str) and table.lower() in columns_seen and isinstance(cond, dict):
            for col in string_filter_cols([{"tool_call": {"tool": "condition_filter",
                                                          "arguments": {"conditions": cond}}}]):
                if col.lower() not in columns_seen[table.lower()]:
                    issues.append(
                        f"step {i + 1} filters on missing column {table}.{col} after schema was known"
                    )
    return (not issues), issues


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


def enrich_one(traj: dict, base: str, key: str, model: str, max_attempts: int = 3,
               api_timeout: int = 180, api_retries: int = 4) -> dict:
    h = Harness(db_path(traj["source"]["db_id"]))
    overview = traj["initial_state"]["dataset_overview"]
    domains = column_domains(h, overview, string_filter_cols(traj["steps"]))
    history: list[dict] = []
    best = None  # perception-only is the reliable floor; errors are added ONLY when they validate,
    feedback = ""  # so the error pass can never drag a trajectory below its perception enrichment.
    for attempt in range(1, max_attempts + 1):
        ctx_domains = dict(domains)
        if feedback:
            ctx_domains["_feedback"] = feedback
        try:
            text, _ = call_retry(
                base, key, model, build_prompt(traj, ctx_domains),
                tries=api_retries, timeout=api_timeout,
            )
        except Exception as e:  # noqa: BLE001
            history.append({"attempt": attempt, "error": f"api: {e}"})
            continue
        ann, rewrites = parse_annotations(text)
        missing_rewrites = [
            i for i in range(len(traj["steps"]))
            if i not in rewrites or not str(rewrites.get(i, "")).strip()
        ]
        if missing_rewrites:
            feedback = (
                "missing semantic first-person rewrites for backbone steps "
                f"{missing_rewrites}; rewrite every original step and explain exact table/column choices"
            )
            history.append({"attempt": attempt, "perc": 0, "errs": 0,
                            "perception_ok": False, "issues": feedback})
            continue
        perc = [a for a in ann if a.get("tool") in PERCEPTION]
        errs = [a for a in ann if a.get("tool") == "error"]
        l2_ok, l2_issues = check_load_bearing(traj, perc)
        l1p_ok, l1p_err, enr_p = replay_validate(traj, spliced_sequence(traj, perc, rewrites))
        q_ok, q_issues = quality_check(enr_p)
        if not (l1p_ok and l2_ok and q_ok):
            feedback = "; ".join(([l1p_err] if not l1p_ok else []) + l2_issues + q_issues) or "no valid perception"
            history.append({"attempt": attempt, "perc": len(perc), "errs": len(errs),
                            "perception_ok": False, "issues": feedback})
            continue
        best = {"steps": enr_p, "mode": "perception_only", "n_perception": len(perc), "n_error": 0}
        if not errs:
            history.append({"attempt": attempt, "perc": len(perc), "errs": 0, "ok": True})
            break
        l1f_ok, l1f_err, enr_f = replay_validate(traj, spliced_sequence(traj, perc + errs, rewrites))
        qf_ok, qf_issues = quality_check(enr_f)
        if l1f_ok and qf_ok:
            best = {"steps": enr_f, "mode": "full", "n_perception": len(perc), "n_error": len(errs)}
            history.append({"attempt": attempt, "perc": len(perc), "errs": len(errs), "ok": True})
            break
        feedback = "perception is correct; fix ONLY the error attempts: " + "; ".join(
            ([l1f_err] if not l1f_ok else []) + qf_issues
        )
        history.append({"attempt": attempt, "perc": len(perc), "errs": len(errs),
                        "perception_ok": True, "errors_ok": False, "issues": feedback})
    if best:
        out = copy.deepcopy(traj)
        out["schema_version"] = "v2-ctx-enriched"
        out["label_status"] = "verified"
        out["initial_state"]["dataset_overview"] = catalog_snapshot(h)
        out["steps"] = best["steps"]
        out["enrichment"] = {"status": "enriched", "mode": best["mode"],
                             "n_perception": best["n_perception"], "n_error": best["n_error"],
                             "annotation_history": history}
        return out
    _, _, base_steps = replay_validate(traj, spliced_sequence(traj, []))
    out = copy.deepcopy(traj)
    out["schema_version"] = "v2-ctx-enriched"
    out["label_status"] = "verified"
    out["initial_state"]["dataset_overview"] = catalog_snapshot(h)
    out["steps"] = base_steps
    out["enrichment"] = {"status": "fallback_skeleton", "mode": "skeleton",
                         "n_perception": 0, "n_error": 0, "annotation_history": history}
    return out


def enrich_one_rewrite_only(traj: dict, base: str, key: str, model: str,
                            api_timeout: int = 180, api_retries: int = 1) -> dict:
    """Lightweight smoke path: deterministic global/local observation scaffold, external LLM only
    rewrites semantic reasoning for backbone actions."""
    h = Harness(db_path(traj["source"]["db_id"]))
    history: list[dict] = []
    try:
        text, _ = call_retry(
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
        out["schema_version"] = "v2-ctx-enriched"
        out["label_status"] = "verified"
        out["initial_state"]["dataset_overview"] = catalog_snapshot(h)
        out["steps"] = base_steps
        out["enrichment"] = {"status": "fallback_skeleton", "mode": "skeleton",
                             "n_perception": 0, "n_error": 0, "annotation_history": history}
        return out

    insertions = deterministic_perception_insertions(traj)
    ok, err, steps = replay_validate(traj, spliced_sequence(traj, insertions, rewrites))
    qok, qissues = quality_check(steps)
    out = copy.deepcopy(traj)
    out["schema_version"] = "v2-ctx-enriched"
    out["label_status"] = "verified"
    out["initial_state"]["dataset_overview"] = catalog_snapshot(h)
    if ok and qok:
        out["steps"] = steps
        out["enrichment"] = {"status": "enriched", "mode": "semantic_rewrite",
                             "n_perception": len(insertions), "n_error": 0,
                             "annotation_history": [{"attempt": 1, "ok": True,
                                                      "insertions": len(insertions)}]}
    else:
        _, _, base_steps = replay_validate(traj, spliced_sequence(traj, []))
        out["steps"] = base_steps
        out["enrichment"] = {"status": "fallback_skeleton", "mode": "skeleton",
                             "n_perception": 0, "n_error": 0,
                             "annotation_history": [{"attempt": 1, "ok": False,
                                                      "issues": "; ".join(([err] if not ok else []) + qissues)}]}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=os.path.join(ROOT, "data", "trajectories", "subset_180.ids.json"))
    ap.add_argument("--which", default="smoke", choices=["smoke", "subset"])
    ap.add_argument("--subset-file", default=os.path.join(ROOT, "data", "trajectories", "subset_180.jsonl"))
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "trajectories", "smoke_enriched.jsonl"))
    ap.add_argument("--mode", choices=["full", "semantic_rewrite"], default="full",
                    help="full asks the LLM for rewrites+insertions; semantic_rewrite uses a deterministic observation scaffold and asks only for semantic thinks")
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
    trajs = [json.loads(l) for l in open(args.subset_file) if l.strip()]
    trajs = [t for t in trajs if t["trajectory_id"] in wanted]
    if args.limit:
        trajs = trajs[: args.limit]
    print(f"enriching {len(trajs)} ({args.which}) with {args.model}\n")

    results = []
    with open(args.out, "w", encoding="utf-8") as f:
        for t in trajs:
            if args.mode == "semantic_rewrite":
                r = enrich_one_rewrite_only(
                    t, base, key, args.model,
                    api_timeout=args.api_timeout, api_retries=args.api_retries,
                )
            else:
                r = enrich_one(
                    t, base, key, args.model,
                    max_attempts=args.max_attempts,
                    api_timeout=args.api_timeout, api_retries=args.api_retries,
                )
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
