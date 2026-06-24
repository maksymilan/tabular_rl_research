#!/usr/bin/env python3
"""Model<->harness interaction protocol — the single source of truth shared by
build_sft_data.py (renders training messages) and src/eval/rollout.py (renders live
messages and parses model output). One module for both guarantees the SFT data format and
the rollout format can never drift apart.

Message protocol (chat roles):
  system   : agent role + tool specs + interaction rules            (SYSTEM_PROMPT)
  user #1  : dataset overview JSON + the question                   (first_user_message)
  assistant: "<think>...</think>\n<tool_call>{...}</tool_call>"     (assistant_message)
  user     : compact JSON of the harness tool_output                (tool_output_message)
  ... repeats; the dialogue ends with the assistant's answer_from_context call
      (terminal — no tool_output follows it).
"""
from __future__ import annotations

import hashlib
import json
import re

# v0 action space = exactly the tools present in the compiled Spider data. Perception / fuzzy tools
# (inspect_column, semantic_match, ...) enter with the v1 data; exposing unlearned tools at eval
# time only invites illegal calls.
TOOL_SPECS: dict[str, str] = {
    "condition_filter":
        'condition_filter(table, conditions) -> new table with the rows that satisfy `conditions`.\n'
        '  conditions: a predicate {"column": c, "op": o, "value": v} with op in '
        '=,!=,>,>=,<,<= | {"op":"in","values":[..]} | {"op":"between","low":x,"high":y} | '
        '{"op":"like","value":pat} | {"op":"contains","value":s} | {"op":"is_null"} | '
        'column-vs-column via {"column":a,"op":o,"column_value":b}; '
        'a scalar computed by an earlier step via {"column":a,"op":o,"value_ref":step_id}; '
        'set membership against a computed table via {"column":a,"op":"in","in_table":table}; '
        'combine with {"and":[..]}, {"or":[..]}, {"not": ..}.',
    "project":
        'project(table, expressions) -> new table with the given columns. `expressions` is a list '
        'of column names or SQL scalar expressions, optionally with "expr AS alias".',
    "join_tables":
        'join_tables(left, right, on, join_type="inner", left_prefix=None, right_prefix=None) -> '
        'new table joining `left` and `right`. on: [{"left": col_in_left, "right": col_in_right}, ..]. '
        'join_type: inner|left|cross. When the two tables share column names, set left_prefix/'
        'right_prefix (short aliases): that side\'s columns are renamed to "<prefix>__<col>"; '
        'a side that is already prefixed (the result of an earlier join) keeps prefix=None and '
        'its columns are referenced as "<prefix>__<col>" in `on`.',
    "group_aggregate":
        'group_aggregate(table, group_by, aggregations, passthrough=None) -> new table grouped by '
        '`group_by` (list of columns; [] = whole table as one group). aggregations: '
        '[{"op": sum|count|count_distinct|mean|min|max, "column": col or "*", "as": name}, ..] '
        '([] with group_by = DISTINCT). passthrough: extra non-grouped columns to carry through.',
    "aggregate":
        'aggregate(table, column, op) -> a single scalar (op: sum|count|count_distinct|mean|min|max; '
        'column "*" allowed for count).',
    "extreme_value_select":
        'extreme_value_select(table, order_by, top_k=None, return_columns=None) -> new table with '
        'the rows ordered by `order_by` (list of "col" or "col DESC") keeping the top `top_k` '
        '(None = all rows, just ordered). return_columns optionally projects.',
    "set_op":
        'set_op(left, right, op) -> new table combining two tables with op: '
        'union|union_all|intersect|except (their columns must align).',
    "describe_table":
        'describe_table(tables) -> the columns, types, primary keys and foreign keys of one or more '
        'tables (tables is a list; pass several at once). The opening overview lists only table names '
        'and relations, so read the schema of the tables you need before operating on them.',
    "inspect_column":
        'inspect_column(table, column) -> the distinct count, most frequent values and NULL flag of a '
        'column. Use it to ground a filter literal (does "France" exist? what is the exact spelling?) '
        'before condition_filter.',
    "read_subtable":
        'read_subtable(table, limit=20) -> the actual rows of a table (bounded). Tool results otherwise '
        'show only a table handle (name, columns, row_count); read_subtable is how you SEE rows, e.g. '
        'the evidence rows before answering.',
    "answer_from_context":
        'answer_from_context(answer, evidence, reason) -> TERMINAL. answer: the '
        'result rows as a list of rows (each row a list of cells; at most 50 rows). evidence: {"table": '
        'name of the table holding the answer rows, or null for a scalar}. reason: one short sentence.',
}

TOOLS = set(TOOL_SPECS)

PROTOCOL_VERSION = "v2b"   # bump when specs, rendering, or the memory model change

# Strict per-tool argument schema (required, optional). Unlisted keys are rejected so the SFT data
# and the live rollout can never silently drift. V2b: a predicate's `value_ref` cites the producing
# step_id directly; there is no add_to_memory tool and no model-authored value.
_ARG_SCHEMA: dict[str, tuple[set, set]] = {
    "condition_filter": ({"table", "conditions"}, {"return_columns", "preview_k"}),
    "project": ({"table", "expressions"}, set()),
    "join_tables": ({"left", "right"}, {"on", "join_type", "left_prefix", "right_prefix", "return_columns"}),
    "group_aggregate": ({"table", "group_by", "aggregations"}, {"passthrough"}),
    "aggregate": ({"table", "column", "op"}, set()),
    "extreme_value_select": ({"table", "order_by"}, {"top_k", "return_columns"}),
    "set_op": ({"left", "right", "op"}, set()),
    "derive_column": ({"table", "new_column", "expression"}, set()),
    "describe_table": ({"tables"}, set()),
    "inspect_column": ({"table", "column"}, {"top_k"}),
    "read_subtable": ({"table"}, {"limit", "columns"}),
    "answer_from_context": ({"answer", "evidence"}, {"reason"}),
}


def validate_arguments(tool: str, args: dict) -> None:
    """Strict per-tool argument schema; raises ProtocolError on any missing/unexpected key."""
    schema = _ARG_SCHEMA.get(tool)
    if schema is None:
        return
    required, optional = schema
    keys = set(args)
    missing = required - keys
    if missing:
        raise ProtocolError(f"{tool}: missing arguments {sorted(missing)}")
    extra = keys - required - optional
    if extra:
        raise ProtocolError(f"{tool}: unexpected arguments {sorted(extra)}")


def protocol_hash() -> str:
    """Stable hash of the model<->harness contract (version + system prompt + tool specs). SFT
    manifests and rollout runs record it so a train/eval protocol mismatch is detectable."""
    payload = json.dumps({"version": PROTOCOL_VERSION, "system": SYSTEM_PROMPT, "tools": TOOL_SPECS},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

SYSTEM_PROMPT = (
    "You are a table-reasoning agent. You answer questions over a relational dataset by calling "
    "tools, one call per turn. The opening overview is a CATALOG: table names + row counts + "
    "foreign-key relations only (no columns) — so it stays small on large databases. Read the "
    "columns of the tables you need with describe_table before operating. Each tool result is an "
    "observation {\"step_id\", \"status\", \"output\"}: step_id names that step so you can cite it "
    "later (e.g. as a predicate's value_ref); a table-creating tool's output is only a HANDLE "
    "(table name, columns, row_count) — use read_subtable to SEE its rows.\n\n"
    "TOOLS\n" + "\n".join(TOOL_SPECS.values()) + "\n\n"
    "RULES\n"
    "1. Each turn, output exactly: <think>brief reasoning</think> then "
    '<tool_call>{"tool": "<name>", "arguments": {...}}</tool_call>. Nothing else.\n'
    "2. describe_table the needed tables first; inspect_column before filtering by a text value.\n"
    "3. To use a computed scalar as a threshold, set the predicate's "
    '{"value_ref": step_id} to the step that produced that scalar.\n'
    "4. read_subtable the evidence table, then finish with answer_from_context citing that table.\n"
)


class ProtocolError(Exception):
    """Model output does not parse into a legal tool call."""


def _compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def first_user_message(overview: dict, question: str) -> str:
    return f"DATASET OVERVIEW\n{_compact(overview)}\n\nQUESTION\n{question}"


def assistant_message(think: str, tool: str, arguments: dict) -> str:
    return (f"<think>{think}</think>\n"
            f"<tool_call>{_compact({'tool': tool, 'arguments': arguments})}</tool_call>")


def tool_output_message(step_id: str, output: dict, status: str = "success") -> str:
    """Observation envelope: a stable `step_id` (so the model can cite it as `source_step_id`),
    a status, and the tool output. Identical offline (SFT) and online (rollout)."""
    return _compact({"step_id": step_id, "status": status, "output": output})


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def parse_assistant(text: str) -> tuple[str, str, dict]:
    """Parse a model turn into (think, tool, arguments). Raises ProtocolError."""
    m = _TOOL_CALL_RE.findall(text)
    if not m:
        raise ProtocolError("no <tool_call>{...}</tool_call> block found")
    try:
        call = json.loads(m[-1])  # last block wins if the model quoted an example
    except json.JSONDecodeError as e:
        raise ProtocolError(f"tool_call is not valid JSON: {e}") from e
    tool = call.get("tool")
    args = call.get("arguments")
    if tool not in TOOLS:
        raise ProtocolError(f"unknown tool {tool!r}; legal tools: {sorted(TOOLS)}")
    if not isinstance(args, dict):
        raise ProtocolError('tool_call must have an "arguments" object')
    validate_arguments(tool, args)
    tm = _THINK_RE.search(text)
    return (tm.group(1).strip() if tm else ""), tool, args


# ---- answer comparison (execution-accuracy scoring) ----
def _cell(x) -> str:
    """Canonical cell: numbers normalized ('2014'==2014, 56.999999->'57'), text stripped."""
    if x is None:
        return ""
    if isinstance(x, bool):
        return str(int(x))
    s = str(x).strip()
    try:
        f = float(s)
    except ValueError:
        return s
    if abs(f - round(f)) < 1e-6:
        return str(int(round(f)))
    return f"{f:.4f}"


def normalize_rows(rows) -> list[tuple]:
    return sorted(tuple(_cell(c) for c in row) for row in rows)


def rows_equal(a, b) -> bool:
    return normalize_rows(a) == normalize_rows(b)
