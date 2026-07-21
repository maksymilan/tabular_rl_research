#!/usr/bin/env python3
"""Model<->harness interaction protocol — the single source of truth shared by
build_sft_data.py (renders training messages) and src/eval/rollout.py (renders live
messages and parses model output). One module for both guarantees the SFT data format and
the rollout format can never drift apart.

Message protocol (chat roles):
  system   : agent role + tool specs + interaction rules            (SYSTEM_PROMPT)
  user #1  : dataset overview JSON + the question                   (first_user_message)
  assistant: "<think>...</think>\n<tool_call>{...}</tool_call>"     (assistant_message)
  user     : CURRENT ENVIRONMENT STATE rendered from harness state  (state_context_message)
  ... repeats; the dialogue ends with the assistant's answer_from_context call.

Tool outputs are kept in the harness/debug event log, but the model-visible context is rebuilt
from resident environment state before each turn. This keeps SFT, eval and RL from accumulating a
duplicated transcript of old observations.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re

# v0 action space = exactly the tools present in the compiled Spider data. Perception / fuzzy tools
# (inspect_column, semantic_match, ...) enter with the v1 data; exposing unlearned tools at eval
# time only invites illegal calls.
TOOL_SPECS: dict[str, str] = {
    "plan":
        'plan(ops) -> update the task plan managed by the harness. Call this first to split the '
        'question into subgoals, and later to add/update/delete subgoals as observations change. '
        'ops: [{"op": create|add|update|delete, "id": "...", "goal": "...", '
        '"status": pending|in_progress|done|blocked, "evidence": "step_k"}]. '
        'A plan item has only goal/status/evidence: goal is the intended subtask, status is progress, '
        'and evidence is a prior step id whose actual tool output the harness will attach to the '
        'plan state. Do NOT write result/conclusion/notes/value fields or final answer values in the '
        'plan. The plan is control state only: it cannot be used as value_ref or final-answer support.',
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
        'join_tables(tables, on, join_types="inner", prefixes=None) -> ONE new table joining several '
        'tables along a path in a single step. tables: ordered list [T1, T2, .., TN] of source names '
        'or earlier step handles. on: a list of length N-1 where on[k] joins tables[k+1] to the tables '
        'already joined, each entry a list [{"left": key_in_accumulated, "right": key_in_next}, ..]. '
        'prefixes: optional COLUMN prefixes [P1, .., PN], not table names; when set, each table i\'s columns are renamed to '
        '"Pi__<col>" so shared / self-join names stay distinct, and `on[k].left` refers to an '
        'accumulated column by its "Pi__<col>" name. join_types: inner|left|cross for all folds, or a '
        'list per fold. Put a whole consecutive join chain in ONE call; joins in different subqueries '
        'stay separate calls.',
    "group_aggregate":
        'group_aggregate(table, group_by, aggregations, passthrough=None) -> new table grouped by '
        '`group_by` (list of columns; [] = whole table as one group, used for scalar count/sum/'
        'avg/min/max answers too). aggregations: '
        '[{"op": sum|count|count_distinct|mean|min|max, "column": col or "*", "as": name}, ..] '
        '([] with group_by = DISTINCT). passthrough: extra non-grouped columns to carry through.',
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
        'read_subtable(table, limit=20) -> up to 20 actual rows of a table (limit must be 1..20). Tool results otherwise '
        'show only a table handle (name, columns, row_count); read_subtable is how you SEE rows, e.g. '
        'the evidence rows before answering.',
    "answer_from_context":
        'answer_from_context(evidence, answer=[], reason="") -> TERMINAL. evidence: {"table": name} '
        'for a table holding the answer rows, or null for a scalar. For table answers, cite the table '
        'and keep answer empty or as a short preview; do NOT handwrite long row lists because the '
        'harness reads the cited evidence table. For scalar answers, put the scalar in answer.',
}

TOOLS = set(TOOL_SPECS)
# ``aggregate`` and the parser repairs below exist only to read historical trajectory artifacts.
# New model turns must use ``TOOLS`` through ``parse_assistant_strict``.  Keeping this distinction
# explicit prevents old data compatibility from quietly widening the live agent interface.
LEGACY_TOOLS = {"aggregate"}
REPLAY_COMPAT_TOOLS = TOOLS | LEGACY_TOOLS
ACCEPTED_TOOLS = REPLAY_COMPAT_TOOLS

PROTOCOL_VERSION = "v2i-state-only-join-feedback-r2"   # bump when specs, rendering, or the memory model change
ROLLING_CONTEXT_VERSION = "v2-bounded-legal-history-resident-observations"
ROLLING_COMPACT_PROMPT_VERSION = "v1-safe-compact"

# Strict per-tool argument schema (required, optional). Unlisted keys are rejected so the SFT data
# and the live rollout can never silently drift. V2b: a predicate's `value_ref` cites the producing
# step_id directly; there is no add_to_memory tool and no model-authored value.
_ARG_SCHEMA: dict[str, tuple[set, set]] = {
    "plan": ({"ops"}, set()),
    "condition_filter": ({"table", "conditions"}, {"return_columns", "preview_k"}),
    "project": ({"table", "expressions"}, set()),
    "join_tables": (set(), {"tables", "on", "join_types", "prefixes",
                            "left", "right", "join_type", "left_prefix", "right_prefix", "return_columns"}),
    "group_aggregate": ({"table", "group_by", "aggregations"}, {"passthrough"}),
    "aggregate": ({"table", "column", "op"}, set()),
    "extreme_value_select": ({"table", "order_by"}, {"top_k", "return_columns"}),
    "set_op": ({"left", "right", "op"}, set()),
    "derive_column": ({"table", "new_column", "expression"}, set()),
    "describe_table": ({"tables"}, set()),
    "inspect_column": ({"table", "column"}, {"top_k"}),
    "read_subtable": ({"table"}, {"limit", "columns"}),
    "answer_from_context": (set(), {"answer", "evidence", "reason"}),
}

# Kept only because old compiled trajectories predate the n-way public join shape. These fields
# are valid for replay, never for a model action in a new episode.
_MODEL_FORBIDDEN_ARGUMENTS: dict[str, set[str]] = {
    "join_tables": {"left", "right", "join_type", "left_prefix", "right_prefix"},
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
    if tool == "answer_from_context" and "answer" not in keys and "evidence" not in keys:
        raise ProtocolError('answer_from_context requires at least "evidence" or "answer"')


def validate_model_arguments(tool: str, args: dict) -> None:
    """Validate the current public action API, excluding replay-only compatibility forms."""
    validate_arguments(tool, args)
    forbidden = sorted(set(args) & _MODEL_FORBIDDEN_ARGUMENTS.get(tool, set()))
    if forbidden:
        raise ProtocolError(f"{tool}: legacy arguments are not valid in new episodes: {forbidden}")
    if tool == "join_tables":
        missing = sorted({"tables", "on"} - set(args))
        if missing:
            raise ProtocolError(f"join_tables: missing arguments {missing}")
    if tool == "read_subtable":
        limit = args.get("limit", 20)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ProtocolError("read_subtable: limit must be an integer from 1 to 20")


def protocol_hash(system_prompt: str | None = None) -> str:
    """Stable hash of the model<->harness contract (version + system prompt + tool specs). SFT
    manifests and rollout runs record it so a train/eval protocol mismatch is detectable."""
    payload = json.dumps({"version": PROTOCOL_VERSION, "system": system_prompt or SYSTEM_PROMPT, "tools": TOOL_SPECS},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

SYSTEM_PROMPT = (
    "You are a table-reasoning agent. You answer questions over a relational dataset by calling "
    "tools, one call per turn. The opening overview is a CATALOG: table names + row counts + "
    "foreign-key relations only (no columns) — so it stays small on large databases. When the task "
    "includes EXTERNAL KNOWLEDGE, treat it as part of the user-provided task context. Read the "
    "columns of the tables you need with describe_table before operating. After each tool call, "
    "the harness updates a CURRENT ENVIRONMENT STATE message. Treat that state as the authoritative "
    "workspace: it contains the resident plan, known schemas, inspected values, table handles, row "
    "reads, scalar values, and the step ids you may cite later (e.g. as a predicate's value_ref). "
    "You will not receive a full transcript of old tool observations; use the state instead of "
    "repeating previous reads. A table-producing tool's state entry is only a HANDLE (table name, "
    "columns, row_count) until you call read_subtable to see rows.\n\n"
    "TOOLS\n" + "\n".join(TOOL_SPECS.values()) + "\n\n"
    "RULES\n"
    "1. Each turn, output exactly: <think>brief reasoning</think> then "
    '<tool_call>{"tool": "<name>", "arguments": {...}}</tool_call>. Nothing else.\n'
    "2. Use plan for multi-step tasks to break the question into subgoals; update it when a subgoal "
    "starts, completes, or changes. Simple direct tasks may proceed without plan.\n"
    "3. describe_table the needed tables first; inspect_column before filtering by a text value.\n"
    "4. To use a computed scalar as a threshold, set the predicate's "
    '{"value_ref": step_id} to the step that produced that scalar.\n'
    "5. read_subtable the evidence table when row values are needed, then finish with "
    "answer_from_context citing that table. "
    "For row-valued answers, you may leave answer empty because the harness reads the evidence table; "
    "do not handwrite long row lists.\n"
)

SYSTEM_PROMPT_COMPACT = (
    "You are a table-tool agent. Answer by calling one tool per turn.\n\n"
    "STRICT FORMAT\n"
    "Each turn output exactly:\n"
    "<think>brief reason for this action</think>\n"
    '<tool_call>{"tool":"...","arguments":{...}}</tool_call>\n'
    "No text outside these tags.\n\n"
    "CONTEXT\n"
    "The opening overview is only a catalog: table names, row counts, and relations. It has no "
    "columns. Use describe_table only for relevant unresolved tables. Tool-created tables return "
    "handles (table, columns, row_count), not rows. Use existing handles instead of restarting from "
    "source tables. Use plan for multi-step tasks, then update it as work completes or changes. "
    "A separate CURRENT ENVIRONMENT STATE message may summarize the current plan and known table "
    "handles before your turn.\n\n"
    "POLICY\n"
    "Inspect a text column before filtering by a literal unless that column was already inspected. "
    "Avoid repeating the same observation. Use read_subtable only when row values are needed; for "
    "scalar aggregate answers, answer directly with evidence=null. Before a row-valued final answer, "
    "read the evidence table and cite it; keep answer empty or short for large tables.\n\n"
    "TOOLS\n"
    "plan(ops), describe_table(tables), inspect_column(table,column,top_k?), condition_filter(table,conditions), "
    "project(table,expressions), join_tables(tables,on,join_types?,prefixes?), "
    "group_aggregate(table,group_by,aggregations,passthrough?), "
    "extreme_value_select(table,order_by,top_k?,return_columns?), set_op(left,right,op), "
    "read_subtable(table,limit?,columns?), answer_from_context(answer,evidence,reason?).\n"
)

ROLLING_HISTORY_SYSTEM_SUFFIX = (
    "\n\nROLLING LEGAL HISTORY\n"
    "In this mode, the user may include a bounded transcript of earlier assistant actions that the "
    "harness executed successfully, paired with their tool-result messages. Continue from that "
    "legal history instead of restarting the task. The transcript is intentionally bounded, so do "
    "not assume it contains every old observation. CURRENT ENVIRONMENT STATE remains the "
    "authoritative factual workspace; LAST TOOL ERROR is the authoritative record of a rejected "
    "action. Never treat a plan item or unexecuted text as factual evidence."
)

# This is a rolling-only ablation. It deliberately retains the public action contract and the
# v2i join/value-reference rules that the generic compact prompt predates.
ROLLING_SYSTEM_PROMPT_COMPACT = (
    "You are a relational table-tool agent. Solve the user question with exactly one tool action "
    "per turn.\n\n"
    "FORMAT\n"
    "Output only <think>specific reason for the next action</think> followed by one complete "
    '<tool_call>{"tool":"name","arguments":{...}}</tool_call>. No prose outside the tags, no '
    "second action, no shorthand JSON, and no legacy tool fields.\n\n"
    "CONTEXT\n"
    "The first user message is a catalog of table names, row counts, and relations, not schemas. "
    "Call describe_table before using unresolved columns. CURRENT ENVIRONMENT STATE is the factual "
    "workspace: use its handles, schemas, inspected values, reads, scalar-producing step ids, and "
    "plan. A bounded transcript may contain only earlier harness-successful actions/results; it is "
    "continuity context, not complete evidence. LAST TOOL ERROR describes a rejected action. Do not "
    "repeat a read already present in state. Table handles expose metadata only until read_subtable "
    "returns rows.\n\n"
    "TOOLS\n"
    "plan(ops); describe_table(tables); inspect_column(table,column,top_k?); "
    "condition_filter(table,conditions); project(table,expressions); "
    "join_tables(tables,on,join_types?,prefixes?); "
    "group_aggregate(table,group_by,aggregations,passthrough?); "
    "extreme_value_select(table,order_by,top_k?,return_columns?); set_op(left,right,op); "
    "read_subtable(table,limit?,columns?); answer_from_context(answer?,evidence?,reason?).\n\n"
    "RULES\n"
    "plan is control only: goals/status/evidence may cite prior step ids, never results or answer "
    "values. Inspect text domains before literal filters unless already inspected. conditions support "
    "comparisons, like, in, between, null, and/or/not; cite a scalar as value_ref:step_id or a "
    "computed table as in_table. Use existing handles rather than restarting from sources. For an "
    "n-way join, tables are ordered and on has one edge list per newly attached table. With prefixes, "
    "use materialized P__column names only for accumulated left keys and bare source columns for the "
    "new right table; never emit L., R., or table.column aliases. For row answers, read the evidence "
    "handle then call answer_from_context with that evidence and an empty/short answer; for scalar "
    "answers cite evidence or give the scalar. The harness strictly validates and executes the action."
)


def get_system_prompt() -> str:
    """Return the default train/eval prompt, or a compact eval-only variant.

    The default remains SYSTEM_PROMPT so SFT data and existing protocol hashes stay stable.
    Set EVAL_SYSTEM_PROMPT_VARIANT=compact for prompt-ablation evaluations.
    """
    variant = os.environ.get("EVAL_SYSTEM_PROMPT_VARIANT", "").strip().lower()
    if variant in {"compact", "short"}:
        return SYSTEM_PROMPT_COMPACT
    return SYSTEM_PROMPT


def rolling_system_prompt(system_prompt: str, *, compact: bool = False) -> str:
    """Return the full or safe-compact bounded-history contract.

    ``compact`` is intentionally rolling-only and opt-in. The state-only protocol and existing
    full-prompt rolling artifacts remain byte-for-byte stable.
    """
    return ROLLING_SYSTEM_PROMPT_COMPACT if compact else system_prompt + ROLLING_HISTORY_SYSTEM_SUFFIX


class ProtocolError(Exception):
    """Model output does not parse into a legal tool call."""


def _compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def first_user_message(
    overview: dict,
    question: str,
    external_knowledge: str | None = None,
) -> str:
    message = f"DATASET OVERVIEW\n{_compact(overview)}\n\nQUESTION\n{question}"
    if external_knowledge and external_knowledge.strip():
        message += f"\n\nEXTERNAL KNOWLEDGE\n{external_knowledge.strip()}"
    return message


def assistant_message(think: str, tool: str, arguments: dict) -> str:
    return (f"<think>{think}</think>\n"
            f"<tool_call>{_compact({'tool': tool, 'arguments': arguments})}</tool_call>")


def tool_output_message(step_id: str, output: dict, status: str = "success",
                        state: dict | None = None) -> str:
    """Observation envelope: a stable `step_id` (so the model can cite it as `source_step_id`),
    a status, and the tool output. Identical offline (SFT) and online (rollout)."""
    msg = {"step_id": step_id, "status": status, "output": output}
    if state is not None:
        msg["state"] = state
    return _compact(msg)


def compact_resident_observation(observation: str) -> str:
    """Replace duplicated historical payloads with a structured resident-state pointer.

    Rolling context still carries the successful action/result pair, but schemas, inspected
    values, row samples, and scalar samples already live in CURRENT ENVIRONMENT STATE. Keeping
    those large facts in both places caused avoidable context overflow. Harness-authored metadata
    remains visible here; the complete factual payload remains visible once in resident state.
    Non-standard legacy strings are left untouched rather than guessed at.
    """
    try:
        envelope = json.loads(observation)
    except (TypeError, json.JSONDecodeError):
        return observation
    if not isinstance(envelope, dict) or envelope.get("status") != "success":
        return observation
    output = envelope.get("output")
    if not isinstance(output, dict):
        return observation

    summary: dict = {}
    for key in ("table", "kind", "columns", "row_count", "column", "distinct_count", "has_null"):
        if output.get(key) is not None:
            summary[key] = output[key]

    if isinstance(output.get("tables"), list):
        summary["tables"] = [
            {
                key: table[key]
                for key in ("table_name", "row_count")
                if isinstance(table, dict) and table.get(key) is not None
            }
            | ({"column_count": len(table.get("columns", []))} if isinstance(table, dict) else {})
            for table in output["tables"]
        ]
    if isinstance(output.get("rows"), list):
        summary["returned_row_count"] = len(output["rows"])
    if isinstance(output.get("result_sample"), list):
        summary["result_sample_row_count"] = len(output["result_sample"])
    if isinstance(output.get("frequent_values"), list):
        summary["frequent_value_count"] = len(output["frequent_values"])
    if isinstance(output.get("changes"), list):
        summary["changes"] = output["changes"]

    # Small outputs that contain no resident factual payload remain useful verbatim. Large factual
    # fields are represented by metadata above and resolved from the appended current state.
    resident_fields = {"tables", "rows", "result_sample", "frequent_values", "plan", "final_answer"}
    if not (set(output) & resident_fields):
        summary = output
    envelope = {
        "step_id": envelope.get("step_id"),
        "status": "success",
        "output_summary": summary,
        "full_output": "resident_in_current_environment_state",
    }
    return _compact(envelope)


def tool_error_message(step_id: str, error_type: str, message: str) -> str:
    return _compact({
        "step_id": step_id,
        "status": "error",
        "error": {"type": error_type, "message": message},
    })


def environment_state_message(state: dict | None, last_error: dict | None = None) -> str:
    text = "CURRENT ENVIRONMENT STATE\n" + _compact(state or {"plan": [], "tables": {}, "values": {}})
    if last_error:
        text += "\n\nLAST TOOL ERROR\n" + _compact(last_error)
    return text


def state_context_message(state: dict | None, last_error: dict | None = None) -> str:
    """Model-visible mutable context between assistant turns."""
    return environment_state_message(state, last_error=last_error)


def _state_is_empty(state: dict | None) -> bool:
    if not isinstance(state, dict):
        return True
    return not state.get("plan") and not state.get("tables") and not state.get("values")


def task_context_message(
    overview: dict,
    question: str,
    state: dict | None,
    last_error: dict | None = None,
    external_knowledge: str | None = None,
) -> str:
    """One-turn user context shared by online rollout and step-level SFT samples."""
    text = first_user_message(overview, question, external_knowledge)
    if not _state_is_empty(state) or last_error:
        text += "\n\n" + state_context_message(state, last_error)
    return text


def model_context_messages(system: str, overview: dict, question: str, state: dict | None,
                           last_error: dict | None = None,
                           external_knowledge: str | None = None) -> list[dict]:
    """Build the complete model-visible context for one turn from resident state.

    This is the state-only path used by online eval/RL. It deliberately ignores any accumulated
    debug transcript so old observations cannot re-enter the prompt.
    """
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": task_context_message(
            overview, question, state, last_error, external_knowledge,
        )},
    ]


def rolling_legal_history_messages(
    system: str,
    overview: dict,
    question: str,
    state: dict | None,
    last_error: dict | None,
    external_knowledge: str | None,
    legal_history: list[dict],
    history_turns: int,
    *,
    compact_observations: bool = True,
) -> list[dict]:
    """Render a bounded transcript of harness-successful assistant/tool pairs.

    Rejected assistant text never becomes context. Its structured error is carried only in the
    current user message, so training and online inference cannot teach the model to imitate an
    invalid call. ``history_turns=0`` retains every legal pair for experiments; production callers
    should use an explicit positive bound.
    """
    if history_turns < 0:
        raise ValueError("history_turns must be non-negative")
    initial = first_user_message(overview, question, external_knowledge)
    if not legal_history:
        content = initial
        if state or last_error:
            content += "\n\n" + state_context_message(state, last_error)
        return [{"role": "system", "content": system}, {"role": "user", "content": content}]

    retained = legal_history if history_turns == 0 else legal_history[-history_turns:]
    messages = [{"role": "system", "content": system}, {"role": "user", "content": initial}]
    for index, item in enumerate(retained):
        assistant = item.get("assistant")
        observation = item.get("observation")
        if not isinstance(assistant, str) or not assistant.strip():
            raise ValueError("rolling legal history has an empty assistant action")
        if not isinstance(observation, str) or not observation.strip():
            raise ValueError("rolling legal history has an empty tool observation")
        messages.append({"role": "assistant", "content": assistant})
        if compact_observations:
            observation = compact_resident_observation(observation)
        if index == len(retained) - 1:
            observation += "\n\n" + state_context_message(state, last_error)
        messages.append({"role": "user", "content": observation})
    return messages


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)


def _extract_balanced_json(text: str, start: int) -> str | None:
    """Extract one balanced JSON object from text[start:], respecting strings.

    This only helps when the model emits a complete JSON object but forgets the closing
    </tool_call> tag. Truncated JSON remains a protocol error.
    """
    begin = text.find("{", start)
    if begin < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(begin, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[begin:i + 1]
    return None


def _tool_call_payloads(text: str) -> list[str]:
    payloads = _TOOL_CALL_RE.findall(text)
    if payloads:
        return payloads
    tag = text.rfind("<tool_call>")
    if tag < 0:
        return []
    balanced = _extract_balanced_json(text, tag + len("<tool_call>"))
    return [balanced] if balanced else []


def _loads_tool_call(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Common unambiguous shorthand:
        #   {"answer_from_context","arguments":{...}}
        # Treat it as {"tool":"answer_from_context","arguments":{...}}.
        m = re.match(r'\s*\{\s*"(?P<tool>[A-Za-z_][\w]*)"\s*,\s*"arguments"\s*:\s*(?P<args>\{.*\})\s*\}\s*$', raw, re.S)
        if m and m.group("tool") in ACCEPTED_TOOLS:
            return {"tool": m.group("tool"), "arguments": json.loads(m.group("args"))}
        raise


_HANDLE_RE = re.compile(r"\b(?:project|filter|join|group|top|setop|derive)_\d{3}\b")


def _repair_truncated_answer_call(text: str) -> dict | None:
    """Recover an answer_from_context call when only the long answer JSON was truncated.

    This is intentionally narrow: it only synthesizes the terminal call when the model clearly
    attempted answer_from_context and cited a concrete evidence table handle. The harness still
    scores the cited table; no answer values are guessed from free text.
    """
    if "answer_from_context" not in text:
        return None
    table = None
    m = re.search(r'"evidence"\s*:\s*\{\s*"table"\s*:\s*"([^"]+)"', text)
    if m:
        table = m.group(1)
    if table is None:
        m = re.search(r"evidence table\s+`?((?:project|filter|join|group|top|setop|derive)_\d{3})`?", text, re.I)
        if m:
            table = m.group(1)
    if table is None:
        handles = _HANDLE_RE.findall(text)
        if handles:
            table = handles[-1]
    if table is None:
        return None
    return {
        "tool": "answer_from_context",
        "arguments": {
            "answer": [],
            "evidence": {"table": table},
            "reason": "Derived by the cited evidence table.",
        },
    }


def _normalize_answer_args(args: dict) -> dict:
    args = dict(args)
    evidence = args.get("evidence")
    if isinstance(evidence, str):
        args["evidence"] = {"table": evidence}
    elif evidence is None and "evidence" not in args:
        args["evidence"] = None
    if "answer" not in args:
        args["answer"] = []
    return args


def parse_assistant(text: str) -> tuple[str, str, dict]:
    """Parse a model turn into (think, tool, arguments). Raises ProtocolError."""
    m = _tool_call_payloads(text)
    if not m:
        call = _repair_truncated_answer_call(text)
        if call is None:
            raise ProtocolError("no <tool_call>{...}</tool_call> block found")
    else:
        try:
            call = _loads_tool_call(m[-1])  # last block wins if the model quoted an example
        except json.JSONDecodeError as e:
            call = _repair_truncated_answer_call(text)
            if call is None:
                raise ProtocolError(f"tool_call is not valid JSON: {e}") from e
    tool = call.get("tool")
    args = call.get("arguments")
    if tool is None:
        # Another common answer shorthand:
        #   {"answer_from_context": {"answer": ..., "evidence": ...}}
        shorthand = [(key, value) for key, value in call.items() if key in ACCEPTED_TOOLS]
        if len(shorthand) == 1:
            tool, value = shorthand[0]
            args = value.get("arguments") if isinstance(value, dict) and isinstance(value.get("arguments"), dict) else value
    if tool not in ACCEPTED_TOOLS:
        raise ProtocolError(f"unknown tool {tool!r}; legal tools: {sorted(TOOLS)}")
    if not isinstance(args, dict):
        raise ProtocolError('tool_call must have an "arguments" object')
    if tool == "answer_from_context":
        args = _normalize_answer_args(args)
    validate_arguments(tool, args)
    tm = _THINK_RE.search(text)
    return (tm.group(1).strip() if tm else ""), tool, args


def parse_assistant_strict(text: str) -> tuple[str, str, dict]:
    """Parse one fully-formed teacher action without repair or argument normalization.

    External teacher generation may use this mode when protocol mistakes should become explicit
    environment feedback. It intentionally rejects the legacy convenience repairs in
    :func:`parse_assistant`: balanced JSON without a closing tag, shorthand tool-call objects,
    truncated terminal answers, and omitted answer fields.
    """
    payloads = _TOOL_CALL_RE.findall(text)
    if len(payloads) != 1:
        raise ProtocolError(
            "expected exactly one complete <tool_call>{...}</tool_call> block; "
            f"received {len(payloads)}"
        )
    try:
        call = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"tool_call is not valid JSON: {exc}") from exc
    if not isinstance(call, dict):
        raise ProtocolError("tool_call JSON must be an object")
    if set(call) != {"tool", "arguments"}:
        raise ProtocolError('tool_call must contain exactly "tool" and "arguments" keys')
    tool = call.get("tool")
    args = call.get("arguments")
    if tool not in TOOLS:
        raise ProtocolError(f"unknown tool {tool!r}; legal tools: {sorted(TOOLS)}")
    if not isinstance(args, dict):
        raise ProtocolError('tool_call must have an "arguments" object')
    think_blocks = _THINK_RE.findall(text)
    if len(think_blocks) != 1 or not think_blocks[0].strip():
        raise ProtocolError("expected exactly one non-empty <think>...</think> block")
    validate_model_arguments(tool, args)
    return think_blocks[0].strip(), tool, args


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
    if math.isnan(f):
        return "nan"
    if math.isinf(f):
        return "inf" if f > 0 else "-inf"
    if abs(f - round(f)) < 1e-6:
        return str(int(round(f)))
    return f"{f:.4f}"


def normalize_rows(rows) -> list[tuple]:
    return sorted(tuple(_cell(c) for c in row) for row in rows)


def rows_equal(a, b) -> bool:
    return normalize_rows(a) == normalize_rows(b)
