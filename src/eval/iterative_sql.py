#!/usr/bin/env python3
"""Evaluate an external model through an iterative-SQL feedback interface.

This is an interface ablation, not a model-visible extension of the typed tool protocol. The model
may start from the same lazy BIRD catalog or a separately versioned full-schema/value-sample context.
It executes read-only SQLite statements to inspect schema, probe data, and debug errors before
submitting one final SQL query for hidden denotation scoring.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "sft"))

from artifacts import ArtifactWriter  # noqa: E402
from action_carrier import (  # noqa: E402
    ACTIVE_ACTION_CARRIER,
    ActionCarrierError,
    parse_action_carrier,
)
from denotation import add_denotation_comparison_argument, compare_denotations  # noqa: E402
from direct_sql_search_protocol import (  # noqa: E402
    DIRECT_SQL_SEARCH_INTERFACE,
    DIRECT_SQL_SEARCH_INTERFACE_V1,
    DIRECT_SQL_SEARCH_INTERFACES,
    DIRECT_SQL_SEARCH_PROTOCOL_VERSION,
    DIRECT_SQL_SEARCH_PROTOCOL_VERSION_V1,
    build_direct_sql_search_system_prompt,
    direct_sql_search_protocol_hash,
    direct_sql_search_tool_schema_hash,
    parse_direct_sql_search_action,
)
from executor import Harness  # noqa: E402
from generate_teacher_rollouts import add_usage, chat_with_retries  # noqa: E402
from protocol import ProtocolError  # noqa: E402
from provider_adapter import (  # noqa: E402
    adapt_provider_response,
    is_deepseek_split_model,
    provider_default_max_tokens,
    provider_instruction,
    provider_request_messages,
    provider_request_options,
    provider_rejection_message,
)
from provider_client import load_api_config  # noqa: E402
from rollout import (  # noqa: E402
    ChatAPIError,
    ContextOverflowError,
    overview,
    task_db_path,
    task_gold_sql,
)
from text2sql import execute_predicted_sql  # noqa: E402


SQL_TOOLS = frozenset({"execute_sql", "submit_sql"})
READ_ONLY_RE = re.compile(r"^(?:SELECT|WITH|PRAGMA|EXPLAIN\s+QUERY\s+PLAN)\b", re.I)
SUBMIT_RE = re.compile(r"^(?:SELECT|WITH)\b", re.I)
ITERATIVE_SQL_PROTOCOL_VERSION = "iterative-sql-v2"
ITERATIVE_SQL_INTERFACE = "execute_sql_submit_sql_v2"
SQL_INTERFACE_CHOICES = (
    ITERATIVE_SQL_INTERFACE,
    *DIRECT_SQL_SEARCH_INTERFACES,
)
LAZY_CATALOG_CONTEXT = "lazy-catalog-v1"
FULL_SCHEMA_SAMPLES_CONTEXT = "full-schema-samples-v1"
SQL_CONTEXT_PROFILES = (
    LAZY_CATALOG_CONTEXT,
    FULL_SCHEMA_SAMPLES_CONTEXT,
)


class NoProgressError(RuntimeError):
    """A successful immutable-database action was submitted again."""

    def __init__(self, prior_step_id: str):
        self.prior_step_id = prior_step_id
        super().__init__(
            f"identical action already succeeded at {prior_step_id}; state is unchanged"
        )

LAZY_CONTEXT_PARAGRAPH = """The opening catalog contains table names, row counts, and foreign-key relations but
not full columns. Inspect schemas with execute_sql using PRAGMA table_info('TableName') or SQLite
catalog queries before relying on column names."""
FULL_SCHEMA_SAMPLES_PARAGRAPH = """The opening database context contains every table and row count,
the complete live column schema, declared foreign-key relations, and a small list of real non-NULL
example values for each column. Example values are samples, not an exhaustive domain."""

SYSTEM_PROMPT = """You are a SQLite data-analysis agent. Solve the user's question by iteratively
executing read-only SQL against the provided database and using the real results or errors to improve
the next query. """ + LAZY_CONTEXT_PARAGRAPH + """ You may also execute bounded SELECT queries to inspect
values and test joins. The harness preserves successful query results in SQL WORKSPACE and returns the
latest rejected action in LAST SQL ERROR.

TOOLS
execute_sql(sql) executes one read-only SELECT, WITH, PRAGMA, or EXPLAIN QUERY PLAN statement and
returns columns plus at most 20 rows. Use it for schema inspection, data exploration, and validating a
candidate query.
submit_sql(sql) is terminal. It accepts exactly one read-only SELECT or WITH query. Submit only after
the query has been executed successfully and you believe its complete result answers the question.
The hidden correctness judge is never shown to you.

RULES
1. Each turn output exactly one non-empty <think> block followed directly by exactly one raw
   {"tool":"...","arguments":{"sql":"..."}} JSON object and nothing else.
2. Make one atomic call per turn. Never emit multiple SQL actions or multiple JSON action objects.
3. Do not use INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, ATTACH, or multiple SQL statements.
4. Use only information in the question, external knowledge, catalog, SQL WORKSPACE, and LAST SQL
   ERROR. Do not assume hidden columns or values.
5. An execution error is feedback: correct the SQL in the next turn rather than restarting.
"""
SQL_CANONICAL_RESPONSE_RULE = """1. Each turn output exactly one non-empty <think> block followed directly by exactly one raw
   {"tool":"...","arguments":{"sql":"..."}} JSON object and nothing else."""
SQL_SPLIT_RESPONSE_RULE = """1. Produce exactly one SQL tool action per turn using the provider-specific response
   envelope at the end of this prompt."""


def build_system_prompt(
    model: str,
    context_profile: str = LAZY_CATALOG_CONTEXT,
    interface: str = ITERATIVE_SQL_INTERFACE,
) -> str:
    """Build a provider-aware prompt containing only this ablation's SQL tools."""
    if interface in DIRECT_SQL_SEARCH_INTERFACES:
        protocol_version = (
            DIRECT_SQL_SEARCH_PROTOCOL_VERSION_V1
            if interface == DIRECT_SQL_SEARCH_INTERFACE_V1
            else DIRECT_SQL_SEARCH_PROTOCOL_VERSION
        )
        return build_direct_sql_search_system_prompt(
            model,
            context_profile,
            protocol_version=protocol_version,
        )
    if interface != ITERATIVE_SQL_INTERFACE:
        raise ValueError(
            f"unknown SQL feedback interface {interface!r}; expected one of "
            f"{SQL_INTERFACE_CHOICES}"
        )
    if context_profile not in SQL_CONTEXT_PROFILES:
        raise ValueError(
            f"unknown iterative-SQL context profile {context_profile!r}; "
            f"expected one of {SQL_CONTEXT_PROFILES}"
        )
    prompt = SYSTEM_PROMPT
    if context_profile == FULL_SCHEMA_SAMPLES_CONTEXT:
        if LAZY_CONTEXT_PARAGRAPH not in prompt:
            raise ValueError("iterative-SQL lazy context paragraph drifted")
        prompt = prompt.replace(
            LAZY_CONTEXT_PARAGRAPH,
            FULL_SCHEMA_SAMPLES_PARAGRAPH,
            1,
        )
    sql_example = (
        '{"tool":"execute_sql","arguments":'
        '{"sql":"PRAGMA table_info(Orders)"}}'
    )
    if not is_deepseek_split_model(model):
        return prompt
    if SQL_CANONICAL_RESPONSE_RULE not in prompt:
        raise ValueError("iterative-SQL response rule drifted")
    prompt = prompt.replace(
        SQL_CANONICAL_RESPONSE_RULE,
        SQL_SPLIT_RESPONSE_RULE,
        1,
    )
    return prompt + provider_instruction(model, example_visible_content=sql_example)


def direct_sql_protocol_version(interface: str) -> str:
    if interface == DIRECT_SQL_SEARCH_INTERFACE_V1:
        return DIRECT_SQL_SEARCH_PROTOCOL_VERSION_V1
    if interface == DIRECT_SQL_SEARCH_INTERFACE:
        return DIRECT_SQL_SEARCH_PROTOCOL_VERSION
    raise ValueError(f"not a direct-SQL-search interface: {interface!r}")


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def quote_identifier(identifier: str) -> str:
    """Quote a live SQLite identifier without interpreting it as SQL."""
    return '"' + identifier.replace('"', '""') + '"'


def sample_column_values(
    h: Harness,
    table_name: str,
    column_name: str,
    *,
    value_count: int,
) -> list:
    """Return a bounded live distinct sample without exposing any verifier data."""
    if value_count <= 0:
        return []
    table_sql = quote_identifier(table_name)
    column_sql = quote_identifier(column_name)
    rows = h.conn.execute(
        f"""
        SELECT {column_sql}
        FROM (
            SELECT DISTINCT {column_sql}
            FROM {table_sql}
            WHERE {column_sql} IS NOT NULL
              AND CAST({column_sql} AS TEXT) != ''
        ) AS sampled_values
        LIMIT ?
        """,
        (value_count,),
    ).fetchall()
    values = []
    for (value,) in rows:
        if isinstance(value, str) and len(value) > 80:
            value = value[:80] + "…"
        elif isinstance(value, bytes):
            prefix = value[:32].hex()
            value = f"blob:0x{prefix}" + ("…" if len(value) > 32 else "")
        values.append(value)
    return values


def build_database_context(
    h: Harness,
    catalog: dict,
    *,
    context_profile: str,
    schema_value_count: int,
) -> dict:
    """Build one versioned model-visible database context from live SQLite state."""
    if context_profile not in SQL_CONTEXT_PROFILES:
        raise ValueError(
            f"unknown iterative-SQL context profile {context_profile!r}; "
            f"expected one of {SQL_CONTEXT_PROFILES}"
        )
    if schema_value_count < 0:
        raise ValueError("schema_value_count must be non-negative")
    if context_profile == LAZY_CATALOG_CONTEXT:
        return catalog

    tables = []
    for table in catalog["tables"]:
        table_name = table["table_name"]
        info = list(h.conn.execute(f"PRAGMA table_info({quote_identifier(table_name)})"))
        columns = []
        for row in info:
            column_name = row[1]
            columns.append({
                "name": column_name,
                "type": (row[2] or "text").lower(),
                "not_null": bool(row[3]),
                "primary_key": bool(row[5]),
                "example_values": sample_column_values(
                    h,
                    table_name,
                    column_name,
                    value_count=schema_value_count,
                ),
            })
        tables.append({
            "table_name": table_name,
            "num_rows": table["num_rows"],
            "columns": columns,
        })
    return {
        "tables": tables,
        "relations": catalog.get("relations", []),
        "example_values_per_column": schema_value_count,
        "example_values_are_exhaustive": False,
    }


def parse_sql_action_strict(
    text: str,
    interface: str = ITERATIVE_SQL_INTERFACE,
) -> tuple[str, str, dict]:
    if interface in DIRECT_SQL_SEARCH_INTERFACES:
        return parse_direct_sql_search_action(text)
    if interface != ITERATIVE_SQL_INTERFACE:
        raise ProtocolError(
            f"unknown SQL feedback interface {interface!r}; expected one of "
            f"{SQL_INTERFACE_CHOICES}"
        )
    try:
        think, call = parse_action_carrier(text)
    except ActionCarrierError as exc:
        raise ProtocolError(str(exc)) from exc
    if not isinstance(call, dict) or set(call) != {"tool", "arguments"}:
        raise ProtocolError('action must contain exactly "tool" and "arguments" keys')
    tool = call.get("tool")
    arguments = call.get("arguments")
    if tool not in SQL_TOOLS:
        raise ProtocolError(f"unknown tool {tool!r}; legal tools: {sorted(SQL_TOOLS)}")
    if not isinstance(arguments, dict) or set(arguments) != {"sql"}:
        raise ProtocolError(f'{tool}: arguments must contain exactly one "sql" field')
    if not isinstance(arguments["sql"], str) or not arguments["sql"].strip():
        raise ProtocolError(f"{tool}: sql must be a non-empty string")
    return think, tool, {"sql": arguments["sql"].strip()}


def validate_read_only_sql(sql: str, *, terminal: bool = False) -> None:
    statement = sql.strip().rstrip(";").strip()
    matcher = SUBMIT_RE if terminal else READ_ONLY_RE
    if not matcher.match(statement):
        allowed = "SELECT or WITH" if terminal else "SELECT, WITH, PRAGMA, or EXPLAIN QUERY PLAN"
        raise ValueError(f"only one read-only {allowed} statement is allowed")
    if ";" in statement:
        raise ValueError("multiple SQL statements are not allowed")


def canonical_sql_text(sql: str) -> str:
    """Normalize only inconsequential outer whitespace and one trailing semicolon."""
    return sql.strip().rstrip(";").strip()


def execute_preview(h: Harness, sql: str, *, limit: int, timeout_seconds: float) -> dict:
    validate_read_only_sql(sql)
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    if deadline is not None:
        h.conn.set_progress_handler(lambda: 1 if time.monotonic() >= deadline else 0, 10_000)
    try:
        cursor = h.conn.execute(sql)
        columns = [item[0] for item in (cursor.description or [])]
        rows = cursor.fetchmany(limit + 1)
    finally:
        h.conn.set_progress_handler(None, 0)
    truncated = len(rows) > limit
    shown = rows[:limit]
    return {
        "sql": sql,
        "columns": columns,
        "rows": [list(row) for row in shown],
        "returned_rows": len(shown),
        "rows_truncated": truncated,
    }


def enrich_direct_sql_preview(output: dict) -> dict:
    """Add bounded fact-only shape diagnostics without executing the query a second time."""
    enriched = deepcopy(output)
    rows = enriched.get("rows") if isinstance(enriched.get("rows"), list) else []
    comparable_rows = [compact_json(row) for row in rows]
    truncated = bool(enriched.get("rows_truncated"))
    returned = int(enriched.get("returned_rows", len(rows)))
    enriched["preview_profile"] = {
        "column_count": len(enriched.get("columns") or []),
        "visible_row_count": returned,
        "result_complete": not truncated,
        "minimum_result_row_count": returned + 1 if truncated else returned,
        "visible_duplicate_row_count": len(comparable_rows) - len(set(comparable_rows)),
        "visible_null_cell_count": sum(
            value is None
            for row in rows
            if isinstance(row, list)
            for value in row
        ),
    }
    enriched["eligible_for_final"] = bool(
        isinstance(enriched.get("sql"), str)
        and SUBMIT_RE.match(enriched["sql"].strip())
    )
    return enriched


_SEARCH_MATCH_TIERS = {
    "exact": 0,
    "normalized_exact": 1,
    "prefix": 2,
    "token": 3,
    "substring": 4,
    "fuzzy": 5,
}


def search_database_values(
    h: Harness,
    catalog: dict,
    *,
    query: str,
    table: str | None = None,
    column: str | None = None,
    limit: int = 10,
    offset: int = 0,
) -> dict:
    """Run deterministic bounded value retrieval over one table or the whole database."""
    table_names = [item["table_name"] for item in catalog.get("tables", [])]
    if table is not None:
        if table not in table_names:
            raise ValueError(
                f"unknown source table {table!r}; available tables: {table_names}"
            )
        searched_tables = [table]
    else:
        searched_tables = table_names
    if column is not None and table is None:
        raise ValueError("search_values.column requires search_values.table")
    if offset > 200:
        raise ValueError("search_values.offset cannot exceed 200")

    needed_per_table = offset + limit
    all_matches: list[dict] = []
    total_matches = 0
    candidate_count = 0
    searched_column_count = 0
    truncated_columns: list[str] = []
    for table_name in searched_tables:
        table_matches: list[dict] = []
        page_offset = 0
        table_total = 0
        while page_offset < needed_per_table:
            page_limit = min(20, needed_per_table - page_offset)
            page = h.search_values_bounded(
                table=table_name,
                query=query,
                column=column if table_name == table else None,
                limit=page_limit,
                offset=page_offset,
            )
            if page_offset == 0:
                table_total = int(page.get("total_matches", 0))
                candidate_count += int(page.get("candidate_count", 0))
                searched_column_count += len(page.get("searched_columns", []))
                truncated_columns.extend(
                    f"{table_name}.{name}"
                    for name in page.get("truncated_columns", [])
                )
            table_matches.extend(page.get("matches", []))
            page_offset += len(page.get("matches", []))
            if not page.get("has_more") or not page.get("matches"):
                break
        total_matches += table_total
        all_matches.extend(table_matches)

    all_matches.sort(key=lambda item: (
        _SEARCH_MATCH_TIERS.get(item.get("match_type"), 99),
        -float(item.get("score", 0.0)),
        -int(item.get("frequency", 0)),
        str(item.get("table", "")).casefold(),
        str(item.get("column", "")).casefold(),
        str(item.get("value", "")).casefold(),
        type(item.get("value")).__name__,
        str(item.get("value", "")),
    ))
    matches = all_matches[offset:offset + limit]
    consumed = offset + len(matches)
    has_more = consumed < total_matches
    return {
        "query": query,
        "scope": "table" if table is not None else "database",
        "table": table,
        "column": column,
        "searched_tables": searched_tables,
        "searched_column_count": searched_column_count,
        "matches": matches,
        "total_matches": total_matches,
        "returned_matches": len(matches),
        "limit": limit,
        "offset": offset,
        "has_more": has_more,
        "next_offset": consumed if has_more else None,
        "candidate_scope": "bounded-v1",
        "candidate_count": candidate_count,
        "candidate_limit_per_column": 4096,
        "truncated_columns": sorted(set(truncated_columns)),
    }


def task_prompt(database_context: dict, ex: dict, *, context_profile: str) -> str:
    heading = (
        "DATABASE CATALOG"
        if context_profile == LAZY_CATALOG_CONTEXT
        else "DATABASE SCHEMA AND VALUE SAMPLES"
    )
    text = (
        heading + "\n" + compact_json(database_context)
        + "\n\nQUESTION\n" + ex["question"]
    )
    if ex.get("external_knowledge"):
        text += "\n\nEXTERNAL KNOWLEDGE\n" + str(ex["external_knowledge"])
    return text


def direct_sql_action_signature(tool: str, arguments: dict) -> str:
    """Return the immutable-database identity of one validated direct-SQL action."""
    normalized = dict(arguments)
    if tool == "execute_sql" and isinstance(normalized.get("sql"), str):
        normalized["sql"] = canonical_sql_text(normalized["sql"])
    return compact_json({"tool": tool, "arguments": normalized})


def workspace_entry_card(entry: dict) -> dict:
    """Keep durable decision evidence without retaining every old row payload."""
    card = {
        key: entry[key]
        for key in ("step_id", "tool")
        if entry.get(key) is not None
    }
    if entry.get("tool") == "search_values":
        for key in (
            "query", "scope", "table", "column", "returned_matches", "total_matches",
            "has_more", "candidate_scope", "candidate_count", "truncated_columns",
        ):
            if entry.get(key) is not None:
                card[key] = entry[key]
        matches = entry.get("matches")
        if isinstance(matches, list):
            card["top_matches"] = matches[:3]
        return card

    for key in ("sql", "mode", "columns", "returned_rows", "rows_truncated"):
        if entry.get(key) is not None:
            card[key] = entry[key]
    rows = entry.get("rows")
    sql = entry.get("sql")
    if isinstance(rows, list):
        table_info = bool(
            isinstance(sql, str)
            and re.match(r"^PRAGMA\s+(?:main\.)?table_info\b", sql.strip(), re.I)
        )
        if table_info:
            card["schema_columns"] = [
                {"name": row[1], "type": row[2]}
                for row in rows
                if isinstance(row, list) and len(row) >= 3
            ]
        elif len(rows) <= 3 and not entry.get("rows_truncated"):
            card["exact_rows"] = rows
        elif rows:
            card["sample_rows"] = rows[:2]
            card["sample_is_complete"] = False
    return card


def compact_direct_sql_observation(observation: str) -> str:
    """Point bounded history at the single authoritative state copy of successful facts."""
    try:
        envelope = json.loads(observation)
    except (TypeError, json.JSONDecodeError):
        return observation
    if not isinstance(envelope, dict) or envelope.get("status") != "success":
        return observation
    output = envelope.get("output")
    if not isinstance(output, dict):
        return observation
    entry = {
        "step_id": envelope.get("step_id"),
        "tool": envelope.get("tool", "execute_sql"),
        **output,
    }
    summary = workspace_entry_card(entry)
    for payload_key in ("top_matches", "exact_rows", "sample_rows", "schema_columns"):
        summary.pop(payload_key, None)
    return compact_json({
        "step_id": envelope.get("step_id"),
        "status": "success",
        "tool": entry["tool"],
        "output_summary": summary,
        "full_output": "resident_in_current_direct_sql_state",
    })


def compact_prior_direct_sql_assistant(assistant: str) -> str:
    """Retain the exact successful action while preventing old rumination from dominating."""
    if "</think>" not in assistant:
        return assistant
    action = assistant.rsplit("</think>", 1)[1].strip()
    try:
        parsed = json.loads(action)
    except json.JSONDecodeError:
        return assistant
    if not isinstance(parsed, dict) or set(parsed) != {"tool", "arguments"}:
        return assistant
    return (
        "<think>[Earlier successful reasoning omitted from bounded continuity; "
        "the exact action and CURRENT DIRECT SQL STATE remain authoritative.]</think>"
        + compact_json(parsed)
    )


def state_prompt(
    workspace: list[dict],
    last_error: dict | None,
    *,
    interface: str = ITERATIVE_SQL_INTERFACE,
) -> str:
    if interface != DIRECT_SQL_SEARCH_INTERFACE:
        text = "CURRENT SQL WORKSPACE\n" + compact_json(
            {"successful_queries": workspace[-8:]}
        )
        if last_error:
            text += "\n\nLAST SQL ERROR\n" + compact_json(last_error)
        return text

    exact_window = 6
    prior_card_window = 6
    prior_start = max(0, len(workspace) - exact_window - prior_card_window)
    older = workspace[prior_start:-exact_window]
    recent = workspace[-exact_window:]
    state = {
        "successful_action_count": len(workspace),
        "archived_action_count": prior_start,
        "prior_action_cards": [workspace_entry_card(item) for item in older],
        "recent_exact_outputs": recent,
        "repeat_semantics": "same successful action on immutable database returns no new fact",
    }
    text = "CURRENT DIRECT SQL STATE\n" + compact_json(state)
    if last_error:
        text += "\n\nLAST TOOL ERROR\n" + compact_json(last_error)
    return text


def context_messages(
    system_prompt: str,
    database_context: dict,
    ex: dict,
    workspace: list[dict],
    last_error: dict | None,
    legal_history: list[dict],
    history_turns: int,
    context_profile: str,
    interface: str = ITERATIVE_SQL_INTERFACE,
) -> list[dict]:
    retained = legal_history[-history_turns:] if history_turns > 0 else legal_history
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": task_prompt(
                database_context,
                ex,
                context_profile=context_profile,
            ),
        },
    ]
    for index, item in enumerate(retained):
        observation = item["observation"]
        assistant = item["assistant"]
        if interface == DIRECT_SQL_SEARCH_INTERFACE:
            observation = compact_direct_sql_observation(observation)
            if index < len(retained) - 1:
                assistant = compact_prior_direct_sql_assistant(assistant)
        messages.extend([
            {"role": "assistant", "content": assistant},
            {"role": "user", "content": observation},
        ])
    messages[-1]["content"] += "\n\n" + state_prompt(
        workspace,
        last_error,
        interface=interface,
    )
    return messages


def error_type(exc: Exception) -> str:
    if isinstance(exc, NoProgressError):
        return "no_progress_error"
    if isinstance(exc, ProtocolError):
        text = str(exc).lower()
        if "arguments" in text or "sql must" in text:
            return "argument_validation_error"
        return "protocol_error"
    return "execution_error"


def structured_error_feedback(event: dict, exc: Exception) -> dict:
    """Render fact-only recovery feedback with the rejected action when it parsed."""
    message = event["message"]
    if isinstance(exc, NoProgressError):
        code = "successful_action_repeated"
        details = {
            "prior_success_step_id": exc.prior_step_id,
            "database_mutability": "immutable_read_only",
            "observation_change": False,
        }
    elif isinstance(exc, ProtocolError):
        lowered = str(exc).lower()
        if "must match a previously successful inspect-mode" in lowered:
            code = "final_sql_not_inspected"
            details = {
                "required_prior_action": "execute_sql with identical sql and mode=inspect",
            }
        elif "provider" in lowered or "response" in lowered:
            code = "provider_action_carrier_rejected"
            details = {
                "required_carrier": "one non-empty reasoning field plus one raw JSON action",
            }
        else:
            code = "invalid_action_contract"
            details = {
                "legal_tools": ["execute_sql", "search_values"],
            }
    else:
        code = "read_only_execution_failed"
        details = {
            "execution_engine": "sqlite",
        }
    details["successful_state_changed"] = False
    feedback = {
        "step_id": event["step_id"],
        "status": "error",
        "error": {
            "type": event["error_type"],
            "code": code,
            "message": message,
            "details": details,
        },
    }
    if event.get("attempted_tool") is not None:
        feedback["attempted_action"] = {
            "tool": event["attempted_tool"],
            "arguments": event.get("attempted_arguments") or {},
        }
    return feedback


def run_one(
    *,
    ex: dict,
    example_index: int,
    base_url: str,
    api_key: str,
    model: str,
    max_steps: int,
    max_errors_per_type: int,
    max_tokens: int,
    api_retries: int,
    api_timeout: int,
    execution_timeout_seconds: float,
    preview_rows: int,
    history_turns: int,
    denotation_comparison: str,
    interface: str = ITERATIVE_SQL_INTERFACE,
    context_profile: str = LAZY_CATALOG_CONTEXT,
    schema_value_count: int = 2,
) -> dict:
    started = time.time()
    h = Harness(task_db_path(ex))
    h.conn.execute("PRAGMA query_only = ON")
    catalog = overview(h)
    context_started = time.time()
    database_context = build_database_context(
        h,
        catalog,
        context_profile=context_profile,
        schema_value_count=schema_value_count,
    )
    database_context_json = compact_json(database_context)
    context_build_seconds = round(time.time() - context_started, 3)
    gold_sql = task_gold_sql(ex)
    system_prompt = build_system_prompt(model, context_profile, interface)
    workspace: list[dict] = []
    successful_action_steps: dict[str, str] = {}
    legal_history: list[dict] = []
    last_error = None
    turns: list[dict] = []
    error_events: list[dict] = []
    counts = collections.Counter()
    usage = collections.Counter()
    record = {
        "example_index": example_index,
        "instance_id": ex.get("instance_id") or ex.get("example_id"),
        "db_id": ex["db_id"],
        "difficulty": (ex.get("metadata") or {}).get("difficulty_proxy"),
        "question": ex["question"],
        "gold_sql": gold_sql,
        "correct": False,
        "legal": False,
        "failure_type": None,
        "turns": turns,
        "error_events": error_events,
        "denotation_comparison": denotation_comparison,
        "tool_scheme": (
            "direct-sql-search"
            if interface in DIRECT_SQL_SEARCH_INTERFACES
            else "iterative-sql"
        ),
        "protocol_version": (
            direct_sql_protocol_version(interface)
            if interface in DIRECT_SQL_SEARCH_INTERFACES
            else ITERATIVE_SQL_PROTOCOL_VERSION
        ),
        "interface": interface,
        "context_profile": context_profile,
        "direct_sql_state_profile": (
            "recent-exact-6-prior-cards-6-v1"
            if interface == DIRECT_SQL_SEARCH_INTERFACE
            else None
        ),
        "schema_value_count": (
            schema_value_count
            if context_profile == FULL_SCHEMA_SAMPLES_CONTEXT
            else None
        ),
        "database_context_chars": len(database_context_json),
        "database_context_tables": len(database_context.get("tables", [])),
        "database_context_columns": sum(
            len(table.get("columns", []))
            for table in database_context.get("tables", [])
        ),
        "database_context_example_values": sum(
            len(column.get("example_values", []))
            for table in database_context.get("tables", [])
            for column in table.get("columns", [])
        ),
        "context_build_seconds": context_build_seconds,
    }

    for action_index in range(1, max_steps + 1):
        model_input = context_messages(
            system_prompt,
            database_context,
            ex,
            workspace,
            last_error,
            legal_history,
            history_turns,
            context_profile,
            interface,
        )
        model_input = provider_request_messages(model, model_input)
        turn = {
            "turn_index": action_index - 1,
            "model_input": deepcopy(model_input),
            "provider_request_options": provider_request_options(model),
        }
        try:
            content, call_usage, reasoning = chat_with_retries(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=model_input,
                max_tokens=max_tokens,
                timeout=api_timeout,
                retries=api_retries,
            )
        except ContextOverflowError as exc:
            turn["api_error"] = f"{type(exc).__name__}: {exc}"
            turns.append(turn)
            record["failure_type"] = "context_overflow"
            break
        except ChatAPIError as exc:
            turn["api_error"] = f"{type(exc).__name__}: {exc}"
            turns.append(turn)
            record["failure_type"] = "api_error"
            break
        except Exception as exc:  # provider clients may surface transport-specific exceptions
            turn["api_error"] = f"{type(exc).__name__}: {exc}"
            turns.append(turn)
            record["failure_type"] = "api_error"
            break

        try:
            add_usage(usage, call_usage)
            turn["api_finish_reason"] = call_usage.get("api_finish_reason")
            if call_usage.get("provider_response_metadata"):
                turn["provider_response_metadata"] = deepcopy(
                    call_usage["provider_response_metadata"]
                )
            adapted, adapter = adapt_provider_response(model, content, reasoning)
            turn.update({
                "raw_model_output": content,
                "provider_reasoning_content": reasoning,
                "response_adapter": adapter,
                "model_output": adapted,
            })
            rejection = provider_rejection_message(adapter)
            if rejection:
                raise ProtocolError(rejection)
            think, tool, arguments = parse_sql_action_strict(adapted, interface)
            turn["parsed"] = {"think": think, "tool": tool, "arguments": arguments}
            turn["feedback_recovery"] = bool(last_error)
            turn["recovered_from_error_type"] = (
                (last_error or {}).get("error", {}).get("type")
            )
            sql = arguments.get("sql")

            is_direct_final = (
                interface in DIRECT_SQL_SEARCH_INTERFACES
                and tool == "execute_sql"
                and arguments.get("mode") == "final"
            )
            action_signature = direct_sql_action_signature(tool, arguments)
            if (
                interface == DIRECT_SQL_SEARCH_INTERFACE
                and not is_direct_final
                and action_signature in successful_action_steps
            ):
                raise NoProgressError(successful_action_steps[action_signature])
            if tool == "submit_sql" or is_direct_final:
                validate_read_only_sql(sql, terminal=True)
                inspected_sql = {
                    canonical_sql_text(item["sql"])
                    for item in workspace
                    if item.get("tool") == "execute_sql" or "sql" in item
                }
                if canonical_sql_text(sql) not in inspected_sql:
                    raise ProtocolError(
                        f"{tool} arguments.sql must match a previously successful inspect-mode "
                        "execute_sql query"
                    )
                predicted = execute_predicted_sql(h, sql, execution_timeout_seconds)
                gold = h.gold(gold_sql)
                record.update({
                    "legal": True,
                    "correct": compare_denotations(predicted, gold, denotation_comparison),
                    "predicted_sql": sql,
                    "predicted_row_count": len(predicted),
                    "predicted_sample": [list(row) for row in predicted[:10]],
                    "gold_row_count": len(gold),
                    "gold_sample": [list(row) for row in gold[:10]],
                })
                if not record["correct"]:
                    record["failure_type"] = "wrong_answer"
                record["outcome"] = (
                    "recovered_success" if record["correct"] and error_events
                    else "clean_success" if record["correct"] else None
                )
                turns.append(turn)
                break

            if tool == "search_values":
                output = search_database_values(h, catalog, **arguments)
                entry = {
                    "step_id": f"step_{action_index}",
                    "tool": tool,
                    **output,
                }
                workspace.append(entry)
                if interface == DIRECT_SQL_SEARCH_INTERFACE:
                    successful_action_steps[action_signature] = entry["step_id"]
                observation = compact_json({
                    "step_id": f"step_{action_index}",
                    "status": "success",
                    "tool": tool,
                    "output": output,
                })
                turn["tool_output"] = output
                turns.append(turn)
                legal_history.append({"assistant": adapted, "observation": observation})
                last_error = None
                continue

            output = execute_preview(
                h, sql, limit=preview_rows, timeout_seconds=execution_timeout_seconds
            )
            if interface == DIRECT_SQL_SEARCH_INTERFACE:
                output = enrich_direct_sql_preview(output)
            workspace.append({
                "step_id": f"step_{action_index}",
                "tool": "execute_sql",
                **({"mode": "inspect"} if interface == DIRECT_SQL_SEARCH_INTERFACE else {}),
                **output,
            })
            if interface == DIRECT_SQL_SEARCH_INTERFACE:
                successful_action_steps[action_signature] = f"step_{action_index}"
            observation = compact_json({
                "step_id": f"step_{action_index}",
                "status": "success",
                **({"tool": "execute_sql"} if interface == DIRECT_SQL_SEARCH_INTERFACE else {}),
                "output": output,
            })
            turn["tool_output"] = output
            turns.append(turn)
            legal_history.append({"assistant": adapted, "observation": observation})
            last_error = None
        except Exception as exc:  # semantic and protocol errors are same-episode feedback
            kind = error_type(exc)
            counts[kind] += 1
            event = {
                "action_index": action_index,
                "step_id": f"step_{action_index}",
                "error_type": kind,
                "message": f"{type(exc).__name__}: {exc}",
            }
            if turn.get("parsed"):
                event["attempted_tool"] = turn["parsed"]["tool"]
                event["attempted_arguments"] = turn["parsed"]["arguments"]
            error_events.append(event)
            turn["error_event"] = event
            turns.append(turn)
            last_error = (
                structured_error_feedback(event, exc)
                if interface == DIRECT_SQL_SEARCH_INTERFACE
                else {
                    "step_id": event["step_id"],
                    "status": "error",
                    "error": {"type": kind, "message": event["message"]},
                }
            )
            if kind != "no_progress_error" and counts[kind] >= max_errors_per_type:
                record["failure_type"] = kind
                break
    else:
        record["failure_type"] = "max_steps"

    if not record["correct"] and record["failure_type"] is None:
        record["failure_type"] = "max_steps"
    record["steps"] = len(turns)
    record["errors"] = len(error_events)
    record["usage"] = dict(usage)
    record["elapsed_seconds"] = round(time.time() - started, 3)
    return record


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-errors-per-type", type=int, default=3)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--execution-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--preview-rows", type=int, default=20)
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument(
        "--interface",
        choices=SQL_INTERFACE_CHOICES,
        default=ITERATIVE_SQL_INTERFACE,
        help="exclusive SQL-feedback public action interface",
    )
    parser.add_argument(
        "--context-profile",
        choices=SQL_CONTEXT_PROFILES,
        default=LAZY_CATALOG_CONTEXT,
        help="versioned initial database-information profile",
    )
    parser.add_argument(
        "--schema-value-count",
        type=int,
        default=2,
        help="distinct live example values per column for full-schema-samples-v1",
    )
    add_denotation_comparison_argument(parser)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_tokens is None:
        args.max_tokens = provider_default_max_tokens(args.model, 1024)

    tasks_path = Path(args.tasks_json)
    if args.start < 0:
        parser.error("--start must be non-negative")
    task_pool = [
        json.loads(line)
        for line in tasks_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tasks = task_pool[args.start:args.start + args.n]
    if args.schema_value_count < 0:
        parser.error("--schema-value-count must be non-negative")
    system_prompt = build_system_prompt(
        args.model,
        args.context_profile,
        args.interface,
    )
    api_key, base_url = load_api_config()
    if not api_key or not base_url:
        parser.error("api.md must define API_KEY and BASE_URL")

    manifest = {
        "runner": "iterative_sql_feedback",
        "protocol_version": (
            direct_sql_protocol_version(args.interface)
            if args.interface in DIRECT_SQL_SEARCH_INTERFACES
            else ITERATIVE_SQL_PROTOCOL_VERSION
        ),
        "protocol_hash": (
            direct_sql_search_protocol_hash(
                system_prompt,
                protocol_version=direct_sql_protocol_version(args.interface),
                interface=args.interface,
            )
            if args.interface in DIRECT_SQL_SEARCH_INTERFACES
            else None
        ),
        "interface": args.interface,
        "tool_scheme": (
            "direct-sql-search"
            if args.interface in DIRECT_SQL_SEARCH_INTERFACES
            else "iterative-sql"
        ),
        "tool_scheme_registry_version": (
            "tool-scheme-registry-v4"
            if args.interface in DIRECT_SQL_SEARCH_INTERFACES
            else None
        ),
        "top_level_tools": (
            ["execute_sql", "search_values"]
            if args.interface in DIRECT_SQL_SEARCH_INTERFACES
            else ["execute_sql", "submit_sql"]
        ),
        "public_tool_schema_sha256": (
            direct_sql_search_tool_schema_hash()
            if args.interface in DIRECT_SQL_SEARCH_INTERFACES
            else None
        ),
        "assistant_carrier": ACTIVE_ACTION_CARRIER,
        "tasks_json": str(tasks_path),
        "tasks_sha256": file_sha256(tasks_path),
        "task_start": args.start,
        "task_count": len(tasks),
        "model": args.model,
        "context_profile": args.context_profile,
        "schema_value_count": (
            args.schema_value_count
            if args.context_profile == FULL_SCHEMA_SAMPLES_CONTEXT
            else None
        ),
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "max_steps": args.max_steps,
        "max_errors_per_type": args.max_errors_per_type,
        "max_tokens": args.max_tokens,
        "api_retries": args.api_retries,
        "execution_timeout_seconds": args.execution_timeout_seconds,
        "preview_rows": args.preview_rows,
        "history_turns": args.history_turns,
        "history_observation_profile": (
            "resident-pointer-v1"
            if args.interface == DIRECT_SQL_SEARCH_INTERFACE
            else "unabridged"
        ),
        "direct_sql_state_profile": (
            "recent-exact-6-prior-cards-6-v1"
            if args.interface == DIRECT_SQL_SEARCH_INTERFACE
            else None
        ),
        "successful_repeat_policy": (
            "reject-any-prior-success-v1"
            if args.interface == DIRECT_SQL_SEARCH_INTERFACE
            else None
        ),
        "preview_feedback_profile": (
            "bounded-shape-facts-v1"
            if args.interface == DIRECT_SQL_SEARCH_INTERFACE
            else None
        ),
        "denotation_comparison": args.denotation_comparison,
        "strict_parser": True,
        "parser_repair": False,
        "gold_visible_to_model": False,
        "interface_ablation": True,
        "sft_export_eligible": False,
        "training_admission": "diagnostic_only",
    }
    writer = ArtifactWriter(args.result_dir, manifest, resume=args.resume)
    work = [ex for ex in tasks if int(ex["example_index"]) not in writer.completed]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                run_one,
                ex=ex,
                example_index=int(ex["example_index"]),
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                max_steps=args.max_steps,
                max_errors_per_type=args.max_errors_per_type,
                max_tokens=args.max_tokens,
                api_retries=args.api_retries,
                api_timeout=args.api_timeout,
                execution_timeout_seconds=args.execution_timeout_seconds,
                preview_rows=args.preview_rows,
                history_turns=args.history_turns,
                denotation_comparison=args.denotation_comparison,
                interface=args.interface,
                context_profile=args.context_profile,
                schema_value_count=args.schema_value_count,
            ): ex for ex in work
        }
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            writer.append(record)
            print(
                f"[{done}/{len(work)}] {'OK' if record['correct'] else 'ERR'} "
                f"steps={record['steps']} errors={record['errors']} "
                f"type={record.get('failure_type')} {record.get('instance_id')}",
                flush=True,
            )
    print(json.dumps(writer.summarize(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
