"""Exclusive two-tool protocol for SQL exploration with out-of-band value search.

This diagnostic scheme deliberately does not expose the atomic relational tools.  It combines a
bounded, deterministic database-value retrieval service with direct read-only SQLite execution.
The terminal decision is a mode of ``execute_sql`` so the public surface remains exactly two tools.
"""
from __future__ import annotations

import hashlib
import json

from tool_modules._bootstrap import activate_legacy_paths

activate_legacy_paths()

from action_carrier import ACTIVE_ACTION_CARRIER, ActionCarrierError, parse_action_carrier
from protocol import ProtocolError
from provider_adapter import is_deepseek_split_model, provider_instruction


DIRECT_SQL_SEARCH_PROTOCOL_VERSION_V1 = "direct-sql-search-v1"
DIRECT_SQL_SEARCH_INTERFACE_V1 = "search-values-execute-sql-v1"
DIRECT_SQL_SEARCH_PROTOCOL_VERSION = "direct-sql-search-v2"
DIRECT_SQL_SEARCH_INTERFACE = "search-values-execute-sql-v2"
DIRECT_SQL_SEARCH_PROTOCOL_VERSIONS = (
    DIRECT_SQL_SEARCH_PROTOCOL_VERSION_V1,
    DIRECT_SQL_SEARCH_PROTOCOL_VERSION,
)
DIRECT_SQL_SEARCH_INTERFACES = (
    DIRECT_SQL_SEARCH_INTERFACE_V1,
    DIRECT_SQL_SEARCH_INTERFACE,
)
DIRECT_SQL_SEARCH_TOOLS = frozenset({"search_values", "execute_sql"})
DIRECT_SQL_SEARCH_ASSISTANT_CARRIER = ACTIVE_ACTION_CARRIER
DIRECT_SQL_SEARCH_MODEL_ARG_SCHEMA = {
    "search_values": {
        "required": ["query"],
        "optional": ["table", "column", "limit", "offset"],
        "additional": False,
        "constraints": {
            "query": "non-empty string, max 256 characters",
            "table": "non-empty source-table name",
            "column": "non-empty column name; requires table",
            "limit": "integer 1..20",
            "offset": "integer 0..200",
        },
    },
    "execute_sql": {
        "required": ["sql", "mode"],
        "optional": [],
        "additional": False,
        "constraints": {
            "sql": "one non-empty read-only SQLite statement",
            "mode": ["inspect", "final"],
        },
    },
}

LAZY_CATALOG_CONTEXT = "lazy-catalog-v1"
FULL_SCHEMA_SAMPLES_CONTEXT = "full-schema-samples-v1"
DIRECT_SQL_SEARCH_CONTEXT_PROFILES = (
    LAZY_CATALOG_CONTEXT,
    FULL_SCHEMA_SAMPLES_CONTEXT,
)

LAZY_CONTEXT_PARAGRAPH = """The opening catalog contains table names, row counts, and foreign-key relations but
not full columns. Inspect schemas with execute_sql(mode=\"inspect\") using PRAGMA table_info or
SQLite catalog queries before relying on column names."""
FULL_SCHEMA_SAMPLES_PARAGRAPH = """The opening database context contains every table and row count,
the complete live column schema, declared foreign-key relations, and a small list of real non-NULL
example values for each column. Example values are samples, not an exhaustive domain."""

CANONICAL_RESPONSE_RULE = """1. Each turn output exactly one non-empty <think> block followed directly by exactly one raw
   JSON action object and nothing else."""
SPLIT_RESPONSE_RULE = """1. Produce exactly one tool action per turn using the provider-specific response envelope at
   the end of this prompt."""

V1_CANONICAL_SYSTEM_PROMPT = """You are a SQLite data-analysis agent. Solve the question by causally
exploring the provided database and using only real tool results or errors to improve the next
action. {context_paragraph} The harness preserves recent successful SQL previews and value-search
observations in CURRENT WORKSPACE and returns the latest rejected action in LAST TOOL ERROR.

TOOLS
search_values(query, table?, column?, limit?, offset?) performs deterministic bounded lexical
retrieval over exact stored values. When table is omitted it searches the whole database; column
requires table. It returns real table/column/value matches, frequencies, match types, scores, and
candidate-scope metadata. This is an out-of-band retrieval service: it does not create a relation,
filter rows, infer semantic equivalence, or change the database.

execute_sql(sql, mode) executes one read-only SQLite statement. mode=\"inspect\" accepts SELECT,
WITH, PRAGMA, or EXPLAIN QUERY PLAN and returns columns plus at most 20 rows. mode=\"final\" is
terminal, accepts SELECT or WITH, and hidden-scores its complete result. A final SQL statement must
first have succeeded in inspect mode in the same episode.

RULES
{response_rule}
2. The raw action has exactly tool and arguments. Make one atomic call per turn.
3. search_values arguments contain query plus only optional table, column, limit, and offset.
   limit is 1..20 and offset is 0..200. A column cannot be supplied without a table.
4. execute_sql arguments contain exactly sql and mode. Use mode=\"inspect\" to explore and validate;
   use mode=\"final\" only for an already successful inspected SELECT/WITH that exactly returns the
   requested rows, columns, column order, representation, and no helper fields.
5. Never use INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, ATTACH, or multiple SQL statements.
6. Treat question/external-knowledge mappings as binding. Do not invent DISTINCT, latest/current,
   singleton, rounding, or extra output columns. An executable plausible result is not proof that
   its population, grain, formula, or output shape answers the question.
7. Use only the question, external knowledge, opening context, CURRENT WORKSPACE, and LAST TOOL
   ERROR. The hidden judge is never shown.
"""

CANONICAL_SYSTEM_PROMPT = """You are a SQLite data-analysis agent. Solve the question by causally
exploring the provided database and using only real tool results or errors to improve the next
action. {context_paragraph} CURRENT DIRECT SQL STATE is the authoritative successful-action
workspace. Bounded history observations point to facts retained there. LAST TOOL ERROR records the
latest rejected action and confirms that successful state was unchanged. Provider history may omit
prior model-authored reasoning while retaining exact successful actions, so use state and tool facts
rather than reconstructing assumptions from old prose.

TOOLS
search_values(query, table?, column?, limit?, offset?) performs deterministic bounded lexical
retrieval over exact stored values. When table is omitted it searches the whole database; column
requires table. It returns real table/column/value matches, frequencies, match types, scores, and
candidate-scope metadata. This is an out-of-band retrieval service: it does not create a relation,
filter rows, infer semantic equivalence, or change the database.

execute_sql(sql, mode) executes one read-only SQLite statement. mode="inspect" accepts SELECT,
WITH, PRAGMA, or EXPLAIN QUERY PLAN and returns columns plus at most 20 rows. mode="final" is
terminal, accepts SELECT or WITH, and hidden-scores its complete result. A final SQL statement must
first have succeeded in inspect mode in the same episode.

ACTION CONTRACT
{response_rule}
2. The raw action has exactly tool and arguments. Make one atomic call per turn.
3. search_values arguments contain query plus only optional table, column, limit, and offset.
   limit is 1..20 and offset is 0..200. A column cannot be supplied without a table.
4. execute_sql arguments contain exactly sql and mode. Use mode="inspect" to explore and validate;
   use mode="final" only for the exact same successful inspected SELECT/WITH.
5. Never use INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, ATTACH, or multiple SQL statements.

SEMANTIC DECISION DISCIPLINE
6. Treat the question and external knowledge as the binding answer specification. Follow every
   explicit phrase-to-column, value, operator, aggregation, formula, identifier, and output-field
   mapping. Do not replace one with a more natural proxy or silently reinterpret a supplied value.
7. Before the first filtering, join, aggregation, or ranking query, identify in <think>: the answer
   entity or measurement unit, what one eligible row represents, all population conditions, and
   the exact requested output fields and order. Preserve that population and grain unless a real
   observation disproves an assumption.
8. Do not invent DISTINCT, earliest/latest/current, active, top-1, mean, rounding, same-year, or a
   singleton restriction merely to obtain a plausible result. COUNT, COUNT DISTINCT, row count,
   and entity count are different. Establish all answer-eligibility joins before aggregation or
   ranking; keep numerator and denominator on the specified population and grain.
9. Treat zero rows, unexpected multiplicity, NULLs, malformed numeric text, impossible dates, and
   implausible arithmetic as evidence to inspect the relevant assumption. Do not change join type,
   target field, identifier representation, or row selector merely to make a query nonempty.
10. Use search_values only when the stored spelling or location of a required literal is genuinely
    unresolved. An explicit mapped literal can be tested directly with SQL. Search matches are
    candidates, not permission to change the requested concept. Do not repeat a successful action:
    the immutable database would return the same fact. State retains recent exact outputs and prior
    cards; the harness also rejects an exact repeat even after its full payload is archived.
11. Immediately before final, audit the inspected result one requested slot at a time. The final
    SQL must return exactly the requested rows, columns, column order, and representation, with no
    diagnostic counts, ranking keys, join identifiers, or other helper fields. A plausible preview
    is not proof that population, grain, formula, duplicate semantics, or output shape is correct.
12. Keep <think> decision-oriented: state the current semantic binding, the one unresolved fact,
    and why this action resolves it. Do not recap the whole transcript, repeatedly debate an
    already observed alternative, or restart schema discovery without a new contradictory fact.
13. Use only the question, external knowledge, opening context, CURRENT DIRECT SQL STATE, and LAST
    TOOL ERROR. Rejected model text is not factual evidence. The hidden judge is never shown.
"""


def build_direct_sql_search_system_prompt(
    model: str | None = None,
    context_profile: str = LAZY_CATALOG_CONTEXT,
    protocol_version: str = DIRECT_SQL_SEARCH_PROTOCOL_VERSION,
) -> str:
    """Build one prompt with exactly the two tools and one selected response carrier."""
    if context_profile not in DIRECT_SQL_SEARCH_CONTEXT_PROFILES:
        raise ValueError(
            f"unknown direct-SQL-search context profile {context_profile!r}; "
            f"expected one of {DIRECT_SQL_SEARCH_CONTEXT_PROFILES}"
        )
    context_paragraph = (
        FULL_SCHEMA_SAMPLES_PARAGRAPH
        if context_profile == FULL_SCHEMA_SAMPLES_CONTEXT
        else LAZY_CONTEXT_PARAGRAPH
    )
    if protocol_version not in DIRECT_SQL_SEARCH_PROTOCOL_VERSIONS:
        raise ValueError(
            f"unknown direct-SQL-search protocol {protocol_version!r}; expected one of "
            f"{DIRECT_SQL_SEARCH_PROTOCOL_VERSIONS}"
        )
    split = bool(model and is_deepseek_split_model(model))
    template = (
        V1_CANONICAL_SYSTEM_PROMPT
        if protocol_version == DIRECT_SQL_SEARCH_PROTOCOL_VERSION_V1
        else CANONICAL_SYSTEM_PROMPT
    )
    prompt = template.format(
        context_paragraph=context_paragraph,
        response_rule=SPLIT_RESPONSE_RULE if split else CANONICAL_RESPONSE_RULE,
    )
    if not split:
        return prompt
    example = (
        '{"tool":"execute_sql","arguments":'
        '{"sql":"PRAGMA table_info(Orders)","mode":"inspect"}}'
    )
    return prompt + provider_instruction(model, example_visible_content=example)


def direct_sql_search_protocol_hash(
    system_prompt: str,
    *,
    protocol_version: str = DIRECT_SQL_SEARCH_PROTOCOL_VERSION,
    interface: str = DIRECT_SQL_SEARCH_INTERFACE,
) -> str:
    payload = (
        protocol_version
        + "\n"
        + interface
        + "\n"
        + DIRECT_SQL_SEARCH_ASSISTANT_CARRIER
        + "\n"
        + system_prompt
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def direct_sql_search_tool_schema_hash() -> str:
    payload = json.dumps(
        DIRECT_SQL_SEARCH_MODEL_ARG_SCHEMA,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_search_arguments(arguments: object) -> dict:
    if not isinstance(arguments, dict):
        raise ProtocolError("search_values: arguments must be an object")
    allowed = {"query", "table", "column", "limit", "offset"}
    if "query" not in arguments or not set(arguments).issubset(allowed):
        raise ProtocolError(
            "search_values: arguments require query and allow only table, column, limit, offset"
        )
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ProtocolError("search_values: query must be a non-empty string")
    if len(query) > 256:
        raise ProtocolError("search_values: query cannot exceed 256 characters")
    table = arguments.get("table")
    column = arguments.get("column")
    if table is not None and (not isinstance(table, str) or not table.strip()):
        raise ProtocolError("search_values: table must be a non-empty string when supplied")
    if column is not None and (not isinstance(column, str) or not column.strip()):
        raise ProtocolError("search_values: column must be a non-empty string when supplied")
    if column is not None and table is None:
        raise ProtocolError("search_values: column requires table")
    limit = arguments.get("limit", 10)
    offset = arguments.get("offset", 0)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
        raise ProtocolError("search_values: limit must be an integer from 1 to 20")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or offset > 200
    ):
        raise ProtocolError("search_values: offset must be an integer from 0 to 200")
    normalized = {"query": query.strip()}
    if table is not None:
        normalized["table"] = table.strip()
    if column is not None:
        normalized["column"] = column.strip()
    normalized.update({"limit": limit, "offset": offset})
    return normalized


def _validate_execute_arguments(arguments: object) -> dict:
    if not isinstance(arguments, dict) or set(arguments) != {"sql", "mode"}:
        raise ProtocolError(
            'execute_sql: arguments must contain exactly "sql" and "mode"'
        )
    sql = arguments.get("sql")
    mode = arguments.get("mode")
    if not isinstance(sql, str) or not sql.strip():
        raise ProtocolError("execute_sql: sql must be a non-empty string")
    if mode not in {"inspect", "final"}:
        raise ProtocolError('execute_sql: mode must be exactly "inspect" or "final"')
    return {"sql": sql.strip(), "mode": mode}


def parse_direct_sql_search_action(text: str) -> tuple[str, str, dict]:
    """Strictly parse and validate one two-tool action."""
    try:
        think, call = parse_action_carrier(text)
    except ActionCarrierError as exc:
        raise ProtocolError(str(exc)) from exc
    if not isinstance(call, dict) or set(call) != {"tool", "arguments"}:
        raise ProtocolError('action must contain exactly "tool" and "arguments" keys')
    tool = call.get("tool")
    if tool not in DIRECT_SQL_SEARCH_TOOLS:
        raise ProtocolError(
            f"unknown tool {tool!r}; legal tools: {sorted(DIRECT_SQL_SEARCH_TOOLS)}"
        )
    arguments = call.get("arguments")
    if tool == "search_values":
        return think, tool, _validate_search_arguments(arguments)
    return think, tool, _validate_execute_arguments(arguments)
