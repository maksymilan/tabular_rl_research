"""Versioned prompt profiles for direct-SQL controls.

The canonical JSON profile is the repository's historical one-shot control.  The SQL-ASTRA
profile reproduces the strongest input details disclosed in that paper's appendix: DDL, BIRD
column descriptions, representative database values, primary/foreign keys, external knowledge,
and an explicit step-by-step SQL instruction.  Keeping this layer separate prevents prompt
experiments from changing decoding, SQL execution, candidate aggregation, or denotation scoring.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from executor import Harness
from rollout import overview

CANONICAL_JSON_PROFILE = "canonical-json-v1"
SQL_ASTRA_APPENDIX_PROFILE = "sql-astra-appendix-v1"
SQL_ASTRA_DISCLOSED_SINGLE_TURN_PROFILE = "sql-astra-disclosed-single-turn-v1"
DIRECT_SQL_PROMPT_PROFILES = (
    CANONICAL_JSON_PROFILE,
    SQL_ASTRA_APPENDIX_PROFILE,
    SQL_ASTRA_DISCLOSED_SINGLE_TURN_PROFILE,
)

CANONICAL_SYSTEM_PROMPT = (
    "You translate natural-language questions into SQLite. Put your final query inside "
    "<answer></answer> tags, e.g. <answer>SELECT ...</answer>. It must be exactly one read-only "
    "SELECT or WITH query. You may reason before the tags; only the query inside them is graded."
)

SQL_ASTRA_SYSTEM_PROMPT = (
    "You are a helpful SQL assistant. Solve the user's question from the supplied SQLite schema. "
    "Think through the query first, then put exactly one read-only SELECT or WITH query inside "
    "<answer></answer> tags. Only the SQL inside those tags is graded."
)

# SQL-ASTRA Appendix F discloses this role identity before its agent-only tool instructions.  The
# paper calls the 47.5 BIRD baseline a single-turn SQL setting, so this isolated reproduction keeps
# only the shared role identity and uses the appendix's verbatim single-turn code-block carrier.
SQL_ASTRA_DISCLOSED_SINGLE_TURN_SYSTEM_PROMPT = "You are a helpful SQL assistant."

_SQL_RESERVED_WORDS = {
    "ALL", "ALTER", "AND", "AS", "ASC", "BETWEEN", "BY", "CASE", "CAST", "CREATE",
    "CROSS", "CURRENT", "DELETE", "DESC", "DISTINCT", "DROP", "ELSE", "END", "EXCEPT",
    "EXISTS", "FALSE", "FILTER", "FOREIGN", "FROM", "FULL", "GROUP", "HAVING", "IN",
    "INDEX", "INNER", "INSERT", "INTERSECT", "INTO", "IS", "JOIN", "LEFT", "LIKE",
    "LIMIT", "NATURAL", "NOT", "NULL", "OFFSET", "ON", "OR", "ORDER", "OUTER", "OVER",
    "PRIMARY", "REFERENCES", "RIGHT", "SELECT", "SET", "TABLE", "THEN", "TRUE", "UNION",
    "UNIQUE", "UPDATE", "USING", "VALUES", "WHEN", "WHERE", "WITH",
}
_SPECIAL_IDENTIFIER = re.compile(r"[^A-Za-z0-9_]")


def add_direct_sql_prompt_arguments(parser) -> None:
    parser.add_argument(
        "--prompt-profile",
        choices=DIRECT_SQL_PROMPT_PROFILES,
        default=CANONICAL_JSON_PROFILE,
        help="versioned direct-SQL input renderer; decoding and scoring are configured separately",
    )
    parser.add_argument(
        "--schema-value-count",
        type=int,
        default=2,
        help="representative distinct values per column for schema-value profiles",
    )
    parser.add_argument(
        "--schema-metadata-json",
        help=(
            "BIRD/Spider tables metadata JSON; SQL-ASTRA profile auto-discovers "
            "dev_tables.json/train_tables.json/tables.json beside the database root when omitted"
        ),
    )


def system_prompt_for_profile(profile: str) -> str:
    if profile == CANONICAL_JSON_PROFILE:
        return CANONICAL_SYSTEM_PROMPT
    if profile == SQL_ASTRA_APPENDIX_PROFILE:
        return SQL_ASTRA_SYSTEM_PROMPT
    if profile == SQL_ASTRA_DISCLOSED_SINGLE_TURN_PROFILE:
        return SQL_ASTRA_DISCLOSED_SINGLE_TURN_SYSTEM_PROMPT
    raise ValueError(f"unknown direct-SQL prompt profile: {profile}")


def canonical_schema_prompt(
    h: Harness,
    question: str,
    external_knowledge: Any = None,
) -> str:
    """Render the repository's historical full-schema JSON prompt unchanged."""
    table_names = [t["table_name"] for t in overview(h)["tables"]]
    user_prompt = (
        "DATABASE SCHEMA\n"
        + json.dumps(h.describe_table(table_names), ensure_ascii=False, separators=(",", ":"))
        + "\n\nQUESTION\n"
        + question
    )
    if external_knowledge:
        user_prompt += "\n\nEXTERNAL KNOWLEDGE\n" + json.dumps(
            external_knowledge, ensure_ascii=False
        )
    return user_prompt


def build_direct_sql_messages(
    h: Harness,
    example: dict,
    *,
    profile: str = CANONICAL_JSON_PROFILE,
    schema_value_count: int = 2,
    schema_metadata_json: str | None = None,
) -> list[dict]:
    if schema_value_count < 0:
        raise ValueError("schema_value_count must be non-negative")
    system_prompt = system_prompt_for_profile(profile)
    if profile == CANONICAL_JSON_PROFILE:
        user_prompt = canonical_schema_prompt(
            h,
            example["question"],
            example.get("external_knowledge"),
        )
    elif profile in {
        SQL_ASTRA_APPENDIX_PROFILE,
        SQL_ASTRA_DISCLOSED_SINGLE_TURN_PROFILE,
    }:
        metadata_path = resolve_schema_metadata_path(
            example,
            explicit_path=schema_metadata_json,
        )
        db_schema = load_database_schema(metadata_path, example["db_id"])
        ddl = render_bird_ddl_with_values(
            h,
            db_schema,
            value_count=schema_value_count,
        )
        renderer = (
            render_sql_astra_disclosed_single_turn_user_prompt
            if profile == SQL_ASTRA_DISCLOSED_SINGLE_TURN_PROFILE
            else render_sql_astra_user_prompt
        )
        user_prompt = renderer(ddl, example["question"], example.get("external_knowledge"))
    else:  # guarded by argparse in CLIs; retained for programmatic callers.
        raise ValueError(f"unknown direct-SQL prompt profile: {profile}")
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def render_sql_astra_user_prompt(
    ddl: str,
    question: str,
    external_knowledge: Any = None,
) -> str:
    """Render the single-turn counterpart of SQL-ASTRA Appendix F's disclosed prompt."""
    if isinstance(external_knowledge, str):
        evidence = external_knowledge.strip()
    elif external_knowledge:
        evidence = json.dumps(external_knowledge, ensure_ascii=False)
    else:
        evidence = ""
    combined_question = f"{evidence}\n{question}" if evidence else question
    return f"""Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{ddl}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{combined_question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.
- Put the final SQL query inside <answer></answer> tags.

Output Format:
<answer>SELECT ...</answer>

Take a deep breath and think step by step to find the correct SQL query."""


def render_sql_astra_disclosed_single_turn_user_prompt(
    ddl: str,
    question: str,
    external_knowledge: Any = None,
) -> str:
    """Render Appendix F's disclosed single-turn text and code-block SQL carrier.

    Appendix F continues with multi-turn ``run_sql_remote`` instructions after this text.  Those
    instructions cannot define the paper's separately reported single-turn baseline, so this
    profile stops at the disclosed code-block output contract and introduces no execution feedback.
    """
    if isinstance(external_knowledge, str):
        evidence = external_knowledge.strip()
    elif external_knowledge:
        evidence = json.dumps(external_knowledge, ensure_ascii=False)
    else:
        evidence = ""
    combined_question = f"{evidence}\n{question}" if evidence else question
    return f"""Task Overview:
You are a data science expert. Below, you are provided with a database schema and a natural language question. Your task is to understand the schema and generate a valid SQL query to answer the question.

Database Engine:
SQLite

Database Schema:
{ddl}
This schema describes the database's structure, including tables, columns, primary keys, foreign keys, and any relevant relationships or constraints.

Question:
{combined_question}

Instructions:
- Make sure you only output the information that is asked in the question. If the question asks for a specific column, make sure to only include that column in the SELECT clause, nothing more.
- The generated query should return all of the information asked in the question without any missing or extra information.
- Before generating the final SQL query, please think through the steps of how to write the query.

Output Format:
In your answer, please enclose the generated SQL query in a code block:
```sql
-- Your SQL query
```

Take a deep breath and think step by step to find the correct SQL query."""


def resolve_schema_metadata_path(
    example: dict,
    *,
    explicit_path: str | None = None,
) -> str:
    if explicit_path:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"schema metadata JSON not found: {path}")
        return str(path)

    db_path = Path(example["db_path"]).expanduser().resolve()
    candidates = []
    for parent in (db_path.parent, *db_path.parents):
        candidates.extend(
            parent / filename
            for filename in ("dev_tables.json", "train_tables.json", "tables.json")
        )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "could not auto-discover schema metadata for "
        f"{db_path}; pass --schema-metadata-json"
    )


@lru_cache(maxsize=8)
def _load_schema_entries(path: str) -> tuple[dict, ...]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"schema metadata must be a JSON list: {path}")
    return tuple(data)


def load_database_schema(path: str, db_id: str) -> dict:
    matches = [entry for entry in _load_schema_entries(path) if entry.get("db_id") == db_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one schema entry for db_id={db_id!r} in {path}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _format_identifier(identifier: str) -> str:
    escaped = identifier.replace("`", "``")
    if (
        identifier.upper() in _SQL_RESERVED_WORDS
        or _SPECIAL_IDENTIFIER.search(identifier)
        or identifier[:1].isdigit()
    ):
        return f"`{escaped}`"
    return identifier


def _sample_column_values(
    h: Harness,
    table_name: str,
    column_name: str,
    *,
    value_count: int,
) -> list:
    if value_count <= 0:
        return []
    table_sql = _format_identifier(table_name)
    column_sql = _format_identifier(column_name)
    rows = h.conn.execute(
        f"""
        SELECT {column_sql}
        FROM (
            SELECT DISTINCT {column_sql}
            FROM {table_sql}
            WHERE {column_sql} IS NOT NULL AND {column_sql} != ''
        ) AS unique_values
        LIMIT ?
        """,
        (value_count,),
    ).fetchall()
    values = [row[0] for row in rows]
    return [value[:40] if isinstance(value, str) else value for value in values]


def _primary_key_indices(db_schema: dict) -> set[int]:
    indices: set[int] = set()
    for value in db_schema.get("primary_keys", []):
        if isinstance(value, int):
            indices.add(value)
        elif isinstance(value, list):
            indices.update(index for index in value if isinstance(index, int))
    return indices


def render_bird_ddl_with_values(
    h: Harness,
    db_schema: dict,
    *,
    value_count: int = 2,
) -> str:
    """Render the BIRD metadata and live values using SQL-R1/SQL-ASTRA DDL conventions."""
    table_names = db_schema["table_names_original"]
    original_columns = db_schema["column_names_original"]
    normalized_columns = db_schema.get("column_names", original_columns)
    column_types = db_schema["column_types"]
    if not (len(original_columns) == len(normalized_columns) == len(column_types)):
        raise ValueError("schema column metadata lengths do not match")

    primary_indices = _primary_key_indices(db_schema)
    foreign_keys = db_schema.get("foreign_keys", [])
    table_ddls = []
    for table_index, table_name in enumerate(table_names):
        column_lines = []
        table_column_indices = []
        for column_index, (
            (owner_index, original_name),
            (_, normalized_name),
            column_type,
        ) in enumerate(zip(original_columns, normalized_columns, column_types)):
            if owner_index != table_index or original_name == "*":
                continue
            table_column_indices.append(column_index)
            values = _sample_column_values(
                h,
                table_name,
                original_name,
                value_count=value_count,
            )
            comment = str(normalized_name).strip()
            redundant_comment = comment.lower() in {
                original_name.lower(),
                original_name.lower().replace(" ", "_"),
                original_name.lower().replace(" ", ""),
            }
            suffix_parts = []
            if comment and not redundant_comment:
                suffix_parts.append(comment)
            if values:
                suffix_parts.append(f"example: {values!r}")
            suffix = f" -- {', '.join(suffix_parts)}" if suffix_parts else ""
            column_lines.append(
                f"    {_format_identifier(original_name)} {column_type},{suffix}"
            )

        primary_columns = [
            original_columns[index][1]
            for index in table_column_indices
            if index in primary_indices
        ]
        constraint_lines = []
        if primary_columns:
            constraint_lines.append(
                "    PRIMARY KEY ("
                + ", ".join(_format_identifier(name) for name in primary_columns)
                + "),"
            )
        for source_index, target_index in foreign_keys:
            if source_index not in table_column_indices:
                continue
            source_table_index, source_column = original_columns[source_index]
            target_table_index, target_column = original_columns[target_index]
            source_table = table_names[source_table_index]
            target_table = table_names[target_table_index]
            constraint_name = re.sub(
                r"[^a-z0-9_]+",
                "_",
                f"fk_{source_table}_{source_column}".lower(),
            ).strip("_")
            constraint_lines.append(
                f"    CONSTRAINT {constraint_name} "
                f"FOREIGN KEY ({_format_identifier(source_column)}) "
                f"REFERENCES {_format_identifier(target_table)} "
                f"({_format_identifier(target_column)}),"
            )

        lines = column_lines + constraint_lines
        if lines:
            lines[-1] = re.sub(r",(?=\s+--)", "", lines[-1]).rstrip(",")
        table_ddls.append(
            f"CREATE TABLE {_format_identifier(table_name)} (\n"
            + "\n".join(lines)
            + "\n);"
        )
    return "\n\n".join(table_ddls)


def prompt_profile_manifest(
    profile: str,
    *,
    schema_value_count: int,
    schema_metadata_json: str | None,
) -> dict:
    return {
        "prompt_profile": profile,
        "schema_value_count": (
            schema_value_count
            if profile
            in {SQL_ASTRA_APPENDIX_PROFILE, SQL_ASTRA_DISCLOSED_SINGLE_TURN_PROFILE}
            else None
        ),
        "schema_metadata_json": (
            os.path.abspath(schema_metadata_json)
            if schema_metadata_json
            and profile
            in {SQL_ASTRA_APPENDIX_PROFILE, SQL_ASTRA_DISCLOSED_SINGLE_TURN_PROFILE}
            else None
        ),
        "system_prompt": system_prompt_for_profile(profile),
    }
