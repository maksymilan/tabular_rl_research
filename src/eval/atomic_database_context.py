"""Diagnostic database-context profiles for the atomic table-tool agent.

The default atomic protocol intentionally starts from a lazy catalog and lets the model acquire
schema/value evidence with ``describe_table`` and ``inspect_column``.  This module owns the
opposite, explicitly diagnostic ablation: reveal the complete BIRD schema, short BIRD column
semantics, and a bounded number of live example values up front, then remove those two perception
tools from the model-visible action surface.

Execution still uses the ordinary atomic harness.  The profile changes only model-visible context,
prompt text, and allowed model-authored tools.
"""
from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from direct_sql_prompt import load_database_schema, resolve_schema_metadata_path
from prompt_contract import (
    CANONICAL_ACTION_RULE,
    ROLLING_HISTORY_CONTRACT,
    SHARED_TOOL_SPECS,
    TEACHER_SEMANTIC_DECISION_DISCIPLINE,
)
from protocol import (
    CANONICAL_CALL_COOKBOOK,
    MODEL_ARG_SCHEMA,
    ProtocolError,
    TEACHER_TOOL_GUIDANCE,
)

CATALOG_CONTEXT_PROFILE = "catalog-v1"
CATALOG_BIRD_INSPECT_SEMANTICS_PROFILE = (
    "catalog-bird-semantics-inspect-only-v1"
)
CATALOG_CONTEXT_PROFILES = (
    CATALOG_CONTEXT_PROFILE,
    CATALOG_BIRD_INSPECT_SEMANTICS_PROFILE,
)
FULL_BIRD_CONTEXT_PROFILE = "full-bird-schema-samples-v1"
FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE = (
    "full-bird-schema-samples-with-inspect-v1"
)
FULL_BIRD_CONTEXT_PROFILES = (
    FULL_BIRD_CONTEXT_PROFILE,
    FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE,
)
DATABASE_CONTEXT_PROFILES = (
    *CATALOG_CONTEXT_PROFILES,
    *FULL_BIRD_CONTEXT_PROFILES,
)
FULL_CONTEXT_DISABLED_TOOLS = frozenset({"describe_table", "inspect_column"})
FULL_CONTEXT_WITH_INSPECT_DISABLED_TOOLS = frozenset({"describe_table"})

def disabled_tools_for_profile(profile: str) -> frozenset[str]:
    if profile in CATALOG_CONTEXT_PROFILES:
        return frozenset()
    if profile == FULL_BIRD_CONTEXT_PROFILE:
        return FULL_CONTEXT_DISABLED_TOOLS
    if profile == FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE:
        return FULL_CONTEXT_WITH_INSPECT_DISABLED_TOOLS
    raise ValueError(f"unknown database context profile: {profile!r}")


def enabled_tools_for_profile(profile: str) -> tuple[str, ...]:
    disabled = disabled_tools_for_profile(profile)
    return tuple(tool for tool in SHARED_TOOL_SPECS if tool not in disabled)


def model_visible_tool_schema_hash(profile: str) -> str:
    """Hash the exact tool names, arguments, and semantics exposed by this profile."""
    enabled = enabled_tools_for_profile(profile)
    payload = {
        "tools": {tool: SHARED_TOOL_SPECS[tool] for tool in enabled},
        "arguments": {
            tool: {
                "required": sorted(MODEL_ARG_SCHEMA[tool][0]),
                "optional": sorted(MODEL_ARG_SCHEMA[tool][1]),
            }
            for tool in enabled
        },
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def validate_profile_tool(profile: str, tool: str, arguments: dict) -> None:
    """Reject a globally valid atomic call that is absent from this visible profile."""
    disabled = disabled_tools_for_profile(profile)
    if tool not in disabled:
        return
    legal_tools = sorted(enabled_tools_for_profile(profile))
    raise ProtocolError(
        f"tool {tool!r} is unavailable in database context profile {profile!r}; "
        f"legal tools: {legal_tools}",
        code="unknown_tool",
        details={
            "database_context_profile": profile,
            "legal_tools": legal_tools,
        },
        attempted_tool=tool,
        attempted_arguments=arguments,
    )


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            with path.open(encoding=encoding, newline="") as handle:
                return [
                    {
                        str(key or "").strip(): _normalize_text(value)
                        for key, value in row.items()
                    }
                    for row in csv.DictReader(handle)
                ]
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"cannot decode {path}")


def _description_file_map(example: dict) -> dict[str, Path]:
    directory = Path(example["db_path"]).expanduser().resolve().parent / "database_description"
    if not directory.is_dir():
        return {}
    return {
        path.stem.casefold(): path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.casefold() == ".csv"
    }


def _bird_column_semantics(
    example: dict,
    table_names: list[str],
    columns_by_table: dict[str, set[str]],
) -> tuple[dict[tuple[str, str], dict[str, str]], list[dict[str, str]]]:
    file_map = _description_file_map(example)
    parsed_files = {
        stem: (path, _read_csv_rows(path))
        for stem, path in file_map.items()
    }
    semantics: dict[tuple[str, str], dict[str, str]] = {}
    sources: list[dict[str, str]] = []
    for table_name in table_names:
        matched = parsed_files.get(table_name.casefold())
        if matched is None:
            expected = {
                name.casefold()
                for name in columns_by_table.get(table_name.casefold(), set())
            }
            ranked = []
            for stem, (candidate_path, rows) in parsed_files.items():
                available = {
                    row.get("original_column_name", "").casefold()
                    for row in rows
                    if row.get("original_column_name")
                }
                intersection = len(expected & available)
                coverage = intersection / max(1, len(expected))
                precision = intersection / max(1, len(available))
                if coverage >= 0.5 and precision >= 0.5:
                    ranked.append(
                        (
                            coverage,
                            precision,
                            intersection,
                            stem,
                            candidate_path,
                            rows,
                        )
                    )
            ranked.sort(reverse=True)
            if ranked and (
                len(ranked) == 1 or ranked[0][:3] > ranked[1][:3]
            ):
                matched = (ranked[0][4], ranked[0][5])
        if matched is None:
            continue
        path, rows = matched
        payload = path.read_bytes()
        sources.append({
            "table_name": table_name,
            "path": str(path),
            "sha256": hashlib.sha256(payload).hexdigest(),
        })
        for row in rows:
            original_name = row.get("original_column_name", "").strip()
            if not original_name:
                continue
            entry = {
                key: row[key]
                for key in ("column_name", "column_description", "data_format")
                if row.get(key)
            }
            if entry:
                semantics[(table_name.casefold(), original_name.casefold())] = entry
    return semantics, sources


_CATALOG_BIRD_INSPECT_SEMANTICS_PROMPT_SUFFIX = """

INSPECT-ONLY BIRD COLUMN SEMANTICS
The source catalog remains lazy, and describe_table returns only the ordinary raw schema without
BIRD semantic names or descriptions. A successful inspect_column observation may attach both
semantic_name and column_description to the one inspected raw column. These BIRD annotations are
meaning hints only: tool arguments must still copy the exact raw table and column names. They are
neither executable aliases nor evidence that a value occurs in the database. Ground literals from
the ordinary inspect_column value-domain fields, search_values matches, or inspect_rows.
""".rstrip()


def catalog_profile_student_prompt(base_prompt: str, profile: str) -> str:
    """Apply only the model-visible prompt delta required by a lazy-catalog profile."""
    if profile == CATALOG_CONTEXT_PROFILE:
        return base_prompt
    if profile == CATALOG_BIRD_INSPECT_SEMANTICS_PROFILE:
        return base_prompt + _CATALOG_BIRD_INSPECT_SEMANTICS_PROMPT_SUFFIX
    raise ValueError(f"not a lazy catalog context profile: {profile!r}")


def enrich_catalog_perception_output(
    profile: str,
    example: dict,
    tool: str,
    arguments: dict,
    output: dict,
) -> tuple[dict, dict | None]:
    """Attach BIRD semantics only to one successful inspect_column observation.

    The overlay is model-visible and audited but never mutates canonical executor output,
    resident state, provenance, or verifier behavior. It exposes neither metadata sample values
    nor ``data_format``/``value_description`` fields.
    """
    if (
        profile != CATALOG_BIRD_INSPECT_SEMANTICS_PROFILE
        or tool != "inspect_column"
    ):
        return output, None

    table_name = str(arguments.get("table") or "").strip()
    column_name = str(
        output.get("column") or arguments.get("column") or ""
    ).strip()
    if not table_name or not column_name:
        return output, None
    semantics, sources = _bird_column_semantics(
        example,
        [table_name],
        {table_name.casefold(): {column_name}},
    )
    annotation = semantics.get(
        (table_name.casefold(), column_name.casefold()),
        {},
    )
    visible = deepcopy(output)
    enriched_fields = 0
    if annotation.get("column_name"):
        visible["semantic_name"] = annotation["column_name"]
        enriched_fields += 1
    if annotation.get("column_description"):
        visible["column_description"] = annotation["column_description"]
        enriched_fields += 1

    def stable_hash(value: dict) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()

    return visible, {
        "tool": tool,
        "enriched_field_count": enriched_fields,
        "metadata_sources": sources,
        "canonical_output_sha256": stable_hash(output),
        "model_visible_output_sha256": stable_hash(visible),
    }


def _format_identifier(identifier: str) -> str:
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _sample_column_values(
    harness,
    table_name: str,
    column_name: str,
    *,
    value_count: int,
) -> list:
    if value_count <= 0:
        return []
    table_sql = _format_identifier(table_name)
    column_sql = _format_identifier(column_name)
    rows = harness.conn.execute(
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
    return [
        value[:40] if isinstance(value, str) else value
        for (value,) in rows
    ]


def _primary_key_indices(db_schema: dict) -> set[int]:
    indices: set[int] = set()
    for value in db_schema.get("primary_keys", []):
        if isinstance(value, int):
            indices.add(value)
        elif isinstance(value, list):
            indices.update(index for index in value if isinstance(index, int))
    return indices


def build_full_bird_database_context(
    harness,
    example: dict,
    catalog: dict,
    *,
    value_count: int = 2,
    schema_metadata_json: str | None = None,
    profile: str = FULL_BIRD_CONTEXT_PROFILE,
) -> tuple[dict, dict]:
    """Return complete structured source context plus an audit record.

    ``column_description`` comes from BIRD's per-table ``database_description`` CSV files.
    ``semantic_name`` falls back to Spider/BIRD's normalized ``column_names`` metadata.
    Representative values are sampled live from the episode database and are examples, not a
    complete value domain.
    """
    if value_count < 0:
        raise ValueError("value_count must be non-negative")
    if profile not in FULL_BIRD_CONTEXT_PROFILES:
        raise ValueError(f"not a full BIRD database context profile: {profile!r}")
    metadata_path = resolve_schema_metadata_path(
        example,
        explicit_path=schema_metadata_json,
    )
    db_schema = load_database_schema(metadata_path, example["db_id"])
    table_names = list(db_schema["table_names_original"])
    original_columns = list(db_schema["column_names_original"])
    normalized_columns = list(db_schema.get("column_names", original_columns))
    column_types = list(db_schema["column_types"])
    if not (len(original_columns) == len(normalized_columns) == len(column_types)):
        raise ValueError("schema column metadata lengths do not match")

    row_counts = {
        str(table.get("table_name")).casefold(): table.get(
            "num_rows", table.get("row_count")
        )
        for table in catalog.get("tables", [])
    }
    primary_keys = _primary_key_indices(db_schema)
    foreign_key_targets = {
        source_index: target_index
        for source_index, target_index in db_schema.get("foreign_keys", [])
    }
    columns_by_table = {
        table_name.casefold(): {
            str(column_name)
            for owner_index, column_name in original_columns
            if owner_index == table_index and column_name != "*"
        }
        for table_index, table_name in enumerate(table_names)
    }
    semantics, description_sources = _bird_column_semantics(
        example,
        table_names,
        columns_by_table,
    )

    tables = []
    for table_index, table_name in enumerate(table_names):
        columns = []
        for column_index, (
            (owner_index, original_name),
            (_, normalized_name),
            column_type,
        ) in enumerate(zip(original_columns, normalized_columns, column_types)):
            if owner_index != table_index or original_name == "*":
                continue
            semantic = dict(
                semantics.get(
                    (table_name.casefold(), str(original_name).casefold()),
                    {},
                )
            )
            normalized_name = _normalize_text(normalized_name)
            if (
                normalized_name
                and normalized_name.casefold()
                not in {
                    str(original_name).casefold(),
                    str(original_name).casefold().replace(" ", "_"),
                    str(original_name).casefold().replace(" ", ""),
                }
                and "column_name" not in semantic
            ):
                semantic["column_name"] = normalized_name
            column = {
                "name": original_name,
                "type": column_type,
            }
            if semantic.get("column_name"):
                column["semantic_name"] = semantic["column_name"]
            if semantic.get("column_description"):
                column["description"] = semantic["column_description"]
            if semantic.get("data_format"):
                column["data_format"] = semantic["data_format"]
            examples = _sample_column_values(
                harness,
                table_name,
                original_name,
                value_count=value_count,
            )
            if examples:
                column["example_values"] = examples
            if column_index in primary_keys:
                column["primary_key"] = True
            target_index = foreign_key_targets.get(column_index)
            if target_index is not None:
                target_table_index, target_column = original_columns[target_index]
                column["references"] = {
                    "table": table_names[target_table_index],
                    "column": target_column,
                }
            columns.append(column)
        tables.append({
            "table_name": table_name,
            "row_count": row_counts.get(table_name.casefold()),
            "columns": columns,
        })

    metadata_bytes = Path(metadata_path).read_bytes()
    context = {
        # Keep the database payload bit-identical across tool-surface ablations. The exact
        # callable profile is recorded separately in the audit and model-visible tool hash.
        "context_profile": FULL_BIRD_CONTEXT_PROFILE,
        "database_engine": "SQLite",
        "schema_scope": "complete_source_schema",
        "example_values_per_column": value_count,
        "example_values_are_exhaustive": False,
        "tables": tables,
        "relations": list(catalog.get("relations", [])),
    }
    context_sha256 = hashlib.sha256(
        json.dumps(
            context,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    audit = {
        "context_profile": profile,
        "schema_metadata_json": metadata_path,
        "schema_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "bird_column_description_sources": description_sources,
        "bird_column_description_file_count": len(description_sources),
        "schema_value_count": value_count,
        "model_visible_database_context_sha256": context_sha256,
        "source_table_count": len(tables),
        "source_column_count": sum(len(table["columns"]) for table in tables),
        "disabled_model_tools": sorted(disabled_tools_for_profile(profile)),
        "model_visible_tool_schema_sha256": model_visible_tool_schema_hash(
            profile
        ),
    }
    return context, audit


_FULL_CONTEXT_CONTRACT = (
    "The opening DATASET OVERVIEW contains every source table, all source columns, types, primary "
    "and foreign keys, BIRD-provided short column semantics when available, and bounded live "
    "example values. It is complete for source schemas but example values are not a complete "
    "domain. EXTERNAL KNOWLEDGE, when present, is user-provided task context. CURRENT ENVIRONMENT "
    "STATE is the authoritative workspace for derived handles, row reads, scalar-producing steps, "
    "and plan control state. A derived table handle exposes columns and row_count but not row "
    "values until read_subtable is called. Harness-authored relation derivation metadata states "
    "executed row/column semantics; it is not policy advice or an additional source of values."
)

def _full_context_runtime_rules(profile: str) -> str:
    if profile == FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE:
        value_guidance = (
            "Example values are representative rather than exhaustive; use inspect_column for an "
            "uncertain literal or value domain and read_subtable when row co-occurrence or "
            "additional rows matter. "
        )
    else:
        value_guidance = (
            "Example values are representative rather than exhaustive; use read_subtable when "
            "exact value spelling, row co-occurrence, or additional rows matter. "
        )
    return (
        "2. Use only names and arguments in TOOLS. One turn contains one action.\n"
        "3. The opening DATASET OVERVIEW already contains the complete source-table schema. Copy "
        "exact table and column names from it. "
        + value_guidance
        + "Use environment handles and exact logical columns; after joins, dotted identifiers are "
        "relation.column, while a bare downstream name is valid only when it resolves uniquely.\n"
        "4. A value_ref cites the step that produced the resident scalar or one-row metric table, "
        "not a perception or plan step. Use value_ref+column for a named metric in a one-row "
        "table.\n"
        "5. Relational operators preserve their declared population and grain. Per-aggregation "
        "where conditions share one input table; output_layout=columns is the "
        "category-row-to-column reshape.\n"
        "6. Terminal evidence is scored from the cited table only. Before answering, derive the "
        "exact requested rows, columns, and column order; observing rows or explaining them does "
        "not change the relation."
    )


def _full_context_teacher_note(profile: str) -> str:
    if profile == FULL_BIRD_CONTEXT_WITH_INSPECT_PROFILE:
        return (
            "FULL DATABASE CONTEXT PROFILE\n"
            "The complete source schema and bounded examples are already visible. describe_table "
            "is deliberately unavailable and is not a legal action in this profile. "
            "inspect_column remains available only for grounding an uncertain literal or value "
            "domain when the bounded examples are insufficient. Use the supplied BIRD "
            "descriptions to resolve column meaning, and use read_subtable when actual row "
            "co-occurrence or additional rows are needed."
        )
    return (
        "FULL DATABASE CONTEXT PROFILE\n"
        "The complete source schema and bounded examples are already visible. describe_table and "
        "inspect_column are deliberately unavailable and are not legal actions in this profile. "
        "Do not recreate either action. Use the supplied BIRD descriptions to resolve column "
        "meaning. When bounded examples do not establish a needed literal or row relationship, "
        "use a targeted read_subtable call, then continue with the ordinary relational tools."
    )


def build_full_context_student_prompt(
    profile: str = FULL_BIRD_CONTEXT_PROFILE,
) -> str:
    if profile not in FULL_BIRD_CONTEXT_PROFILES:
        raise ValueError(f"not a full BIRD database context profile: {profile!r}")
    tools = enabled_tools_for_profile(profile)
    return (
        "You are a relational table-tool agent. Answer the question by executing one typed tool "
        "action per turn.\n\n"
        "CONTEXT\n"
        f"{_FULL_CONTEXT_CONTRACT}\n\n"
        "TOOLS\n"
        + "\n".join(SHARED_TOOL_SPECS[tool] for tool in tools)
        + "\n\nRULES\n"
        + CANONICAL_ACTION_RULE
        + "\n"
        + _full_context_runtime_rules(profile)
    )


def build_full_context_teacher_prompt(
    student_prompt: str,
    profile: str = FULL_BIRD_CONTEXT_PROFILE,
) -> str:
    if profile not in FULL_BIRD_CONTEXT_PROFILES:
        raise ValueError(f"not a full BIRD database context profile: {profile!r}")
    tools = enabled_tools_for_profile(profile)
    guidance = []
    for tool in tools:
        text = TEACHER_TOOL_GUIDANCE[tool]
        if tool == "scalar_compute":
            text = text.replace(
                "never a plan, describe_table, inspect_column, or read_subtable observation step",
                "never a plan or read-only perception observation step",
            )
        guidance.append(text)
    return (
        student_prompt
        + "\n\nTEACHER-ONLY DATA GENERATION GUIDANCE\n"
        "The following elaborations and examples are quality controls for producing causal "
        "demonstrations. They are not part of the student runtime prompt and do not add any public "
        "tool, argument, state field, or execution behavior.\n"
        + _full_context_teacher_note(profile)
        + "\n\n"
        + TEACHER_SEMANTIC_DECISION_DISCIPLINE
        + "\n\n"
        + "\n".join(guidance)
        + "\n\n"
        + CANONICAL_CALL_COOKBOOK
    )


def rolling_full_context_student_prompt(
    profile: str = FULL_BIRD_CONTEXT_PROFILE,
) -> str:
    return build_full_context_student_prompt(profile) + ROLLING_HISTORY_CONTRACT
