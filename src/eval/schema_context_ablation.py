"""Isolated initial-context profiles for the version24 BIRD tool-agent ablation.

The historical condition is deliberately byte-for-byte unchanged: ``lazy-catalog-v1``
returns the existing catalog and contributes no system-prompt suffix. Every experimental
condition has one explicit renderer/profile ID and one dedicated prompt contract so schema
names, semantic aliases, and sampled values cannot be silently mixed across arms.
"""
from __future__ import annotations

import csv
import hashlib
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

from catalog import build_catalog


CONTEXT_RENDERER_VERSION = "bird-tool-context-ablation-v1"

LAZY_CATALOG_PROFILE = "lazy-catalog-v1"
FULL_SCHEMA_PROFILE = "full-schema-v1"
FULL_SCHEMA_SEMANTIC_PROFILE = "full-schema-bird-semantic-v1"
FULL_SCHEMA_SEMANTIC_DESCRIPTIONS_PROFILE = (
    "full-schema-bird-semantic-sql-astra-descriptions-v1"
)
FULL_SCHEMA_SEMANTIC_VALUES_PROFILE = "full-schema-bird-semantic-values2-v1"
LAZY_SEMANTIC_DESCRIBE_PROFILE = "lazy-catalog-semantic-describe-v1"

INITIAL_CONTEXT_PROFILES = (
    LAZY_CATALOG_PROFILE,
    FULL_SCHEMA_PROFILE,
    FULL_SCHEMA_SEMANTIC_PROFILE,
    FULL_SCHEMA_SEMANTIC_DESCRIPTIONS_PROFILE,
    FULL_SCHEMA_SEMANTIC_VALUES_PROFILE,
    LAZY_SEMANTIC_DESCRIBE_PROFILE,
)

_SEMANTIC_PROFILES = {
    FULL_SCHEMA_SEMANTIC_PROFILE,
    FULL_SCHEMA_SEMANTIC_DESCRIPTIONS_PROFILE,
    FULL_SCHEMA_SEMANTIC_VALUES_PROFILE,
    LAZY_SEMANTIC_DESCRIBE_PROFILE,
}
_FULL_SCHEMA_PROFILES = {
    FULL_SCHEMA_PROFILE,
    FULL_SCHEMA_SEMANTIC_PROFILE,
    FULL_SCHEMA_SEMANTIC_DESCRIPTIONS_PROFILE,
    FULL_SCHEMA_SEMANTIC_VALUES_PROFILE,
}

_PROFILE_PROMPT_SUFFIXES = {
    LAZY_CATALOG_PROFILE: "",
    FULL_SCHEMA_PROFILE: (
        "\n\nINITIAL CONTEXT PROFILE: full-schema-v1\n"
        "The opening DATASET OVERVIEW lists every source table and every executable raw column "
        "name with its SQLite type and primary-key flag. Foreign-key edges are in relations. "
        "It contains no semantic aliases and no values. Use the exact table_name and column name "
        "strings in tool arguments. Source describe_table calls are legal but are unnecessary "
        "when they would only repeat schema already present in the opening overview. Use "
        "inspect_column or read_subtable when actual values are needed."
    ),
    FULL_SCHEMA_SEMANTIC_PROFILE: (
        "\n\nINITIAL CONTEXT PROFILE: full-schema-bird-semantic-v1\n"
        "The opening DATASET OVERVIEW lists every source table and every executable raw column "
        "name with its SQLite type and primary-key flag, plus BIRD's semantic_table_name and "
        "semantic_name aliases. Foreign-key edges are in relations. Semantic names are hints for "
        "understanding only: they are not executable identifiers. Every tool argument must use "
        "the exact raw table_name and raw column name. No database values are provided. Source "
        "describe_table returns the same raw-name/semantic-name distinction. Use inspect_column "
        "or read_subtable when actual values are needed."
    ),
    FULL_SCHEMA_SEMANTIC_DESCRIPTIONS_PROFILE: (
        "\n\nINITIAL CONTEXT PROFILE: "
        "full-schema-bird-semantic-sql-astra-descriptions-v1\n"
        "The opening DATASET OVERVIEW lists every source table and every executable raw column "
        "name with its SQLite type and primary-key flag, plus BIRD's semantic_table_name and "
        "semantic_name aliases and the BIRD column description used by SQL-ASTRA-style prompts. "
        "Foreign-key edges are in relations. Semantic names and descriptions are task-"
        "understanding metadata only: they are not executable identifiers, observed database "
        "values, or evidence that a predicate is true. Every tool argument must use the exact raw "
        "table_name and raw column name. No example values are provided. Source describe_table "
        "returns the same raw/semantic/description distinction. Use inspect_column or "
        "read_subtable when actual values are needed."
    ),
    FULL_SCHEMA_SEMANTIC_VALUES_PROFILE: (
        "\n\nINITIAL CONTEXT PROFILE: full-schema-bird-semantic-values2-v1\n"
        "The opening DATASET OVERVIEW lists every source table and every executable raw column "
        "name with its SQLite type and primary-key flag, plus BIRD's semantic_table_name and "
        "semantic_name aliases and up to two deterministic live example_values per column. "
        "Foreign-key edges are in relations. Semantic names are understanding hints only and "
        "example_values are non-exhaustive samples only; neither changes the executable schema. "
        "Every tool argument must use the exact raw table_name and raw column name. Source "
        "describe_table returns the same raw/semantic/sample fields. Use inspect_column when a "
        "filter literal must be checked beyond the shown samples."
    ),
    LAZY_SEMANTIC_DESCRIBE_PROFILE: (
        "\n\nINITIAL CONTEXT PROFILE: lazy-catalog-semantic-describe-v1\n"
        "The opening DATASET OVERVIEW remains the historical lazy catalog: table names, row "
        "counts, and foreign-key relations only, with no columns, aliases, or values. Acquire "
        "needed source schemas through describe_table. Its returned columns include both the "
        "executable raw name and BIRD's semantic_name; tables may include semantic_table_name. "
        "Semantic names are understanding hints only. Every later tool argument must use the "
        "exact raw table_name and raw column name. Values still require inspect_column or "
        "read_subtable."
    ),
}


def context_prompt_suffix(profile: str) -> str:
    try:
        return _PROFILE_PROMPT_SUFFIXES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown initial context profile: {profile}") from exc


def align_base_system_prompt(system_prompt: str, profile: str) -> str:
    """Remove version24's lazy-only instructions from full-schema experimental arms.

    Appending a later exception is not sufficient: it leaves the model with contradictory schema
    acquisition instructions. These exact replacements fail closed if the frozen version24 prompt
    changes, while both lazy profiles preserve the historical base prompt byte-for-byte.
    """
    if profile not in INITIAL_CONTEXT_PROFILES:
        raise ValueError(f"unknown initial context profile: {profile}")
    if profile not in _FULL_SCHEMA_PROFILES:
        return system_prompt

    replacements = (
        (
            "The opening overview is a CATALOG: table names + row counts + foreign-key relations "
            "only (no columns) — so it stays small on large databases.",
            "The opening overview is a DATABASE CONTEXT object whose exact schema information is "
            "defined by the INITIAL CONTEXT PROFILE below.",
        ),
        (
            "Read the columns of the tables you need with describe_table before operating.",
            "Follow the schema-acquisition policy in the INITIAL CONTEXT PROFILE below.",
        ),
        (
            "The opening overview lists only table names and relations, so read the schema of the "
            "tables you need before operating on them.",
            "Whether source schema needs to be read is defined by the INITIAL CONTEXT PROFILE "
            "below.",
        ),
        (
            "3. describe_table the needed tables first; inspect_column before filtering by a text "
            "value.",
            "3. Follow the INITIAL CONTEXT PROFILE for schema acquisition; inspect_column before "
            "filtering by a text value.",
        ),
    )
    aligned = system_prompt
    for old, new in replacements:
        count = aligned.count(old)
        if count != 1:
            raise ValueError(
                "frozen version24 prompt alignment expected one occurrence, "
                f"found {count}: {old}"
            )
        aligned = aligned.replace(old, new)
    return aligned


def context_contract_sha256(profile: str) -> str:
    payload = {
        "renderer_version": CONTEXT_RENDERER_VERSION,
        "profile": profile,
        "prompt_suffix": context_prompt_suffix(profile),
    }
    if profile in _FULL_SCHEMA_PROFILES:
        payload["base_prompt_alignment"] = "replace-version24-lazy-schema-instructions-v1"
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
            for filename in ("train_tables.json", "dev_tables.json", "tables.json")
        )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        f"could not discover BIRD schema metadata for {db_path}; "
        "pass --schema-metadata-json"
    )


@lru_cache(maxsize=8)
def _load_schema_entries(path: str) -> tuple[dict, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"schema metadata must be a JSON list: {path}")
    return tuple(payload)


def load_database_schema(path: str, db_id: str) -> dict:
    matches = [entry for entry in _load_schema_entries(path) if entry.get("db_id") == db_id]
    if len(matches) != 1:
        raise ValueError(
            f"expected one schema entry for db_id={db_id!r} in {path}, found {len(matches)}"
        )
    return matches[0]


def _primary_key_indices(db_schema: dict) -> set[int]:
    indices: set[int] = set()
    for item in db_schema.get("primary_keys", []):
        if isinstance(item, int):
            indices.add(item)
        elif isinstance(item, list):
            indices.update(value for value in item if isinstance(value, int))
    return indices


def _semantic_maps(db_schema: dict) -> tuple[dict[str, str], dict[tuple[str, str], str]]:
    original_tables = db_schema.get("table_names_original", [])
    semantic_tables = db_schema.get("table_names", original_tables)
    table_aliases = {
        str(original).lower(): str(semantic)
        for original, semantic in zip(original_tables, semantic_tables)
    }

    original_columns = db_schema.get("column_names_original", [])
    semantic_columns = db_schema.get("column_names", original_columns)
    column_aliases: dict[tuple[str, str], str] = {}
    for original, semantic in zip(original_columns, semantic_columns):
        if not (
            isinstance(original, list)
            and len(original) == 2
            and isinstance(semantic, list)
            and len(semantic) == 2
        ):
            continue
        table_index, raw_name = original
        if not isinstance(table_index, int) or table_index < 0:
            continue
        if table_index >= len(original_tables):
            continue
        column_aliases[
            (str(original_tables[table_index]).lower(), str(raw_name).lower())
        ] = str(semantic[1])
    return table_aliases, column_aliases


@lru_cache(maxsize=128)
def _load_column_descriptions(directory: str) -> dict[tuple[str, str], str]:
    """Load BIRD's per-table CSV descriptions without treating value metadata as evidence."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"BIRD database_description directory not found: {root}")
    descriptions: dict[tuple[str, str], str] = {}
    for csv_path in sorted(root.glob("*.csv"), key=lambda path: path.name.lower()):
        table_key = csv_path.stem.lower()
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"original_column_name", "column_description"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(
                    f"BIRD description CSV lacks {sorted(required)}: {csv_path}"
                )
            for row in reader:
                raw_column = str(row.get("original_column_name") or "").strip()
                description = str(row.get("column_description") or "").strip()
                if not raw_column or not description:
                    continue
                key = (table_key, raw_column.lower())
                previous = descriptions.get(key)
                if previous is not None and previous != description:
                    raise ValueError(
                        f"conflicting BIRD descriptions for {key}: {csv_path}"
                    )
                descriptions[key] = description
    return descriptions


def resolve_database_description_dir(example: dict) -> str:
    db_path = Path(example["db_path"]).expanduser().resolve()
    path = db_path.parent / "database_description"
    if not path.is_dir():
        raise FileNotFoundError(
            f"could not discover BIRD database_description beside {db_path}"
        )
    return str(path)


def _quote_identifier(value: str) -> str:
    return '"' + str(value).replace('"', '""') + '"'


def _sample_column_values(
    harness: Any,
    table_name: str,
    column_name: str,
    *,
    value_count: int,
    scan_limit: int = 200,
) -> list:
    """Return bounded deterministic live examples without full-column scans on large BIRD DBs."""
    if value_count <= 0:
        return []
    table_sql = _quote_identifier(table_name)
    column_sql = _quote_identifier(column_name)
    rows = harness.conn.execute(
        f"SELECT {column_sql} FROM {table_sql} "
        f"WHERE {column_sql} IS NOT NULL LIMIT ?",
        (max(value_count, scan_limit),),
    ).fetchall()
    values = []
    seen = set()
    for (value,) in rows:
        if value == "":
            continue
        normalized = value
        if isinstance(normalized, bytes):
            normalized = normalized.decode("utf-8", "replace")
        if isinstance(normalized, str) and len(normalized) > 80:
            normalized = normalized[:80] + "…"
        key = (type(normalized).__name__, repr(normalized))
        if key in seen:
            continue
        seen.add(key)
        values.append(normalized)
        if len(values) >= value_count:
            break
    return values


def _enrich_describe_output(
    harness: Any,
    output: dict,
    db_schema: dict,
    *,
    add_semantics: bool,
    column_descriptions: dict[tuple[str, str], str] | None = None,
    value_count: int,
) -> dict:
    visible = deepcopy(output)
    table_aliases, column_aliases = _semantic_maps(db_schema)
    primary_indices = _primary_key_indices(db_schema)
    original_tables = db_schema.get("table_names_original", [])
    original_columns = db_schema.get("column_names_original", [])

    primary_by_name = set()
    for index in primary_indices:
        if not isinstance(index, int) or not 0 <= index < len(original_columns):
            continue
        table_index, column_name = original_columns[index]
        if isinstance(table_index, int) and 0 <= table_index < len(original_tables):
            primary_by_name.add(
                (str(original_tables[table_index]).lower(), str(column_name).lower())
            )

    for table in visible.get("tables", []):
        raw_table = str(table.get("table_name", ""))
        table_key = raw_table.lower()
        if table_key not in table_aliases:
            continue
        if add_semantics:
            table["semantic_table_name"] = table_aliases[table_key]
        for column in table.get("columns", []):
            raw_column = str(column.get("name", ""))
            column_key = (table_key, raw_column.lower())
            column["pk"] = bool(column.get("pk") or column_key in primary_by_name)
            if add_semantics:
                column["semantic_name"] = column_aliases.get(column_key, raw_column)
            if column_descriptions:
                description = column_descriptions.get(column_key)
                if description:
                    column["description"] = description
            if value_count:
                column["example_values"] = _sample_column_values(
                    harness,
                    raw_table,
                    raw_column,
                    value_count=value_count,
                )
    return visible


def build_initial_context(
    harness: Any,
    example: dict,
    *,
    profile: str,
    schema_metadata_json: str | None = None,
    value_count: int = 2,
) -> dict:
    if profile not in INITIAL_CONTEXT_PROFILES:
        raise ValueError(f"unknown initial context profile: {profile}")
    lazy_catalog = build_catalog(harness)
    if profile in {LAZY_CATALOG_PROFILE, LAZY_SEMANTIC_DESCRIBE_PROFILE}:
        return lazy_catalog

    table_names = [table["table_name"] for table in lazy_catalog["tables"]]
    described = harness.describe_table(table_names)
    if profile == FULL_SCHEMA_PROFILE:
        enriched = deepcopy(described)
    else:
        schema_path = resolve_schema_metadata_path(
            example,
            explicit_path=schema_metadata_json,
        )
        db_schema = load_database_schema(schema_path, example["db_id"])
        column_descriptions = (
            _load_column_descriptions(resolve_database_description_dir(example))
            if profile == FULL_SCHEMA_SEMANTIC_DESCRIPTIONS_PROFILE
            else None
        )
        enriched = _enrich_describe_output(
            harness,
            described,
            db_schema,
            add_semantics=True,
            column_descriptions=column_descriptions,
            value_count=(
                value_count if profile == FULL_SCHEMA_SEMANTIC_VALUES_PROFILE else 0
            ),
        )

    rows_by_name = {
        table["table_name"]: table["num_rows"] for table in lazy_catalog["tables"]
    }
    tables = []
    for table in enriched.get("tables", []):
        item = {
            "table_name": table["table_name"],
            "num_rows": rows_by_name.get(table["table_name"], table.get("row_count")),
            "columns": table.get("columns", []),
        }
        if "semantic_table_name" in table:
            item["semantic_table_name"] = table["semantic_table_name"]
        tables.append(item)
    return {
        "context_profile": profile,
        "tables": tables,
        "relations": lazy_catalog["relations"],
    }


def align_tool_output(
    harness: Any,
    tool: str,
    output: dict,
    example: dict,
    *,
    profile: str,
    schema_metadata_json: str | None = None,
    value_count: int = 2,
) -> dict:
    """Align describe_table feedback with the semantic contract of one experimental arm."""
    if tool != "describe_table" or profile not in _SEMANTIC_PROFILES:
        return output
    schema_path = resolve_schema_metadata_path(
        example,
        explicit_path=schema_metadata_json,
    )
    db_schema = load_database_schema(schema_path, example["db_id"])
    column_descriptions = (
        _load_column_descriptions(resolve_database_description_dir(example))
        if profile == FULL_SCHEMA_SEMANTIC_DESCRIPTIONS_PROFILE
        else None
    )
    return _enrich_describe_output(
        harness,
        output,
        db_schema,
        add_semantics=True,
        column_descriptions=column_descriptions,
        value_count=(
            value_count if profile == FULL_SCHEMA_SEMANTIC_VALUES_PROFILE else 0
        ),
    )
