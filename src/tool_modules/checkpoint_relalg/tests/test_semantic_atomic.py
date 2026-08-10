from __future__ import annotations

import json
import sqlite3

import pytest

from src.tool_modules.checkpoint_relalg.environment_state import EnvironmentState
from src.tool_modules.checkpoint_relalg.executors import (
    SQLiteRelationalExecutor,
    artifact_backing_name,
)
from src.tool_modules.checkpoint_relalg.expression import RelAlgValidationError
from src.tool_modules.checkpoint_relalg.protocol import (
    ATOMIC_OPERATOR_PROFILE_SEMANTIC,
    MODE_TOOLS,
    ProtocolValidationError,
    SEMANTIC_ATOMIC_TOOLS,
    capability_manifest,
    get_system_prompt,
    provider_tool_definitions,
    tool_schema_hash,
    validate_tool_call,
)
from src.tool_modules.checkpoint_relalg.relation_artifact import Column, SourceRelation
from src.tool_modules.checkpoint_relalg.runtime import CheckpointRelalgRuntime


def _fixture() -> tuple[sqlite3.Connection, EnvironmentState, SQLiteRelationalExecutor]:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE facts(category TEXT, amount INTEGER, active INTEGER)")
    rows = [("a", 10, 1), ("a", 20, 0), ("b", 30, 1), ("b", 30, 1)]
    connection.executemany("INSERT INTO facts VALUES (?, ?, ?)", rows)
    source = SourceRelation(
        name="facts",
        columns=(
            Column("category", "TEXT"),
            Column("amount", "INTEGER"),
            Column("active", "INTEGER"),
        ),
        row_count=len(rows),
    )
    state = EnvironmentState([source])
    return connection, state, SQLiteRelationalExecutor(connection, state)


def _rows(connection: sqlite3.Connection, artifact) -> list[tuple[object, ...]]:
    columns = ", ".join(f'"{column.name}"' for column in artifact.columns)
    order = ' ORDER BY "__relalg_ordinal"' if artifact.ordered_by else ""
    return connection.execute(
        f'SELECT {columns} FROM temp."{artifact_backing_name(artifact.table)}"{order}'
    ).fetchall()


def test_semantic_profile_is_isolated_and_keeps_checkpoint_control() -> None:
    names = [
        item["function"]["name"]
        for item in provider_tool_definitions("atomic", ATOMIC_OPERATOR_PROFILE_SEMANTIC)
    ]
    assert names == [
        "describe_table",
        "inspect_column",
        "read_rows",
        *SEMANTIC_ATOMIC_TOOLS,
        "commit_checkpoint",
        "restore_checkpoint",
        "answer",
    ]
    assert [item["function"]["name"] for item in provider_tool_definitions("atomic")] == list(
        MODE_TOOLS["atomic"]
    )
    assert "sort" not in names and "rank_select" in names
    manifest = capability_manifest(
        "atomic", atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC
    )
    assert manifest["checkpoint_tools_model_visible"] is True
    assert manifest["tool_schema_sha256"] == tool_schema_hash(
        "atomic", ATOMIC_OPERATOR_PROFILE_SEMANTIC
    )
    prompt = get_system_prompt(
        "atomic",
        teacher=True,
        atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
    )
    assert "MODE: ATOMIC / SEMANTIC-V2" in prompt
    assert "TEACHER CHECKPOINT GUIDANCE" in prompt
    assert len(
        json.dumps(
            provider_tool_definitions("atomic", ATOMIC_OPERATOR_PROFILE_SEMANTIC),
            separators=(",", ":"),
        )
    ) < len(json.dumps(provider_tool_definitions("atomic"), separators=(",", ":")))


def test_semantic_validator_accepts_metric_where_and_rejects_micro_leak() -> None:
    action = validate_tool_call(
        "atomic",
        "group_aggregate",
        {
            "table": "facts",
            "group_by": [],
            "metrics": [
                {
                    "op": "sum",
                    "column": "amount",
                    "as": "active_sum",
                    "where": {"column": "active", "op": "=", "value": 1},
                }
            ],
        },
        atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
    )
    assert action["metrics"][0]["as"] == "active_sum"
    with pytest.raises(ProtocolValidationError, match="unavailable"):
        validate_tool_call(
            "atomic",
            "sort",
            {"table": "facts", "keys": [{"column": "amount", "direction": "desc"}]},
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
        )


def test_group_aggregate_and_scalar_compute_lower_to_grounded_artifacts() -> None:
    connection, _, executor = _fixture()
    aggregate = executor.execute(
        "group_aggregate",
        {
            "table": "facts",
            "group_by": [],
            "metrics": [
                {
                    "op": "sum",
                    "column": "amount",
                    "as": "active_sum",
                    "where": {"column": "active", "op": "=", "value": 1},
                },
                {"op": "count", "column": "*", "as": "all_rows"},
            ],
        },
    )
    assert _rows(connection, aggregate) == [(70, 4)]
    scalar = executor.execute(
        "scalar_compute",
        {
            "table": aggregate.table,
            "outputs": [
                {
                    "op": "divide",
                    "left": {"column": "active_sum"},
                    "right": {"column": "all_rows"},
                    "multiplier": 100,
                    "round_digits": 1,
                    "as": "percentage",
                }
            ],
        },
    )
    assert _rows(connection, scalar) == [(1750.0,)]
    assert scalar.scalar_cell == 1750.0

    before = executor.state.logical_hash()
    with pytest.raises(RelAlgValidationError) as captured:
        executor.execute(
            "scalar_compute",
            {
                "table": "facts",
                "outputs": [
                    {
                        "op": "divide",
                        "left": {"column": "amount"},
                        "right": {"value": 1},
                        "as": "amount",
                    }
                ],
            },
        )
    assert captured.value.code == "scalar_input_not_single_row"
    assert executor.state.logical_hash() == before


def test_rank_select_ties_and_shape_distinct() -> None:
    connection, _, executor = _fixture()
    ranked = executor.execute(
        "rank_select",
        {
            "table": "facts",
            "order_by": [{"column": "amount", "direction": "desc"}],
            "top_k": 1,
            "with_ties": True,
            "outputs": [{"column": "category"}, {"column": "amount"}],
        },
    )
    assert _rows(connection, ranked) == [("b", 30), ("b", 30)]
    shaped = executor.execute(
        "shape_rows",
        {
            "table": ranked.table,
            "outputs": [{"column": "category"}],
            "distinct": True,
        },
    )
    assert _rows(connection, shaped) == [("b",)]


def test_semantic_projection_is_not_poisoned_by_unconsumed_dirty_column() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE dirty(value INTEGER, malformed_date DATE)")
    connection.execute("INSERT INTO dirty VALUES (7, 'not-a-date')")
    source = SourceRelation(
        name="dirty",
        columns=(Column("value", "INTEGER"), Column("malformed_date", "DATE")),
        row_count=1,
    )
    state = EnvironmentState([source])
    executor = SQLiteRelationalExecutor(connection, state)
    shaped = executor.execute(
        "shape_rows",
        {"table": "dirty", "outputs": [{"column": "value"}]},
    )
    assert _rows(connection, shaped) == [(7,)]


def test_semantic_runtime_checkpoint_restore_remains_exact() -> None:
    connection, _, _ = _fixture()
    runtime = CheckpointRelalgRuntime(
        connection,
        mode="atomic",
        atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
    )
    committed = runtime.apply(
        "commit_checkpoint",
        {
            "progress_summary": ["population identified"],
            "remaining_uncertainties": ["ranking"],
            "next_targets": ["rank active rows"],
        },
    )
    checkpoint_id = committed["checkpoint_id"]
    disposable = runtime.apply(
        "shape_rows",
        {
            "table": "facts",
            "outputs": [{"column": "category"}],
        },
    )
    assert disposable["status"] == "success"
    restored = runtime.apply(
        "restore_checkpoint",
        {
            "checkpoint_id": checkpoint_id,
            "reason": "discard the exploratory projection",
            "next_targets": ["rank active rows with ties"],
        },
    )
    assert restored["status"] == "success"
    assert not runtime.state.active_artifact_ids
