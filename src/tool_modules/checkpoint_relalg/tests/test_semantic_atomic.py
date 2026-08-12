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
    ATOMIC_OPERATOR_PROFILE_SEMANTIC_V3,
    ATOMIC_OPERATOR_PROFILE_SEMANTIC_V4,
    ATOMIC_OPERATOR_PROFILE_SEMANTIC_V5,
    CARRIER_TEXT_JSON,
    MODE_TOOLS,
    ProtocolValidationError,
    SEMANTIC_ATOMIC_TOOLS,
    capability_manifest,
    get_system_prompt,
    provider_tool_definitions,
    tool_schema_hash,
    provider_phase_history_policy,
    trim_provider_phase_history,
    validate_tool_call,
    validate_model_action,
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


def test_semantic_v3_reuses_execution_surface_with_v24_compact_prompt() -> None:
    v2_tools = provider_tool_definitions("atomic", ATOMIC_OPERATOR_PROFILE_SEMANTIC)
    v3_tools = provider_tool_definitions("atomic", ATOMIC_OPERATOR_PROFILE_SEMANTIC_V3)
    assert v3_tools == v2_tools
    assert tool_schema_hash("atomic", ATOMIC_OPERATOR_PROFILE_SEMANTIC_V3) == tool_schema_hash(
        "atomic", ATOMIC_OPERATOR_PROFILE_SEMANTIC
    )

    v2_prompt = get_system_prompt(
        "atomic",
        teacher=True,
        carrier=CARRIER_TEXT_JSON,
        atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
    )
    v3_prompt = get_system_prompt(
        "atomic",
        teacher=True,
        carrier=CARRIER_TEXT_JSON,
        atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC_V3,
    )
    assert "MODE: ATOMIC / SEMANTIC-V3-V24" in v3_prompt
    assert "COMPACT EXECUTABLE CONTRACT" in v3_prompt
    assert "EXACT OUTPUT CONTRACT" in v3_prompt
    assert "EXACT TOOL SCHEMAS FOR ATOMIC MODE" not in v3_prompt
    assert len(v3_prompt) < len(v2_prompt) / 2

    manifest = capability_manifest(
        "atomic",
        carrier=CARRIER_TEXT_JSON,
        atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC_V3,
    )
    assert manifest["provider_phase_history_policy"] == "recent-4-turns-v1"
    assert manifest["model_schema_delivery"] == "v24-compact-operational-contract-v1"
    assert provider_phase_history_policy(ATOMIC_OPERATOR_PROFILE_SEMANTIC_V3) == (
        "recent-4-turns-v1"
    )


def test_semantic_v3_recent_four_history_keeps_complete_text_turns() -> None:
    messages = [
        {"role": role, "content": f"turn{turn}-{role}"}
        for turn in range(5)
        for role in ("user", "assistant", "user-result")
    ]
    trimmed = trim_provider_phase_history(
        messages,
        ATOMIC_OPERATOR_PROFILE_SEMANTIC_V3,
    )
    assert len(trimmed) == 12
    assert trimmed[0]["content"] == "turn1-user"
    assert trim_provider_phase_history(
        messages,
        ATOMIC_OPERATOR_PROFILE_SEMANTIC,
    ) == messages


def test_semantic_v3_canonical_calls_are_executable_schema_valid() -> None:
    contract = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "prompts"
        / "text_json_semantic_v3_contract.txt"
    ).read_text(encoding="utf-8")
    examples = [json.loads(line) for line in contract.splitlines() if line.startswith("{\"tool\"")]
    assert len(examples) >= 8
    for example in examples:
        validate_model_action(
            example,
            mode="atomic",
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC_V3,
        )


def test_semantic_v4_preserves_exact_logical_output_names_without_alias() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute('CREATE TABLE left_rows(id INTEGER, "Product Name" TEXT)')
    connection.execute('CREATE TABLE right_rows(id INTEGER, value INTEGER)')
    connection.execute('INSERT INTO left_rows VALUES (1, "widget")')
    connection.execute('INSERT INTO right_rows VALUES (1, 9)')
    runtime = CheckpointRelalgRuntime(
        connection,
        mode="atomic",
        atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC_V4,
    )
    runtime.apply("describe_table", {"tables": ["left_rows", "right_rows"]})
    joined = runtime.apply(
        "join",
        {
            "left": "left_rows",
            "right": "right_rows",
            "left_role": "l",
            "right_role": "r",
            "type": "inner",
            "on": [{"left_column": "id", "op": "=", "right_column": "id"}],
        },
    )
    table = joined["artifact"]["table"]
    shaped = runtime.apply(
        "shape_rows",
        {"table": table, "outputs": [{"column": "l.Product Name"}]},
    )
    assert shaped["status"] == "success"
    assert shaped["artifact"]["columns"] == [
        {"name": "l.Product Name", "canonical_type": "TEXT"}
    ]


def test_semantic_v3_replay_still_rejects_dotted_implicit_alias() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE left_rows(id INTEGER, value TEXT)")
    connection.execute("CREATE TABLE right_rows(id INTEGER)")
    runtime = CheckpointRelalgRuntime(
        connection,
        mode="atomic",
        atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC_V3,
    )
    runtime.apply("describe_table", {"tables": ["left_rows", "right_rows"]})
    joined = runtime.apply(
        "join",
        {
            "left": "left_rows",
            "right": "right_rows",
            "left_role": "l",
            "right_role": "r",
            "type": "inner",
            "on": [{"left_column": "id", "op": "=", "right_column": "id"}],
        },
    )
    result = runtime.apply(
        "shape_rows",
        {
            "table": joined["artifact"]["table"],
            "outputs": [{"column": "l.value"}],
        },
    )
    assert result["status"] == "error"
    assert result["error"]["code"] == "invalid_identifier"


def test_semantic_v4_contract_examples_and_identity_are_bound() -> None:
    contract = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "prompts"
        / "text_json_semantic_v4_contract.txt"
    ).read_text(encoding="utf-8")
    examples = [json.loads(line) for line in contract.splitlines() if line.startswith('{"tool"')]
    for example in examples:
        validate_model_action(
            example,
            mode="atomic",
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC_V4,
        )
    v3_manifest = capability_manifest(
        "atomic", carrier=CARRIER_TEXT_JSON,
        atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC_V3,
    )
    v4_manifest = capability_manifest(
        "atomic", carrier=CARRIER_TEXT_JSON,
        atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC_V4,
    )
    assert v4_manifest["tool_schema_sha256"] == v3_manifest["tool_schema_sha256"]
    assert v4_manifest["student_prompt_sha256"] != v3_manifest["student_prompt_sha256"]
    assert v4_manifest["model_schema_delivery"] == (
        "v24-compact-operational-contract-v2"
    )
    assert v4_manifest["semantic_output_name_policy"] == (
        "preserve-exact-logical-column-when-as-omitted-v1"
    )


def test_semantic_v5_changes_only_compact_output_guidance_identity() -> None:
    v4 = capability_manifest(
        "atomic", carrier=CARRIER_TEXT_JSON,
        atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC_V4,
    )
    v5 = capability_manifest(
        "atomic", carrier=CARRIER_TEXT_JSON,
        atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC_V5,
    )
    assert v5["tool_schema_sha256"] == v4["tool_schema_sha256"]
    assert v5["student_prompt_sha256"] != v4["student_prompt_sha256"]
    assert v5["semantic_output_name_policy"] == v4["semantic_output_name_policy"]
    assert v5["model_schema_delivery"] == "v24-compact-operational-contract-v3"
    prompt = get_system_prompt(
        "atomic", teacher=True, carrier=CARRIER_TEXT_JSON,
        atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC_V5,
    )
    assert "mechanical role prefix" in prompt
    assert "space-containing source column name" in prompt


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
