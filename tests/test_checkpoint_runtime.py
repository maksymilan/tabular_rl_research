from __future__ import annotations

import sqlite3

import pytest

from tool_modules.checkpoint_relalg.checkpoint_store import (
    CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V2,
    CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1,
)
from tool_modules.checkpoint_relalg.runtime import (
    CheckpointRelalgRuntime,
    RuntimeConfig,
    load_source_catalog,
)


def _connection() -> sqlite3.Connection:
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        """
        CREATE TABLE customers(
          customer_id INTEGER PRIMARY KEY,
          country TEXT,
          amount REAL
        );
        INSERT INTO customers VALUES
          (1, 'UK', 10.0), (2, 'US', 20.0), (3, 'UK', 30.0);
        """
    )
    return connection


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 0.0, -1.0])
def test_runtime_rejects_non_finite_or_non_positive_sql_timeouts(value):
    with pytest.raises(ValueError, match="positive finite"):
        RuntimeConfig(sql_timeout_seconds=value)


def test_direct_mode_executes_and_answers_exact_artifact():
    runtime = CheckpointRelalgRuntime(_connection(), mode="direct")
    described = runtime.apply("describe_table", {"tables": ["customers"]})
    assert described["status"] == "success"
    executed = runtime.apply(
        "execute_sql",
        {"sql": "SELECT COUNT(*) AS n FROM customers WHERE country = 'UK'"},
    )
    table = executed["artifact"]["table"]
    answered = runtime.apply("answer", {"table": table})
    assert answered["episode_ended"]
    assert runtime.answer_rows() == (["n"], [[2]])


def test_atomic_and_direct_artifacts_interoperate_in_hybrid_mode():
    runtime = CheckpointRelalgRuntime(_connection(), mode="hybrid")
    runtime.apply("describe_table", {"tables": ["customers"]})
    direct = runtime.apply(
        "execute_sql", {"sql": "SELECT customer_id, amount FROM customers"}
    )
    projected = runtime.apply(
        "project",
        {
            "table": direct["artifact"]["table"],
            "outputs": [{"expression": {"column": "customer_id"}}],
        },
    )
    assert projected["status"] == "success"
    assert projected["artifact"]["row_count"] == 3


def test_rejected_call_preserves_environment_hash():
    runtime = CheckpointRelalgRuntime(_connection(), mode="atomic")
    before = runtime.state.logical_hash()
    result = runtime.apply("distinct", {"table": "missing"})
    assert result["status"] == "error"
    assert result["error"]["code"] == "unknown_table"
    assert runtime.state.logical_hash() == before


def test_repeated_checkpoint_goal_is_a_state_preserving_structured_error():
    runtime = CheckpointRelalgRuntime(_connection(), mode="atomic")
    first = runtime.apply(
        "commit_checkpoint",
        {
            "progress_summary": ["Fixed the requested population."],
            "remaining_uncertainties": [],
            "next_targets": ["Compute the final aggregate."],
        },
    )
    assert first["status"] == "success"
    before = runtime.state.logical_hash()
    repeated = runtime.apply(
        "commit_checkpoint",
        {
            "progress_summary": ["Restated the same phase goal."],
            "remaining_uncertainties": [],
            "next_targets": [" compute the final AGGREGATE! "],
        },
    )
    assert repeated["status"] == "error"
    assert repeated["error"]["code"] == "checkpoint_goal_not_distinct"
    assert repeated["error"]["type"] == "argument_validation_error"
    assert runtime.checkpoints.checkpoint_count == 1
    assert runtime.state.logical_hash() == before


def test_runtime_rejects_checkpoint_budgets_above_eight():
    with pytest.raises(ValueError, match="must not exceed 8"):
        RuntimeConfig(max_checkpoints=9)


def test_runtime_checkpoint_commit_eligibility_is_semantic_atomic_only():
    with pytest.raises(ValueError, match="checkpoint_commit_eligibility_policy"):
        RuntimeConfig(checkpoint_commit_eligibility_policy="unsupported")

    config = RuntimeConfig(
        checkpoint_commit_eligibility_policy=(
            f" {CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1} "
        )
    )
    assert (
        config.checkpoint_commit_eligibility_policy
        == CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1
    )
    with pytest.raises(ValueError, match="atomic semantic-v2"):
        CheckpointRelalgRuntime(_connection(), mode="direct", config=config)
    with pytest.raises(ValueError, match="atomic semantic-v2"):
        CheckpointRelalgRuntime(_connection(), mode="atomic", config=config)


def test_runtime_enforces_first_semantic_milestone_commit_quota_without_mutation():
    runtime = CheckpointRelalgRuntime(
        _connection(),
        mode="atomic",
        atomic_operator_profile="semantic-v2",
        config=RuntimeConfig(
            checkpoint_commit_eligibility_policy=(
                CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1
            )
        ),
    )
    before = runtime.state.logical_hash()
    rejected = runtime.apply(
        "commit_checkpoint",
        {
            "progress_summary": ["No grounded milestone yet."],
            "remaining_uncertainties": [],
            "next_targets": ["Build a grounded population."],
        },
    )
    assert rejected["status"] == "error"
    assert rejected["error"]["type"] == "state_validation_error"
    assert rejected["error"]["code"] == "checkpoint_phase_progress_insufficient"
    assert rejected["error"]["details"]["quota"] == 2
    assert runtime.state.logical_hash() == before
    assert runtime.checkpoints.checkpoint_count == 0

    for country in ("UK", "US"):
        produced = runtime.apply(
            "filter_rows",
            {
                "table": "customers",
                "conditions": {
                    "op": "=",
                    "left": {"column": "country"},
                    "right": {"value": country},
                },
            },
        )
        assert produced["status"] == "success"
    committed = runtime.apply(
        "commit_checkpoint",
        {
            "progress_summary": ["Two candidate populations are grounded."],
            "remaining_uncertainties": ["The requested branch remains to be selected."],
            "next_targets": ["Select and aggregate the requested population."],
        },
    )
    assert committed["status"] == "success"
    assert runtime.checkpoints.checkpoint_count == 1
    assert runtime.audit_state()["checkpoint_commit_eligibility_policy"] == (
        CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1
    )


def test_runtime_requires_ordered_target_transition_before_next_producer():
    runtime = CheckpointRelalgRuntime(
        _connection(),
        mode="atomic",
        atomic_operator_profile="semantic-v2",
        config=RuntimeConfig(
            checkpoint_commit_eligibility_policy=(
                CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V2
            )
        ),
    )
    active_target = "Ground the requested population."
    remaining_target = "Construct and verify the exact answer relation."
    bootstrap = runtime.apply(
        "commit_checkpoint",
        {
            "progress_summary": [
                "Initial decomposition only; no database evidence exists yet."
            ],
            "remaining_uncertainties": ["The requested population is unresolved."],
            "next_targets": [active_target, remaining_target],
        },
    )
    assert bootstrap["status"] == "success"
    for country in ("UK", "US"):
        produced = runtime.apply(
            "filter_rows",
            {
                "table": "customers",
                "conditions": {
                    "op": "=",
                    "left": {"column": "country"},
                    "right": {"value": country},
                },
            },
        )
        assert produced["status"] == "success"

    before = runtime.state.logical_hash()
    blocked = runtime.apply(
        "group_aggregate",
        {
            "table": "customers",
            "group_by": [],
            "metrics": [{"op": "count", "column": "*", "as": "n"}],
        },
    )
    assert blocked["status"] == "error"
    assert blocked["error"]["type"] == "state_validation_error"
    assert blocked["error"]["code"] == "checkpoint_target_transition_required"
    assert runtime.state.logical_hash() == before

    advanced = runtime.apply(
        "commit_checkpoint",
        {
            "progress_summary": ["The active population target is grounded."],
            "remaining_uncertainties": ["The exact answer relation remains."],
            "next_targets": [remaining_target],
        },
    )
    assert advanced["status"] == "success"
    assert runtime.state.current_targets == (remaining_target,)
    final_status = runtime.checkpoints.ordered_target_status()
    assert final_status["remaining_targets"] == []
    assert final_status["target_transition_required"] is False


def test_checkpoint_restore_deactivates_later_artifact_and_keeps_monotonic_ids():
    runtime = CheckpointRelalgRuntime(_connection(), mode="atomic")
    runtime.apply("describe_table", {"tables": ["customers"]})
    commit = runtime.apply(
        "commit_checkpoint",
        {
            "progress_summary": ["Confirmed the customer schema."],
            "remaining_uncertainties": ["The answer population is not fixed."],
            "next_targets": ["Construct the requested population."],
        },
    )
    later = runtime.apply(
        "filter_rows",
        {
            "table": "customers",
            "conditions": {
                "op": "=",
                "left": {"column": "country"},
                "right": {"value": "UK"},
            },
        },
    )
    handle = later["artifact"]["table"]
    restore = runtime.apply(
        "restore_checkpoint",
        {
            "checkpoint_id": "root",
            "reason": (
                "The population after the checkpoint used an unsupported assumption; "
                "this contradicts the schema-only milestone, so return to root."
            ),
            "next_targets": ["Rebuild from the catalog."],
        },
    )
    assert restore["status"] == "success"
    rejected = runtime.apply("distinct", {"table": handle})
    assert rejected["error"]["code"] == "inactive_handle"
    assert int(rejected["step_id"].split("_")[1]) > int(commit["step_id"].split("_")[1])


def test_restore_can_roll_current_phase_back_to_its_starting_checkpoint():
    runtime = CheckpointRelalgRuntime(_connection(), mode="direct")
    committed = runtime.apply(
        "commit_checkpoint",
        {
            "progress_summary": ["The source catalog is the stable phase boundary."],
            "remaining_uncertainties": ["The exact population remains unresolved."],
            "next_targets": ["Test the candidate population."],
        },
    )
    checkpoint_id = committed["checkpoint_id"]
    candidate = runtime.apply(
        "execute_sql", {"sql": "SELECT customer_id FROM customers WHERE country = 'UK'"}
    )
    handle = candidate["artifact"]["table"]
    restored = runtime.apply(
        "restore_checkpoint",
        {
            "checkpoint_id": checkpoint_id,
            "reason": (
                "The candidate population introduced in this phase contradicts the "
                "phase-start uncertainty, so discard it and retry from that snapshot."
            ),
            "next_targets": ["Construct a different candidate population."],
        },
    )
    assert restored["status"] == "success"
    assert restored["restored_from"] == checkpoint_id
    rejected = runtime.apply("answer", {"table": handle})
    assert rejected["error"]["code"] == "inactive_handle"


def test_checkpoint_clears_tool_transcripts_but_keeps_grounded_producer_cards():
    runtime = CheckpointRelalgRuntime(_connection(), mode="direct")
    runtime.apply("describe_table", {"tables": ["customers"]})
    produced = runtime.apply("execute_sql", {"sql": "SELECT customer_id FROM customers"})
    assert produced["artifact"]["table"] == "sql_001"
    committed = runtime.apply(
        "commit_checkpoint",
        {
            "progress_summary": ["Constructed the customer identifier relation."],
            "remaining_uncertainties": ["The final requested shape is not confirmed."],
            "next_targets": ["Confirm the required output shape."],
        },
    )
    checkpoint_id = committed["checkpoint_id"]
    snapshot_hash = runtime.checkpoints.get(checkpoint_id).snapshot.environment_state_hash
    assert runtime.state.logical_hash() == snapshot_hash

    context = runtime.render_context("Return customer identifiers.")
    assert "step_001 describe_table" not in context
    assert "step_002 execute_sql -> sql_001" in context
    assert "step_003 commit_checkpoint" not in context
    assert "next_phase_started" not in context

    runtime.apply("execute_sql", {"sql": "SELECT COUNT(*) AS n FROM customers"})
    runtime.apply(
        "commit_checkpoint",
        {
            "progress_summary": ["Tested an aggregate alternative."],
            "remaining_uncertainties": ["It conflicts with the requested row shape."],
            "next_targets": ["Return to the identifier relation."],
        },
    )
    restored = runtime.apply(
        "restore_checkpoint",
        {
            "checkpoint_id": checkpoint_id,
            "reason": (
                "The later aggregate collapsed the requested identifier rows, contradicting "
                "the earlier row-grain conclusion, so restore that milestone."
            ),
            "next_targets": ["Use the restored identifier relation."],
        },
    )
    assert restored["status"] == "success"
    assert runtime.state.logical_hash() == snapshot_hash
    assert runtime.state.usable_step_ids == {"step_002"}


def test_mode_surface_and_resource_limit_are_state_preserving():
    runtime = CheckpointRelalgRuntime(
        _connection(), mode="direct", config=RuntimeConfig(max_primitive_calls=1)
    )
    runtime.apply("describe_table", {"tables": ["customers"]})
    before = runtime.state.logical_hash()
    exhausted = runtime.apply("describe_table", {"tables": ["customers"]})
    assert exhausted["error"]["type"] == "resource_limit_error"
    assert runtime.state.logical_hash() == before
    assert runtime.done


def test_rejected_native_turn_is_bounded_by_the_same_primitive_budget():
    runtime = CheckpointRelalgRuntime(
        _connection(), mode="direct", config=RuntimeConfig(max_primitive_calls=1)
    )
    first = runtime.reject_native_turn(
        code="invalid_tool_call_count",
        message="expected exactly one call",
        details={"call_count": 2},
    )
    assert first["error"]["type"] == "protocol_error"
    before = runtime.state.logical_hash()
    exhausted = runtime.reject_native_turn(
        code="invalid_tool_call_count",
        message="expected exactly one call",
        details={"call_count": 2},
    )
    assert exhausted["error"]["type"] == "resource_limit_error"
    assert exhausted["error"]["code"] == "primitive_call_limit_reached"
    assert runtime.state.logical_hash() == before
    assert runtime.done


def test_catalog_resolves_implicit_foreign_key_target_to_parent_primary_key():
    connection = sqlite3.connect(":memory:")
    connection.executescript(
        "CREATE TABLE parent(id INTEGER PRIMARY KEY);"
        "CREATE TABLE child(parent_id INTEGER REFERENCES parent);"
    )
    catalog = load_source_catalog(connection)
    foreign_key = catalog["child"].foreign_keys[0]
    assert foreign_key.columns == ("parent_id",)
    assert foreign_key.ref_table == "parent"
    assert foreign_key.ref_columns == ("id",)


def test_perception_text_collation_is_stable_before_and_after_projection():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE labels(value TEXT COLLATE NOCASE)")
    connection.executemany("INSERT INTO labels VALUES (?)", [("a",), ("A",)])
    runtime = CheckpointRelalgRuntime(connection, mode="atomic")

    source = runtime.apply(
        "inspect_column", {"table": "labels", "column": "value", "top_k": 10}
    )
    projected = runtime.apply(
        "project",
        {
            "table": "labels",
            "outputs": [{"expression": {"column": "value"}}],
        },
    )
    artifact = runtime.apply(
        "inspect_column",
        {
            "table": projected["artifact"]["table"],
            "column": "value",
            "top_k": 10,
        },
    )
    assert source["distinct_count"] == artifact["distinct_count"] == 2
    assert source["values"] == artifact["values"] == ["A", "a"]
    assert source["frequencies"] == artifact["frequencies"] == [1, 1]

    source_rows = runtime.apply(
        "read_rows",
        {
            "table": "labels",
            "columns": ["value"],
            "order_by": [
                {"column": "value", "direction": "asc", "nulls": "last"}
            ],
        },
    )
    artifact_rows = runtime.apply(
        "read_rows",
        {
            "table": projected["artifact"]["table"],
            "columns": ["value"],
            "order_by": [
                {"column": "value", "direction": "asc", "nulls": "last"}
            ],
        },
    )
    assert source_rows["rows"] == artifact_rows["rows"] == [["A"], ["a"]]


def test_read_rows_rejects_explicit_blob_ordering_but_allows_unordered_read():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE blobs(payload BLOB)")
    connection.executemany("INSERT INTO blobs VALUES (?)", [(b"b",), (b"a",)])
    runtime = CheckpointRelalgRuntime(connection, mode="atomic")
    before = runtime.state.logical_hash()
    rejected = runtime.apply(
        "read_rows",
        {
            "table": "blobs",
            "order_by": [
                {"column": "payload", "direction": "asc", "nulls": "last"}
            ],
        },
    )
    assert rejected["error"]["code"] == "type_mismatch"
    assert runtime.state.logical_hash() == before
    allowed = runtime.apply("read_rows", {"table": "blobs", "limit": 2})
    assert allowed["status"] == "success"
    assert allowed["returned_count"] == 2


def test_perception_uses_shared_timeout_and_scalar_error_mapping():
    runtime = CheckpointRelalgRuntime(_connection(), mode="atomic")
    before = runtime.state.logical_hash()
    divided = runtime.apply(
        "read_rows",
        {
            "table": "customers",
            "conditions": {
                "op": ">",
                "left": {
                    "op": "divide",
                    "args": [{"column": "amount"}, {"value": 0}],
                },
                "right": {"value": 1},
            },
        },
    )
    assert divided["error"]["code"] == "divide_by_zero"
    assert runtime.state.logical_hash() == before

    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE big(x INTEGER)")
    connection.executemany("INSERT INTO big VALUES (?)", [(index,) for index in range(5000)])
    timed = CheckpointRelalgRuntime(
        connection,
        mode="atomic",
        config=RuntimeConfig(sql_timeout_seconds=1e-12),
    )
    before = timed.state.logical_hash()
    result = timed.apply("inspect_column", {"table": "big", "column": "x"})
    assert result["error"]["code"] == "sql_timeout"
    assert timed.state.logical_hash() == before
    assert not timed.state.active_observation_ids


def test_non_finite_perception_and_answer_are_state_preserving_errors():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE weird(x REAL)")
    connection.execute("INSERT INTO weird VALUES (?)", (float("inf"),))
    runtime = CheckpointRelalgRuntime(connection, mode="direct")
    before = runtime.state.logical_hash()
    observed = runtime.apply("inspect_column", {"table": "weird", "column": "x"})
    assert observed["status"] == "error"
    assert runtime.state.logical_hash() == before
    assert not runtime.state.active_observation_ids
    answered = runtime.apply("answer", {"table": "weird"})
    assert answered["error"]["code"] == "non_finite_result"
    assert runtime.state.logical_hash() == before
    assert not runtime.done
