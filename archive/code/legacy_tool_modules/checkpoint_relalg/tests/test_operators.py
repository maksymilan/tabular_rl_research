from __future__ import annotations

import sqlite3

import pytest

from src.tool_modules.checkpoint_relalg.environment_state import EnvironmentState
from src.tool_modules.checkpoint_relalg.errors import CheckpointRelalgError
from src.tool_modules.checkpoint_relalg.executors import (
    SQLiteRelationalExecutor,
    artifact_backing_name,
)
from src.tool_modules.checkpoint_relalg.expression import RelAlgValidationError
from src.tool_modules.checkpoint_relalg.operator_registry import OPERATOR_SPECS
from src.tool_modules.checkpoint_relalg.relation_artifact import Column, SourceRelation


def _source(name: str, columns: list[tuple[str, str]], rows: list[tuple[object, ...]]):
    return SourceRelation(
        name=name,
        columns=tuple(Column(column_name, kind) for column_name, kind in columns),
        row_count=len(rows),
    )


@pytest.fixture()
def engine():
    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE people(id INTEGER, dept TEXT, amount REAL, joined DATE, note TEXT)"
    )
    people_rows = [
        (1, "a", 10.0, "2024-01-02", " Alpha "),
        (1, "a", 10.0, "2024-01-02", " Alpha "),
        (2, "a", None, "2024-01-03", None),
        (3, "b", 7.5, None, "beta"),
        (4, None, 7.5, "2023-12-31", "gamma"),
    ]
    connection.executemany("INSERT INTO people VALUES (?, ?, ?, ?, ?)", people_rows)
    connection.execute("CREATE TABLE depts(dept TEXT, label TEXT)")
    dept_rows = [("a", "A-one"), ("a", "A-two"), ("c", "C")]
    connection.executemany("INSERT INTO depts VALUES (?, ?)", dept_rows)
    sources = [
        _source(
            "people",
            [
                ("id", "INTEGER"),
                ("dept", "TEXT"),
                ("amount", "REAL"),
                ("joined", "DATE"),
                ("note", "TEXT"),
            ],
            people_rows,
        ),
        _source("depts", [("dept", "TEXT"), ("label", "TEXT")], dept_rows),
    ]
    state = EnvironmentState(sources)
    return connection, state, SQLiteRelationalExecutor(connection, state)


def _rows(connection: sqlite3.Connection, artifact) -> list[tuple[object, ...]]:
    names = ", ".join('"' + column.name.replace('"', '""') + '"' for column in artifact.columns)
    order = ' ORDER BY "__relalg_ordinal"' if artifact.ordered_by else ""
    backing = artifact_backing_name(artifact.table)
    return connection.execute(
        f'SELECT {names} FROM temp."{backing}"{order}'
    ).fetchall()


def test_registry_is_the_frozen_nine_operator_semantic_source():
    assert tuple(OPERATOR_SPECS) == (
        "filter_rows",
        "project",
        "join",
        "aggregate",
        "distinct",
        "set_operation",
        "sort",
        "limit",
        "add_rank",
    )
    assert OPERATOR_SPECS["filter_rows"].ordering_effect == "preserve"
    assert OPERATOR_SPECS["sort"].ordering_effect == "establish"
    assert OPERATOR_SPECS["join"].input_arity == 2


def test_filter_sql_three_valued_logic_duplicates_and_exact_columns(engine):
    connection, state, executor = engine
    artifact = executor.execute(
        "filter_rows",
        {
            "table": "people",
            "conditions": {
                "op": "and",
                "args": [
                    {
                        "op": ">",
                        "left": {"column": "amount"},
                        "right": {"value": 7.5},
                    },
                    {
                        "op": "not_in",
                        "value": {"column": "dept"},
                        "values": [{"value": "b"}],
                    },
                ],
            },
        },
    )
    assert _rows(connection, artifact) == [
        (1, "a", 10.0, "2024-01-02", " Alpha "),
        (1, "a", 10.0, "2024-01-02", " Alpha "),
    ]
    assert artifact.derivation["schema"] == "relation-derivation-v2"
    assert artifact.derivation["operator"] == "filter_rows"

    before = state.logical_hash()
    with pytest.raises(RelAlgValidationError) as captured:
        executor.execute(
            "filter_rows",
            {
                "table": "people",
                "conditions": {
                    "op": "=",
                    "left": {"column": "people.id"},
                    "right": {"value": 1},
                },
            },
        )
    assert captured.value.code == "unknown_column"
    assert captured.value.error_type == "state_validation_error"
    assert state.logical_hash() == before


def test_project_typed_expression_null_date_case_and_no_text_numeric_coercion(engine):
    connection, state, executor = engine
    artifact = executor.execute(
        "project",
        {
            "table": "people",
            "outputs": [
                {"expression": {"column": "id"}},
                {
                    "expression": {
                        "op": "divide",
                        "args": [{"column": "amount"}, {"value": 2}],
                    },
                    "as": "half",
                },
                {
                    "expression": {
                        "op": "case_when",
                        "branches": [
                            {
                                "when": {"op": "is_null", "value": {"column": "amount"}},
                                "then": {"value": "missing"},
                            }
                        ],
                        "else": {"value": "known"},
                    },
                    "as": "quality",
                },
                {
                    "expression": {"op": "extract_year", "args": [{"column": "joined"}]},
                    "as": "year",
                },
                {
                    "expression": {
                        "op": "date_diff_days",
                        "args": [
                            {
                                "op": "cast",
                                "args": [{"value": "2024-01-01T00:00:00"}],
                                "to": "DATETIME",
                            },
                            {
                                "op": "cast",
                                "args": [{"value": "2024-01-02T12:00:00"}],
                                "to": "DATETIME",
                            },
                        ],
                    },
                    "as": "elapsed_days",
                },
            ],
        },
    )
    assert [column.canonical_type for column in artifact.columns] == [
        "INTEGER",
        "REAL",
        "TEXT",
        "INTEGER",
        "REAL",
    ]
    assert _rows(connection, artifact)[2] == (2, None, "missing", 2024, 1.5)

    before = state.logical_hash()
    with pytest.raises(RelAlgValidationError) as captured:
        executor.execute(
            "project",
            {
                "table": "people",
                "outputs": [
                    {
                        "expression": {
                            "op": "add",
                            "args": [{"column": "dept"}, {"value": 1}],
                        },
                        "as": "bad",
                    }
                ],
            },
        )
    assert captured.value.code == "type_mismatch"
    assert state.logical_hash() == before

    with pytest.raises(CheckpointRelalgError) as divided:
        executor.execute(
            "project",
            {
                "table": "people",
                "outputs": [
                    {
                        "expression": {
                            "op": "divide",
                            "args": [{"column": "amount"}, {"value": 0}],
                        },
                        "as": "bad_division",
                    }
                ],
            },
        )
    assert divided.value.code == "divide_by_zero"
    assert state.logical_hash() == before


def test_join_multiplicity_left_semi_anti_and_recursive_prefix(engine):
    connection, _, executor = engine
    inner = executor.execute(
        "join",
        {
            "left": "people",
            "right": "depts",
            "left_role": "person",
            "right_role": "department",
            "type": "inner",
            "on": [{"left_column": "dept", "op": "=", "right_column": "dept"}],
        },
    )
    assert inner.row_count == 6  # three bag rows with dept=a x two matching department rows
    assert inner.columns[0].name == "person.id"
    assert inner.columns[-1].name == "department.label"

    nested = executor.execute(
        "join",
        {
            "left": inner.table,
            "right": "depts",
            "left_role": "prior",
            "right_role": "again",
            "type": "cross",
            "on": [],
        },
    )
    assert nested.columns[0].name == "prior.person.id"

    semi = executor.execute(
        "join",
        {
            "left": "people",
            "right": "depts",
            "type": "semi",
            "on": [{"left_column": "dept", "op": "=", "right_column": "dept"}],
        },
    )
    assert semi.row_count == 3
    assert [column.name for column in semi.columns] == [column.name for column in state_source(executor, "people").columns]
    anti = executor.execute(
        "join",
        {
            "left": "people",
            "right": "depts",
            "type": "anti",
            "on": [{"left_column": "dept", "op": "=", "right_column": "dept"}],
        },
    )
    assert [row[0] for row in _rows(connection, anti)] == [3, 4]


def state_source(executor: SQLiteRelationalExecutor, name: str):
    return executor.state.get_relation(name)


def test_aggregate_empty_null_and_count_semantics(engine):
    connection, _, executor = engine
    empty = executor.execute(
        "filter_rows",
        {
            "table": "people",
            "conditions": {"op": "=", "left": {"column": "id"}, "right": {"value": 999}},
        },
    )
    global_aggregate = executor.execute(
        "aggregate",
        {
            "table": empty.table,
            "group_by": [],
            "metrics": [
                {"op": "count", "column": "*", "as": "n"},
                {"op": "sum", "column": "amount", "as": "total"},
                {"op": "avg", "column": "amount", "as": "mean"},
            ],
        },
    )
    assert _rows(connection, global_aggregate) == [(0, None, None)]
    grouped = executor.execute(
        "aggregate",
        {
            "table": "people",
            "group_by": ["dept"],
            "metrics": [
                {"op": "count", "column": "amount", "as": "non_null"},
                {"op": "count", "column": "id", "distinct": True, "as": "unique_ids"},
            ],
        },
    )
    assert sorted(_rows(connection, grouped), key=lambda row: str(row[0])) == [
        (None, 1, 1),
        ("a", 2, 2),
        ("b", 1, 1),
    ]


def test_distinct_and_set_null_equality_numeric_promotion(engine):
    connection, state, executor = engine
    left = executor.execute(
        "project", {"table": "people", "outputs": [{"expression": {"column": "amount"}}]}
    )
    unique = executor.execute("distinct", {"table": left.table})
    assert set(_rows(connection, unique)) == {(10.0,), (7.5,), (None,)}

    connection.execute("CREATE TABLE integer_values(v INTEGER)")
    int_rows = [(10,), (None,), (10,)]
    connection.executemany("INSERT INTO integer_values VALUES (?)", int_rows)
    state.add_source(_source("integer_values", [("v", "INTEGER")], int_rows))
    union = executor.execute(
        "set_operation", {"left": "integer_values", "right": left.table, "op": "union"}
    )
    assert union.columns[0].name == "v"
    assert union.columns[0].canonical_type == "REAL"
    assert set(_rows(connection, union)) == {(10.0,), (7.5,), (None,)}
    union_all = executor.execute(
        "set_operation", {"left": "integer_values", "right": left.table, "op": "union_all"}
    )
    assert union_all.row_count == len(int_rows) + 5


def test_sort_deterministic_ties_limit_and_order_propagation(engine):
    connection, state, executor = engine
    sorted_artifact = executor.execute(
        "sort",
        {
            "table": "people",
            "keys": [{"column": "amount", "direction": "desc", "nulls": "last"}],
        },
    )
    assert [row[0] for row in _rows(connection, sorted_artifact)] == [1, 1, 3, 4, 2]
    assert [key.column for key in sorted_artifact.ordered_by] == ["amount"]
    limited = executor.execute(
        "limit", {"table": sorted_artifact.table, "count": 2, "offset": 1}
    )
    assert [row[0] for row in _rows(connection, limited)] == [1, 3]
    projected = executor.execute(
        "project",
        {
            "table": limited.table,
            "outputs": [
                {"expression": {"column": "id"}},
                {"expression": {"column": "amount"}},
            ],
        },
    )
    assert projected.ordered_by == limited.ordered_by
    assert _rows(connection, projected) == [(1, 10.0), (3, 7.5)]

    # Ordering is a relation property carried by the hidden ordinal.  It stays
    # valid even when generalized projection removes the visible sort key.
    dropped_key = executor.execute(
        "project",
        {
            "table": sorted_artifact.table,
            "outputs": [{"expression": {"column": "id"}}],
        },
    )
    assert dropped_key.ordered_by[0].column == "amount"
    after_drop_limit = executor.execute(
        "limit", {"table": dropped_key.table, "count": 3}
    )
    assert _rows(connection, after_drop_limit) == [(1,), (1,), (3,)]

    before = state.logical_hash()
    with pytest.raises(RelAlgValidationError) as captured:
        executor.execute("limit", {"table": "people", "count": 1})
    assert captured.value.code == "unordered_limit_input"
    assert state.logical_hash() == before


def test_rank_ties_partition_and_clears_order(engine):
    connection, _, executor = engine
    ranked = executor.execute(
        "add_rank",
        {
            "table": "people",
            "partition_by": ["dept"],
            "order_by": [{"column": "amount", "direction": "desc", "nulls": "last"}],
            "method": "dense_rank",
            "as": "r",
        },
    )
    a_rows = sorted(row for row in _rows(connection, ranked) if row[1] == "a")
    assert [row[-1] for row in a_rows] == [1, 1, 2]
    assert ranked.ordered_by == ()
    assert ranked.columns[-1] == Column("r", "INTEGER")


def test_inactive_unknown_and_execution_failure_preserve_logical_hash(engine):
    connection, state, executor = engine
    artifact = executor.execute("distinct", {"table": "people"})
    state.active_artifact_ids.remove(artifact.table)
    before = state.logical_hash()
    with pytest.raises(CheckpointRelalgError) as inactive:
        executor.execute("distinct", {"table": artifact.table})
    assert inactive.value.code == "inactive_handle"
    assert state.logical_hash() == before
    with pytest.raises(CheckpointRelalgError) as unknown:
        executor.execute("distinct", {"table": "missing"})
    assert unknown.value.code == "unknown_table"
    assert state.logical_hash() == before


def test_direct_sql_readonly_order_full_materialization_and_limits(engine):
    connection, state, executor = engine
    artifact = executor.execute_sql(
        "SELECT id, amount FROM people WHERE amount IS NOT NULL ORDER BY amount DESC, id ASC",
        max_rows=10,
    )
    assert artifact.kind == "sql"
    assert [key.column for key in artifact.ordered_by] == ["amount", "id"]
    assert _rows(connection, artifact) == [(1, 10.0), (1, 10.0), (3, 7.5), (4, 7.5)]
    assert list(artifact.derivation["inputs"]) == ["people"]
    hidden_order = executor.execute_sql(
        "SELECT id FROM people ORDER BY amount * -1 ASC NULLS LAST, id ASC", max_rows=10
    )
    assert hidden_order.ordered_by[0].column == "<opaque-direct-order>"
    assert "__relalg_ordinal" not in str(hidden_order.to_payload())
    assert _rows(connection, hidden_order) == [(1,), (1,), (3,), (4,), (2,)]
    cte = executor.execute_sql(
        "WITH eligible AS (SELECT id, amount FROM people WHERE dept = 'a') "
        "SELECT id, amount FROM eligible ORDER BY id",
        max_rows=10,
    )
    assert _rows(connection, cte) == [(1, 10.0), (1, 10.0), (2, None)]
    assert list(cte.derivation["inputs"]) == ["people"]
    before = state.logical_hash()
    with pytest.raises(CheckpointRelalgError) as readonly:
        executor.execute_sql("DELETE FROM people")
    assert readonly.value.code == "unsafe_sql"
    assert state.logical_hash() == before
    with pytest.raises(CheckpointRelalgError) as too_many:
        executor.execute_sql("SELECT id FROM people", max_rows=2)
    assert too_many.value.code == "result_too_large"
    assert state.logical_hash() == before


def test_hybrid_sql_sees_only_visible_artifact_columns(engine):
    connection, state, executor = engine
    ordered = executor.execute(
        "sort",
        {
            "table": "people",
            "keys": [{"column": "id", "direction": "asc", "nulls": "last"}],
        },
    )
    direct = executor.execute_sql(f'SELECT * FROM "{ordered.table}"')
    assert [column.name for column in direct.columns] == [
        "id",
        "dept",
        "amount",
        "joined",
        "note",
    ]
    assert direct.row_count == ordered.row_count
    before = state.logical_hash()
    with pytest.raises(CheckpointRelalgError):
        executor.execute_sql(
            f'SELECT "__relalg_ordinal" AS leaked FROM "{ordered.table}"'
        )
    assert state.logical_hash() == before


def test_sqlite_identifier_equivalence_cannot_shadow_sources_or_join_columns():
    connection = sqlite3.connect(":memory:")
    connection.execute('CREATE TABLE "Filter_001"(x INTEGER)')
    connection.executemany('INSERT INTO "Filter_001" VALUES (?)', [(1,), (2,)])
    connection.execute("CREATE TABLE rhs(x INTEGER)")
    connection.execute("INSERT INTO rhs VALUES (3)")
    state = EnvironmentState(
        [
            _source("Filter_001", [("x", "INTEGER")], [(1,), (2,)]),
            _source("rhs", [("x", "INTEGER")], [(3,)]),
        ]
    )
    executor = SQLiteRelationalExecutor(connection, state)
    filtered = executor.execute(
        "filter_rows",
        {
            "table": "Filter_001",
            "conditions": {
                "op": "=",
                "left": {"column": "x"},
                "right": {"value": 1},
            },
        },
    )
    assert filtered.table == "filter_002"
    assert connection.execute('SELECT x FROM main."Filter_001" ORDER BY x').fetchall() == [
        (1,),
        (2,),
    ]
    before = state.logical_hash()
    with pytest.raises(CheckpointRelalgError) as roles:
        executor.execute(
            "join",
            {
                "left": "Filter_001",
                "right": "rhs",
                "left_role": "a",
                "right_role": "A",
                "type": "cross",
                "on": [],
            },
        )
    assert roles.value.code == "invalid_join_roles"
    assert state.logical_hash() == before


def test_atomic_timeout_and_non_finite_results_roll_back_before_publish():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE big(x REAL)")
    connection.executemany("INSERT INTO big VALUES (?)", [(float(index),) for index in range(5000)])
    state = EnvironmentState(
        [_source("big", [("x", "REAL")], [(float(index),) for index in range(5000)])]
    )
    executor = SQLiteRelationalExecutor(connection, state, timeout_seconds=1e-12)
    before = state.logical_hash()
    with pytest.raises(CheckpointRelalgError) as timed_out:
        executor.execute(
            "sort",
            {
                "table": "big",
                "keys": [{"column": "x", "direction": "desc", "nulls": "last"}],
            },
        )
    assert timed_out.value.code == "sql_timeout"
    assert state.logical_hash() == before
    assert not state.active_artifact_ids

    finite_connection = sqlite3.connect(":memory:")
    finite_connection.execute("CREATE TABLE values_with_inf(x REAL)")
    finite_connection.execute("INSERT INTO values_with_inf VALUES (?)", (float("inf"),))
    finite_state = EnvironmentState(
        [_source("values_with_inf", [("x", "REAL")], [(float("inf"),)])]
    )
    finite_executor = SQLiteRelationalExecutor(finite_connection, finite_state)
    before = finite_state.logical_hash()
    with pytest.raises(CheckpointRelalgError) as non_finite:
        finite_executor.execute("distinct", {"table": "values_with_inf"})
    assert non_finite.value.code == "non_finite_result"
    assert finite_state.logical_hash() == before
    assert not finite_state.active_artifact_ids


def test_atomic_rejects_dynamic_values_that_violate_declared_canonical_types():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE dirty(x INTEGER)")
    connection.executemany("INSERT INTO dirty VALUES (?)", [("5",), ("abc",)])
    # The metadata is intentionally what a real SQLite catalog declares; the
    # second row remains TEXT storage despite INTEGER affinity.
    state = EnvironmentState(
        [_source("dirty", [("x", "INTEGER")], [("5",), ("abc",)])]
    )
    executor = SQLiteRelationalExecutor(connection, state)
    before = state.logical_hash()

    with pytest.raises(CheckpointRelalgError) as arithmetic:
        executor.execute(
            "project",
            {
                "table": "dirty",
                "outputs": [
                    {
                        "expression": {
                            "op": "add",
                            "args": [{"column": "x"}, {"value": 1}],
                        },
                        "as": "y",
                    }
                ],
            },
        )
    assert arithmetic.value.code == "canonical_type_mismatch"
    assert state.logical_hash() == before
    assert not state.active_artifact_ids

    with pytest.raises(CheckpointRelalgError) as aggregate:
        executor.execute(
            "aggregate",
            {
                "table": "dirty",
                "group_by": [],
                "metrics": [{"op": "sum", "column": "x", "as": "total"}],
            },
        )
    assert aggregate.value.code == "canonical_type_mismatch"
    assert state.logical_hash() == before


def test_atomic_rejects_integer_overflow_that_changes_materialized_storage_type():
    connection = sqlite3.connect(":memory:")
    maximum = 9_223_372_036_854_775_807
    connection.execute("CREATE TABLE ints(x INTEGER)")
    connection.execute("INSERT INTO ints VALUES (?)", (maximum,))
    state = EnvironmentState([_source("ints", [("x", "INTEGER")], [(maximum,)])])
    executor = SQLiteRelationalExecutor(connection, state)
    before = state.logical_hash()

    with pytest.raises(CheckpointRelalgError) as overflow:
        executor.execute(
            "project",
            {
                "table": "ints",
                "outputs": [
                    {
                        "expression": {
                            "op": "add",
                            "args": [{"column": "x"}, {"value": 1}],
                        },
                        "as": "y",
                    }
                ],
            },
        )
    assert overflow.value.code == "canonical_type_mismatch"
    assert state.logical_hash() == before
    assert not state.active_artifact_ids


def test_direct_sql_rejects_attached_and_temp_relations_that_shadow_sources():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE secret(x INTEGER)")
    connection.execute("INSERT INTO secret VALUES (42)")
    state = EnvironmentState([_source("secret", [("x", "INTEGER")], [(42,)])])
    executor = SQLiteRelationalExecutor(connection, state)

    connection.execute("ATTACH DATABASE ':memory:' AS other")
    connection.execute("CREATE TABLE other.secret(x INTEGER)")
    connection.execute("INSERT INTO other.secret VALUES (777)")
    before = state.logical_hash()
    with pytest.raises(CheckpointRelalgError) as attached:
        executor.execute_sql("SELECT x FROM other.secret")
    assert attached.value.code == "unsafe_sql"
    assert state.logical_hash() == before

    connection.execute("CREATE TEMP TABLE secret(x INTEGER)")
    connection.execute("INSERT INTO temp.secret VALUES (999)")
    with pytest.raises(CheckpointRelalgError) as shadowed:
        executor.execute_sql("SELECT x FROM secret")
    assert shadowed.value.code == "unsafe_sql"
    assert state.logical_hash() == before

    allowed = executor.execute_sql("SELECT x FROM main.secret")
    backing = artifact_backing_name(allowed.table)
    assert connection.execute(
        f'SELECT x FROM temp."{backing}"'
    ).fetchall() == [(42,)]


@pytest.mark.parametrize(
    "name", ["__relalg_ordinal", "__RELALG_ORDINAL", "__checkpoint_relalg_data_x"]
)
def test_direct_sql_rejects_harness_reserved_output_columns(name):
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE t(x INTEGER)")
    connection.execute("INSERT INTO t VALUES (7)")
    state = EnvironmentState([_source("t", [("x", "INTEGER")], [(7,)])])
    executor = SQLiteRelationalExecutor(connection, state)
    before = state.logical_hash()
    with pytest.raises(CheckpointRelalgError) as reserved:
        executor.execute_sql(f'SELECT x AS "{name}" FROM main.t')
    assert reserved.value.code == "reserved_output_column"
    assert state.logical_hash() == before
    assert not state.active_artifact_ids


def test_atomic_executor_rejects_harness_reserved_output_columns_without_state_change():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE t(x INTEGER)")
    connection.execute("INSERT INTO t VALUES (7)")
    state = EnvironmentState([_source("t", [("x", "INTEGER")], [(7,)])])
    executor = SQLiteRelationalExecutor(connection, state)
    before = state.logical_hash()

    with pytest.raises(CheckpointRelalgError) as reserved:
        executor.execute(
            "project",
            {
                "table": "t",
                "outputs": [
                    {"expression": {"value": 1}, "as": "__relalg_ordinal"}
                ],
            },
        )

    assert reserved.value.code == "reserved_output_column"
    assert state.logical_hash() == before
    assert not state.active_artifact_ids


def test_atomic_text_semantics_ignore_source_ddl_collation_across_artifacts():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE t(x TEXT COLLATE NOCASE)")
    connection.executemany("INSERT INTO t VALUES (?)", [("A",), ("a",)])
    state = EnvironmentState([_source("t", [("x", "TEXT")], [("A",), ("a",)])])
    executor = SQLiteRelationalExecutor(connection, state)

    condition = {
        "op": "=",
        "left": {"column": "x"},
        "right": {"value": "a"},
    }
    source_filtered = executor.execute(
        "filter_rows", {"table": "t", "conditions": condition}
    )
    projected = executor.execute(
        "project",
        {"table": "t", "outputs": [{"expression": {"column": "x"}}]},
    )
    artifact_filtered = executor.execute(
        "filter_rows", {"table": projected.table, "conditions": condition}
    )
    assert _rows(connection, source_filtered) == [("a",)]
    assert _rows(connection, artifact_filtered) == [("a",)]

    source_distinct = executor.execute("distinct", {"table": "t"})
    artifact_distinct = executor.execute("distinct", {"table": projected.table})
    assert sorted(_rows(connection, source_distinct)) == [("A",), ("a",)]
    assert sorted(_rows(connection, artifact_distinct)) == [("A",), ("a",)]


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        (
            "sort",
            {
                "table": "blobs",
                "keys": [{"column": "payload", "direction": "asc", "nulls": "last"}],
            },
        ),
        (
            "add_rank",
            {
                "table": "blobs",
                "partition_by": [],
                "order_by": [
                    {"column": "payload", "direction": "asc", "nulls": "last"}
                ],
                "method": "row_number",
                "as": "rank_id",
            },
        ),
    ],
)
def test_atomic_blob_columns_cannot_be_order_keys(tool, arguments):
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE blobs(payload BLOB)")
    connection.executemany("INSERT INTO blobs VALUES (?)", [(b"b",), (b"a",)])
    state = EnvironmentState(
        [_source("blobs", [("payload", "BLOB")], [(b"b",), (b"a",)])]
    )
    executor = SQLiteRelationalExecutor(connection, state)
    before = state.logical_hash()
    with pytest.raises(CheckpointRelalgError) as rejected:
        executor.execute(tool, arguments)
    assert rejected.value.code == "type_mismatch"
    assert state.logical_hash() == before
    assert not state.active_artifact_ids


def test_atomic_row_limit_rolls_back_before_publish_and_direct_empty_keeps_declared_type(engine):
    connection, state, _ = engine
    executor = SQLiteRelationalExecutor(connection, state, max_rows=2)
    before = state.logical_hash()
    with pytest.raises(CheckpointRelalgError) as too_large:
        executor.execute("distinct", {"table": "people"})
    assert too_large.value.error_type == "resource_limit_error"
    assert too_large.value.code == "result_too_large"
    assert state.logical_hash() == before
    assert not state.active_artifact_ids

    empty = executor.execute_sql(
        "SELECT id, joined FROM people WHERE 0 ORDER BY id", max_rows=2
    )
    assert empty.row_count == 0
    assert [column.canonical_type for column in empty.columns] == ["INTEGER", "DATE"]
    assert empty.ordered_by[0].column == "id"

    before = state.logical_hash()
    with pytest.raises(CheckpointRelalgError) as multiple:
        executor.execute_sql("SELECT 1 AS x; SELECT 2 AS x")
    assert multiple.value.code == "multiple_sql_statements"
    assert state.logical_hash() == before


def test_replay_determinism_and_sql_equivalence():
    def run_once():
        connection = sqlite3.connect(":memory:")
        connection.execute("CREATE TABLE t(k INTEGER, v REAL)")
        rows = [(2, 1.0), (1, None), (1, 3.0), (1, 3.0)]
        connection.executemany("INSERT INTO t VALUES (?, ?)", rows)
        state = EnvironmentState([_source("t", [("k", "INTEGER"), ("v", "REAL")], rows)])
        executor = SQLiteRelationalExecutor(connection, state)
        filtered = executor.execute(
            "filter_rows",
            {
                "table": "t",
                "conditions": {"op": ">=", "left": {"column": "v"}, "right": {"value": 1}},
            },
        )
        sorted_artifact = executor.execute(
            "sort",
            {
                "table": filtered.table,
                "keys": [
                    {"column": "k", "direction": "asc", "nulls": "last"},
                    {"column": "v", "direction": "desc", "nulls": "last"},
                ],
            },
        )
        result = _rows(connection, sorted_artifact)
        equivalent = connection.execute(
            "SELECT k, v FROM t WHERE v >= 1 ORDER BY k ASC, v DESC"
        ).fetchall()
        return result, equivalent, sorted_artifact.to_payload()

    first = run_once()
    second = run_once()
    assert first[0] == first[1]
    assert first == second
