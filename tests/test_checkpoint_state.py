from __future__ import annotations

import pytest

from src.tool_modules.checkpoint_relalg.checkpoint_store import (
    MAX_CHECKPOINTS,
    CheckpointStore,
)
from src.tool_modules.checkpoint_relalg.environment_renderer import EnvironmentRenderer
from src.tool_modules.checkpoint_relalg.environment_state import (
    EnvironmentState,
    Observation,
    StateError,
    StepRecord,
)
from src.tool_modules.checkpoint_relalg.relation_artifact import (
    Column,
    ForeignKey,
    OrderingKey,
    RelationArtifact,
    SourceRelation,
)


def catalog() -> tuple[SourceRelation, ...]:
    customers = SourceRelation(
        "customers",
        (Column("customer_id", "INTEGER"), Column("name", "TEXT")),
        3,
        primary_key=("customer_id",),
    )
    orders = SourceRelation(
        "orders",
        (Column("order_id", "INTEGER"), Column("customer_id", "INTEGER")),
        10,
        primary_key=("order_id",),
        foreign_keys=(ForeignKey(("customer_id",), "customers", ("customer_id",)),),
    )
    return customers, orders


def artifact(table: str, *, input_table: str = "customers", row_count: int = 2) -> RelationArtifact:
    return RelationArtifact(
        table=table,
        kind="filter",
        columns=(Column("customer_id", "INTEGER"), Column("name", "TEXT")),
        row_count=row_count,
        ordered_by=(OrderingKey("customer_id"),),
        derivation={
            "schema": "relation-derivation-v2",
            "operator": "filter_rows",
            "inputs": [input_table],
        },
    )


def add_fact_bundle(state: EnvironmentState, table: str) -> None:
    state.add_artifact(artifact(table))
    observation_id = state.allocate_observation_id()
    state.add_observation(
        Observation(observation_id, "read_rows", table, {"rows": [[1, "Ada"]]})
    )
    state.add_step(
        StepRecord(
            state.allocate_step_id(),
            "filter_rows",
            "success",
            {"table": table},
            phase_id=state.phase_id,
            checkpoint_id=state.checkpoint_id,
            produced_artifact_ids=(table,),
        )
    )


def test_bootstrap_supports_zero_checkpoint_and_fixed_render_order() -> None:
    state = EnvironmentState(catalog())
    store = CheckpointStore(state)
    text = EnvironmentRenderer().render("Who?", "none", state, store)

    assert store.checkpoint_count == 0
    assert store.active_checkpoint_path == ("root",)
    assert state.current_targets == ()
    headings = [
        "QUESTION",
        "EXTERNAL KNOWLEDGE",
        "CURRENT PHASE TARGETS",
        "CHECKPOINT HISTORY",
        "CURRENT ENVIRONMENT STATE",
    ]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "LAST ERROR" not in text


def test_opening_catalog_hides_undiscovered_columns_and_storeless_render_is_pure() -> None:
    state = EnvironmentState(catalog())
    before = (state.checkpoint_id, state.phase_id, state.logical_hash())
    text = EnvironmentRenderer().render("Who?", "none", state)

    assert "CATALOG\ncustomers[3]\norders[10] FK->customers" in text
    assert "customer_id INTEGER" not in text
    assert (state.checkpoint_id, state.phase_id, state.logical_hash()) == before

    store = CheckpointStore(state)
    store.commit(["Completed bootstrap."], [], ["Inspect customers."])
    with pytest.raises(ValueError, match="checkpoint store is required"):
        EnvironmentRenderer().render("Who?", "none", state)


def test_commit_chain_captures_hash_and_advances_phase() -> None:
    state = EnvironmentState(catalog())
    store = CheckpointStore(state)
    state.discover_schema("customers")
    add_fact_bundle(state, state.allocate_artifact_handle("filter"))

    first = store.commit(["Found the customer population."], [], ["Compute totals."])
    assert first.parent_id == "root"
    assert first.snapshot.environment_state_hash == state.logical_hash()
    assert state.current_targets == ("Compute totals.",)
    assert state.phase_id == "phase_001"

    second = store.commit(
        ["Fixed one row per customer."],
        ["Totals remain unknown."],
        ["Aggregate order values."],
    )
    assert second.parent_id == first.checkpoint_id
    assert store.active_checkpoint_path == ("root", first.checkpoint_id, second.checkpoint_id)
    assert all(store.get(item).status == "active_path" for item in store.active_checkpoint_path)


def test_restore_root_is_exact_and_creates_recovery_checkpoint() -> None:
    state = EnvironmentState(catalog())
    store = CheckpointStore(state)
    state.discover_schema("customers")
    handle = state.allocate_artifact_handle("filter")
    add_fact_bundle(state, handle)
    committed = store.commit(["Built a candidate relation."], [], ["Verify it."])

    restored = store.restore(
        "root",
        "The later candidate used the wrong population, contradicting the bootstrap scope.",
        ["Find the intended population."],
    )

    assert restored.created_by == "restore_checkpoint"
    assert restored.parent_id == "root"
    assert restored.restored_from == "root"
    assert restored.checkpoint_id != committed.checkpoint_id
    assert restored.abandoned_checkpoints == (committed.checkpoint_id,)
    assert state.discovered_schema_ids == set()
    assert state.active_artifact_ids == set()
    assert state.active_observation_ids == set()
    assert state.usable_step_ids == set()
    assert state.current_targets == ("Find the intended population.",)
    assert store.active_checkpoint_path == ("root", restored.checkpoint_id)


def test_abandoned_artifact_is_known_but_cannot_be_resolved_or_rendered() -> None:
    state = EnvironmentState(catalog())
    store = CheckpointStore(state)
    first = store.commit(["Bootstrap complete."], [], ["Build candidate."])
    handle = state.allocate_artifact_handle("filter")
    state.add_artifact(artifact(handle))
    second = store.commit(["Built a candidate."], [], ["Check shape."])
    assert state.get_relation(handle).table == handle

    store.restore(
        first.checkpoint_id,
        "The candidate changed the intended grain, contradicting the earlier entity-level target.",
        ["Rebuild at entity grain."],
    )
    assert store.get(second.checkpoint_id).status == "abandoned"
    assert handle in state.artifacts
    with pytest.raises(StateError) as caught:
        state.get_relation(handle)
    assert caught.value.code == "inactive_handle"

    text = EnvironmentRenderer().render("q", "k", state, store)
    assert handle not in text


def test_restore_never_reuses_step_artifact_observation_or_checkpoint_ids() -> None:
    state = EnvironmentState(catalog())
    store = CheckpointStore(state)
    handle_1 = state.allocate_artifact_handle("filter")
    add_fact_bundle(state, handle_1)
    first_step = max(state.steps)
    first_observation = max(state.observations)
    first_checkpoint = store.commit(["Recorded candidate."], [], ["Continue."])

    handle_2 = state.allocate_artifact_handle("join")
    add_fact_bundle(state, handle_2)
    second_step = max(state.steps)
    second_observation = max(state.observations)
    recovery = store.restore(
        "root",
        "The later artifacts used a mistaken branch that contradicts the intended population.",
        ["Restart exploration."],
    )

    handle_3 = state.allocate_artifact_handle("project")
    step_3 = state.allocate_step_id()
    observation_3 = state.allocate_observation_id()
    assert int(handle_3.rsplit("_", 1)[1]) > int(handle_2.rsplit("_", 1)[1])
    assert int(step_3.rsplit("_", 1)[1]) > int(second_step.rsplit("_", 1)[1])
    assert int(observation_3.rsplit("_", 1)[1]) > int(second_observation.rsplit("_", 1)[1])
    assert first_step != step_3 and first_observation != observation_3
    assert recovery.checkpoint_id > first_checkpoint.checkpoint_id


def test_logical_hash_restores_exactly_and_ignores_failed_audit_steps() -> None:
    state = EnvironmentState(catalog())
    store = CheckpointStore(state)
    state.discover_schema("customers")
    first_handle = state.allocate_artifact_handle("filter")
    add_fact_bundle(state, first_handle)
    first = store.commit(["Established population."], [], ["Test alternative."])
    expected_hash = first.snapshot.environment_state_hash

    second_handle = state.allocate_artifact_handle("join")
    add_fact_bundle(state, second_handle)
    store.commit(["Tried an alternative join."], [], ["Audit grain."])
    before_error = state.logical_hash()
    state.add_step(
        StepRecord(
            state.allocate_step_id(),
            "project",
            "error",
            None,
            error={"code": "unknown_column"},
        )
    )
    assert state.logical_hash() == before_error
    assert state.last_error == {"code": "unknown_column"}

    store.restore(
        first.checkpoint_id,
        "The later join changed row grain, contradicting the population fixed at this checkpoint.",
        ["Use the earlier population."],
    )
    assert state.logical_hash() == expected_hash
    assert state.last_error is None


def test_restore_accepts_current_phase_start_checkpoint_and_honors_budgets() -> None:
    state = EnvironmentState(catalog())
    store = CheckpointStore(state, max_checkpoints=2, max_restores=1)
    first = store.commit(["One milestone."], [], ["Continue."])
    recovered = store.restore(
        first.checkpoint_id,
        "Work in this phase contradicted its starting milestone, so roll the phase back.",
        ["Rebuild from the same milestone."],
    )
    assert recovered.parent_id == first.checkpoint_id
    assert recovered.restored_from == first.checkpoint_id
    assert recovered.abandoned_checkpoints == ()
    with pytest.raises(StateError) as caught:
        store.commit(["Another milestone."], [], ["Continue."])
    assert caught.value.code == "checkpoint_limit_reached"

    second_state = EnvironmentState(catalog())
    second_store = CheckpointStore(second_state, max_checkpoints=4, max_restores=1)
    milestone = second_store.commit(["One milestone."], [], ["Continue."])
    second_store.restore(milestone.checkpoint_id, "Roll back this phase.", ["Retry."])
    with pytest.raises(StateError) as caught:
        second_store.restore("root", "Only one restore is permitted.", ["Restart."])
    assert caught.value.code == "restore_limit_reached"


def test_checkpoint_text_validation_is_structural_and_bounded() -> None:
    state = EnvironmentState(catalog())
    store = CheckpointStore(state)
    with pytest.raises(StateError):
        store.commit([], [], ["target"])
    with pytest.raises(StateError):
        store.commit(["x" * 513], [], ["target"])
    with pytest.raises(StateError):
        store.commit(["progress"], [], [])


def test_checkpoint_goals_must_be_distinct_across_the_active_path() -> None:
    state = EnvironmentState(catalog())
    store = CheckpointStore(state)
    store.commit(["Population fixed."], [], ["Compute customer totals."])
    store.commit(["Totals computed."], [], ["Rank the resulting customers."])
    before_hash = state.logical_hash()
    before_phase = state.phase_id
    before_count = store.checkpoint_count

    with pytest.raises(StateError) as caught:
        store.commit(
            ["A later phase tried to cycle back."],
            [],
            ["  COMPUTE   CUSTOMER TOTALS  "],
        )

    assert caught.value.code == "checkpoint_goal_not_distinct"
    assert "checkpoint_001" in caught.value.details["conflicts"]
    assert state.logical_hash() == before_hash
    assert state.phase_id == before_phase
    assert store.checkpoint_count == before_count


def test_checkpoint_goal_order_and_cosmetic_punctuation_do_not_create_novelty() -> None:
    state = EnvironmentState(catalog())
    store = CheckpointStore(state)
    store.commit(
        ["Stable milestone."],
        [],
        ["Confirm the output grain!", "Compute the final metric."],
    )
    with pytest.raises(StateError) as caught:
        store.commit(
            ["No new milestone."],
            [],
            [" compute the final metric ", "CONFIRM THE OUTPUT GRAIN"],
        )
    assert caught.value.code == "checkpoint_goal_not_distinct"

    with pytest.raises(StateError) as duplicate:
        CheckpointStore(EnvironmentState(catalog())).commit(
            ["Milestone."],
            [],
            ["Rank customers.", " rank   CUSTOMERS "],
        )
    assert duplicate.value.code == "checkpoint_goal_not_distinct"


def test_checkpoint_store_has_a_global_eight_checkpoint_ceiling() -> None:
    assert MAX_CHECKPOINTS == 8
    with pytest.raises(ValueError, match="0 to 8"):
        CheckpointStore(EnvironmentState(catalog()), max_checkpoints=9)


def test_renderer_keeps_all_history_as_working_memory_not_evidence() -> None:
    state = EnvironmentState(catalog())
    store = CheckpointStore(state)
    first = store.commit(
        ["A model-authored value was 17."],
        ["The value is not yet grounded."],
        ["Inspect the database value."],
    )
    store.commit(["Inspected a different branch."], [], ["Continue."])
    store.restore(
        first.checkpoint_id,
        "The later branch contradicted the earlier grain, so return before its transformation.",
        ["Rebuild carefully."],
    )
    text = EnvironmentRenderer().render("q", "k", state, store)
    for checkpoint_id in store.nodes:
        assert checkpoint_id in text
    assert "+ A model-authored value was 17." in text
    assert "? The value is not yet grounded." in text
    assert "! The later branch contradicted" in text
    assert "-> Rebuild carefully." in text
    assert "not database evidence" in text
    assert "CURRENT ENVIRONMENT STATE" in text


def test_renderer_no_leak_and_only_active_facts() -> None:
    state = EnvironmentState(catalog())
    state.discover_schema("orders")
    table = state.allocate_artifact_handle("filter")
    leaked = RelationArtifact(
        table,
        "filter",
        (Column("order_id", "INTEGER"),),
        1,
        derivation={
            "operator": "filter_rows",
            "gold_sql": "SELECT super_secret_gold",
            "evaluator_result": "super_secret_score",
        },
        scalar_cell=7,
    )
    state.add_artifact(leaked)
    state.add_observation(
        Observation(
            state.allocate_observation_id(),
            "read_rows",
            table,
            {"rows": [[7]], "gold_rows": [[999]], "evaluator": {"correct": True}},
        )
    )
    store = CheckpointStore(state)
    text = EnvironmentRenderer().render(
        {"question": "Return the order.", "gold_sql": "super_secret_question_sql"},
        {"evidence": "Use active orders.", "gold_rows": [[999]]},
        state,
        store,
    )
    lowered = text.lower()
    assert "return the order" in lowered
    assert "use active orders" in lowered
    assert "super_secret" not in lowered
    assert "gold_sql" not in lowered
    assert "gold_rows" not in lowered
    assert "evaluator" not in lowered
    assert "[[7]]" in text


def test_snapshots_share_records_instead_of_copying_table_payloads() -> None:
    state = EnvironmentState(catalog())
    store = CheckpointStore(state)
    handle = state.allocate_artifact_handle("filter")
    state.add_artifact(artifact(handle))
    node = store.commit(["Built relation."], [], ["Continue."])
    assert node.snapshot.active_artifact_ids == frozenset({handle})
    assert not hasattr(node.snapshot, "artifacts")
    assert state.artifacts[handle] is state.get_relation(handle)


def test_renderer_shows_grounded_null_scalar_for_one_by_one_relation() -> None:
    state = EnvironmentState(catalog())
    state.add_artifact(
        RelationArtifact(
            "aggregate_001",
            "aggregate",
            (Column("total", "REAL"),),
            1,
            derivation={"operator": "aggregate"},
            scalar_cell=None,
        )
    )
    text = EnvironmentRenderer().render_environment(state)
    assert "scalar: null" in text


def test_fact_records_are_recursively_immutable_and_isolated_from_inputs() -> None:
    derivation = {"operator": "filter_rows", "inputs": ["customers"]}
    item = RelationArtifact(
        "filter_001",
        "filter",
        (Column("customer_id", "INTEGER"),),
        1,
        derivation=derivation,
        scalar_cell=1,
    )
    derivation["inputs"].append("orders")
    assert item.derivation["inputs"] == ("customers",)
    with pytest.raises(TypeError):
        item.derivation["operator"] = "project"  # type: ignore[index]

    rows = {"rows": [[1]]}
    observation = Observation("observation_001", "read_rows", "filter_001", rows)
    rows["rows"][0][0] = 9
    assert observation.payload["rows"] == ((1,),)
    with pytest.raises(TypeError):
        observation.payload["rows"][0] = (9,)  # type: ignore[index]
