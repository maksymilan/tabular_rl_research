#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import sqlite3
import tempfile
import unittest
from pathlib import Path

RL_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = RL_DIR.parent
sys.path.insert(0, str(SRC_DIR))

from rl.objectives.process_credit import (  # noqa: E402
    ProcessRewardConfig,
    StepFeature,
    _expression_columns,
    _expression_predicate_literals,
    _grounding_reference_redundant_with_task,
    _visible_output_values,
    allocate_process_rewards,
    normalized_search_reduction,
    replay_step_features,
)
from rl.objectives.process_objective import process_policy_loss, sampled_forward_kl  # noqa: E402
from provenance import build_grounding_references, build_references  # noqa: E402
from rl.scenarios.audits.review_grounding_edges_external import build_review_package  # noqa: E402
from rl.runtime.target_support import (  # noqa: E402
    TargetSupport,
    build_target_support,
    normalized_row_unit,
    observed_target_row_units,
)
from rl.runtime.terminal_reward import terminal_result_reward  # noqa: E402
from rl.runtime.task_support import task_text_supports_literal  # noqa: E402


def feature(index: int, **kwargs) -> StepFeature:
    return StepFeature(
        action_index=index,
        step_id=f"step_{index}",
        tool=kwargs.pop("tool", "condition_filter"),
        legal_success=kwargs.pop("legal_success", True),
        **kwargs,
    )


class ProcessRewardTests(unittest.TestCase):
    def test_dense_uniform_rewards_every_clean_correct_turn(self):
        config = ProcessRewardConfig(allocation_mode="dense_uniform")
        result = allocate_process_rewards(
            "dense-correct",
            [feature(1), feature(2), feature(3), feature(4, tool="answer_from_context")],
            correct=True,
            config=config,
        )
        self.assertEqual([step.reward for step in result.steps], [0.25] * 4)
        self.assertAlmostEqual(result.total_reward, 1.0)
        self.assertTrue(result.process_update)
        self.assertEqual(result.diagnostics["dense_credit_scope"], "every_authored_turn")

    def test_dense_uniform_mildly_penalizes_every_clean_failed_turn(self):
        config = ProcessRewardConfig(allocation_mode="dense_uniform")
        result = allocate_process_rewards(
            "dense-failed",
            [feature(1), feature(2), feature(3), feature(4, tool="answer_from_context")],
            correct=False,
            config=config,
        )
        self.assertEqual([step.reward for step in result.steps], [-0.125] * 4)
        self.assertAlmostEqual(result.total_reward, -0.5)
        self.assertTrue(all(step.p_outcome == 0.5 for step in result.steps))

    def test_dense_severe_categories_share_one_non_stacking_maximum_penalty(self):
        config = ProcessRewardConfig(allocation_mode="dense_uniform")
        result = allocate_process_rewards(
            "dense-severe",
            [
                feature(1, legal_success=False, error_type="execution_error"),
                feature(2, adjacent_repeat=True),
                feature(3, legal_no_state_change=1.0),
                feature(
                    4,
                    legal_success=False,
                    error_type="execution_error",
                    adjacent_repeat=True,
                    legal_no_state_change=1.0,
                ),
            ],
            correct=False,
            config=config,
        )
        self.assertEqual([step.p_local for step in result.steps], [2.0] * 4)
        self.assertEqual([step.reward for step in result.steps], [-0.5] * 4)
        self.assertAlmostEqual(result.total_reward, -2.0)
        self.assertEqual(result.diagnostics["dense_severe_turns"], 4)

    def test_dense_strategic_adds_only_grounded_positive_bonuses(self):
        config = ProcessRewardConfig(
            allocation_mode="dense_strategic",
            dense_observation_bonus_weight=0.5,
            dense_backslice_bonus_weight=0.5,
        )
        result = allocate_process_rewards(
            "dense-strategic",
            [
                feature(1, tool="read_subtable", new_used_evidence=1.0),
                feature(2, tool="condition_filter", back_slice=1.0),
                feature(3, tool="answer_from_context", is_terminal=True),
            ],
            correct=True,
            config=config,
        )
        self.assertEqual([step.g_positive for step in result.steps], [1.5, 1.5, 1.0])
        self.assertEqual(result.diagnostics["dense_observation_bonus_turns"], 1)
        self.assertEqual(result.diagnostics["dense_backslice_bonus_turns"], 1)
        self.assertAlmostEqual(result.total_reward, 4.0 / 3.0)

    def test_sql_expression_support_extracts_sources_and_domain_predicates(self):
        expression = (
            'CASE WHEN "SalesPerson.SalesQuota" > 300000 THEN 1 ELSE 0 END'
        )
        self.assertEqual(
            _expression_columns(expression),
            {"SalesPerson.SalesQuota"},
        )
        self.assertEqual(
            _expression_predicate_literals(expression),
            [("SalesPerson.SalesQuota", 300000)],
        )

    def test_visible_literal_support_accepts_canonical_derivations_only(self):
        text = (
            "Born after 1970/1/1; during July, 2014; in year 2003; "
            "no more than one car; puree of split peas; Basketball Men''s."
        )
        self.assertTrue(task_text_supports_literal("1970-01-01", text))
        self.assertTrue(task_text_supports_literal("2014-07-01", text))
        self.assertTrue(task_text_supports_literal("2014-07-31", text))
        self.assertTrue(task_text_supports_literal("2003-12-31", text))
        self.assertTrue(task_text_supports_literal(1, text))
        self.assertTrue(task_text_supports_literal("%puree%split%peas%", text))
        self.assertTrue(task_text_supports_literal("Basketball Men's", text))
        self.assertTrue(task_text_supports_literal(12882, "menu ID12882"))
        self.assertTrue(
            task_text_supports_literal(
                25.746,
                "a place with 25746 inhabitants",
                column="INHABITANTS_K",
            )
        )
        self.assertTrue(task_text_supports_literal("New Jersey", 'schools in "NJ"'))
        self.assertFalse(task_text_supports_literal(559, text))
        self.assertFalse(task_text_supports_literal("C001035", text))

    def test_only_factual_rendered_cells_count_as_visible_literal_support(self):
        output = {
            "table": "filter_001",
            "columns": ["movie_id"],
            "row_count": 1,
            "rows": [[559]],
            "derivation": {"predicate": {"value": "not-visible-as-a-row"}},
        }
        self.assertEqual(_visible_output_values(output), [559])

    def test_task_literal_does_not_create_redundant_row_evidence(self):
        reference = {
            "type": "grounding",
            "role": "row_observation",
            "step": "step_3",
            "target": {
                "values": ["Zentral Theater Terrace", "Young's Hotel"],
                "column_matches": [{"target_column": "name"}],
            },
        }
        self.assertTrue(
            _grounding_reference_redundant_with_task(
                reference,
                'Compare "Zentral Theater Terrace" with "Young\'s Hotel".',
            )
        )
        self.assertFalse(
            _grounding_reference_redundant_with_task(
                {
                    **reference,
                    "target": {
                        "values": [35487],
                        "column_matches": [{"target_column": "menu_id"}],
                    },
                },
                'Compare "Zentral Theater Terrace" with "Young\'s Hotel".',
            )
        )

    def test_result_only_control_is_exactly_binary(self):
        self.assertEqual(terminal_result_reward(True), 1.0)
        self.assertEqual(terminal_result_reward(False), 0.0)

    def test_execution_ladder_distinguishes_wrong_terminal_from_invalid(self):
        self.assertEqual(
            terminal_result_reward(
                True,
                executable=True,
                profile="execution-ladder",
            ),
            1.0,
        )
        self.assertEqual(
            terminal_result_reward(
                False,
                executable=True,
                profile="execution-ladder",
            ),
            0.2,
        )
        self.assertEqual(
            terminal_result_reward(
                False,
                executable=False,
                profile="execution-ladder",
            ),
            0.0,
        )

    def test_four_level_uses_result_sign_and_structured_error_tier(self):
        self.assertEqual(
            terminal_result_reward(True, profile="four-level", has_errors=False),
            1.5,
        )
        self.assertEqual(
            terminal_result_reward(True, profile="four-level", has_errors=True),
            1.0,
        )
        self.assertEqual(
            terminal_result_reward(False, profile="four-level", has_errors=False),
            -0.5,
        )
        self.assertEqual(
            terminal_result_reward(False, profile="four-level", has_errors=True),
            -1.0,
        )

    def test_three_level_clean_weighted_profile(self):
        self.assertEqual(
            terminal_result_reward(
                True,
                profile="three-level-clean-weighted",
                has_errors=False,
            ),
            1.25,
        )
        self.assertEqual(
            terminal_result_reward(
                True,
                profile="three-level-clean-weighted",
                has_errors=True,
            ),
            0.75,
        )
        self.assertEqual(
            terminal_result_reward(
                False,
                profile="three-level-clean-weighted",
                has_errors=False,
            ),
            -1.0,
        )
        self.assertEqual(
            terminal_result_reward(
                False,
                profile="three-level-clean-weighted",
                has_errors=True,
            ),
            -1.0,
        )

    def test_target_row_units_use_bird_raw_cell_equality(self):
        self.assertNotEqual(normalized_row_unit(["1"]), normalized_row_unit([1]))
        self.assertEqual(normalized_row_unit([1]), normalized_row_unit([1.0]))

    def test_read_observation_discovers_only_rows_actually_exposed(self):
        target = TargetSupport(
            tables=frozenset({"items"}),
            columns=frozenset({"items.name"}),
            row_units=frozenset({("a",), ("b",)}),
            output_columns=("name",),
            sql_parse_complete=True,
            rows_complete=True,
        )
        self.assertEqual(
            observed_target_row_units(["id", "name"], [[1, "a"]], target),
            {("a",)},
        )

    def test_target_support_excludes_cte_names_from_physical_tables(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            connection = sqlite3.connect(tmp.name)
            connection.execute("CREATE TABLE items(id INTEGER, value TEXT)")
            connection.execute("INSERT INTO items VALUES (1, 'x')")
            connection.commit()
            connection.close()
            from executor import Harness

            harness = Harness(tmp.name)
            try:
                target = build_target_support(
                    harness,
                    "WITH selected AS (SELECT id, value FROM items) "
                    "SELECT value FROM selected",
                )
            finally:
                harness.conn.close()
        self.assertEqual(target.tables, frozenset({"items"}))
        self.assertNotIn("selected", target.tables)
        self.assertEqual(target.columns, frozenset({"items.id", "items.value"}))

    def test_process_reward_rejects_historical_denotation_metric(self):
        with self.assertRaisesRegex(ValueError, "bird-set"):
            replay_step_features({}, denotation_comparison="strict-multiset")

    def test_version5_join_emits_one_data_edge_per_relation_input(self):
        args = {
            "base": "orders",
            "joins": [
                {
                    "table": "filter_001",
                    "on": [{"left": "orders.customer_id", "right": "id"}],
                },
                {
                    "table": "regions",
                    "on": [{"left": "filter_001.region_id", "right": "id"}],
                },
            ],
        }
        refs = build_references(
            "join_tables",
            args,
            lambda ref: "step_2" if ref == "filter_001" else None,
        )
        self.assertEqual(
            refs,
            [
                {"type": "data", "source": "orders", "role": "table",
                 "target": {"table": "orders"}},
                {"type": "data", "step": "step_2", "role": "table",
                 "target": {"handle": "filter_001"}},
                {"type": "data", "source": "regions", "role": "table",
                 "target": {"table": "regions"}},
            ],
        )

    def test_conditional_aggregate_emits_value_and_grounding_edges(self):
        args = {
            "table": "items",
            "group_by": [],
            "aggregations": [{
                "op": "count",
                "column": "*",
                "as": "active_above_threshold",
                "where": {
                    "and": [
                        {"column": "status", "op": "=", "value": "active"},
                        {"column": "score", "op": ">", "value_ref": "step_2"},
                    ],
                },
            }],
        }
        refs = build_references(
            "group_aggregate",
            args,
            lambda ref: ref if ref == "step_2" else None,
        )
        self.assertIn(
            {
                "type": "value",
                "step": "step_2",
                "role": "aggregate_where",
                "target": {"aggregation_index": 0, "column": "score"},
            },
            refs,
        )
        grounded = build_grounding_references(
            "group_aggregate",
            args,
            {
                "step_1": {
                    "tool": "inspect_column",
                    "arguments": {"table": "items", "column": "status"},
                    "output": {
                        "column": "status",
                        "distinct_count": 2,
                        "frequent_values": ["active", "inactive"],
                    },
                },
            },
        )
        self.assertEqual(grounded[0]["role"], "domain_observation")
        self.assertEqual(grounded[0]["target"]["values"], ["active"])

    def test_wide_aggregate_category_values_reuse_domain_grounding(self):
        refs = build_grounding_references(
            "group_aggregate",
            {
                "table": "patients",
                "group_by": ["gender"],
                "aggregations": [{
                    "op": "count",
                    "column": "*",
                    "as": "patient_count",
                }],
                "output_layout": "columns",
                "category_values": ["M", "F"],
                "output_columns": ["male_count", "female_count"],
            },
            {
                "step_1": {
                    "tool": "inspect_column",
                    "arguments": {"table": "patients", "column": "gender"},
                    "output": {
                        "column": "gender",
                        "distinct_count": 2,
                        "frequent_values": ["M", "F"],
                    },
                },
            },
        )
        self.assertEqual(refs[0]["role"], "domain_observation")
        self.assertEqual(refs[0]["target"]["values"], ["M", "F"])

    def test_named_scalar_operands_keep_distinct_column_provenance(self):
        refs = build_references(
            "scalar_compute",
            {
                "operation": "percent",
                "operands": [
                    {"value_ref": "step_7", "column": "usa_nominees"},
                    {"value_ref": "step_7", "column": "total_nominees"},
                ],
            },
            lambda ref: ref if ref == "step_7" else None,
        )
        self.assertEqual(
            refs,
            [
                {
                    "type": "value",
                    "step": "step_7",
                    "role": "operand",
                    "target": {"operand_index": 0, "column": "usa_nominees"},
                },
                {
                    "type": "value",
                    "step": "step_7",
                    "role": "operand",
                    "target": {"operand_index": 1, "column": "total_nominees"},
                },
            ],
        )

    def test_harness_infers_final_table_and_perception_chain_without_model_evidence(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            conn = sqlite3.connect(tmp.name)
            conn.executescript(
                """
                CREATE TABLE people(first_name TEXT, last_name TEXT, bioguide TEXT);
                CREATE TABLE social(bioguide TEXT, instagram TEXT);
                INSERT INTO people VALUES ('Bob', 'Corker', 'C001071');
                INSERT INTO people VALUES ('Jane', 'Doe', 'D000001');
                INSERT INTO social VALUES ('C001071', 'senbobcorker');
                INSERT INTO social VALUES ('D000001', 'janedoe');
                """
            )
            conn.commit()
            conn.close()

            def step(index, tool, arguments):
                return {
                    "step_id": f"step_{index}",
                    "tool_call": {"tool": tool, "arguments": arguments},
                }

            trajectory = {
                "trajectory_id": "automatic-grounding",
                "question": "Where is Bob Corker on Instagram?",
                "source": {
                    "db_path": tmp.name,
                    "external_knowledge": "Instagram handle refers to social.instagram.",
                    "gold_sql": (
                        "SELECT social.instagram FROM people JOIN social "
                        "ON people.bioguide = social.bioguide "
                        "WHERE people.first_name = 'Bob' AND people.last_name = 'Corker'"
                    ),
                },
                "steps": [
                    step(1, "describe_table", {"tables": ["people", "social"]}),
                    step(2, "condition_filter", {
                        "table": "people",
                        "conditions": {"and": [
                            {"column": "first_name", "op": "=", "value": "Bob"},
                            {"column": "last_name", "op": "=", "value": "Corker"},
                        ]},
                    }),
                    step(3, "read_subtable", {"table": "filter_001", "limit": 5}),
                    step(4, "condition_filter", {
                        "table": "social",
                        "conditions": {"column": "bioguide", "op": "=", "value": "C001071"},
                    }),
                    step(5, "read_subtable", {"table": "filter_002", "limit": 5}),
                    step(6, "answer_from_context", {
                        "evidence": None,
                        "answer": ["senbobcorker"],
                    }),
                ],
            }
            features, diagnostics = replay_step_features(trajectory)
            review_package = build_review_package(trajectory)
            result = allocate_process_rewards(
                trajectory["trajectory_id"],
                features,
                correct=diagnostics["replay_correct"],
                diagnostics=diagnostics,
            )

        self.assertEqual(diagnostics["grounding_method"], "automatic_last_table")
        self.assertEqual(diagnostics["grounding_handle"], "filter_002")
        self.assertEqual(
            diagnostics["back_slice_step_ids"],
            ["step_2", "step_3", "step_4", "step_5"],
        )
        self.assertFalse(result.fallback_terminal_credit)
        self.assertEqual(result.steps[-1].reward, 0.02)
        self.assertEqual(features[0].back_slice, 0.0)
        self.assertEqual(features[0].new_used_evidence, 1.0)  # schema evidence, not data B
        self.assertGreater(result.steps[0].reward, 0.0)  # describe_table receives E only
        self.assertGreater(result.steps[2].reward, 0.0)  # first read used as a later literal
        self.assertGreater(result.steps[4].reward, 0.0)  # final row observation
        self.assertTrue(features[2].state_changed)
        self.assertEqual(features[2].legal_no_state_change, 0.0)
        dependency_roles = {edge["role"] for edge in review_package["dependency_edges"]}
        self.assertIn("automatic_final_table", dependency_roles)
        self.assertTrue(review_package["grounding_edges"])
        self.assertEqual(
            review_package["external_knowledge"],
            "Instagram handle refers to social.instagram.",
        )

    def test_inspected_domain_value_builds_harness_grounding_edge(self):
        history = {
            "step_1": {
                "tool": "inspect_column",
                "arguments": {"table": "items", "column": "status"},
                "output": {
                    "column": "status",
                    "distinct_count": 2,
                    "frequent_values": ["active", "inactive"],
                    "truncated": False,
                },
            }
        }
        refs = build_grounding_references(
            "condition_filter",
            {
                "table": "items",
                "conditions": {"column": "status", "op": "=", "value": "active"},
            },
            history,
        )
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0]["step"], "step_1")
        self.assertEqual(refs[0]["role"], "domain_observation")
        self.assertEqual(refs[0]["target"]["values"], ["active"])

    def test_schema_grounding_uses_only_latest_description(self):
        described = {
            "tool": "describe_table",
            "arguments": {"tables": ["items"]},
            "output": {"tables": [{
                "table_name": "items",
                "columns": [{"name": "status"}],
            }]},
        }
        refs = build_grounding_references(
            "condition_filter",
            {
                "table": "items",
                "conditions": {"column": "status", "op": "=", "value": "active"},
            },
            {"step_1": described, "step_2": described},
        )
        self.assertEqual([ref["step"] for ref in refs], ["step_2"])

    def test_row_grounding_is_column_aware_and_uses_foreign_keys(self):
        history = {
            "step_1": {
                "tool": "describe_table",
                "arguments": {"tables": ["Match", "Season", "Team"]},
                "output": {"tables": [
                    {
                        "table_name": "Match",
                        "columns": [{"name": "Match_Winner"}],
                        "foreign_keys": [{"column": "Match_Winner", "references": "Team.Team_Id"}],
                    },
                    {"table_name": "Season", "columns": [{"name": "Season_Id"}], "foreign_keys": []},
                    {"table_name": "Team", "columns": [{"name": "Team_Id"}], "foreign_keys": []},
                ]},
                "references": [],
            },
            "step_2": {
                "tool": "condition_filter",
                "arguments": {"table": "Team", "conditions": {"column": "Team_Name", "op": "=", "value": "Mumbai Indians"}},
                "output": {"table": "filter_001", "columns": ["Team_Id", "Team_Name"], "row_count": 1},
                "references": [{"type": "data", "source": "Team", "role": "table", "target": {"table": "Team"}}],
            },
            "step_3": {
                "tool": "read_subtable",
                "arguments": {"table": "filter_001", "limit": 5},
                "output": {"table": "filter_001", "columns": ["Team_Id", "Team_Name"], "rows": [[7, "Mumbai Indians"]], "row_count": 1},
                "references": [],
            },
            "step_4": {
                "tool": "read_subtable",
                "arguments": {"table": "Season", "limit": 10},
                "output": {"table": "Season", "columns": ["Season_Id", "Orange_Cap"], "rows": [[7, 305]], "row_count": 1},
                "references": [],
            },
        }
        refs = build_grounding_references(
            "condition_filter",
            {"table": "Match", "conditions": {"column": "Match_Winner", "op": "=", "value": 7}},
            history,
        )
        row_refs = [ref for ref in refs if ref["role"] == "row_observation"]
        self.assertEqual([ref["step"] for ref in row_refs], ["step_3"])
        match = row_refs[0]["target"]["column_matches"][0]
        self.assertEqual(match["source_column"], "Team_Id")
        self.assertEqual(match["target_column"], "Match_Winner")

    def test_model_visible_relation_preview_can_ground_later_foreign_key_literal(self):
        history = {
            "step_1": {
                "tool": "describe_table",
                "arguments": {"tables": ["movie", "movie_cast"]},
                "output": {"tables": [
                    {
                        "table_name": "movie",
                        "columns": [
                            {"name": "movie_id", "pk": True},
                            {"name": "title", "pk": False},
                        ],
                        "foreign_keys": [],
                    },
                    {
                        "table_name": "movie_cast",
                        "columns": [{"name": "movie_id", "pk": False}],
                        "foreign_keys": [
                            {"column": "movie_id", "references": "movie.movie_id"}
                        ],
                    },
                ]},
            },
            "step_2": {
                "tool": "condition_filter",
                "arguments": {
                    "table": "movie",
                    "conditions": {"column": "title", "op": "=", "value": "Spider-Man 3"},
                    "return_columns": ["movie_id"],
                },
                "output": {
                    "table": "filter_001",
                    "columns": ["movie_id"],
                    "rows": [[559]],
                    "row_count": 1,
                },
                "references": [
                    {
                        "type": "data",
                        "source": "movie",
                        "role": "table",
                        "target": {"table": "movie"},
                    }
                ],
            },
        }
        refs = build_grounding_references(
            "condition_filter",
            {
                "table": "movie_cast",
                "conditions": {"column": "movie_id", "op": "=", "value": 559},
            },
            history,
        )
        row_refs = [ref for ref in refs if ref["role"] == "row_observation"]
        self.assertEqual([ref["step"] for ref in row_refs], ["step_2"])
        self.assertEqual(row_refs[0]["target"]["values"], [559])
        match = row_refs[0]["target"]["column_matches"][0]
        self.assertEqual(match["argument_path"], ["conditions", "value"])
        self.assertEqual(match["source_row_index"], 0)
        self.assertEqual(match["source_column_index"], 0)

    def test_unique_visible_cell_copy_survives_missing_foreign_key_metadata(self):
        refs = build_grounding_references(
            "condition_filter",
            {
                "table": "Player",
                "conditions": {"column": "Player_Id", "op": "=", "value": 9},
            },
            {
                "step_1": {
                    "tool": "condition_filter",
                    "arguments": {
                        "table": "Ball_by_Ball",
                        "conditions": {"column": "Match_Id", "op": "=", "value": 419169},
                    },
                    "output": {
                        "table": "filter_001",
                        "columns": ["Striker"],
                        "rows": [[9]],
                        "row_count": 1,
                    },
                    "references": [{
                        "type": "data",
                        "source": "Ball_by_Ball",
                        "role": "table",
                        "target": {"table": "Ball_by_Ball"},
                    }],
                },
            },
        )
        row_refs = [ref for ref in refs if ref["role"] == "row_observation"]
        self.assertEqual([ref["step"] for ref in row_refs], ["step_1"])
        match = row_refs[0]["target"]["column_matches"][0]
        self.assertEqual(match["match_kind"], "visible_literal_copy")
        self.assertEqual(match["argument_path"], ["conditions", "value"])
        self.assertEqual(match["source_row_index"], 0)
        self.assertEqual(match["source_column_index"], 0)
        self.assertEqual(
            match["replay_binding"],
            {
                "step": "step_1",
                "row_index": 0,
                "column_index": 0,
                "column": "Striker",
            },
        )

    def test_replay_binding_uses_the_selected_grounding_producer(self):
        refs = build_grounding_references(
            "condition_filter",
            {
                "table": "Player",
                "conditions": {"column": "Player_Id", "op": "=", "value": 9},
            },
            {
                "step_1": {
                    "tool": "read_subtable",
                    "arguments": {"table": "Player", "limit": 1},
                    "output": {
                        "rows": [[9]],
                        "row_count": 1,
                    },
                    "observed_columns": ["Player_Id"],
                },
                "step_2": {
                    "tool": "read_subtable",
                    "arguments": {"table": "unrelated", "limit": 1},
                    "output": {
                        "rows": [[9]],
                        "row_count": 1,
                    },
                    "observed_columns": ["other_value"],
                },
            },
        )
        row_ref = next(ref for ref in refs if ref["role"] == "row_observation")
        match = row_ref["target"]["column_matches"][0]
        self.assertEqual(row_ref["step"], "step_1")
        self.assertEqual(match["match_kind"], "column_equivalent")
        self.assertEqual(match["replay_binding"]["step"], row_ref["step"])

    def test_ambiguous_unrelated_visible_cells_do_not_create_copy_edge(self):
        refs = build_grounding_references(
            "condition_filter",
            {
                "table": "Player",
                "conditions": {"column": "Player_Id", "op": "=", "value": 9},
            },
            {
                "step_1": {
                    "tool": "read_subtable",
                    "arguments": {"table": "unrelated", "limit": 1},
                    "output": {"rows": [[9, 9]], "row_count": 1},
                    "observed_columns": ["left_value", "right_value"],
                },
            },
        )
        self.assertFalse(
            any(ref["role"] == "row_observation" for ref in refs)
        )

    def test_computed_scalar_result_can_ground_a_later_predicate_literal(self):
        refs = build_grounding_references(
            "condition_filter",
            {
                "table": "players",
                "conditions": {"column": "dob", "op": "=", "value": "1998-07-18"},
            },
            {
                "step_1": {
                    "tool": "group_aggregate",
                    "arguments": {
                        "table": "players",
                        "group_by": [],
                        "aggregations": [{"op": "max", "column": "dob", "as": "max_dob"}],
                    },
                    "output": {
                        "table": "group_001",
                        "columns": ["max_dob"],
                        "rows": [["1998-07-18"]],
                        "row_count": 1,
                    },
                }
            },
        )
        row_refs = [ref for ref in refs if ref["role"] == "row_observation"]
        self.assertEqual([ref["step"] for ref in row_refs], ["step_1"])
        self.assertEqual(
            row_refs[0]["target"]["column_matches"][0]["match_kind"],
            "computed_result",
        )

    def test_answer_value_equality_does_not_create_direct_row_edge(self):
        history = {
            "step_1": {
                "tool": "read_subtable",
                "arguments": {"table": "shipping_method", "limit": 10},
                "output": {"table": "shipping_method", "columns": ["id"], "rows": [[2]], "row_count": 1},
            },
        }
        refs = build_grounding_references(
            "answer_from_context",
            {"answer": [2], "evidence": None},
            history,
        )
        self.assertEqual(refs, [])

    def test_numeric_task_literal_uses_token_boundaries(self):
        self.assertFalse(task_text_supports_literal(2, "orders placed in 2021"))
        self.assertTrue(task_text_supports_literal(2021, "orders placed in 2021"))
        self.assertTrue(task_text_supports_literal(392194, "match ID 392194."))

    def test_compound_filter_can_link_multiple_prior_row_observations(self):
        history = {
            "step_1": {
                "tool": "read_subtable",
                "arguments": {"table": "customer", "limit": 5},
                "output": {"table": "customer", "columns": ["customer_id"], "rows": [[88]], "row_count": 1},
            },
            "step_2": {
                "tool": "read_subtable",
                "arguments": {"table": "shipping_method", "limit": 5},
                "output": {"table": "shipping_method", "columns": ["shipping_method_id"], "rows": [[2]], "row_count": 1},
            },
        }
        refs = build_grounding_references(
            "condition_filter",
            {
                "table": "cust_order",
                "conditions": {"and": [
                    {"column": "customer_id", "op": "=", "value": 88},
                    {"column": "shipping_method_id", "op": "=", "value": 2},
                ]},
            },
            history,
        )
        row_refs = [ref for ref in refs if ref["role"] == "row_observation"]
        self.assertEqual({ref["step"] for ref in row_refs}, {"step_1", "step_2"})

    def test_missing_fk_target_column_resolves_to_unique_primary_key(self):
        history = {
            "step_1": {
                "tool": "describe_table",
                "arguments": {"tables": ["cust_order", "shipping_method"]},
                "output": {"tables": [
                    {
                        "table_name": "cust_order",
                        "columns": [{"name": "shipping_method_id", "pk": False}],
                        "foreign_keys": [{"column": "shipping_method_id", "references": "shipping_method.None"}],
                    },
                    {
                        "table_name": "shipping_method",
                        "columns": [{"name": "method_id", "pk": True}],
                        "foreign_keys": [],
                    },
                ]},
            },
            "step_2": {
                "tool": "read_subtable",
                "arguments": {"table": "shipping_method", "limit": 5},
                "output": {"rows": [[2]], "row_count": 1},
                "observed_columns": ["method_id"],
            },
        }
        refs = build_grounding_references(
            "condition_filter",
            {
                "table": "cust_order",
                "conditions": {"column": "shipping_method_id", "op": "=", "value": 2},
            },
            history,
        )
        row_refs = [ref for ref in refs if ref["role"] == "row_observation"]
        self.assertEqual([ref["step"] for ref in row_refs], ["step_2"])
        self.assertEqual(row_refs[0]["target"]["column_matches"][0]["source_column"], "method_id")

    def test_multi_value_answer_collects_multiple_read_handles(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            conn = sqlite3.connect(tmp.name)
            conn.executescript(
                """
                CREATE TABLE venue(id INTEGER, venue TEXT);
                CREATE TABLE team(id INTEGER, team TEXT);
                INSERT INTO venue VALUES (12, 'Kingsmead');
                INSERT INTO team VALUES (6, 'Delhi Daredevils');
                """
            )
            conn.commit()
            conn.close()

            def step(index, tool, arguments):
                return {"step_id": f"step_{index}", "tool_call": {"tool": tool, "arguments": arguments}}

            trajectory = {
                "trajectory_id": "multi-final",
                "source": {
                    "db_path": tmp.name,
                    "gold_sql": "SELECT venue.venue, team.team FROM venue CROSS JOIN team",
                },
                "steps": [
                    step(1, "describe_table", {"tables": ["venue", "team"]}),
                    step(2, "condition_filter", {"table": "venue", "conditions": {"column": "id", "op": "=", "value": 12}}),
                    step(3, "read_subtable", {"table": "filter_001", "limit": 1}),
                    step(4, "condition_filter", {"table": "team", "conditions": {"column": "id", "op": "=", "value": 6}}),
                    step(5, "read_subtable", {"table": "filter_002", "limit": 1}),
                    step(6, "answer_from_context", {"answer": [["Kingsmead", "Delhi Daredevils"]], "evidence": None}),
                ],
            }
            _, diagnostics = replay_step_features(trajectory)

        self.assertTrue(diagnostics["replay_correct"])
        self.assertEqual(diagnostics["grounding_method"], "automatic_multi_table")
        self.assertEqual(diagnostics["grounding_handles"], ["filter_002", "filter_001"])
        self.assertTrue(diagnostics["final_value_grounding_complete"])
        roles = {ref["role"] for ref in diagnostics["final_references"]}
        self.assertIn("automatic_additional_final_table", roles)

    def test_replay_remaps_handles_after_recoverable_error_consumed_online_suffix(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            conn = sqlite3.connect(tmp.name)
            conn.executescript(
                """
                CREATE TABLE items(id INTEGER, name TEXT);
                INSERT INTO items VALUES (1, 'one'), (2, 'two');
                """
            )
            conn.commit()
            conn.close()

            trajectory = {
                "trajectory_id": "gapped-handle-replay",
                "question": "List names for positive item ids.",
                "source": {
                    "db_path": tmp.name,
                    "gold_sql": "SELECT name FROM items WHERE id > 0",
                },
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool_call": {
                            "tool": "condition_filter",
                            "arguments": {
                                "table": "items",
                                "conditions": {"column": "id", "op": ">", "value": 0},
                            },
                        },
                        "tool_output": {"table": "filter_001"},
                    },
                    {
                        "step_id": "step_3",
                        "tool_call": {
                            "tool": "project",
                            "arguments": {
                                "table": "filter_001",
                                "expressions": ["name"],
                            },
                        },
                        "tool_output": {"table": "project_003"},
                    },
                    {
                        "step_id": "step_4",
                        "tool_call": {
                            "tool": "answer_from_context",
                            "arguments": {
                                "evidence": {"table": "project_003"},
                                "answer": [],
                            },
                        },
                    },
                ],
                "rollout_generation": {
                    "error_events": [{
                        "action_index": 2,
                        "step_id": "step_2",
                        "error_type": "execution_error",
                        "attempted_tool": "project",
                        "attempted_arguments": {
                            "table": "filter_001",
                            "expressions": ["invalid syntax"],
                        },
                        "state_before_hash": "same",
                        "state_after_hash": "same",
                    }],
                },
            }
            _, diagnostics = replay_step_features(trajectory)

        self.assertTrue(diagnostics["replay_correct"])
        self.assertEqual(
            diagnostics["replay_handle_map"],
            {"filter_001": "filter_001", "project_003": "project_002"},
        )
        self.assertEqual(diagnostics["grounding_handle"], "project_002")
        self.assertEqual(diagnostics["back_slice_step_ids"], ["step_1", "step_3"])

    def test_linear_positive_normalization_keeps_zero_credit_zero(self):
        result = allocate_process_rewards(
            "t1",
            [feature(1, back_slice=1), feature(2), feature(3, is_terminal=True)],
            correct=True,
        )
        self.assertEqual([step.c_positive for step in result.steps], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(result.total_reward, 1.0)

    def test_no_normalize_uses_clipped_raw_positive_credit(self):
        result = allocate_process_rewards(
            "t1-raw",
            [
                feature(1, back_slice=1, search_reduction=0.5),
                feature(2, back_slice=0.25),
            ],
            correct=True,
            config=ProcessRewardConfig(normalize_positive=False),
        )
        self.assertEqual([step.c_positive for step in result.steps], [1.0, 0.25])
        self.assertAlmostEqual(result.total_reward, 1.25)
        self.assertEqual(
            result.diagnostics["positive_allocation"],
            "raw_clipped_per_step",
        )

    def test_simple_process_config_uses_only_declared_positive_and_penalty_terms(self):
        config_path = SRC_DIR / "rl" / "configs" / "simple_process_reward.json"
        values = {
            key: value
            for key, value in json.loads(config_path.read_text(encoding="utf-8")).items()
            if not key.startswith("_")
        }
        config = ProcessRewardConfig(**values)
        config.validate()
        result = allocate_process_rewards(
            "simple-process",
            [
                feature(
                    1,
                    back_slice=1,
                    search_reduction=0.5,
                    new_used_evidence=1,
                    state_changed=True,
                ),
                feature(
                    2,
                    legal_success=False,
                    error_type="execution_error",
                    adjacent_repeat=True,
                ),
                feature(3, state_changed=False, legal_no_state_change=1),
                feature(4, tool="answer_from_context", is_terminal=True),
            ],
            correct=True,
            config=config,
        )
        self.assertEqual([step.g_positive for step in result.steps], [1.5, 0.0, 0.0, 1.0])
        self.assertAlmostEqual(result.steps[1].p_local, 0.14)
        self.assertAlmostEqual(result.steps[2].p_local, 0.03)
        self.assertEqual(result.steps[3].p_outcome, 0.0)
        self.assertAlmostEqual(result.total_reward, 0.83)

    def test_simple_process_config_allows_zero_reward_clean_failure(self):
        config_path = SRC_DIR / "rl" / "configs" / "simple_process_reward.json"
        values = {
            key: value
            for key, value in json.loads(config_path.read_text(encoding="utf-8")).items()
            if not key.startswith("_")
        }
        config = ProcessRewardConfig(**values)
        result = allocate_process_rewards(
            "simple-clean-failure",
            [feature(1, state_changed=True)],
            correct=False,
            config=config,
        )
        self.assertEqual(result.total_reward, 0.0)

    def test_adjacent_repeat_feature_does_not_mark_nonadjacent_same_call(self):
        with tempfile.NamedTemporaryFile(suffix=".sqlite") as tmp:
            conn = sqlite3.connect(tmp.name)
            conn.execute("CREATE TABLE items(id INTEGER)")
            conn.execute("INSERT INTO items VALUES (1)")
            conn.commit()
            conn.close()
            same_call = {
                "attempted_tool": "plan",
                "attempted_arguments": {"goal": "inspect"},
            }
            trajectory = {
                "trajectory_id": "adjacent-repeat-only",
                "source": {
                    "db_path": tmp.name,
                    "gold_sql": "SELECT id FROM items",
                },
                "question": "Return item ids.",
                "steps": [],
                "rollout_generation": {
                    "error_events": [
                        {
                            "action_index": 1,
                            "error_type": "protocol_error",
                            **same_call,
                        },
                        {
                            "action_index": 2,
                            "error_type": "protocol_error",
                            **same_call,
                        },
                        {
                            "action_index": 3,
                            "error_type": "protocol_error",
                            "attempted_tool": "plan",
                            "attempted_arguments": {"goal": "other"},
                        },
                        {
                            "action_index": 4,
                            "error_type": "protocol_error",
                            **same_call,
                        },
                    ],
                },
            }
            features, _ = replay_step_features(trajectory)
        self.assertEqual(
            [feature.adjacent_repeat for feature in features],
            [False, True, False, False],
        )

    def test_correct_zero_grounded_signal_is_excluded_without_fallback(self):
        result = allocate_process_rewards(
            "t2",
            [
                feature(1, state_changed=True),
                feature(2, tool="answer_from_context", is_terminal=True),
            ],
            correct=True,
        )
        self.assertFalse(result.fallback_terminal_credit)
        self.assertFalse(result.process_update)
        self.assertEqual([step.c_positive for step in result.steps], [0.0, 0.0])
        self.assertEqual([step.reward for step in result.steps], [0.0, 0.0])

    def test_penalty_cap_preserves_positive_correct_total(self):
        config = ProcessRewardConfig(penalty_cap=0.8, lambda_tool_error=10.0)
        result = allocate_process_rewards(
            "t3",
            [
                feature(
                    1,
                    legal_success=False,
                    error_type="execution_error",
                    back_slice=1,
                )
            ],
            correct=True,
            config=config,
        )
        self.assertAlmostEqual(result.capped_penalty, 0.8)
        self.assertAlmostEqual(result.total_reward, 0.2)

    def test_penalty_cap_allows_strong_local_ablation_up_to_one_point_five(self):
        config = ProcessRewardConfig(penalty_cap=1.5, lambda_tool_error=10.0)
        config.validate()
        result = allocate_process_rewards(
            "strong-local-cap",
            [
                feature(
                    1,
                    legal_success=False,
                    error_type="execution_error",
                )
            ],
            correct=False,
            config=config,
        )
        self.assertAlmostEqual(result.capped_penalty, 1.5)
        self.assertAlmostEqual(result.total_reward, -1.5)
        with self.assertRaisesRegex(ValueError, "P_max <= 1.5"):
            ProcessRewardConfig(penalty_cap=1.500001).validate()

    def test_failed_trajectory_puts_outcome_penalty_at_terminal_boundary(self):
        result = allocate_process_rewards(
            "t4",
            [
                feature(1, attempted_back_slice=1, state_changed=True),
                feature(
                    2,
                    tool="answer_from_context",
                    is_terminal=True,
                    attempted_back_slice=1,
                ),
            ],
            correct=False,
        )
        self.assertLess(result.total_reward, 0)
        self.assertAlmostEqual(result.total_reward, -0.3)
        self.assertAlmostEqual(result.total_reward, -result.capped_penalty)
        self.assertEqual(result.steps[0].reward, 0.0)
        self.assertEqual(result.steps[0].p_outcome, 0.0)
        self.assertAlmostEqual(result.steps[1].reward, -0.3)
        self.assertAlmostEqual(result.steps[1].p_outcome, 0.3)

    def test_terminal_failure_and_local_error_remain_step_local(self):
        result = allocate_process_rewards(
            "t4-distributed",
            [
                feature(1, attempted_back_slice=1, state_changed=True),
                feature(2, legal_success=False, error_type="protocol_error"),
                feature(
                    3,
                    tool="answer_from_context",
                    is_terminal=True,
                    attempted_back_slice=1,
                ),
            ],
            correct=False,
        )
        self.assertAlmostEqual(result.total_reward, -0.38)
        self.assertEqual(result.steps[0].reward, 0.0)
        self.assertEqual(result.steps[0].p_outcome, 0)
        self.assertEqual(result.steps[1].p_outcome, 0)
        self.assertAlmostEqual(result.steps[1].p_local, 0.08)
        self.assertAlmostEqual(result.steps[1].reward, -0.08)
        self.assertAlmostEqual(result.steps[2].p_outcome, 0.3)
        self.assertAlmostEqual(result.steps[2].reward, -0.3)

    def test_active_atomic_config_matches_dataclass_and_reward_bounds(self):
        config_path = SRC_DIR / "rl" / "configs" / "atomic_process_reward.json"
        values = {
            key: value
            for key, value in json.loads(config_path.read_text(encoding="utf-8")).items()
            if not key.startswith("_")
        }
        config = ProcessRewardConfig(**values)
        config.validate()
        result = allocate_process_rewards(
            "atomic-control",
            [
                feature(1, attempted_back_slice=1, state_changed=True),
                feature(2, tool="answer_from_context", is_terminal=True),
            ],
            correct=False,
            config=config,
        )
        self.assertEqual([step.reward for step in result.steps], [0.0, -0.3])

    def test_fixed_root_search_reduction_telescopes(self):
        first = normalized_search_reduction(100, 100, 10)
        second = normalized_search_reduction(100, 10, 2)
        direct = normalized_search_reduction(100, 100, 2)
        self.assertAlmostEqual(first + second, direct)

    def test_search_reduction_rejects_empty_or_expanding_outputs(self):
        self.assertEqual(normalized_search_reduction(100, 10, 0), 0.0)
        self.assertEqual(normalized_search_reduction(100, 10, 11), 0.0)

    def test_feedback_credit_is_separate_from_penalty(self):
        result = allocate_process_rewards(
            "t5",
            [
                feature(1, legal_success=False, error_type="protocol_error"),
                feature(2, feedback_response=1),
                feature(3, tool="answer_from_context", is_terminal=True),
            ],
            correct=True,
        )
        self.assertGreater(result.steps[1].c_positive, 0)
        self.assertGreater(result.steps[0].c_negative, 0)
        self.assertAlmostEqual(result.total_reward, 0.92)

    def test_process_objective_uses_step_rewards_and_fixed_reference_kl(self):
        class Scalar(float):
            pass

        loss = process_policy_loss(
            [[Scalar(-2.0), Scalar(-1.0)]],
            [[0.75, 0.25]],
            episode_step_kls=[[Scalar(0.1), Scalar(0.2)]],
            beta=0.5,
        )
        self.assertAlmostEqual(loss, 0.95)

    def test_process_objective_excludes_correct_zero_signal_episode(self):
        loss = process_policy_loss(
            [[-5.0], [-2.0, -1.0]],
            [[0.0], [0.75, 0.25]],
            episode_update_mask=[False, True],
        )
        self.assertAlmostEqual(loss, 0.875)

    def test_positive_beta_requires_frozen_reference_kl(self):
        with self.assertRaisesRegex(ValueError, "frozen SFT-2 reference"):
            process_policy_loss([[-1.0]], [[1.0]], beta=0.1)

    def test_sampled_forward_kl_is_nonnegative_and_zero_at_reference(self):
        self.assertEqual(sampled_forward_kl(-2.0, -2.0), 0.0)
        self.assertGreater(sampled_forward_kl(-1.0, -2.0), 0.0)
        self.assertGreater(sampled_forward_kl(-3.0, -2.0), 0.0)


if __name__ == "__main__":
    unittest.main()
