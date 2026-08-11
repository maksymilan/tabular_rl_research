#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from tool_modules.checkpoint_relalg.protocol import (  # noqa: E402
    ATOMIC_OPERATOR_PROFILE_SEMANTIC,
    ATOMIC_TOOLS,
    CARRIER_NATIVE_TOOL_CALLS,
    CARRIER_TEXT_JSON,
    CHECKPOINT_GUIDANCE_PROFILES,
    CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET,
    CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET_V2,
    CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET_V3,
    CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET_V4,
    CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE,
    CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE_V2,
    CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE_V3,
    CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE_V4,
    CHECKPOINT_GUIDANCE_PROFILE_DISABLED,
    CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V5,
    CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V6,
    MAX_EXPRESSION_DEPTH,
    MODE_TOOLS,
    MODES,
    PROTOCOL_VERSION,
    SCHEME,
    ProtocolValidationError,
    capability_manifest,
    carrier_protocol_hash,
    checkpoint_commit_eligibility_for_guidance_profile,
    checkpoint_commit_eligibility_manifest,
    get_system_prompt,
    parameter_schema,
    prompt_hash,
    provider_tool_definitions,
    tool_schema_hash,
    validate_tool_call,
)
from tool_modules.checkpoint_relalg.checkpoint_store import (  # noqa: E402
    CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V1,
    CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V2,
    CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V3,
    CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V4,
    CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
    CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1,
)
from tool_modules.checkpoint_relalg import protocol as checkpoint_protocol  # noqa: E402
from tool_modules.checkpoint_relalg.provider_tools import (  # noqa: E402
    NativeToolCallError,
    validate_native_tool_calls,
)


def _comparison() -> dict:
    return {
        "op": "=",
        "left": {"column": "status"},
        "right": {"value": "active"},
    }


def _nested_unary(count: int) -> dict:
    expression: dict = {"column": "name"}
    for _ in range(count):
        expression = {"op": "upper", "args": [expression]}
    return expression


def _nested_not(count: int) -> dict:
    predicate = _comparison()
    for _ in range(count):
        predicate = {"op": "not", "arg": predicate}
    return predicate


class CheckpointRelalgProtocolTests(unittest.TestCase):
    def test_version_scheme_and_strict_mode_surfaces(self):
        self.assertEqual(PROTOCOL_VERSION, "checkpoint-relalg-v1")
        self.assertEqual(SCHEME, "checkpoint-relalg")
        self.assertEqual(MODES, ("direct", "atomic", "hybrid"))
        self.assertEqual(len(ATOMIC_TOOLS), 9)
        self.assertIn("execute_sql", MODE_TOOLS["direct"])
        self.assertNotIn("execute_sql", MODE_TOOLS["atomic"])
        self.assertTrue(set(ATOMIC_TOOLS).isdisjoint(MODE_TOOLS["direct"]))
        self.assertEqual(
            set(MODE_TOOLS["hybrid"]),
            set(MODE_TOOLS["direct"]) | set(MODE_TOOLS["atomic"]),
        )
        with self.assertRaisesRegex(ProtocolValidationError, "unavailable"):
            validate_tool_call("direct", "filter_rows", {"table": "t", "conditions": _comparison()})
        with self.assertRaisesRegex(ProtocolValidationError, "unavailable"):
            validate_tool_call("atomic", "execute_sql", {"sql": "SELECT 1"})

    def test_provider_surface_is_closed_and_manifest_matches(self):
        for mode in MODES:
            tools = provider_tool_definitions(mode)
            names = [item["function"]["name"] for item in tools]
            self.assertEqual(names, list(MODE_TOOLS[mode]))
            self.assertEqual(capability_manifest(mode)["tools"], names)
            self.assertEqual(
                capability_manifest(mode)["tool_schema_sha256"],
                tool_schema_hash(mode),
            )
            for tool in tools:
                parameters = tool["function"]["parameters"]
                self.assertFalse(parameters["additionalProperties"])
                self.assertLessEqual(len(tool["function"]["description"]), 100)

    def test_semantic_milestone_v5_and_v6_bind_to_fixed_eligibility(self):
        prompt = get_system_prompt(
            "atomic",
            teacher=True,
            carrier=CARRIER_TEXT_JSON,
            checkpoint_guidance_profile=(
                CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V5
            ),
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
        )
        self.assertIn("TEACHER SEMANTIC MILESTONE V5 CHECKPOINT GUIDANCE", prompt)
        self.assertIn("The first checkpoint requires two counted producers", prompt)
        self.assertIn("Every later checkpoint requires three counted producers", prompt)
        self.assertTrue(
            prompt.endswith(
                "At most eight commits are available; restore is optional."
            )
        )
        self.assertEqual(
            prompt_hash(
                "atomic",
                teacher=True,
                carrier=CARRIER_TEXT_JSON,
                checkpoint_guidance_profile=(
                    CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V5
                ),
                atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
            ),
            "b3f8ed39e0a03be67d99c1a298f3cf110e0c8f96a0e25170562104012ebe26eb",
        )

        v6_prompt = get_system_prompt(
            "atomic",
            teacher=True,
            carrier=CARRIER_TEXT_JSON,
            checkpoint_guidance_profile=(
                CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V6
            ),
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
        )
        self.assertIn("TEACHER SEMANTIC MILESTONE V6 CHECKPOINT GUIDANCE", v6_prompt)
        self.assertIn("has no accepted checkpoint", v6_prompt)
        self.assertIn("the current-phase quota is two counted producers", v6_prompt)
        self.assertIn("the quota is three counted producers", v6_prompt)
        self.assertIn("the next action MUST be commit_checkpoint", v6_prompt)
        self.assertGreater(
            v6_prompt.rfind("MILESTONE V6 TURN CHECK"),
            v6_prompt.rfind("EXACT TOOL SCHEMAS FOR ATOMIC MODE"),
        )
        self.assertTrue(
            v6_prompt.endswith(
                "At most eight commits are available; restore is optional."
            )
        )
        self.assertNotEqual(v6_prompt, prompt)

        for profile in (
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V5,
            CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V6,
        ):
            self.assertEqual(
                checkpoint_commit_eligibility_for_guidance_profile(profile),
                CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1,
            )
        policy = CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1
        self.assertEqual(
            checkpoint_commit_eligibility_manifest(policy),
            {
                "policy": CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1,
                "first_commit_min_milestone_producers": 2,
                "later_commit_min_milestone_producers": 3,
                "milestone_producer_tools": [
                    "filter_rows",
                    "group_aggregate",
                    "join",
                    "set_operation",
                ],
                "requires_new_active_artifacts_at_least_quota": True,
            },
        )
        self.assertIsNone(
            checkpoint_commit_eligibility_manifest(
                CHECKPOINT_COMMIT_ELIGIBILITY_NONE
            )
        )
        for old_profile in CHECKPOINT_GUIDANCE_PROFILES:
            if old_profile in {
                CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET,
                CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET_V2,
                CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET_V3,
                CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET_V4,
                CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V5,
                CHECKPOINT_GUIDANCE_PROFILE_SEMANTIC_MILESTONE_V6,
            }:
                continue
            self.assertEqual(
                checkpoint_commit_eligibility_for_guidance_profile(old_profile),
                CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
                old_profile,
            )

    def test_initial_target_profile_separates_simple_and_hard_bootstrap(self):
        prompt = get_system_prompt(
            "atomic",
            teacher=True,
            carrier=CARRIER_TEXT_JSON,
            checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET,
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
        )
        self.assertIn("TEACHER INITIAL-TARGET V1 CHECKPOINT GUIDANCE", prompt)
        self.assertIn("simple task", prompt)
        self.assertIn("zero initial checkpoint", prompt)
        self.assertIn("For a hard task only, the first action MUST", prompt)
        self.assertIn("two or three distinct semantic work targets", prompt)
        self.assertGreater(
            prompt.rfind("INITIAL-TARGET V1 TURN CHECK"),
            prompt.rfind("EXACT TOOL SCHEMAS FOR ATOMIC MODE"),
        )
        policy = checkpoint_commit_eligibility_for_guidance_profile(
            CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET
        )
        self.assertEqual(policy, CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V1)
        manifest = checkpoint_commit_eligibility_manifest(policy)
        self.assertEqual(manifest["first_commit_min_milestone_producers"], 2)
        self.assertEqual(manifest["later_commit_min_milestone_producers"], 3)
        self.assertIs(manifest["allows_initial_target_checkpoint"], True)
        self.assertIs(
            manifest["initial_target_checkpoint_must_be_first_action"], True
        )
        self.assertEqual(manifest["initial_target_min_targets"], 2)
        self.assertIs(
            manifest["initial_target_checkpoint_counts_as_milestone_commit"],
            False,
        )
        self.assertNotEqual(
            carrier_protocol_hash(
                "atomic",
                CARRIER_TEXT_JSON,
                ATOMIC_OPERATOR_PROFILE_SEMANTIC,
                policy,
            ),
            carrier_protocol_hash(
                "atomic",
                CARRIER_TEXT_JSON,
                ATOMIC_OPERATOR_PROFILE_SEMANTIC,
                CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1,
            ),
        )

    def test_initial_target_v2_binds_ordered_transition_state_machine(self):
        prompt = get_system_prompt(
            "atomic",
            teacher=True,
            carrier=CARRIER_TEXT_JSON,
            checkpoint_guidance_profile=(
                CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET_V2
            ),
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
        )
        self.assertIn("TEACHER ORDERED-TARGET V2 CHECKPOINT GUIDANCE", prompt)
        self.assertIn("the first item is the active target", prompt)
        self.assertIn("TARGET TRANSITION REQUIRED", prompt)
        self.assertGreater(
            prompt.rfind("ORDERED-TARGET V2 TURN CHECK"),
            prompt.rfind("EXACT TOOL SCHEMAS FOR ATOMIC MODE"),
        )
        policy = checkpoint_commit_eligibility_for_guidance_profile(
            CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET_V2
        )
        self.assertEqual(policy, CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V2)
        manifest = checkpoint_commit_eligibility_manifest(policy)
        self.assertIs(manifest["ordered_target_lifecycle"], True)
        self.assertEqual(manifest["active_target_position"], 0)
        self.assertIs(
            manifest["transition_required_when_quota_met_and_targets_remain"],
            True,
        )
        self.assertEqual(
            manifest["blocked_while_transition_required"],
            ["answer", "filter_rows", "group_aggregate", "join", "set_operation"],
        )
        self.assertIs(manifest["next_targets_must_remove_active_target"], True)
        self.assertIs(
            manifest["next_targets_must_retain_exact_remaining_target"], True
        )
        self.assertNotEqual(
            carrier_protocol_hash(
                "atomic",
                CARRIER_TEXT_JSON,
                ATOMIC_OPERATOR_PROFILE_SEMANTIC,
                policy,
            ),
            carrier_protocol_hash(
                "atomic",
                CARRIER_TEXT_JSON,
                ATOMIC_OPERATOR_PROFILE_SEMANTIC,
                CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V1,
            ),
        )

    def test_initial_target_v3_allows_only_preproduction_perception_bootstrap(self):
        prompt = get_system_prompt(
            "atomic",
            teacher=True,
            carrier=CARRIER_TEXT_JSON,
            checkpoint_guidance_profile=(
                CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET_V3
            ),
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
        )
        self.assertIn("PERCEPTION-BOOTSTRAP ORDERED-TARGET V3", prompt)
        self.assertIn("describe_table, inspect_column, or read_rows", prompt)
        self.assertGreater(
            prompt.rfind("PERCEPTION-BOOTSTRAP V3 TURN CHECK"),
            prompt.rfind("EXACT TOOL SCHEMAS FOR ATOMIC MODE"),
        )
        policy = checkpoint_commit_eligibility_for_guidance_profile(
            CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET_V3
        )
        self.assertEqual(policy, CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V3)
        manifest = checkpoint_commit_eligibility_manifest(policy)
        self.assertIs(manifest["initial_target_checkpoint_must_be_first_action"], False)
        self.assertIs(
            manifest["allows_perception_before_initial_target_checkpoint"], True
        )
        self.assertEqual(
            manifest["pre_bootstrap_perception_tools"],
            ["describe_table", "inspect_column", "read_rows"],
        )
        self.assertIs(
            manifest["initial_target_checkpoint_must_precede_relation_artifacts"],
            True,
        )
        self.assertNotEqual(
            carrier_protocol_hash(
                "atomic",
                CARRIER_TEXT_JSON,
                ATOMIC_OPERATOR_PROFILE_SEMANTIC,
                policy,
            ),
            carrier_protocol_hash(
                "atomic",
                CARRIER_TEXT_JSON,
                ATOMIC_OPERATOR_PROFILE_SEMANTIC,
                CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V2,
            ),
        )

    def test_initial_target_v4_keeps_bootstrap_open_after_rejected_calls(self):
        prompt = get_system_prompt(
            "atomic",
            teacher=True,
            carrier=CARRIER_TEXT_JSON,
            checkpoint_guidance_profile=(
                CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET_V4
            ),
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
        )
        self.assertIn("ERROR-TOLERANT PERCEPTION-BOOTSTRAP V4", prompt)
        self.assertIn("rejected state-preserving call", prompt)
        self.assertGreater(
            prompt.rfind("ERROR-TOLERANT BOOTSTRAP V4 TURN CHECK"),
            prompt.rfind("EXACT TOOL SCHEMAS FOR ATOMIC MODE"),
        )
        policy = checkpoint_commit_eligibility_for_guidance_profile(
            CHECKPOINT_GUIDANCE_PROFILE_INITIAL_TARGET_V4
        )
        self.assertEqual(policy, CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V4)
        manifest = checkpoint_commit_eligibility_manifest(policy)
        self.assertIs(
            manifest["state_preserving_failures_keep_bootstrap_window_open"],
            True,
        )
        self.assertNotEqual(
            carrier_protocol_hash(
                "atomic",
                CARRIER_TEXT_JSON,
                ATOMIC_OPERATOR_PROFILE_SEMANTIC,
                policy,
            ),
            carrier_protocol_hash(
                "atomic",
                CARRIER_TEXT_JSON,
                ATOMIC_OPERATOR_PROFILE_SEMANTIC,
                CHECKPOINT_COMMIT_ELIGIBILITY_INITIAL_TARGET_V3,
            ),
        )

    def test_model_choice_and_disabled_profiles_leave_timing_to_the_model(self):
        model_choice = get_system_prompt(
            "atomic",
            teacher=True,
            carrier=CARRIER_TEXT_JSON,
            checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE,
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
        )
        disabled = get_system_prompt(
            "atomic",
            teacher=True,
            carrier=CARRIER_TEXT_JSON,
            checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_DISABLED,
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
        )
        self.assertIn("You decide whether and when commit_checkpoint is useful", model_choice)
        self.assertIn("There is no producer quota", model_choice)
        self.assertIn("restore_checkpoint is deprecated and disabled", model_choice)
        self.assertIn("Both commit_checkpoint and restore_checkpoint are disabled", disabled)
        self.assertGreater(
            model_choice.rfind("MODEL-CHOICE TURN CHECK"),
            model_choice.rfind("EXACT TOOL SCHEMAS FOR ATOMIC MODE"),
        )
        self.assertGreater(
            disabled.rfind("NO-CHECKPOINT TURN CHECK"),
            disabled.rfind("EXACT TOOL SCHEMAS FOR ATOMIC MODE"),
        )
        self.assertEqual(
            checkpoint_commit_eligibility_for_guidance_profile(
                CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE
            ),
            CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
        )
        self.assertEqual(
            checkpoint_commit_eligibility_for_guidance_profile(
                CHECKPOINT_GUIDANCE_PROFILE_DISABLED
            ),
            CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
        )
        self.assertNotEqual(
            prompt_hash(
                "atomic",
                teacher=True,
                carrier=CARRIER_TEXT_JSON,
                checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE,
                atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
            ),
            prompt_hash(
                "atomic",
                teacher=True,
                carrier=CARRIER_TEXT_JSON,
                checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_DISABLED,
                atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
            ),
        )
        with self.assertRaises(ValueError):
            get_system_prompt(
                "direct",
                teacher=True,
                checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE,
            )

    def test_model_choice_v2_requires_model_selected_stage_commit_without_harness_gate(self):
        prompt = get_system_prompt(
            "atomic",
            teacher=True,
            carrier=CARRIER_TEXT_JSON,
            checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE_V2,
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
        )
        self.assertIn("decide for yourself whether the task is single-stage or multi-stage", prompt)
        self.assertIn("your next action must be commit_checkpoint", prompt)
        self.assertIn("The Harness supplies no producer quota or trigger", prompt)
        self.assertIn("Never call restore_checkpoint", prompt)
        self.assertGreater(
            prompt.rfind("MODEL-CHOICE STAGE DECISION V2"),
            prompt.rfind("EXACT TOOL SCHEMAS FOR ATOMIC MODE"),
        )
        self.assertEqual(
            checkpoint_commit_eligibility_for_guidance_profile(
                CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE_V2
            ),
            CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
        )
        self.assertNotEqual(
            prompt_hash(
                "atomic",
                teacher=True,
                carrier=CARRIER_TEXT_JSON,
                checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE,
                atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
            ),
            prompt_hash(
                "atomic",
                teacher=True,
                carrier=CARRIER_TEXT_JSON,
                checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE_V2,
                atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
            ),
        )

    def test_model_choice_v3_requires_model_verified_stage_without_harness_gate(self):
        prompt = get_system_prompt(
            "atomic",
            teacher=True,
            carrier=CARRIER_TEXT_JSON,
            checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE_V3,
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
        )
        self.assertIn("decide for yourself whether the task is single-stage or multi-stage", prompt)
        self.assertIn("successfully call read_rows or inspect_column on that exact artifact", prompt)
        self.assertIn("a producer's metadata alone is not this verification", prompt)
        self.assertIn("If any check fails, continue or answer directly", prompt)
        self.assertIn("The Harness supplies no producer quota or trigger", prompt)
        self.assertIn("Never call restore_checkpoint", prompt)
        self.assertGreater(
            prompt.rfind("MODEL-CHOICE VERIFIED STAGE DECISION V3"),
            prompt.rfind("EXACT TOOL SCHEMAS FOR ATOMIC MODE"),
        )
        self.assertEqual(
            checkpoint_commit_eligibility_for_guidance_profile(
                CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE_V3
            ),
            CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
        )
        self.assertNotEqual(
            prompt_hash(
                "atomic",
                teacher=True,
                carrier=CARRIER_TEXT_JSON,
                checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE_V2,
                atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
            ),
            prompt_hash(
                "atomic",
                teacher=True,
                carrier=CARRIER_TEXT_JSON,
                checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE_V3,
                atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
            ),
        )

    def test_model_choice_v4_uses_only_narrow_model_selected_vetoes(self):
        prompt = get_system_prompt(
            "atomic",
            teacher=True,
            carrier=CARRIER_TEXT_JSON,
            checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE_V4,
            atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
        )
        self.assertIn("decide for yourself whether the task is single-stage or multi-stage", prompt)
        self.assertIn("there is no mandatory extra read or inspect before every commit", prompt)
        self.assertIn("its correction is not yet demonstrated", prompt)
        self.assertIn("only one remaining relational operator", prompt)
        self.assertIn("they do not define a producer quota", prompt)
        self.assertIn("Never call restore_checkpoint", prompt)
        self.assertGreater(
            prompt.rfind("MODEL-CHOICE STAGE DECISION V4"),
            prompt.rfind("EXACT TOOL SCHEMAS FOR ATOMIC MODE"),
        )
        self.assertEqual(
            checkpoint_commit_eligibility_for_guidance_profile(
                CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE_V4
            ),
            CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
        )
        self.assertNotEqual(
            prompt_hash(
                "atomic",
                teacher=True,
                carrier=CARRIER_TEXT_JSON,
                checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE_V3,
                atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
            ),
            prompt_hash(
                "atomic",
                teacher=True,
                carrier=CARRIER_TEXT_JSON,
                checkpoint_guidance_profile=CHECKPOINT_GUIDANCE_PROFILE_MODEL_CHOICE_V4,
                atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
            ),
        )

    def test_commit_eligibility_changes_protocol_identity_not_tool_schema(self):
        for carrier in (CARRIER_NATIVE_TOOL_CALLS, CARRIER_TEXT_JSON):
            baseline_hash = carrier_protocol_hash(
                "atomic",
                carrier,
                ATOMIC_OPERATOR_PROFILE_SEMANTIC,
                CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
            )
            v5_hash = carrier_protocol_hash(
                "atomic",
                carrier,
                ATOMIC_OPERATOR_PROFILE_SEMANTIC,
                CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1,
            )
            self.assertNotEqual(v5_hash, baseline_hash)

            baseline = capability_manifest(
                "atomic",
                carrier=carrier,
                atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
                checkpoint_commit_eligibility_policy=(
                    CHECKPOINT_COMMIT_ELIGIBILITY_NONE
                ),
            )
            v5 = capability_manifest(
                "atomic",
                carrier=carrier,
                atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_SEMANTIC,
                checkpoint_commit_eligibility_policy=(
                    CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1
                ),
            )
            self.assertNotEqual(v5, baseline)
            self.assertNotIn("checkpoint_commit_eligibility", baseline)
            self.assertEqual(
                v5["checkpoint_commit_eligibility"],
                checkpoint_commit_eligibility_manifest(
                    CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1
                ),
            )
            for invariant in (
                "tools",
                "tool_capabilities",
                "tool_schema_sha256",
                "student_prompt_sha256",
            ):
                self.assertEqual(v5[invariant], baseline[invariant], invariant)
            self.assertEqual(
                v5["tool_schema_sha256"],
                tool_schema_hash("atomic", ATOMIC_OPERATOR_PROFILE_SEMANTIC),
            )

    def test_every_declared_object_shape_is_closed(self):
        def walk(value):
            if isinstance(value, dict):
                if value.get("type") == "object":
                    self.assertIs(value.get("additionalProperties"), False)
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        for mode in MODES:
            walk(provider_tool_definitions(mode))

    def test_expression_and_predicate_use_closed_finite_shared_refs(self):
        def refs(value):
            found = set()
            if isinstance(value, dict):
                ref = value.get("$ref")
                if isinstance(ref, str) and ref.startswith("#/$defs/"):
                    found.add(ref.removeprefix("#/$defs/"))
                for child in value.values():
                    found.update(refs(child))
            elif isinstance(value, list):
                for child in value:
                    found.update(refs(child))
            return found

        for tool in ("read_rows", "filter_rows", "project"):
            schema = parameter_schema(tool)
            self.assertIn("$defs", schema)
            self.assertTrue(refs(schema))
            self.assertTrue(refs(schema).issubset(schema["$defs"]))
            encoded = json.dumps(schema, sort_keys=True)
            self.assertNotIn('"$ref": "#/$defs/expression"', encoded)
            self.assertNotIn('"$ref": "#/$defs/predicate"', encoded)
        self.assertEqual(
            parameter_schema("filter_rows")["properties"]["conditions"],
            {"$ref": "#/$defs/predicate_d1"},
        )
        project_defs = parameter_schema("project")["$defs"]
        self.assertIn("computed_expression_d1", project_defs)
        self.assertIn("expression_d5", project_defs)
        self.assertNotIn("expression_d1", project_defs)

    def test_expression_acceptance_and_frozen_arity(self):
        accepted = [
            ({"value": 1}, "constant_one"),
            ({"op": "add", "args": [{"column": "x"}, {"value": 1}]}, "sum_x"),
            ({"op": "abs", "args": [{"column": "x"}]}, "abs_x"),
            ({"op": "round", "args": [{"column": "x"}]}, "rounded"),
            ({"op": "round", "args": [{"column": "x"}, {"value": 2}]}, "rounded"),
            ({"op": "concat", "args": [{"column": "a"}, {"column": "b"}]}, "joined"),
            ({"op": "coalesce", "args": [{"column": "a"}, {"value": None}]}, "filled"),
            ({"op": "cast", "args": [{"column": "x"}], "to": "REAL"}, "real_x"),
            (
                {
                    "op": "case_when",
                    "branches": [{"when": _comparison(), "then": {"value": 1}}],
                    "else": {"value": 0},
                },
                "flag",
            ),
        ]
        for expression, alias in accepted:
            with self.subTest(expression=expression):
                validate_tool_call(
                    "atomic",
                    "project",
                    {"table": "t", "outputs": [{"expression": expression, "as": alias}]},
                )

        rejected = [
            {"op": "add", "args": [{"column": "x"}]},
            {"op": "abs", "args": [{"column": "x"}, {"value": 1}]},
            {"op": "round", "args": [{"column": "x"}, {"value": 1}, {"value": 2}]},
            {"op": "concat", "args": [{"column": "x"}]},
            {"op": "coalesce", "args": [{"value": 1}] * 9},
            {"op": "cast", "args": [{"column": "x"}], "to": "NUMBER"},
        ]
        for expression in rejected:
            with self.subTest(expression=expression), self.assertRaises(ProtocolValidationError):
                validate_tool_call(
                    "atomic",
                    "project",
                    {"table": "t", "outputs": [{"expression": expression, "as": "bad"}]},
                )
        with self.assertRaises(ProtocolValidationError):
            validate_tool_call(
                "atomic",
                "project",
                {"table": "t", "outputs": [{"expression": {"value": 1}}]},
            )

    def test_predicate_frozen_shapes_and_literal_membership(self):
        predicates = [
            _comparison(),
            {"op": "and", "args": [_comparison(), {"op": "is_not_null", "value": {"column": "x"}}]},
            {"op": "not", "arg": _comparison()},
            {"op": "between", "value": {"column": "x"}, "lower": {"value": 1}, "upper": {"value": 5}},
            {"op": "in", "value": {"column": "x"}, "values": [{"value": 1}, {"value": None}]},
            {"op": "like", "value": {"column": "x"}, "pattern": {"value": "A%"}},
            {"op": "contains", "value": {"column": "x"}, "pattern": {"value": "A"}},
            {"op": "is_null", "value": {"column": "x"}},
        ]
        for predicate in predicates:
            with self.subTest(predicate=predicate):
                validate_tool_call("atomic", "filter_rows", {"table": "t", "conditions": predicate})

        rejected = [
            {"op": "and", "args": [_comparison()]},
            {"op": "not", "args": [_comparison()]},
            {"op": "between", "expression": {"column": "x"}, "lower": {"value": 1}, "upper": {"value": 5}},
            {"op": "in", "value": {"column": "x"}, "values": [{"column": "other"}]},
            {"op": "like", "left": {"column": "x"}, "right": {"value": "A%"}},
        ]
        for predicate in rejected:
            with self.subTest(predicate=predicate), self.assertRaises(ProtocolValidationError):
                validate_tool_call("atomic", "filter_rows", {"table": "t", "conditions": predicate})

    def test_runtime_ast_depth_is_five(self):
        self.assertEqual(MAX_EXPRESSION_DEPTH, 5)
        validate_tool_call(
            "atomic",
            "project",
            {"table": "t", "outputs": [{"expression": _nested_unary(4), "as": "x"}]},
        )
        with self.assertRaisesRegex(ProtocolValidationError, "depth exceeds 5") as raised:
            validate_tool_call(
                "atomic",
                "project",
                {"table": "t", "outputs": [{"expression": _nested_unary(5), "as": "x"}]},
            )
        self.assertEqual(raised.exception.code, "expression_depth_exceeded")

        schema = parameter_schema("project")
        encoded = json.dumps(schema, sort_keys=True)
        self.assertNotIn('"$ref": "#/$defs/expression"', encoded)
        self.assertIn('"$ref": "#/$defs/computed_expression_d1"', encoded)
        self.assertIn("expression_d5", schema["$defs"])
        self.assertNotIn("computed_expression_d5", schema["$defs"])

        valid_arguments = {
            "table": "t",
            "outputs": [{"expression": _nested_unary(4), "as": "x"}],
        }
        checkpoint_protocol._validate_schema(
            valid_arguments,
            schema,
            root=schema,
            path="$.arguments",
        )
        invalid_arguments = {
            "table": "t",
            "outputs": [{"expression": _nested_unary(5), "as": "x"}],
        }
        with self.assertRaises(ProtocolValidationError):
            checkpoint_protocol._validate_schema(
                invalid_arguments,
                schema,
                root=schema,
                path="$.arguments",
            )

        predicate_schema = parameter_schema("filter_rows")
        valid_predicate_arguments = {"table": "t", "conditions": _nested_not(3)}
        validate_tool_call("atomic", "filter_rows", valid_predicate_arguments)
        checkpoint_protocol._validate_schema(
            valid_predicate_arguments,
            predicate_schema,
            root=predicate_schema,
            path="$.arguments",
        )
        invalid_predicate_arguments = {"table": "t", "conditions": _nested_not(4)}
        with self.assertRaises(ProtocolValidationError) as raised_predicate:
            validate_tool_call("atomic", "filter_rows", invalid_predicate_arguments)
        self.assertEqual(raised_predicate.exception.code, "expression_depth_exceeded")
        with self.assertRaises(ProtocolValidationError):
            checkpoint_protocol._validate_schema(
                invalid_predicate_arguments,
                predicate_schema,
                root=predicate_schema,
                path="$.arguments",
            )

    def test_project_literal_is_accepted_by_runtime_and_provider_schema(self):
        arguments = {
            "table": "t",
            "outputs": [{"expression": {"value": 1}, "as": "constant_one"}],
        }
        self.assertEqual(
            validate_tool_call("atomic", "project", arguments),
            arguments,
        )
        schema = parameter_schema("project")
        checkpoint_protocol._validate_schema(
            arguments,
            schema,
            root=schema,
            path="$.arguments",
        )

    def test_unsupported_ast_operator_has_stable_domain_code(self):
        with self.assertRaises(ProtocolValidationError) as raised:
            validate_tool_call(
                "atomic",
                "project",
                {
                    "table": "t",
                    "outputs": [
                        {
                            "expression": {"op": "sqrt", "args": [{"column": "x"}]},
                            "as": "root",
                        }
                    ],
                },
            )
        self.assertEqual(raised.exception.code, "unsupported_expression_operator")

    def test_operator_shapes_and_defaults_are_strict(self):
        validate_tool_call(
            "direct",
            "execute_sql",
            {"sql": "WITH one AS (\n  SELECT 1 AS value\n)\nSELECT value FROM one"},
        )
        validate_tool_call(
            "atomic",
            "sort",
            {"table": "t", "keys": [{"column": "x", "direction": "asc"}]},
        )
        with self.assertRaises(ProtocolValidationError):
            validate_tool_call("atomic", "sort", {"table": "t", "keys": [{"column": "x"}]})
        validate_tool_call(
            "atomic",
            "add_rank",
            {
                "table": "t",
                "order_by": [{"column": "x", "direction": "desc"}],
                "method": "rank",
                "as": "position",
            },
        )
        validate_tool_call(
            "atomic",
            "aggregate",
            {
                "table": "t",
                "group_by": [],
                "metrics": [{"op": "count", "column": "*", "as": "n"}],
            },
        )
        with self.assertRaises(ProtocolValidationError):
            validate_tool_call(
                "atomic",
                "aggregate",
                {"table": "t", "group_by": [], "metrics": [{"op": "count", "as": "n"}]},
            )
        with self.assertRaises(ProtocolValidationError):
            validate_tool_call(
                "atomic",
                "aggregate",
                {
                    "table": "t",
                    "group_by": [],
                    "metrics": [{"op": "count", "column": "*", "distinct": True, "as": "n"}],
                },
            )

    def test_cross_join_and_ordered_offset_contracts_are_in_schema(self):
        validate_tool_call(
            "atomic",
            "join",
            {"left": "a", "right": "b", "type": "cross", "on": []},
        )
        with self.assertRaises(ProtocolValidationError):
            validate_tool_call(
                "atomic",
                "join",
                {"left": "a", "right": "b", "type": "cross", "on": [{"left_column": "x", "op": "=", "right_column": "y"}]},
            )
        with self.assertRaises(ProtocolValidationError):
            validate_tool_call("atomic", "join", {"left": "a", "right": "b", "on": []})
        with self.assertRaises(ProtocolValidationError):
            validate_tool_call("direct", "read_rows", {"table": "a", "offset": 1})
        validate_tool_call(
            "direct",
            "read_rows",
            {"table": "a", "offset": 1, "order_by": [{"column": "x", "direction": "asc"}]},
        )

    def test_provider_and_runtime_share_reserved_output_namespace(self):
        calls = [
            (
                "project",
                {
                    "table": "t",
                    "outputs": [
                        {"expression": {"value": 1}, "as": "__relalg_ordinal"}
                    ],
                },
            ),
            (
                "aggregate",
                {
                    "table": "t",
                    "group_by": [],
                    "metrics": [
                        {"op": "count", "column": "*", "as": "__private"}
                    ],
                },
            ),
            (
                "add_rank",
                {
                    "table": "t",
                    "order_by": [
                        {"column": "x", "direction": "asc", "nulls": "last"}
                    ],
                    "method": "row_number",
                    "as": "__checkpoint_relalg_data_rank",
                },
            ),
        ]
        for tool, arguments in calls:
            with self.subTest(tool=tool):
                with self.assertRaises(ProtocolValidationError) as raised:
                    validate_tool_call("atomic", tool, arguments)
                self.assertEqual(raised.exception.code, "reserved_output_column")
                schema = parameter_schema(tool)
                with self.assertRaises(ProtocolValidationError):
                    checkpoint_protocol._validate_schema(
                        arguments, schema, root=schema, path="$.arguments"
                    )

    def test_checkpoint_shapes_are_bounded(self):
        validate_tool_call(
            "direct",
            "commit_checkpoint",
            {
                "progress_summary": ["Established customer grain."],
                "remaining_uncertainties": [],
                "next_targets": ["Compute customer totals."],
            },
        )
        with self.assertRaises(ProtocolValidationError):
            validate_tool_call(
                "direct",
                "commit_checkpoint",
                {"progress_summary": [], "remaining_uncertainties": [], "next_targets": ["x"]},
            )
        with self.assertRaises(ProtocolValidationError):
            validate_tool_call(
                "direct",
                "restore_checkpoint",
                {"checkpoint_id": "root", "reason": "x", "next_targets": ["a", "b", "c", "d"]},
            )

    def test_native_carrier_requires_exactly_one_call(self):
        call = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "answer", "arguments": json.dumps({"table": "result_1"})},
        }
        action = validate_native_tool_calls("direct", [call])
        self.assertEqual(action["tool"], "answer")
        self.assertEqual(action["arguments"], {"table": "result_1"})
        self.assertEqual(action["tool_call_id"], "call_1")
        for invalid in ([], [call, call], None):
            with self.subTest(invalid=invalid), self.assertRaises(NativeToolCallError):
                validate_native_tool_calls("direct", invalid)

    def test_hashes_are_stable_and_provider_results_are_defensive_copies(self):
        first = tool_schema_hash("atomic")
        tools = provider_tool_definitions("atomic")
        tools[0]["function"]["parameters"]["properties"].clear()
        self.assertEqual(tool_schema_hash("atomic"), first)
        self.assertEqual(tool_schema_hash("atomic"), tool_schema_hash("atomic"))
        self.assertEqual(len(first), 64)
        self.assertNotEqual(tool_schema_hash("direct"), first)
        self.assertNotEqual(prompt_hash("atomic"), prompt_hash("atomic", teacher=True))

    def test_prompts_are_layered_compact_and_keep_evidence_boundary(self):
        direct = get_system_prompt("direct")
        atomic = get_system_prompt("atomic")
        teacher = get_system_prompt("atomic", teacher=True)
        self.assertIn("exactly one native tool call", direct)
        self.assertIn("cannot ground", direct)
        self.assertIn("may guide work", direct)
        self.assertNotIn("Do not write answer values", direct)
        self.assertIn("execute_sql", direct)
        self.assertNotIn("execute_sql", atomic)
        self.assertNotIn("TEACHER CHECKPOINT GUIDANCE", atomic)
        self.assertIn("TEACHER CHECKPOINT GUIDANCE", teacher)
        stress = get_system_prompt(
            "atomic",
            teacher=True,
            checkpoint_guidance_profile="checkpoint-stress-v1",
        )
        self.assertIn("TEACHER CHECKPOINT STRESS GUIDANCE", stress)
        self.assertIn("commit exactly once", stress)
        self.assertIn("restore to the phase-start checkpoint", stress)
        self.assertNotEqual(stress, teacher)
        self.assertNotEqual(
            prompt_hash(
                "atomic",
                teacher=True,
                checkpoint_guidance_profile="checkpoint-stress-v1",
            ),
            prompt_hash("atomic", teacher=True),
        )
        restore_trigger = get_system_prompt(
            "atomic",
            teacher=True,
            checkpoint_guidance_profile="restore-trigger-v1",
        )
        self.assertIn("TEACHER RESTORE-TRIGGER DIAGNOSTIC GUIDANCE", restore_trigger)
        self.assertIn("before a third branch attempt", restore_trigger)
        self.assertIn(
            "same non-carrier type, schema, or execution error occurs twice",
            restore_trigger,
        )
        self.assertIn("never trigger restore", restore_trigger)
        self.assertNotEqual(restore_trigger, stress)
        self.assertNotEqual(
            prompt_hash(
                "atomic",
                teacher=True,
                checkpoint_guidance_profile="restore-trigger-v1",
            ),
            prompt_hash(
                "atomic",
                teacher=True,
                checkpoint_guidance_profile="checkpoint-stress-v1",
            ),
        )
        with self.assertRaisesRegex(ValueError, "teacher-only"):
            get_system_prompt(
                "atomic",
                checkpoint_guidance_profile="restore-trigger-v1",
            )
        restore_target = get_system_prompt(
            "atomic",
            teacher=True,
            checkpoint_guidance_profile="restore-target-v2",
        )
        self.assertIn("TEACHER RESTORE-TARGET V2 DIAGNOSTIC GUIDANCE", restore_target)
        self.assertIn("Never call restore_checkpoint before a successful commit", restore_target)
        self.assertIn("Never restore root", restore_target)
        self.assertIn("A successful action clears", restore_target)
        self.assertNotEqual(restore_target, restore_trigger)
        self.assertNotEqual(
            prompt_hash(
                "atomic",
                teacher=True,
                checkpoint_guidance_profile="restore-target-v2",
            ),
            prompt_hash(
                "atomic",
                teacher=True,
                checkpoint_guidance_profile="restore-trigger-v1",
            ),
        )
        with self.assertRaisesRegex(ValueError, "teacher-only"):
            get_system_prompt(
                "atomic",
                checkpoint_guidance_profile="restore-target-v2",
            )
        restore_probe = get_system_prompt(
            "atomic",
            teacher=True,
            checkpoint_guidance_profile="restore-probe-v3",
        )
        self.assertIn("TEACHER RESTORE-PROBE V3 DIAGNOSTIC GUIDANCE", restore_probe)
        self.assertIn("__restore_probe_missing_column__", restore_probe)
        self.assertIn("perform exactly one restore probe", restore_probe)
        self.assertIn("must never be used as an SFT or RL target", restore_probe)
        self.assertNotEqual(restore_probe, restore_target)
        self.assertNotEqual(
            prompt_hash(
                "atomic",
                teacher=True,
                checkpoint_guidance_profile="restore-probe-v3",
            ),
            prompt_hash(
                "atomic",
                teacher=True,
                checkpoint_guidance_profile="restore-target-v2",
            ),
        )
        with self.assertRaisesRegex(ValueError, "teacher-only"):
            get_system_prompt(
                "atomic",
                checkpoint_guidance_profile="restore-probe-v3",
            )
        semantic_milestone = get_system_prompt(
            "atomic",
            teacher=True,
            checkpoint_guidance_profile="semantic-milestone-v1",
            atomic_operator_profile="semantic-v2",
        )
        self.assertIn(
            "TEACHER SEMANTIC MILESTONE V1 CHECKPOINT GUIDANCE",
            semantic_milestone,
        )
        self.assertIn("before a fifth successful", semantic_milestone)
        self.assertIn("does not require restore_checkpoint", semantic_milestone)
        self.assertIn("never asks for a synthetic error", semantic_milestone)
        self.assertNotEqual(semantic_milestone, teacher)
        self.assertNotEqual(
            prompt_hash(
                "atomic",
                teacher=True,
                checkpoint_guidance_profile="semantic-milestone-v1",
                atomic_operator_profile="semantic-v2",
            ),
            prompt_hash(
                "atomic",
                teacher=True,
                checkpoint_guidance_profile="adaptive-v1",
                atomic_operator_profile="semantic-v2",
            ),
        )
        with self.assertRaisesRegex(ValueError, "requires atomic mode"):
            get_system_prompt(
                "atomic",
                teacher=True,
                checkpoint_guidance_profile="semantic-milestone-v1",
                atomic_operator_profile="micro-v1",
            )
        with self.assertRaisesRegex(ValueError, "requires atomic mode"):
            get_system_prompt(
                "direct",
                teacher=True,
                checkpoint_guidance_profile="semantic-milestone-v1",
            )
        semantic_milestone_v2 = get_system_prompt(
            "atomic",
            teacher=True,
            carrier="text-json",
            checkpoint_guidance_profile="semantic-milestone-v2",
            atomic_operator_profile="semantic-v2",
        )
        self.assertIn(
            "TEACHER SEMANTIC MILESTONE V2 CHECKPOINT GUIDANCE",
            semantic_milestone_v2,
        )
        self.assertTrue(
            semantic_milestone_v2.endswith(
                "If the exact answer artifact is ready, answer instead. Restore is not required."
            )
        )
        self.assertIn("two counted producers", semantic_milestone_v2)
        self.assertIn("next action MUST be commit_checkpoint", semantic_milestone_v2)
        self.assertNotEqual(semantic_milestone_v2, semantic_milestone)
        with self.assertRaisesRegex(ValueError, "requires atomic mode"):
            get_system_prompt(
                "atomic",
                teacher=True,
                checkpoint_guidance_profile="semantic-milestone-v2",
                atomic_operator_profile="micro-v1",
            )
        semantic_milestone_v3 = get_system_prompt(
            "atomic",
            teacher=True,
            carrier="text-json",
            checkpoint_guidance_profile="semantic-milestone-v3",
            atomic_operator_profile="semantic-v2",
        )
        self.assertIn("first inspect CHECKPOINT HISTORY", semantic_milestone_v3)
        self.assertIn("commit_checkpoint is forbidden", semantic_milestone_v3)
        self.assertTrue(
            semantic_milestone_v3.endswith(
                "If the exact answer is ready, answer. Restore is not required."
            )
        )
        self.assertNotEqual(semantic_milestone_v3, semantic_milestone_v2)
        semantic_milestone_v4 = get_system_prompt(
            "atomic",
            teacher=True,
            carrier="text-json",
            checkpoint_guidance_profile="semantic-milestone-v4",
            atomic_operator_profile="semantic-v2",
        )
        self.assertIn("up to the Harness limit of eight", semantic_milestone_v4)
        self.assertIn("substantively new semantic goal", semantic_milestone_v4)
        self.assertIn("restart the count for the new phase", semantic_milestone_v4)
        self.assertTrue(
            semantic_milestone_v4.endswith(
                "At most eight commits are available; restore is not required."
            )
        )
        self.assertNotEqual(semantic_milestone_v4, semantic_milestone_v3)
        for implementation_detail in ("snapshot hash", "reward", "SQLite", "error code", "provider validator"):
            self.assertNotIn(implementation_detail, teacher)
        self.assertLess(len(teacher), 5000)


if __name__ == "__main__":
    unittest.main()
