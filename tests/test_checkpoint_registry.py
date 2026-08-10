from __future__ import annotations

import pytest

from tool_modules.checkpoint_relalg.checkpoint_store import (
    CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
    CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1,
)
from eval.run_tool_scheme import runner_argv
from rl.tool_environment import create_tool_use_env
from sft.generate_tool_scheme_rollouts import generator_argv
from tool_modules.registry import (
    CHECKPOINT_RELALG_TOOL_SCHEME,
    FORWARD_TOOL_SCHEME,
    TOOL_SCHEME_REGISTRY_VERSION,
    build_checkpoint_relalg_tool_scheme,
    build_tool_scheme,
)


def test_checkpoint_scheme_is_forward_but_diagnostic_and_mode_explicit():
    assert TOOL_SCHEME_REGISTRY_VERSION == "tool-scheme-registry-v12"
    assert FORWARD_TOOL_SCHEME == CHECKPOINT_RELALG_TOOL_SCHEME
    with pytest.raises(ValueError, match="requires mode"):
        build_tool_scheme(CHECKPOINT_RELALG_TOOL_SCHEME)
    direct = build_checkpoint_relalg_tool_scheme(mode="direct")
    atomic = build_checkpoint_relalg_tool_scheme(mode="atomic")
    hybrid = build_checkpoint_relalg_tool_scheme(mode="hybrid")
    assert direct.admission_status == "diagnostic-only"
    assert direct.max_batch_calls == 1
    assert "execute_sql" in direct.top_level_tools
    assert "execute_sql" not in atomic.top_level_tools
    assert set(hybrid.top_level_tools) == set(direct.top_level_tools) | set(atomic.top_level_tools)
    assert len({direct.protocol_hash, atomic.protocol_hash, hybrid.protocol_hash}) == 3


def test_semantic_atomic_profile_has_distinct_identity_and_keeps_checkpoints():
    micro = build_checkpoint_relalg_tool_scheme(mode="atomic")
    semantic = build_checkpoint_relalg_tool_scheme(
        mode="atomic",
        atomic_operator_profile="semantic-v2",
    )
    assert semantic.atomic_operator_profile == "semantic-v2"
    assert semantic.protocol_hash != micro.protocol_hash
    assert semantic.tool_schema_hash != micro.tool_schema_hash
    assert "group_aggregate" in semantic.atomic_tools
    assert "rank_select" in semantic.atomic_tools
    assert "sort" not in semantic.top_level_tools
    assert "commit_checkpoint" in semantic.top_level_tools
    assert "restore_checkpoint" in semantic.top_level_tools


def test_semantic_checkpoint_eligibility_changes_only_registry_protocol_identity():
    baseline = build_checkpoint_relalg_tool_scheme(
        mode="atomic",
        atomic_operator_profile="semantic-v2",
        checkpoint_commit_eligibility_policy=CHECKPOINT_COMMIT_ELIGIBILITY_NONE,
    )
    ordinal = build_checkpoint_relalg_tool_scheme(
        mode="atomic",
        atomic_operator_profile="semantic-v2",
        checkpoint_commit_eligibility_policy=(
            CHECKPOINT_COMMIT_ELIGIBILITY_ORDINAL_MILESTONE_V1
        ),
    )

    assert ordinal.protocol_hash != baseline.protocol_hash
    assert ordinal.tool_schema_hash == baseline.tool_schema_hash
    assert ordinal.student_prompt_hash == baseline.student_prompt_hash
    assert ordinal.system_prompt == baseline.system_prompt
    assert ordinal.top_level_tools == baseline.top_level_tools


def test_unified_launchers_route_checkpoint_scheme_to_its_own_runner():
    forwarded = ["--", "--mode", "hybrid", "--result-dir", "/tmp/result", "--n", "1"]
    evaluated = runner_argv(CHECKPOINT_RELALG_TOOL_SCHEME, forwarded)
    generated = generator_argv(CHECKPOINT_RELALG_TOOL_SCHEME, forwarded)
    assert evaluated == generated
    assert evaluated[1].endswith("/tool_modules/checkpoint_relalg/runner.py")
    assert evaluated[-6:] == ["--mode", "hybrid", "--result-dir", "/tmp/result", "--n", "1"]


def test_legacy_rl_environment_rejects_checkpoint_scheme_explicitly():
    with pytest.raises(ValueError, match="diagnostic-only"):
        create_tool_use_env(
            {"db_id": "unused", "question": "unused"},
            tool_scheme=CHECKPOINT_RELALG_TOOL_SCHEME,
        )
