from __future__ import annotations

import json

from tool_modules.checkpoint_relalg.protocol import get_system_prompt
from tool_modules.checkpoint_relalg.sft_export import (
    ADMISSION_POLICY_VERSION,
    CARRIER,
    GUIDANCE,
    MODE,
    PROFILE,
    convert_turn,
    episode_rejection_reasons,
    is_successful_target_turn,
)
from tool_modules.registry import (
    CHECKPOINT_RELALG_TOOL_SCHEME,
    build_checkpoint_relalg_tool_scheme,
)


def _record() -> dict:
    scheme = build_checkpoint_relalg_tool_scheme(
        mode=MODE,
        carrier=CARRIER,
        atomic_operator_profile=PROFILE,
    )
    action = {
        "tool": "describe_table",
        "arguments": {"tables": ["orders"]},
        "tool_call_id": None,
    }
    assistant = {
        "role": "assistant",
        "reasoning_content": "Inspect the source schema first.",
        "content": json.dumps(
            {"tool": action["tool"], "arguments": action["arguments"]},
            separators=(",", ":"),
        ),
    }
    return {
        "example_id": "bird_train_00001",
        "task_position": 1,
        "tool_scheme": CHECKPOINT_RELALG_TOOL_SCHEME,
        "mode": MODE,
        "final_mode": MODE,
        "atomic_operator_profile": PROFILE,
        "carrier": CARRIER,
        "checkpoint_guidance_profile": GUIDANCE,
        "protocol_hash": scheme.protocol_hash,
        "student_prompt_sha256": scheme.student_prompt_hash,
        "tool_schema_sha256": scheme.tool_schema_hash,
        "runtime_config": {"max_checkpoints": 0, "max_restores": 0},
        "checkpoint_count": 0,
        "restore_count": 0,
        "denotation_comparison": "bird-set",
        "correct": True,
        "legal": True,
        "strict_artifact_accuracy": False,
        "schema_match": False,
        "sft_export_eligible": False,
        "turns": [
            {
                "model_input": [
                    {"role": "system", "content": "teacher prompt"},
                    {"role": "user", "content": "causal current state"},
                ],
                "assistant_message": assistant,
                "action": action,
                "result": {"status": "success", "step_id": "step_001"},
            }
        ],
    }


def test_bird_set_correct_episode_ignores_strict_schema_and_legacy_marker() -> None:
    record = _record()
    assert record["strict_artifact_accuracy"] is False
    assert record["schema_match"] is False
    assert record["sft_export_eligible"] is False
    assert episode_rejection_reasons(record) == []


def test_identity_and_correctness_remain_hard_gates() -> None:
    record = _record()
    record["correct"] = False
    record["atomic_operator_profile"] = "semantic-v2"
    reasons = episode_rejection_reasons(record)
    assert "not_correct" in reasons
    assert "identity:atomic_operator_profile" in reasons


def test_zero_row_intermediate_is_context_only() -> None:
    turn = {
        "assistant_message": {"role": "assistant"},
        "action": {"tool": "filter_rows", "arguments": {}},
        "result": {"status": "success", "artifact": {"row_count": 0}},
    }
    assert is_successful_target_turn(turn) == (
        False,
        "zero_row_intermediate_artifact",
    )


def test_convert_preserves_native_reasoning_and_replaces_teacher_system(tmp_path) -> None:
    record = _record()
    candidate, index = convert_turn(record, 0, source_result_dir=tmp_path)
    expected_system = get_system_prompt(
        MODE,
        teacher=False,
        carrier=CARRIER,
        checkpoint_guidance_profile=GUIDANCE,
        atomic_operator_profile=PROFILE,
    )
    assert candidate["messages"][0] == {
        "role": "system",
        "content": expected_system,
    }
    assert candidate["messages"][-1] == record["turns"][0]["assistant_message"]
    assert candidate["loss_message_index"] == len(candidate["messages"]) - 1
    assert candidate["metadata"]["admission_policy_version"] == ADMISSION_POLICY_VERSION
    assert index["tool_name"] == "describe_table"
    assert "reasoning_content" not in json.dumps(index)
