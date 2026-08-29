from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.sft.project_checkpoint_relalg_qwen3_sft import project
from tool_modules.checkpoint_relalg.qwen3_carrier import (
    QWEN3_INLINE_CARRIER_VERSION,
    provider_student_system_prompt,
    qwen3_student_prompt_sha256,
    qwen3_student_system_prompt,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_projection_merges_adjacent_users_and_preserves_targets(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    index = tmp_path / "index.jsonl"
    source_manifest = tmp_path / "source.manifest.json"
    output = tmp_path / "train.jsonl"
    output_index = tmp_path / "train_index.jsonl"
    row = {
        "schema_version": "checkpoint-relalg-native-reasoning-sft-candidate-v1",
        "loss_message_index": 5,
        "messages": [
            {"role": "system", "content": provider_student_system_prompt()},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "reasoning_content": "inspect first",
                "content": '{"tool":"describe_table","arguments":{"table":"t"}}',
            },
            {"role": "user", "content": "feedback"},
            {"role": "user", "content": "current context"},
            {
                "role": "assistant",
                "reasoning_content": "now answer",
                "content": '{"tool":"answer","arguments":{"table":"x"}}',
            },
        ],
        "metadata": {"record_id": "r1", "source_episode_id": "e1"},
    }
    source_item = {"record_id": "r1", "model_input_sha256": "old", "target_visible_content_sha256": "visible"}
    _write_jsonl(source, [row])
    _write_jsonl(index, [source_item])
    source_manifest.write_text(json.dumps({"output_sha256": _sha(source), "index_sha256": _sha(index), "records": 1}))

    result = project(source, index, source_manifest, output, output_index, "dataset")
    projected = json.loads(output.read_text())
    assert projected["system"] == qwen3_student_system_prompt()
    assert "provider reasoning_content" not in projected["system"]
    assert "QWEN3 INLINE ACTION CARRIER" in projected["system"]
    assert [item["from"] for item in projected["conversations"]] == ["human", "gpt", "human", "gpt"]
    assert projected["conversations"][1]["value"] == (
        '{"tool":"describe_table","arguments":{"table":"t"}}'
    )
    assert projected["conversations"][2]["value"] == "feedback\n\ncurrent context"
    assert projected["conversations"][-1]["value"] == (
        '<think>\nnow answer\n</think>\n\n{"tool":"answer","arguments":{"table":"x"}}'
    )
    derived_index = json.loads(output_index.read_text())
    assert derived_index["source_model_input_sha256"] == "old"
    assert derived_index["model_input_sha256"] != "old"
    assert result["merged_adjacent_user_boundaries"] == 1
    assert result["projection"] == "checkpoint-relalg-native-reasoning-to-qwen3-sharegpt-v2"
    assert result["qwen3_inline_carrier_version"] == QWEN3_INLINE_CARRIER_VERSION
    assert result["qwen3_student_prompt_sha256"] == qwen3_student_prompt_sha256()
    assert result["visible_action_json_changed"] == 0
    assert result["assistant_carrier_projection_applied"] is True
    assert result["system_carrier_clause_projection_applied"] is True
    assert result["historical_assistant_reasoning_removed"] == 1
    assert result["historical_assistant_policy"] == "json-action-only-v1"


def test_projection_can_drop_only_invalid_history(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    index = tmp_path / "index.jsonl"
    manifest = tmp_path / "manifest.json"
    out = tmp_path / "out.jsonl"
    out_index = tmp_path / "out_index.jsonl"
    base = {
        "schema_version": "checkpoint-relalg-native-reasoning-sft-candidate-v1",
        "loss_message_index": 4,
        "messages": [
            {"role": "system", "content": provider_student_system_prompt()},
            {"role": "user", "content": "u"},
            {"role": "assistant", "reasoning_content": "r", "content": '{"tool":"x","arguments":{}} trailing'},
            {"role": "user", "content": "u2"},
            {"role": "assistant", "reasoning_content": "r2", "content": '{"tool":"answer","arguments":{}}'},
        ],
        "metadata": {"record_id": "bad-history", "source_episode_id": "episode"},
    }
    _write_jsonl(source, [base])
    _write_jsonl(index, [{"record_id": "bad-history"}])
    manifest.write_text(json.dumps({"output_sha256": _sha(source), "index_sha256": _sha(index), "records": 1}))
    result = project(source, index, manifest, out, out_index, "dataset", True)
    assert result["records"] == 0
    assert result["dropped_invalid_history_records"] == 1
    assert out.read_text() == ""


def test_projection_rejects_source_prompt_identity_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    index = tmp_path / "index.jsonl"
    manifest = tmp_path / "manifest.json"
    row = {
        "schema_version": "checkpoint-relalg-native-reasoning-sft-candidate-v1",
        "loss_message_index": 2,
        "messages": [
            {"role": "system", "content": "provider prompt silently changed"},
            {"role": "user", "content": "question"},
            {
                "role": "assistant",
                "reasoning_content": "answer from grounded state",
                "content": '{"tool":"answer","arguments":{"table":"x"}}',
            },
        ],
        "metadata": {"record_id": "drift", "source_episode_id": "episode"},
    }
    _write_jsonl(source, [row])
    _write_jsonl(index, [{"record_id": "drift"}])
    manifest.write_text(
        json.dumps(
            {"output_sha256": _sha(source), "index_sha256": _sha(index), "records": 1}
        )
    )

    try:
        project(
            source,
            index,
            manifest,
            tmp_path / "out.jsonl",
            tmp_path / "out_index.jsonl",
            "dataset",
        )
    except ValueError as exc:
        assert "source student system prompt identity drifted" in str(exc)
    else:
        raise AssertionError("source prompt drift must fail closed")
