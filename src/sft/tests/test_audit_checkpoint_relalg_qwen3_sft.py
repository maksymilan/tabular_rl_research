from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.sft.audit_checkpoint_relalg_qwen3_sft import audit
from tool_modules.checkpoint_relalg.qwen3_carrier import (
    QWEN3_INLINE_CARRIER_VERSION,
    qwen3_student_prompt_sha256,
    qwen3_student_system_prompt,
    render_qwen3_action,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    canonical = tmp_path / "canonical.jsonl"
    index = tmp_path / "index.jsonl"
    training = tmp_path / "training.jsonl"
    projection_manifest = tmp_path / "projection.json"
    training_manifest = tmp_path / "training.json"
    conversations = [
        {"from": "human", "value": "question"},
        {"from": "gpt", "value": '{"tool":"describe_table","arguments":{"tables":["t"]}}'},
        {"from": "human", "value": "feedback and current state"},
        {
            "from": "gpt",
            "value": render_qwen3_action(
                "The exact resident artifact is ready.",
                '{"tool":"answer","arguments":{"table":"shape_001"}}',
            ),
        },
    ]
    source = {
        "system": qwen3_student_system_prompt(),
        "conversations": conversations,
        "metadata": {"record_id": "r1"},
    }
    _write(canonical, [source])
    _write(index, [{"record_id": "r1"}])
    _write(training, [{"system": source["system"], "conversations": conversations}])
    projection_manifest.write_text(
        json.dumps(
            {
                "projection": "checkpoint-relalg-native-reasoning-to-qwen3-sharegpt-v2",
                "qwen3_inline_carrier_version": QWEN3_INLINE_CARRIER_VERSION,
                "qwen3_student_prompt_sha256": qwen3_student_prompt_sha256(),
            }
        )
    )
    training_manifest.write_text(
        json.dumps(
            {
                "canonical_input_sha256": _sha(canonical),
                "canonical_index_sha256": _sha(index),
                "output_sha256": _sha(training),
                "records": 1,
            }
        )
    )
    return canonical, index, training, projection_manifest, training_manifest


def test_qwen3_sft_audit_passes_exact_carrier_projection(tmp_path: Path) -> None:
    result = audit(*_fixture(tmp_path))
    assert result["passed"] is True
    assert result["historical_json_actions"] == 1
    assert result["inline_think_json_targets"] == 1


def test_qwen3_sft_audit_rejects_provider_system_leak(tmp_path: Path) -> None:
    canonical, index, training, projection_manifest, training_manifest = _fixture(tmp_path)
    row = json.loads(canonical.read_text())
    row["system"] += " provider reasoning_content"
    _write(canonical, [row])
    _write(training, [{"system": row["system"], "conversations": row["conversations"]}])
    manifest = json.loads(training_manifest.read_text())
    manifest.update(
        canonical_input_sha256=_sha(canonical),
        output_sha256=_sha(training),
    )
    training_manifest.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="Qwen3 system prompt differs"):
        audit(canonical, index, training, projection_manifest, training_manifest)
