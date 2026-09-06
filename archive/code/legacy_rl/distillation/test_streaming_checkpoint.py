import json

import pytest

from src.rl.distillation.streaming_checkpoint import (
    SCHEMA_VERSION,
    complete_checkpoints,
    jsonl_line_count,
    latest_complete_checkpoint,
    load_checkpoint_state,
    prune_complete_checkpoints,
    truncate_jsonl,
)


def _checkpoint(root, batch):
    path = root / f"checkpoint-batch-{batch:05d}"
    adapter = path / "student_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "optimizer.pt").write_bytes(b"optimizer")
    (path / "checkpoint_state.json").write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "run_identity_sha256": "abc",
                "completed_batches": batch,
                "completed_tasks": batch * 2,
                "optimizer_steps": batch,
                "generator_call_index": batch * 3,
                "updates_lines": batch,
                "tasks_lines": batch * 2,
                "student_adapter_relative_path": "student_adapter",
            }
        )
    )
    (path / "COMPLETE").write_text("complete\n")
    return path


def test_checkpoint_discovery_validation_and_retention(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    first = _checkpoint(checkpoints, 5)
    second = _checkpoint(checkpoints, 10)
    third = _checkpoint(checkpoints, 15)
    incomplete = checkpoints / "checkpoint-batch-00020"
    incomplete.mkdir()

    assert complete_checkpoints(checkpoints) == [first, second, third]
    assert latest_complete_checkpoint(checkpoints) == third
    assert load_checkpoint_state(second)["completed_tasks"] == 20
    assert prune_complete_checkpoints(checkpoints, keep=2) == [first]
    assert complete_checkpoints(checkpoints) == [second, third]
    assert incomplete.is_dir()


def test_jsonl_truncation_is_exact_and_rejects_short_logs(tmp_path):
    path = tmp_path / "tasks.jsonl"
    path.write_text('{"i":0}\n{"i":1}\n{"i":2}\n')
    assert jsonl_line_count(path) == 3
    truncate_jsonl(path, 2)
    assert path.read_text() == '{"i":0}\n{"i":1}\n'
    with pytest.raises(ValueError, match="shorter than checkpoint"):
        truncate_jsonl(path, 3)
