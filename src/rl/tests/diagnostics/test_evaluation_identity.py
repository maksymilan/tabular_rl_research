import json
from pathlib import Path

import pytest

from rl.diagnostics.evaluation_identity import ensure_identity


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    result = tmp_path / "result"
    adapter = tmp_path / "adapter"
    model = tmp_path / "model"
    adapter.mkdir()
    model.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-v1")
    (adapter / "adapter_config.json").write_text("{}\n")
    return result, adapter, model


def _ensure(result: Path, adapter: Path, model: Path) -> dict:
    return ensure_identity(
        result_dir=result,
        adapter=adapter,
        base_model=model,
        served_model="candidate",
        protocol_version="version36",
        protocol_hash="20a8d3b4356d883c",
    )


def test_identity_is_atomic_and_idempotent(tmp_path: Path) -> None:
    result, adapter, model = _inputs(tmp_path)
    first = _ensure(result, adapter, model)
    second = _ensure(result, adapter, model)
    assert first == second
    assert json.loads((result / "evaluation_identity.json").read_text()) == first
    assert not (result / "evaluation_identity.json.tmp").exists()


def test_changed_adapter_is_rejected(tmp_path: Path) -> None:
    result, adapter, model = _inputs(tmp_path)
    _ensure(result, adapter, model)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights-v2")
    with pytest.raises(ValueError, match="evaluation identity mismatch"):
        _ensure(result, adapter, model)


def test_unbound_partial_result_is_rejected(tmp_path: Path) -> None:
    result, adapter, model = _inputs(tmp_path)
    result.mkdir()
    (result / "all.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="refusing to infer provenance"):
        _ensure(result, adapter, model)
