import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from rl.scenarios.diagnostics.compare_lora_updates import compare


def _write_adapter(path: Path, a: torch.Tensor, b: torch.Tensor) -> None:
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps({"r": 2, "lora_alpha": 4}), encoding="utf-8"
    )
    save_file(
        {
            "layer.lora_A.weight": a,
            "layer.lora_B.weight": b,
        },
        path / "adapter_model.safetensors",
    )


def test_effective_update_ignores_reciprocal_factor_rescaling(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    equivalent = tmp_path / "equivalent"
    changed = tmp_path / "changed"
    a = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    b = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    _write_adapter(reference, a, b)
    _write_adapter(equivalent, 2.0 * a, 0.5 * b)
    _write_adapter(changed, a, b + 1.0)

    result = compare(reference, [equivalent, changed])

    equivalent_stats = result["effective_lora"][str(equivalent)]
    changed_stats = result["effective_lora"][str(changed)]
    assert equivalent_stats["update_norm"] == 0.0
    assert changed_stats["update_norm"] > 0.0
    assert result["raw_adapter"][str(equivalent)]["update_norm"] > 0.0
    assert result["reference_artifact"]["path"] == str(reference)
    assert len(result["reference_artifact"]["adapter_sha256"]) == 64
    assert result["checkpoint_artifacts"][str(changed)]["path"] == str(changed)
    assert len(
        result["checkpoint_artifacts"][str(changed)]["adapter_sha256"]
    ) == 64
