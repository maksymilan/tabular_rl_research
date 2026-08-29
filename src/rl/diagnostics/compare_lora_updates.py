#!/usr/bin/env python3
"""Compare LoRA checkpoints without loading the base model.

The diagnostic reports both raw adapter-parameter movement and movement of the
effective LoRA matrices ``scale * B @ A``.  The latter is invariant to the
common reciprocal rescaling of LoRA factors and can be computed from rank-size
Gram matrices without materializing every dense base-model-shaped matrix.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file


def _config(path: Path) -> dict[str, Any]:
    with (path / "adapter_config.json").open(encoding="utf-8") as source:
        return json.load(source)


def _weights(path: Path) -> dict[str, torch.Tensor]:
    return {
        key: value.float()
        for key, value in load_file(path / "adapter_model.safetensors").items()
    }


def _artifact(path: Path) -> dict[str, str]:
    weights = path / "adapter_model.safetensors"
    digest = hashlib.sha256()
    with weights.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "adapter_sha256": digest.hexdigest()}


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def _safe_cosine(dot: float, left2: float, right2: float) -> float | None:
    denominator = math.sqrt(left2 * right2)
    return dot / denominator if denominator else None


def _adapter_name(key: str) -> str:
    return key.replace(".lora_A.weight", "").replace(".lora_B.weight", "")


def _lora_pairs(weights: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    pairs: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for key, value in weights.items():
        if not key.endswith(".lora_A.weight"):
            continue
        name = _adapter_name(key)
        b_key = key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in weights:
            raise ValueError(f"missing LoRA B tensor for {key}")
        pairs[name] = (value, weights[b_key])
    if not pairs:
        raise ValueError("adapter contains no LoRA A/B tensor pairs")
    return pairs


def _effective_inner(
    left: tuple[torch.Tensor, torch.Tensor],
    right: tuple[torch.Tensor, torch.Tensor],
    left_scale: float,
    right_scale: float,
) -> float:
    left_a, left_b = left
    right_a, right_b = right
    # <B_l A_l, B_r A_r>_F = sum((B_l^T B_r) * (A_l A_r^T)).
    b_gram = left_b.T @ right_b
    a_gram = left_a @ right_a.T
    return float((b_gram * a_gram).sum().item() * left_scale * right_scale)


def compare(reference_path: Path, checkpoint_paths: list[Path]) -> dict[str, Any]:
    reference_config = _config(reference_path)
    reference = _weights(reference_path)
    checkpoints = {str(path): _weights(path) for path in checkpoint_paths}
    configs = {str(path): _config(path) for path in checkpoint_paths}
    reference_keys = set(reference)
    for name, weights in checkpoints.items():
        if set(weights) != reference_keys:
            missing = sorted(reference_keys - set(weights))
            extra = sorted(set(weights) - reference_keys)
            raise ValueError(f"adapter key mismatch for {name}: missing={missing[:3]} extra={extra[:3]}")

    reference_norm2 = sum(float((tensor * tensor).sum().item()) for tensor in reference.values())
    raw: dict[str, dict[str, float | None]] = {}
    updates: dict[str, dict[str, torch.Tensor]] = {}
    for name, weights in checkpoints.items():
        delta = {key: weights[key] - reference[key] for key in reference}
        updates[name] = delta
        delta2 = sum(float((tensor * tensor).sum().item()) for tensor in delta.values())
        checkpoint2 = sum(float((tensor * tensor).sum().item()) for tensor in weights.values())
        raw[name] = {
            "reference_norm": math.sqrt(reference_norm2),
            "checkpoint_norm": math.sqrt(checkpoint2),
            "update_norm": math.sqrt(delta2),
            "update_over_reference": _safe_ratio(math.sqrt(delta2), math.sqrt(reference_norm2)),
        }

    raw_cosines: dict[str, float | None] = {}
    names = list(checkpoints)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            dot = sum(
                float((updates[left][key] * updates[right][key]).sum().item())
                for key in reference
            )
            left2 = raw[left]["update_norm"] ** 2  # type: ignore[operator]
            right2 = raw[right]["update_norm"] ** 2  # type: ignore[operator]
            raw_cosines[f"{left} :: {right}"] = _safe_cosine(dot, left2, right2)

    reference_pairs = _lora_pairs(reference)
    checkpoint_pairs = {name: _lora_pairs(weights) for name, weights in checkpoints.items()}
    module_names = sorted(reference_pairs)
    if any(set(pairs) != set(module_names) for pairs in checkpoint_pairs.values()):
        raise ValueError("LoRA module names do not match")
    reference_scale = float(reference_config["lora_alpha"]) / float(reference_config["r"])
    scales = {
        name: float(configs[name]["lora_alpha"]) / float(configs[name]["r"])
        for name in names
    }

    effective_reference2 = 0.0
    effective_update2 = {name: 0.0 for name in names}
    per_checkpoint_modules: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    effective_update_cross = {
        (left, right): 0.0
        for left_index, left in enumerate(names)
        for right in names[left_index + 1 :]
    }
    for module_name in module_names:
        reference_pair = reference_pairs[module_name]
        rr = _effective_inner(reference_pair, reference_pair, reference_scale, reference_scale)
        effective_reference2 += rr
        update_stats: dict[str, tuple[float, float, float]] = {}
        for name in names:
            checkpoint_pair = checkpoint_pairs[name][module_name]
            cc = _effective_inner(checkpoint_pair, checkpoint_pair, scales[name], scales[name])
            cr = _effective_inner(checkpoint_pair, reference_pair, scales[name], reference_scale)
            update2 = max(0.0, cc + rr - 2.0 * cr)
            effective_update2[name] += update2
            update_stats[name] = (cc, cr, update2)
            per_checkpoint_modules[name].append(
                {
                    "module": module_name,
                    "reference_norm": math.sqrt(max(0.0, rr)),
                    "checkpoint_norm": math.sqrt(max(0.0, cc)),
                    "update_norm": math.sqrt(update2),
                    "update_over_reference": _safe_ratio(
                        math.sqrt(update2), math.sqrt(max(0.0, rr))
                    ),
                }
            )
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                checkpoint_cross = _effective_inner(
                    checkpoint_pairs[left][module_name],
                    checkpoint_pairs[right][module_name],
                    scales[left],
                    scales[right],
                )
                _, left_ref, _ = update_stats[left]
                _, right_ref, _ = update_stats[right]
                # <C_l-R, C_r-R> = <C_l,C_r>-<C_l,R>-<R,C_r>+<R,R>.
                effective_update_cross[(left, right)] += (
                    checkpoint_cross - left_ref - right_ref + rr
                )

    effective = {}
    for name in names:
        update_norm = math.sqrt(max(0.0, effective_update2[name]))
        reference_norm = math.sqrt(max(0.0, effective_reference2))
        effective[name] = {
            "reference_norm": reference_norm,
            "update_norm": update_norm,
            "update_over_reference": _safe_ratio(update_norm, reference_norm),
            "modules": per_checkpoint_modules[name],
        }
    effective_cosines = {
        f"{left} :: {right}": _safe_cosine(
            dot,
            effective_update2[left],
            effective_update2[right],
        )
        for (left, right), dot in effective_update_cross.items()
    }
    return {
        "schema_version": "lora-checkpoint-update-comparison-v1",
        "reference": str(reference_path),
        "reference_artifact": _artifact(reference_path),
        "checkpoints": names,
        "checkpoint_artifacts": {
            name: _artifact(Path(name)) for name in names
        },
        "tensor_count": len(reference),
        "lora_module_count": len(module_names),
        "raw_adapter": raw,
        "raw_update_cosines": raw_cosines,
        "effective_lora": effective,
        "effective_update_cosines": effective_cosines,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path, action="append")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.reference, args.checkpoint)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
