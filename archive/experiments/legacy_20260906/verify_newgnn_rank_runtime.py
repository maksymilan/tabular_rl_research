#!/usr/bin/env python3
"""Fail-closed readiness audit for SFT2-based Exp8/Exp9 on NewGNN."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


EXPECTED_PACKAGES = {
    "torch": "2.9.0",
    "torchvision": "0.24.0",
    "torchaudio": "2.9.0",
    "transformers": "4.57.6",
    "trl": "0.29.0",
    "accelerate": "1.14.0",
    "peft": "0.19.1",
    "bitsandbytes": "0.50.0",
    "vllm": "0.12.0",
    "datasets": "5.0.0",
}
EXPECTED_MODEL_FILES = {
    "model-00001-of-00004.safetensors": 4_877_660_776,
    "model-00002-of-00004.safetensors": 4_932_751_008,
    "model-00003-of-00004.safetensors": 4_330_865_200,
    "model-00004-of-00004.safetensors": 1_089_994_880,
}
EXPECTED_ADAPTER_SHA256 = (
    "d880e2d7cc3203fdb0d11a7c188d8f607fd297b174eff23f741b6fe73cc3ce6e"
)
EXPECTED_GATE_SHA256 = (
    "5624c75c6ebbca556f1e0946069fdb95e923dc95d2d7692c8d929bedebd9568b"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/dengyan/tabular_rl_outputs"),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/home/dengyan/tabular_rl_project"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("/home/dengyan/models/Qwen2.5-Coder-7B-Instruct"),
    )
    parser.add_argument("--ready-marker", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    runtime = output_root / "rl_runtime_process_gated_v2_20260730"
    eval_runtime = output_root / "eval_runtime_version36_20260728"
    adapter = (
        output_root
        / "checkpoints"
        / "qwen2.5-coder-7b-bird-sft2-batch2-cp560-repeated-only-6400-single-gpu-qlora"
        / "checkpoint-1682"
    )
    gate = (
        output_root
        / "process_gate_v2_20260730"
        / "counterfactual_suite23_fullcopy_candidate_v2"
        / "counterfactual_suite_v2.passed.json"
    )
    examples = (
        runtime
        / "data/rl/bird_sft2_step1682_version26_train_first50_passk8_mixed_outcomes_nonempty.json"
    )
    eval_examples = eval_runtime / "data/eval_inputs/bird_dev_20240627.jsonl"
    eval_indices = (
        eval_runtime
        / "data/eval_inputs/bird_dev_equal_difficulty300_seed20260729.indices.json"
    )

    assert args.model.name == "Qwen2.5-Coder-7B-Instruct", args.model
    for name, expected_size in EXPECTED_MODEL_FILES.items():
        path = args.model / name
        assert path.stat().st_size == expected_size, path
    assert not list(args.model.rglob("*.aria2")), "model snapshot has incomplete aria2 files"

    adapter_config = json.loads((adapter / "adapter_config.json").read_text())
    assert Path(adapter_config["base_model_name_or_path"]).name == args.model.name
    assert sha256(adapter / "adapter_model.safetensors") == EXPECTED_ADAPTER_SHA256
    assert sha256(gate) == EXPECTED_GATE_SHA256

    package_versions = {
        name: importlib.metadata.version(name) for name in EXPECTED_PACKAGES
    }
    assert package_versions == EXPECTED_PACKAGES, package_versions

    payload = json.loads(examples.read_text(encoding="utf-8"))
    rows = payload.get("examples") or payload.get("records")
    assert len(rows) == 23
    missing_train_databases = [row["db_path"] for row in rows if not Path(row["db_path"]).is_file()]
    assert not missing_train_databases, missing_train_databases[:3]

    eval_rows = [json.loads(line) for line in eval_examples.open() if line.strip()]
    selected = set(json.loads(eval_indices.read_text())["indices"])
    assert len(eval_rows) == 1534
    assert len(selected) == 300
    selected_rows = [row for row in eval_rows if int(row["example_index"]) in selected]
    assert len(selected_rows) == 300
    missing_eval_databases = [
        row["db_path"] for row in selected_rows if not Path(row["db_path"]).is_file()
    ]
    assert not missing_eval_databases, missing_eval_databases[:3]

    result = {
        "schema_version": "newgnn-sft2-rank-runtime-ready-v1",
        "base_model": str(args.model),
        "sft2_adapter": str(adapter),
        "sft2_adapter_sha256": EXPECTED_ADAPTER_SHA256,
        "counterfactual_manifest": str(gate),
        "counterfactual_manifest_sha256": EXPECTED_GATE_SHA256,
        "train_records": len(rows),
        "eval_records": len(selected_rows),
        "package_versions": package_versions,
    }
    args.ready_marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.ready_marker.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.ready_marker)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
