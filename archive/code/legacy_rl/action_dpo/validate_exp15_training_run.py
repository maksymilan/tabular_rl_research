#!/usr/bin/env python3
"""Verify an expanded Exp15 Action-DPO run and freeze a completion audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--verification-audit", required=True, type=Path)
    parser.add_argument("--expected-initial-adapter-sha256", required=True)
    parser.add_argument("--expected-learning-rate", required=True, type=float)
    parser.add_argument("--output-audit", required=True, type=Path)
    args = parser.parse_args()

    manifest_path = args.run_dir / "run_manifest.json"
    metrics_path = args.run_dir / "training_metrics.jsonl"
    reference_path = args.run_dir / "reference_scores.jsonl"
    adapter_path = args.run_dir / "final" / "adapter_model.safetensors"
    required = [manifest_path, metrics_path, reference_path, adapter_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"incomplete Action-DPO run: {missing}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verification = json.loads(args.verification_audit.read_text(encoding="utf-8"))
    dataset_rows = load_jsonl(args.dataset)
    metrics = load_jsonl(metrics_path)
    reference = load_jsonl(reference_path)
    expected = {
        "schema_version": "fixed-prefix-action-dpo-run-v1",
        "dataset_sha256": sha256_file(args.dataset),
        "verification_audit_sha256": sha256_file(args.verification_audit),
        "initial_adapter_sha256": args.expected_initial_adapter_sha256,
        "beta": 0.1,
        "learning_rate": args.expected_learning_rate,
        "epochs": 1,
        "gradient_clip": 1.0,
        "tool_token_only": True,
        "question_balanced_sampling": True,
        "seed": 101,
    }
    for key, value in expected.items():
        actual = manifest.get(key)
        if isinstance(value, float):
            valid = math.isclose(float(actual), value, rel_tol=0.0, abs_tol=1e-15)
        else:
            valid = actual == value
        if not valid:
            raise SystemExit(f"training manifest mismatch for {key}: {actual} != {value}")

    questions = {str(row["question_id"]) for row in dataset_rows}
    if manifest.get("pairs") != len(dataset_rows):
        raise SystemExit("training pair count mismatch")
    if manifest.get("questions") != len(questions):
        raise SystemExit("training question count mismatch")
    if manifest.get("optimizer_steps") != len(metrics) or len(metrics) != len(questions):
        raise SystemExit("training optimizer-step count mismatch")
    if len(reference) != len(dataset_rows):
        raise SystemExit("reference-score count mismatch")
    if verification.get("status") != "passed":
        raise SystemExit("verification audit is not passed")
    if verification.get("verified_pairs") != len(dataset_rows):
        raise SystemExit("verification audit pair count mismatch")

    for metric in metrics:
        for key in ("loss", "policy_margin", "grad_norm"):
            if not math.isfinite(float(metric[key])):
                raise SystemExit(f"non-finite training metric: {key}")
    clip_fraction = sum(bool(row["gradient_clipped"]) for row in metrics) / len(metrics)
    if not math.isclose(
        clip_fraction,
        float(manifest["gradient_clip_fraction"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise SystemExit("gradient clipping fraction mismatch")

    audit = {
        "schema_version": "expanded-exp15-training-completion-audit-v1",
        "status": "passed",
        "run_dir": str(args.run_dir.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_sha256": sha256_file(args.dataset),
        "verification_audit_sha256": sha256_file(args.verification_audit),
        "initial_adapter_sha256": manifest["initial_adapter_sha256"],
        "final_adapter_sha256": sha256_file(adapter_path),
        "pairs": len(dataset_rows),
        "questions": len(questions),
        "optimizer_steps": len(metrics),
        "beta": manifest["beta"],
        "learning_rate": manifest["learning_rate"],
        "epochs": manifest["epochs"],
        "gradient_clip": manifest["gradient_clip"],
        "gradient_clip_fraction": manifest["gradient_clip_fraction"],
        "tool_token_only": manifest["tool_token_only"],
        "question_balanced_sampling": manifest["question_balanced_sampling"],
        "seed": manifest["seed"],
    }
    args.output_audit.parent.mkdir(parents=True, exist_ok=True)
    args.output_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
