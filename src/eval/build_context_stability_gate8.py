#!/usr/bin/env python3
"""Freeze the version39/version49 K=3 context-stability cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SENSITIVE_IDS = (
    "bird_train_05873",
    "bird_train_02918",
    "bird_train_06299",
)
CONTROL_IDS = (
    "bird_train_02196",
    "bird_train_02179",
    "bird_train_03768",
    "bird_train_00166",
    "bird_train_00808",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    source_hash = sha256(args.source)
    if source_hash != source_manifest.get("output_sha256"):
        raise ValueError("source cohort hash does not match its frozen manifest")

    rows = load_jsonl(args.source)
    by_id = {row.get("example_id") or row.get("instance_id"): row for row in rows}
    selected_ids = (*SENSITIVE_IDS, *CONTROL_IDS)
    missing = [example_id for example_id in selected_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"source cohort is missing examples: {missing}")

    selected = [by_id[example_id] for example_id in selected_ids]
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in selected
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")

    manifest = {
        "schema_version": "context-stability-gate8-v1",
        "cohort": "version39-version49-context-stability-k3-gate8",
        "source": str(args.source),
        "source_sha256": source_hash,
        "source_manifest": str(args.source_manifest),
        "source_manifest_sha256": sha256(args.source_manifest),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "selection": {
            "context_sensitive": list(SENSITIVE_IDS),
            "controls": list(CONTROL_IDS),
        },
        "selection_rule": (
            "Three predeclared context-sensitive tasks from the frozen Gate16 "
            "(two prior observation-churn controls and one dormant-branch task), plus "
            "five Gate16 controls that version39 and version49 both solved."
        ),
        "evaluation_rule": {
            "arms": ["version39", "version49"],
            "independent_runs_per_arm": 3,
            "attempts_per_example_per_run": 1,
            "aggregation": "all attempts; no pass@K or verifier selection",
            "denotation_comparison": "bird-set",
            "training_admission": "diagnostic_only_pending_protocol_scale_gate",
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
