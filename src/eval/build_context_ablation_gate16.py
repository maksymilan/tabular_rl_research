#!/usr/bin/env python3
"""Build the frozen read-heavy target/control cohort for context renderer ablations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


TARGET_IDS = (
    "bird_train_06165",
    "bird_train_06246",
    "bird_train_02868",
    "bird_train_02438",
    "bird_train_00582",
    "bird_train_00074",
    "bird_train_06299",
    "bird_train_01213",
)
CONTROL_IDS = (
    "bird_train_02196",
    "bird_train_02179",
    "bird_train_03768",
    "bird_train_00166",
    "bird_train_02918",
    "bird_train_00808",
    "bird_train_05873",
    "bird_train_05558",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source_rows = load_jsonl(args.source)
    by_id = {row.get("example_id") or row.get("instance_id"): row for row in source_rows}
    ordered_ids = (*TARGET_IDS, *CONTROL_IDS)
    missing = [example_id for example_id in ordered_ids if example_id not in by_id]
    if missing:
        raise ValueError(f"source cohort is missing examples: {missing}")
    selected = [by_id[example_id] for example_id in ordered_ids]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in selected
    )
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "cohort": "context-handle-card-read-heavy-gate16-v1",
        "source": str(args.source),
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "output": str(args.output),
        "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "selection": {
            "targets": list(TARGET_IDS),
            "controls": list(CONTROL_IDS),
        },
        "selection_rule": (
            "Eight version24 failures with the largest final resident row-read payload, plus the "
            "eight correct trajectories with at least nine actions and the largest row-read payload."
        ),
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
