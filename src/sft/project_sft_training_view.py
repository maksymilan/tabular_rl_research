#!/usr/bin/env python3
"""Project an audited ShareGPT JSONL into a schema-stable training view.

Audit metadata can legitimately evolve between independently produced data lanes.
Hugging Face Datasets infers one Arrow struct for that unused metadata and may reject
later rows with additional fields. This projection retains only the model-visible
``system`` and ``conversations`` fields while binding the output to the canonical
source and index hashes in a manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from sft_dataset_registry import write_sharegpt_dataset_info


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project(
    input_path: Path,
    index_path: Path,
    output_path: Path,
    dataset_name: str,
) -> dict[str, Any]:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset_name):
        raise ValueError("dataset_name must contain only letters, digits, '.', '_' or '-'")

    index_rows = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    index_ids = [row.get("record_id") for row in index_rows]
    if any(not isinstance(item, str) or not item for item in index_ids):
        raise ValueError("index is missing record_id")
    if len(index_ids) != len(set(index_ids)):
        raise ValueError("index contains duplicate record ids")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    source_ids: list[str] = []
    metadata_key_hist: Counter[str] = Counter()
    try:
        with input_path.open(encoding="utf-8") as source, temporary.open(
            "w", encoding="utf-8"
        ) as target:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                metadata = row.get("metadata")
                if not isinstance(metadata, dict):
                    raise ValueError(f"line {line_number} is missing metadata")
                item_id = metadata.get("record_id")
                if not isinstance(item_id, str) or not item_id:
                    raise ValueError(f"line {line_number} is missing metadata.record_id")
                system = row.get("system")
                conversations = row.get("conversations")
                if not isinstance(system, str) or not system:
                    raise ValueError(f"{item_id}: system must be a non-empty string")
                if not isinstance(conversations, list) or not conversations:
                    raise ValueError(f"{item_id}: conversations must be a non-empty list")
                source_ids.append(item_id)
                metadata_key_hist.update(metadata.keys())
                target.write(
                    json.dumps(
                        {"system": system, "conversations": conversations},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        if source_ids != index_ids:
            raise ValueError("canonical source and index are not in identical record-id order")
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    snippet, registry = write_sharegpt_dataset_info(output_path, dataset_name)
    manifest = {
        "projection": "sharegpt-training-fields-v1",
        "mutation": (
            "none to model-visible training content; audit-only metadata is omitted from "
            "the Arrow training view"
        ),
        "canonical_input": str(input_path),
        "canonical_input_sha256": sha256(input_path),
        "canonical_index": str(index_path),
        "canonical_index_sha256": sha256(index_path),
        "output": str(output_path),
        "output_sha256": sha256(output_path),
        "records": len(source_ids),
        "unique_record_ids": len(set(source_ids)),
        "omitted_metadata_key_occurrences": dict(sorted(metadata_key_hist.items())),
        "retained_fields": ["system", "conversations"],
        "dataset_name": dataset_name,
        "dataset_info_snippet": str(snippet),
        "dataset_info_registry": str(registry),
    }
    output_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    args = parser.parse_args()
    manifest = project(
        args.input.resolve(),
        args.index.resolve(),
        args.out.resolve(),
        args.dataset_name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
