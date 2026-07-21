#!/usr/bin/env python3
"""Select tasks whose prior pass@k attempt was absent or transport-incomplete."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_transport_complete(record: dict) -> bool:
    samples = record.get("samples") or []
    expected = int(record.get("n_samples") or 0)
    if not samples or (expected and len(samples) != expected):
        return False
    return all(sample.get("failure_type") != "api_error" for sample in samples)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--prior-results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    source = read_jsonl(args.source)
    prior = read_jsonl(args.prior_results)
    source_ids = [int(row["example_index"]) for row in source]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"duplicate example_index in {args.source}")

    prior_by_id = {int(row["example_index"]): row for row in prior}
    unknown = sorted(set(prior_by_id) - set(source_ids))
    if unknown:
        raise ValueError(f"prior results contain ids outside source: {unknown[:10]}")

    completed_ids = {index for index, row in prior_by_id.items() if is_transport_complete(row)}
    transport_failed_ids = set(prior_by_id) - completed_ids
    unattempted_ids = set(source_ids) - set(prior_by_id)
    retry = [row for row in source if int(row["example_index"]) not in completed_ids]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in retry),
        encoding="utf-8",
    )
    manifest = {
        "source": str(args.source),
        "source_count": len(source),
        "prior_results": str(args.prior_results),
        "prior_record_count": len(prior),
        "transport_complete_count": len(completed_ids),
        "transport_failed_count": len(transport_failed_ids),
        "unattempted_count": len(unattempted_ids),
        "retry_count": len(retry),
        "transport_failed_example_indices": sorted(transport_failed_ids),
        "unattempted_example_indices": sorted(unattempted_ids),
        "output": str(args.out),
        "output_sha256": hashlib.sha256(args.out.read_bytes()).hexdigest(),
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
