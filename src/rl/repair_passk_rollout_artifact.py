#!/usr/bin/env python3
"""Build a clean pass@k artifact by replacing infrastructure-contaminated task rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


INFRASTRUCTURE_FAILURE_TYPES = {
    "api_error",
    "context_overflow",
    "generation_oom",
    "incomplete_api_response",
    "provider_carrier_error",
    "task_timeout",
    "transport_error",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def row_infrastructure_failures(row: dict[str, Any]) -> list[str]:
    failures = []
    if row.get("failure_type") in INFRASTRUCTURE_FAILURE_TYPES:
        failures.append(str(row["failure_type"]))
    failures.extend(
        str(sample["failure_type"])
        for sample in row.get("samples") or []
        if sample.get("failure_type") in INFRASTRUCTURE_FAILURE_TYPES
    )
    return failures


def repaired_rows(
    base_rows: list[dict[str, Any]],
    replacement_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    base_ids = [int(row["example_index"]) for row in base_rows]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError("base pass@k artifact contains duplicate example indices")
    replacement_map = {
        int(row["example_index"]): row for row in replacement_rows
    }
    if len(replacement_map) != len(replacement_rows):
        raise ValueError("replacement pass@k artifacts contain duplicate example indices")
    missing = sorted(set(replacement_map) - set(base_ids))
    if missing:
        raise ValueError(f"replacement rows are absent from the base artifact: {missing}")
    contaminated = {
        index: row_infrastructure_failures(row)
        for index, row in replacement_map.items()
        if row_infrastructure_failures(row)
    }
    if contaminated:
        raise ValueError(f"replacement rows still contain infrastructure failures: {contaminated}")
    output = [replacement_map.get(int(row["example_index"]), row) for row in base_rows]
    final_contamination = {
        int(row["example_index"]): row_infrastructure_failures(row)
        for row in output
        if row_infrastructure_failures(row)
    }
    if final_contamination:
        raise ValueError(
            f"repaired artifact still contains infrastructure failures: {final_contamination}"
        )
    return output, sorted(replacement_map)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-all", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--replacement-all", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base_rows = read_jsonl(args.base_all)
    replacement_rows = [
        row for path in args.replacement_all for row in read_jsonl(path)
    ]
    rows, replaced_ids = repaired_rows(base_rows, replacement_rows)
    manifest = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    manifest["operational_repair"] = {
        "schema_version": "passk-operational-row-repair-v1",
        "base_all": {
            "path": str(args.base_all),
            "sha256": hashlib.sha256(args.base_all.read_bytes()).hexdigest(),
        },
        "replacement_artifacts": [
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in args.replacement_all
        ],
        "replaced_example_indices": replaced_ids,
        "infrastructure_failures_remaining": 0,
    }
    pass_keys = sorted({
        int(key)
        for row in rows
        for key in (row.get("pass_at") or {})
    })
    summary = {
        "total": len(rows),
        "unique_example_indices": len({int(row["example_index"]) for row in rows}),
        "replaced_example_indices": replaced_ids,
        "pass_at": {
            str(key): {
                "correct": sum(bool((row.get("pass_at") or {}).get(str(key))) for row in rows),
                "total": len(rows),
            }
            for key in pass_keys
        },
        "infrastructure_failures": 0,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "all.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
