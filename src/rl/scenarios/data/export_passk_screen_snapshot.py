#!/usr/bin/env python3
"""Export a compact, mergeable snapshot from a raw K-sample screening artifact."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rl.data_selection.passk_artifact import (  # noqa: E402
    compact_jsonl_text,
    export_snapshot,
)
from rl.shared.io import atomic_write_text, sha256_file  # noqa: E402


UPSTREAM_IDENTITY_KEYS = (
    "schema_version",
    "status",
    "model",
    "model_checkpoint",
    "protocol_version",
    "protocol_hash",
    "carrier",
    "selection",
    "sources",
    "created_at_utc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--expected-group-size", type=int, default=8)
    parser.add_argument("--protocol-version")
    parser.add_argument("--protocol-hash")
    parser.add_argument("--upstream-manifest", type=Path)
    parser.add_argument("--exported-at")
    return parser.parse_args()


def _upstream_block(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    identity = {key: raw[key] for key in UPSTREAM_IDENTITY_KEYS if key in raw}
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "identity": identity,
    }


def main() -> int:
    args = parse_args()
    upstream = (
        _upstream_block(args.upstream_manifest)
        if args.upstream_manifest is not None
        else None
    )
    rows, manifest = export_snapshot(
        input_path=args.input,
        result_path=args.result,
        source_label=args.source_label,
        expected_group_size=args.expected_group_size,
        protocol_version=args.protocol_version,
        protocol_hash=args.protocol_hash,
        upstream_manifest=upstream,
        exported_at_utc=args.exported_at,
    )
    atomic_write_text(args.output, compact_jsonl_text(rows))
    manifest["output"] = {
        "path": str(args.output),
        "records": len(rows),
        "sha256": sha256_file(args.output),
    }
    atomic_write_text(
        args.manifest_output,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "source_label": manifest["source_label"],
                "records": manifest["records"],
                "correct_count_histogram": manifest["correct_count_histogram"],
                "output_sha256": manifest["output"]["sha256"],
                "manifest": str(args.manifest_output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
