#!/usr/bin/env python3
"""Build the versioned, auditable RL screening-candidate inventory."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from rl.data_selection.screened_pool import build_pool  # noqa: E402
from rl.shared.io import atomic_write_text, sha256_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--current-pointer", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_spec = args.source_spec.resolve()
    output_dir = args.output_dir.resolve()
    manifest = build_pool(
        repo_root=REPO_ROOT,
        source_spec_path=source_spec,
        output_dir=output_dir,
    )
    manifest_path = output_dir / "manifest.json"
    if args.current_pointer is not None:
        pointer_path = args.current_pointer.resolve()
        pointer = {
            "schema_version": "atomic-v26-screened-rl-current-pointer-v1",
            "active_manifest": str(manifest_path.relative_to(REPO_ROOT)),
            "active_manifest_sha256": sha256_file(manifest_path),
            "status": manifest["status"],
            "summary": manifest["summary"],
        }
        atomic_write_text(
            pointer_path,
            json.dumps(pointer, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "summary": manifest["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
