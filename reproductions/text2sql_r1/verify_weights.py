#!/usr/bin/env python3
"""Verify downloaded Hugging Face LFS files against a saved model API response."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--api-manifest", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.api_manifest.read_text())
    failures: list[str] = []
    verified: list[dict[str, object]] = []
    for sibling in metadata["siblings"]:
        lfs = sibling.get("lfs")
        if not lfs:
            continue
        path = args.model_dir / sibling["rfilename"]
        if not path.is_file():
            failures.append(f"missing: {path}")
            continue
        actual_size = path.stat().st_size
        expected_size = int(lfs["size"])
        if actual_size != expected_size:
            failures.append(
                f"size mismatch: {path.name}: {actual_size} != {expected_size}"
            )
            continue
        actual_hash = sha256(path)
        expected_hash = lfs["sha256"]
        if actual_hash != expected_hash:
            failures.append(
                f"sha256 mismatch: {path.name}: {actual_hash} != {expected_hash}"
            )
            continue
        verified.append(
            {"file": path.name, "bytes": actual_size, "sha256": actual_hash}
        )

    print(
        json.dumps(
            {
                "repo_id": metadata["id"],
                "revision": metadata["sha"],
                "verified": verified,
                "failures": failures,
            },
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
