#!/usr/bin/env python3
"""Create or validate immutable model identity for a result directory."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "evaluation-model-identity-v1"
IDENTITY_NAME = "evaluation_identity.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_identity(
    *,
    adapter: Path,
    base_model: Path,
    served_model: str,
    protocol_version: str,
    protocol_hash: str,
) -> dict[str, Any]:
    adapter = adapter.resolve()
    base_model = base_model.resolve()
    weights = adapter / "adapter_model.safetensors"
    config = adapter / "adapter_config.json"
    if not weights.is_file() or not config.is_file():
        raise ValueError(f"adapter is incomplete: {adapter}")
    if not base_model.is_dir():
        raise ValueError(f"base model is missing: {base_model}")
    return {
        "schema_version": SCHEMA_VERSION,
        "adapter_path": str(adapter),
        "adapter_sha256": sha256_file(weights),
        "adapter_config_sha256": sha256_file(config),
        "base_model_path": str(base_model),
        "served_model": served_model,
        "protocol_version": protocol_version,
        "protocol_hash": protocol_hash,
    }


def ensure_identity(
    *,
    result_dir: Path,
    adapter: Path,
    base_model: Path,
    served_model: str,
    protocol_version: str,
    protocol_hash: str,
) -> dict[str, Any]:
    result_dir.mkdir(parents=True, exist_ok=True)
    identity_path = result_dir / IDENTITY_NAME
    expected = expected_identity(
        adapter=adapter,
        base_model=base_model,
        served_model=served_model,
        protocol_version=protocol_version,
        protocol_hash=protocol_hash,
    )
    if identity_path.exists():
        observed = json.loads(identity_path.read_text(encoding="utf-8"))
        if observed != expected:
            raise ValueError(
                "evaluation identity mismatch: "
                f"expected={expected!r} observed={observed!r}"
            )
        return observed
    existing = sorted(path.name for path in result_dir.iterdir())
    if existing:
        raise ValueError(
            "result directory has artifacts but no evaluation identity; "
            f"refusing to infer provenance: {existing[:10]}"
        )
    temporary = identity_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(identity_path)
    return expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--protocol-version", required=True)
    parser.add_argument("--protocol-hash", required=True)
    args = parser.parse_args()
    result = ensure_identity(
        result_dir=args.result_dir,
        adapter=args.adapter,
        base_model=args.base_model,
        served_model=args.served_model,
        protocol_version=args.protocol_version,
        protocol_hash=args.protocol_hash,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
