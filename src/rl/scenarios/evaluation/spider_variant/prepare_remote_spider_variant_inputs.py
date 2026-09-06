#!/usr/bin/env python3
"""Rewrite prepared Spider-variant tasks for a remote table_rl asset layout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("data/spider_variants/eval"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--local-root", type=Path, default=Path.cwd())
    parser.add_argument("--remote-spider-root", required=True)
    parser.add_argument("--remote-dk-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    local_root = args.local_root.resolve()
    output_dir = args.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"refusing to overwrite non-empty {output_dir}; pass --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    remote_manifest = {
        "schema_version": "spider-variant-remote-inputs-v1",
        "source_manifest_sha256": sha256(args.input_dir / "manifest.json"),
        "remote_spider_root": args.remote_spider_root,
        "remote_dk_root": args.remote_dk_root,
        "datasets": {},
    }
    for source in sorted(args.input_dir.glob("spider_*_scored.jsonl")):
        rows = []
        for line in source.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            path = Path(row["db_path"]).resolve()
            try:
                relative = path.relative_to(local_root / "data" / "spider_data")
                remote_path = f"{args.remote_spider_root.rstrip('/')}/{relative.as_posix()}"
            except ValueError:
                try:
                    relative = path.relative_to(local_root / "data" / "spider_variants" / "upstream" / "Spider-DK")
                    remote_path = f"{args.remote_dk_root.rstrip('/')}/{relative.as_posix()}"
                except ValueError as exc:
                    raise SystemExit(f"unmapped local db_path: {path}") from exc
            row["db_path"] = remote_path
            rows.append(row)
        target = output_dir / source.name
        if target.exists() and not args.overwrite:
            raise SystemExit(f"refusing to overwrite {target}; pass --overwrite")
        with target.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        remote_manifest["datasets"][source.stem] = {
            "records": len(rows),
            "output": str(target),
            "output_sha256": sha256(target),
            "db_paths": sorted({row["db_path"] for row in rows}),
        }
        print(f"{source.name}: {len(rows)} records -> {target}")
    manifest_path = output_dir / "remote_manifest.json"
    if manifest_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite {manifest_path}; pass --overwrite")
    manifest_path.write_text(json.dumps(remote_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"manifest -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
