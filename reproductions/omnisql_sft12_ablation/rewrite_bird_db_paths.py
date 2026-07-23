#!/usr/bin/env python3
"""Rewrite DatasetTask db_path values without changing task identity or order."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--db-root", type=Path, required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for source_index, source_row in enumerate(rows):
            row = dict(source_row)
            if int(row.get("example_index", source_index)) != source_index:
                raise ValueError(f"unexpected example_index at source row {source_index}")
            row["db_path"] = str(
                args.db_root / row["db_id"] / f"{row['db_id']}.sqlite"
            )
            if not Path(row["db_path"]).is_file():
                raise FileNotFoundError(row["db_path"])
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    manifest = {
        "source": str(args.input),
        "source_sha256": sha256(args.input),
        "db_root": str(args.db_root),
        "records": len(rows),
        "output": str(args.out),
        "output_sha256": sha256(args.out),
        "task_content_change": "db_path only",
    }
    args.out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
