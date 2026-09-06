#!/usr/bin/env python3
"""Remap frozen BIRD RL tasks and emit the minimal database transfer manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--files-from", type=Path, required=True)
    parser.add_argument("--remote-db-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--smallest", action="store_true")
    parser.add_argument("--example-index", type=int)
    return parser.parse_args()


def load_payload(path: Path) -> tuple[Any, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples") if isinstance(payload, dict) else payload
    if not isinstance(examples, list):
        raise ValueError("input must be a JSON list or an object containing examples")
    return payload, examples


def main() -> None:
    args = parse_args()
    payload, examples = load_payload(args.input)
    if args.example_index is not None:
        examples = [item for item in examples if int(item["example_index"]) == args.example_index]
        if not examples:
            raise ValueError(f"example_index {args.example_index} is not present in the input")
    if args.smallest:
        examples = sorted(examples, key=lambda item: Path(item["db_path"]).stat().st_size)
    if args.limit:
        examples = examples[: args.limit]

    files: dict[str, Path] = {}
    remapped: list[dict[str, Any]] = []
    for example in examples:
        local_path = Path(example["db_path"]).resolve()
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        relative = Path(local_path.parent.name) / local_path.name
        prior = files.setdefault(relative.as_posix(), local_path)
        if prior != local_path:
            raise ValueError(f"conflicting database path for {relative}")
        record = dict(example)
        record["db_path"] = str(args.remote_db_root / relative)
        remapped.append(record)

    if isinstance(payload, dict):
        output_payload = dict(payload)
        output_payload["examples"] = remapped
        output_payload["count"] = len(remapped)
        output_payload["remote_bundle"] = {
            "source": str(args.input),
            "database_count": len(files),
            "database_bytes": sum(path.stat().st_size for path in files.values()),
        }
    else:
        output_payload = remapped

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.files_from.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    args.files_from.write_text("\n".join(sorted(files)) + "\n", encoding="utf-8")
    print(json.dumps({
        "examples": len(remapped),
        "databases": len(files),
        "bytes": sum(path.stat().st_size for path in files.values()),
        "output_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
