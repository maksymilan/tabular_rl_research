#!/usr/bin/env python3
"""Add or update one ShareGPT dataset entry without replacing unrelated registrations."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--file-name", required=True)
    args = parser.parse_args()

    registry = args.registry.resolve()
    data = json.loads(registry.read_text(encoding="utf-8")) if registry.exists() else {}
    if not isinstance(data, dict):
        raise ValueError("dataset registry must be a JSON object")
    data[args.name] = {
        "file_name": args.file_name,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }
    tmp = registry.with_suffix(registry.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, registry)
    print(json.dumps({"registry": str(registry), "datasets": len(data), "updated": args.name}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
