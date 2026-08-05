#!/usr/bin/env python3
"""Record source schemas and gold outputs needed to design sparse counterfactual DBs."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any


TABLE_PATTERN = re.compile(
    r"\b(?:FROM|JOIN)\s+(?:`([^`]+)`|\"([^\"]+)\"|([A-Za-z_][A-Za-z0-9_]*))",
    re.IGNORECASE,
)


def read_examples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["examples"] if isinstance(payload, dict) else payload


def quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = []
    for example in read_examples(args.examples_json):
        sql = str(example["gold_sql"])
        table_names = sorted(
            {
                next(value for value in match.groups() if value is not None)
                for match in TABLE_PATTERN.finditer(sql)
            },
            key=str.lower,
        )
        connection = sqlite3.connect(str(example["db_path"]))
        try:
            schemas = {}
            for table_name in table_names:
                table_exists = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)",
                    (table_name,),
                ).fetchone()
                if table_exists is None:
                    continue
                actual_name = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)",
                    (table_name,),
                ).fetchone()[0]
                table = quoted(actual_name)
                schemas[actual_name] = {
                    "columns": [
                        {
                            "cid": row[0],
                            "name": row[1],
                            "type": row[2],
                            "notnull": row[3],
                            "default": row[4],
                            "pk": row[5],
                        }
                        for row in connection.execute(f"PRAGMA table_info({table})")
                    ],
                    "row_count": connection.execute(
                        f"SELECT COUNT(*) FROM {table}"
                    ).fetchone()[0],
                    "sample_rows": [
                        list(row)
                        for row in connection.execute(f"SELECT * FROM {table} LIMIT 3")
                    ],
                }
            gold_rows = [list(row) for row in connection.execute(sql).fetchall()]
        finally:
            connection.close()
        output.append(
            {
                "example_index": example["example_index"],
                "db_id": example["db_id"],
                "db_path": example["db_path"],
                "question": example["question"],
                "gold_sql": sql,
                "gold_rows": gold_rows,
                "referenced_tables": schemas,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"tasks": len(output), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
