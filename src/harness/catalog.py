"""Build the bounded, model-visible catalog for a tool-use episode."""

from __future__ import annotations

from typing import Any


def build_catalog(harness: Any) -> dict:
    """Return table row counts and foreign-key relations without exposing columns.

    Table schemas are acquired causally through ``describe_table``. Keeping them out of
    the opening catalog bounds prompt size by the number of tables rather than columns.
    """
    table_names = [
        name
        for (name,) in harness.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    ]
    tables = [
        {
            "table_name": name,
            "num_rows": harness.conn.execute(
                f'SELECT COUNT(*) FROM "{name}"'
            ).fetchone()[0],
        }
        for name in table_names
    ]
    relations = []
    for name in table_names:
        for relation in harness.conn.execute(f'PRAGMA foreign_key_list("{name}")'):
            relations.append(
                {
                    "from": f"{name}.{relation[3]}",
                    "to": f"{relation[2]}.{relation[4]}",
                }
            )
    return {"tables": tables, "relations": relations}
