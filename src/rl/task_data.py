"""Framework-neutral task records for table-tool RL experiments.

Records contain the model-visible initial prompt plus hidden harness metadata.  Framework adapters
may serialize them to Parquet, JSONL, or an in-memory dataset, but must never expose ``gold_sql``
to the model.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from executor import Harness
from protocol import SYSTEM_PROMPT, first_user_message
from rollout import db_path, overview


def load_selection(path: Path | None) -> list[int] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as source:
        return [int(json.loads(line)["example_index"]) for line in source if line.strip()]


def load_result_only_task_records(
    project_root: Path,
    *,
    split: str,
    selection: Path | None = None,
    examples_json: Path | None = None,
    limit: int = 0,
    seed: int = 20260710,
) -> list[dict[str, Any]]:
    """Create generic records for the binary terminal-result baseline."""
    if selection is not None and examples_json is not None:
        raise ValueError("selection and examples_json are mutually exclusive")

    source_path = project_root / "data" / "spider_data" / ("train_spider.json" if split == "train" else "dev.json")
    if examples_json is not None:
        if split != "train":
            raise ValueError("examples_json is only valid for the training split")
        payload = json.loads(examples_json.read_text(encoding="utf-8"))
        examples = payload.get("examples") if isinstance(payload, dict) else payload
        if not isinstance(examples, list):
            raise ValueError("examples_json must be a JSON list or an object with an examples list")
        indexed = [(int(example["example_index"]), example) for example in examples]
    else:
        examples = json.loads(source_path.read_text(encoding="utf-8"))
        selected = load_selection(selection)
        if selected is not None:
            if split != "train":
                raise ValueError("selection is only valid for the training split")
            indexed = [(index, examples[index]) for index in selected]
        else:
            indexed = list(enumerate(examples))
            random.Random(seed).shuffle(indexed)

    if limit:
        indexed = indexed[:limit]

    records: list[dict[str, Any]] = []
    catalogs: dict[str, dict[str, Any]] = {}
    for index, example in indexed:
        db_id = example["db_id"]
        catalog = catalogs.get(db_id)
        if catalog is None:
            catalog = overview(Harness(db_path(db_id)))
            catalogs[db_id] = catalog
        records.append(
            {
                "data_source": "spider_table_result_only",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": first_user_message(catalog, example["question"])},
                ],
                "environment": {
                    "dataset_split": split,
                    "example_index": index,
                    "db_id": db_id,
                    "question": example["question"],
                    "gold_sql": example["query"],
                },
            }
        )
    return records
