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

from catalog import build_catalog
from executor import Harness
from protocol import first_user_message, student_runtime_system_prompt


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def _task_db_path(example: dict[str, Any]) -> str:
    provided = example.get("db_path")
    if provided:
        return str(provided)
    return str(Path("data/spider_data/database") / example["db_id"] / f"{example['db_id']}.sqlite")


def _task_gold_sql(example: dict[str, Any]) -> str | None:
    return example.get("gold_sql") or example.get("query")


def _task_id(example: dict[str, Any], *, split: str, index: int) -> str:
    provided = (
        example.get("example_id")
        or example.get("instance_id")
        or example.get("trajectory_id")
    )
    if provided:
        return str(provided)
    dataset = str(example.get("dataset") or "spider").replace("-", "_")
    return f"{dataset}_{split}_{index:05d}"


def load_selection(path: Path | None) -> list[int] | None:
    if path is None:
        return None
    with path.open(encoding="utf-8") as source:
        return [int(json.loads(line)["example_index"]) for line in source if line.strip()]


def load_rl_task_records(
    project_root: Path,
    *,
    split: str,
    selection: Path | None = None,
    examples_json: Path | None = None,
    limit: int = 0,
    seed: int = 20260710,
    context_mode: str = "rolling-legal-history",
) -> list[dict[str, Any]]:
    """Create generic hidden-label task records shared by both RL reward conditions."""
    if selection is not None and examples_json is not None:
        raise ValueError("selection and examples_json are mutually exclusive")

    source_path = project_root / "data" / "spider_data" / ("train_spider.json" if split == "train" else "dev.json")
    if examples_json is not None:
        if split != "train":
            raise ValueError("examples_json is only valid for the training split")
        if examples_json.suffix == ".jsonl":
            examples = _load_jsonl(examples_json)
        else:
            payload = json.loads(examples_json.read_text(encoding="utf-8"))
            examples = payload.get("examples") if isinstance(payload, dict) else payload
            if not isinstance(examples, list):
                raise ValueError("examples_json must be a JSON list/JSONL or an object with examples")
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
    system_prompt = student_runtime_system_prompt(
        context_mode=context_mode,
        compact=False,
    )
    catalogs: dict[str, dict[str, Any]] = {}
    for index, example in indexed:
        db_id = example["db_id"]
        db_path = _task_db_path(example)
        if not Path(db_path).is_absolute():
            db_path = str(project_root / db_path)
        catalog = catalogs.get(db_id)
        if catalog is None:
            harness = Harness(db_path)
            try:
                catalog = build_catalog(harness)
            finally:
                harness.conn.close()
            catalogs[db_id] = catalog
        records.append(
            {
                "data_source": f"{example.get('dataset', 'spider')}_table_rl",
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": first_user_message(
                        catalog, example["question"], example.get("external_knowledge"),
                    )},
                ],
                "environment": {
                    "dataset_split": split,
                    "example_index": index,
                    "task_id": _task_id(example, split=split, index=index),
                    "db_id": db_id,
                    "db_path": db_path,
                    "question": example["question"],
                    "gold_sql": _task_gold_sql(example),
                    "external_knowledge": example.get("external_knowledge"),
                },
            }
        )
    return records
