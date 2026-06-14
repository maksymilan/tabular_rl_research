#!/usr/bin/env python3
"""Extract small JSON samples from downloaded table QA datasets."""

from __future__ import annotations

import argparse
import ast
import json
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    repo_id: str
    split: str
    input_path: str
    output_name: str
    source_format: str
    description: str
    postprocess: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    order_column: str | None = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _try_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "{[(":
        return value
    try:
        return _json_safe(ast.literal_eval(stripped))
    except (SyntaxError, ValueError):
        return value


def _compact_table_bundle(value: Any, max_tables: int = 3, max_rows: int = 12) -> Any:
    if not isinstance(value, dict):
        return value

    compacted: dict[str, Any] = {}
    table_items = list(value.items())
    for table_name, table_value in table_items[:max_tables]:
        if not isinstance(table_value, dict):
            compacted[table_name] = table_value
            continue

        table_copy = dict(table_value)
        data = table_copy.get("data")
        if isinstance(data, list):
            table_copy["row_count"] = len(data)
            table_copy["data"] = data[:max_rows]
            if len(data) > max_rows:
                table_copy["truncated_rows"] = len(data) - max_rows
        compacted[table_name] = table_copy

    if len(table_items) > max_tables:
        compacted["_omitted_tables"] = [name for name, _ in table_items[max_tables:]]
    return compacted


def _postprocess_tqabench(record: dict[str, Any]) -> dict[str, Any]:
    processed = dict(record)
    raw_table = processed.pop("table", "")
    raw_text = processed.pop("text", "")
    parsed_table = _try_literal(raw_table)
    parsed_text = _try_literal(raw_text)
    processed["table"] = _compact_table_bundle(parsed_table)
    processed["text"] = parsed_text
    processed["raw_field_lengths"] = {
        "table_chars": len(raw_table) if isinstance(raw_table, str) else None,
        "text_chars": len(raw_text) if isinstance(raw_text, str) else None,
    }
    return processed


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "tqabench": DatasetConfig(
        key="tqabench",
        repo_id="table-benchmark/tqabench",
        split="train",
        input_path="tqabench/data/train-*.parquet",
        output_name="tqabench_sample.json",
        source_format="parquet",
        description="70k table QA examples with serialized database-like tables.",
        postprocess=_postprocess_tqabench,
        order_column="original_dataset_id",
    ),
    "fetaqa": DatasetConfig(
        key="fetaqa",
        repo_id="DongfuJiang/FeTaQA",
        split="train",
        input_path="FeTaQA/fetaQA-v1_train.jsonl",
        output_name="fetaqa_sample.json",
        source_format="jsonl",
        description="Free-form table question answering examples.",
    ),
    "tablebench": DatasetConfig(
        key="tablebench",
        repo_id="Multilingual-Multimodal-NLP/TableBench",
        split="TQA_test",
        input_path="TableBench/TableBench.jsonl",
        output_name="tablebench_sample.json",
        source_format="jsonl",
        description="Core TableBench test examples across table reasoning types.",
    ),
    "tablebench_dp": DatasetConfig(
        key="tablebench_dp",
        repo_id="Multilingual-Multimodal-NLP/TableBench",
        split="Instruct_test/DP",
        input_path="TableBench/TableBench_DP.jsonl",
        output_name="tablebench_dp_sample.json",
        source_format="jsonl",
        description="TableBench direct-prompting instruction examples.",
    ),
    "tablebench_scot": DatasetConfig(
        key="tablebench_scot",
        repo_id="Multilingual-Multimodal-NLP/TableBench",
        split="Instruct_test/SCoT",
        input_path="TableBench/TableBench_SCoT.jsonl",
        output_name="tablebench_scot_sample.json",
        source_format="jsonl",
        description="TableBench symbolic chain-of-thought instruction examples.",
    ),
    "tablebench_tcot": DatasetConfig(
        key="tablebench_tcot",
        repo_id="Multilingual-Multimodal-NLP/TableBench",
        split="Instruct_test/TCoT",
        input_path="TableBench/TableBench_TCoT.jsonl",
        output_name="tablebench_tcot_sample.json",
        source_format="jsonl",
        description="TableBench textual chain-of-thought instruction examples.",
    ),
    "tablebench_pot": DatasetConfig(
        key="tablebench_pot",
        repo_id="Multilingual-Multimodal-NLP/TableBench",
        split="Instruct_test/PoT",
        input_path="TableBench/TableBench_PoT.jsonl",
        output_name="tablebench_pot_sample.json",
        source_format="jsonl",
        description="TableBench program-of-thought instruction examples.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract reproducible JSON samples from table QA datasets.")
    parser.add_argument(
        "--dataset",
        default="all",
        help="Dataset key to extract, or 'all'. Available: " + ", ".join(DATASET_CONFIGS),
    )
    parser.add_argument("--num", type=int, default=5, help="Number of records per output file.")
    parser.add_argument("--seed", type=int, default=20260602, help="Seed used by random sampling mode.")
    parser.add_argument(
        "--mode",
        choices=("first", "random"),
        default="first",
        help="Use the first stable rows or deterministic random samples.",
    )
    parser.add_argument("--base-dir", type=Path, default=PROJECT_ROOT / "data", help="Downloaded data directory.")
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "data_sample", help="Sample output directory.")
    return parser.parse_args()


def load_jsonl(path: Path, num: int, mode: str, seed: int) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if mode == "random":
        rng = random.Random(seed)
        rng.shuffle(records)

    return records[:num]


def _sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def load_parquet_with_duckdb(config: DatasetConfig, path: Path, num: int, mode: str, seed: int) -> list[dict[str, Any]]:
    if shutil.which("duckdb") is None:
        raise RuntimeError("duckdb CLI is required to read parquet samples without pyarrow/pandas.")

    source = _sql_string(str(path))
    order_column = config.order_column or "1"
    if mode == "random":
        order_by = f"hash(CAST({order_column} AS VARCHAR) || '{seed}')"
    else:
        order_by = order_column

    sql = f"SELECT * FROM read_parquet({source}) ORDER BY {order_by} LIMIT {num}"
    result = subprocess.run(
        ["duckdb", "-json", "-c", sql],
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def load_records(config: DatasetConfig, base_dir: Path, num: int, mode: str, seed: int) -> list[dict[str, Any]]:
    input_path = base_dir / config.input_path
    if config.source_format == "jsonl":
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input file: {input_path}")
        records = load_jsonl(input_path, num, mode, seed)
    elif config.source_format == "parquet":
        if not list(input_path.parent.glob(input_path.name)):
            raise FileNotFoundError(f"Missing parquet files matching: {input_path}")
        records = load_parquet_with_duckdb(config, input_path, num, mode, seed)
    else:
        raise ValueError(f"Unsupported source format: {config.source_format}")

    if config.postprocess:
        records = [config.postprocess(record) for record in records]
    return records


def write_sample(config: DatasetConfig, records: list[dict[str, Any]], args: argparse.Namespace) -> Path:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.out_dir / config.output_name
    selection = {
        "mode": args.mode,
        "seed": args.seed if args.mode == "random" else None,
        "requested_records": args.num,
        "actual_records": len(records),
    }
    if config.key == "tqabench":
        selection["notes"] = "Tables are compacted for inspection; use raw parquet files for complete rows."

    payload = {
        "dataset": config.key,
        "description": config.description,
        "source": {
            "repo_id": config.repo_id,
            "local_path": str((args.base_dir / config.input_path).relative_to(PROJECT_ROOT)),
            "format": config.source_format,
            "split": config.split,
        },
        "selection": selection,
        "records": records,
    }
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
    return output_path


def main() -> None:
    args = parse_args()
    if args.num < 1:
        raise ValueError("--num must be >= 1")

    if args.dataset == "all":
        keys = list(DATASET_CONFIGS)
    else:
        if args.dataset not in DATASET_CONFIGS:
            raise KeyError(f"Unknown dataset '{args.dataset}'. Available: {', '.join(DATASET_CONFIGS)}")
        keys = [args.dataset]

    for key in keys:
        config = DATASET_CONFIGS[key]
        records = load_records(config, args.base_dir, args.num, args.mode, args.seed)
        output_path = write_sample(config, records, args)
        print(f"wrote {len(records):>2} records: {output_path}")


if __name__ == "__main__":
    main()
