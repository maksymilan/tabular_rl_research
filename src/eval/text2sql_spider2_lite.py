#!/usr/bin/env python3
"""Direct-SQL baseline for Spider 2.0-Lite local SQLite tasks.

This is a dataset-adapter baseline, not a trajectory generator. It consumes the common
DatasetTask shape emitted by `src/harness/spider2_adapter.py`, gives the model full SQLite schema
for each local database, and grades only examples with released gold SQL.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "sft"))

from artifacts import ArtifactWriter  # noqa: E402
from executor import Harness  # noqa: E402
from protocol import rows_equal  # noqa: E402
from rollout import ContextOverflowError, chat, overview  # noqa: E402
from spider2_adapter import DEFAULT_ROOT, load_examples, load_local_map, to_task  # noqa: E402
from text2sql import SYSTEM_PROMPT, extract_sql  # noqa: E402


def load_tasks(root: Path, gold_sql_only: bool = True) -> list[dict]:
    local_map = load_local_map(root)
    out = []
    for ex in load_examples(root):
        task = to_task(root, ex, local_map)
        if task.backend != "sqlite" or not task.db_path:
            continue
        if gold_sql_only and not task.gold_sql:
            continue
        out.append(task.to_json())
    return out


def schema_prompt(h: Harness, question: str, external_knowledge=None) -> str:
    table_names = [t["table_name"] for t in overview(h)["tables"]]
    prompt = (
        "DATABASE SCHEMA\n"
        + json.dumps(h.describe_table(table_names), ensure_ascii=False, separators=(",", ":"))
        + "\n\nQUESTION\n"
        + question
    )
    if external_knowledge:
        prompt += "\n\nEXTERNAL KNOWLEDGE FILES\n" + json.dumps(external_knowledge, ensure_ascii=False)
    return prompt


def run_one(task: dict, position: int, base_url: str, model: str, max_tokens: int) -> dict:
    started = time.time()
    h = Harness(task["db_path"])
    h.conn.execute("PRAGMA query_only = ON")
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": schema_prompt(h, task["question"], task.get("external_knowledge"))},
    ]
    record = {
        "example_index": position,
        "dataset_index": task["index"],
        "instance_id": task["instance_id"],
        "db_id": task["db_id"],
        "backend": task["backend"],
        "question": task["question"],
        "gold_sql": task.get("query"),
        "gold_sql_path": task.get("gold_sql_path"),
        "gold_exec_results": task.get("gold_exec_results", []),
        "model_input": messages,
        "correct": False,
        "failure_type": None,
    }
    try:
        output = chat(base_url, model, messages, max_tokens=max_tokens)
    except ContextOverflowError as exc:
        record["failure_type"] = "context_overflow"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = round(time.time() - started, 3)
        return record
    except Exception as exc:  # noqa: BLE001
        record["failure_type"] = "api_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = round(time.time() - started, 3)
        return record

    record["model_output"] = output
    sql = extract_sql(output)
    record["predicted_sql"] = sql
    if sql is None:
        record["failure_type"] = "no_sql"
        record["elapsed_seconds"] = round(time.time() - started, 3)
        return record
    if not task.get("query"):
        record["failure_type"] = "missing_gold_sql"
        record["elapsed_seconds"] = round(time.time() - started, 3)
        return record
    try:
        predicted_rows = h.gold(sql)
        gold_rows = h.gold(task["query"])
    except Exception as exc:  # noqa: BLE001
        record["failure_type"] = "execution_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = round(time.time() - started, 3)
        return record
    record["predicted_row_count"] = len(predicted_rows)
    record["gold_row_count"] = len(gold_rows)
    record["predicted_sample"] = [list(row) for row in predicted_rows[:10]]
    record["gold_sample"] = [list(row) for row in gold_rows[:10]]
    record["correct"] = rows_equal(predicted_rows, gold_rows)
    if not record["correct"]:
        record["failure_type"] = "wrong_result"
    record["elapsed_seconds"] = round(time.time() - started, 3)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=0, help="number of gold-SQL local examples; 0 means all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    tasks = load_tasks(args.root, gold_sql_only=True)
    if args.n:
        tasks = tasks[: args.n]
    writer = ArtifactWriter(str(args.result_dir), {
        "runner": "direct_sql_spider2_lite_local",
        "dataset": "spider2-lite",
        "subset": "local_sqlite_gold_sql",
        "model": args.model,
        "base_url": args.base_url,
        "dev_size": len(tasks),
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "enable_thinking": os.environ.get("EVAL_ENABLE_THINKING"),
        "execution_feedback": False,
        "system_prompt": SYSTEM_PROMPT,
    }, args.resume)
    pending = [(i, task) for i, task in enumerate(tasks) if i not in writer.completed]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, task, i, args.base_url, args.model, args.max_tokens) for i, task in pending]
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            writer.append(record)
            flag = "OK" if record["correct"] else record["failure_type"]
            print(f"[{done}/{len(pending)}] {flag} {record['instance_id']} {record['question'][:70]}", flush=True)
    summary = writer.summarize()
    print(f"EXEC-ACC {summary['correct']}/{summary['total']} ({100 * summary['accuracy']:.1f}%)")
    print(f"-> {writer.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
