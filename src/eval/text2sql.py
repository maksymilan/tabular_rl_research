#!/usr/bin/env python3
"""Zero-shot direct-SQL baseline with complete model I/O and split artifacts."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "harness"))
sys.path.insert(0, os.path.join(ROOT, "src", "sft"))

from artifacts import ArtifactWriter                          # noqa: E402
from executor import Harness                                  # noqa: E402
from protocol import DENOTATION_COMPARISONS, compare_denotations  # noqa: E402
from rollout import (  # noqa: E402
    ContextOverflowError,
    chat,
    load_tasks_json,
    overview,
    task_db_path,
    task_gold_sql,
)

SPIDER = os.path.join(ROOT, "data", "spider_data")
SYSTEM_PROMPT = (
    "You translate natural-language questions into SQLite. Put your final query inside "
    "<answer></answer> tags, e.g. <answer>SELECT ...</answer>. It must be exactly one read-only "
    "SELECT or WITH query. You may reason before the tags; only the query inside them is graded."
)


def extract_sql(text: str) -> str | None:
    # Thinking models reason first, so grade only the query inside <answer></answer> when present;
    # otherwise fall back to scanning the whole reply (also covers non-thinking direct output).
    answer = re.search(r"<answer>(.*?)</answer>", text, re.S | re.I)
    candidate = answer.group(1) if answer else text
    candidate = re.sub(r"<think>.*?</think>", " ", candidate, flags=re.S | re.I)
    cleaned = re.sub(r"```(?:sql)?", "", candidate, flags=re.I).strip()
    match = re.search(r"\b(?:WITH|SELECT)\b.*", cleaned, re.S | re.I)
    if not match:
        return None
    sql = match.group(0)
    if ";" in sql:
        sql = sql.split(";", 1)[0]
    sql = sql.strip()
    return sql if re.match(r"^(WITH|SELECT)\b", sql, re.I) else None


def schema_prompt(h: Harness, question: str, external_knowledge=None) -> str:
    """Render the complete SQLite schema needed by the one-shot SQL baseline."""
    table_names = [t["table_name"] for t in overview(h)["tables"]]
    user_prompt = (
        "DATABASE SCHEMA\n"
        + json.dumps(h.describe_table(table_names), ensure_ascii=False, separators=(",", ":"))
        + "\n\nQUESTION\n"
        + question
    )
    if external_knowledge:
        user_prompt += "\n\nEXTERNAL KNOWLEDGE\n" + json.dumps(external_knowledge, ensure_ascii=False)
    return user_prompt


def execute_predicted_sql(h: Harness, sql: str, timeout_seconds: float) -> list:
    """Execute one model SQL with an explicit SQLite VM deadline.

    Generated Cartesian products can otherwise occupy a BIRD large database indefinitely. Gold SQL
    is evaluated separately without this generated-query guard.
    """
    if timeout_seconds <= 0:
        return h.gold(sql)
    deadline = time.monotonic() + timeout_seconds
    h.conn.set_progress_handler(lambda: 1 if time.monotonic() >= deadline else 0, 10_000)
    try:
        return h.gold(sql)
    finally:
        h.conn.set_progress_handler(None, 0)


def run_one(
    ex: dict,
    example_index: int,
    base_url: str,
    model: str,
    max_tokens: int = 512,
    execution_timeout_seconds: float = 5.0,
    denotation_comparison: str = "strict-multiset",
) -> dict:
    gold_sql = task_gold_sql(ex)
    h = Harness(task_db_path(ex))
    h.conn.execute("PRAGMA query_only = ON")
    # Direct-SQL needs the FULL schema (columns/types/PK/FK) up front — the model cannot probe with
    # tools here. overview()/_catalog is the v2-ctx lazy catalog (names + row counts only), which
    # starves text-to-SQL and forces column hallucination; describe_table gives the complete schema.
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": schema_prompt(h, ex["question"], ex.get("external_knowledge"))},
    ]
    started = time.time()
    record = {
        "example_index": example_index,
        "dataset_index": ex.get("index"),
        "instance_id": ex.get("instance_id"),
        "db_id": ex["db_id"],
        "question": ex["question"],
        "gold_sql": gold_sql,
        "denotation_comparison": denotation_comparison,
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
    if not gold_sql:
        record["failure_type"] = "missing_gold_sql"
        record["elapsed_seconds"] = round(time.time() - started, 3)
        return record
    try:
        predicted_rows = execute_predicted_sql(h, sql, execution_timeout_seconds)
        gold_rows = h.gold(gold_sql)
    except Exception as exc:  # noqa: BLE001
        record["failure_type"] = "execution_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = round(time.time() - started, 3)
        return record
    record["predicted_row_count"] = len(predicted_rows)
    record["gold_row_count"] = len(gold_rows)
    record["predicted_sample"] = [list(row) for row in predicted_rows[:10]]
    record["gold_sample"] = [list(row) for row in gold_rows[:10]]
    record["correct"] = compare_denotations(
        predicted_rows, gold_rows, denotation_comparison
    )
    if not record["correct"]:
        record["failure_type"] = "wrong_result"
    record["elapsed_seconds"] = round(time.time() - started, 3)
    return record


def _run_one_worker(sender, kwargs: dict) -> None:
    try:
        sender.send(run_one(**kwargs))
    except BaseException as exc:  # pragma: no cover - parent preserves the failure for audit.
        sender.send({"worker_error": f"{type(exc).__name__}: {exc}"})
    finally:
        sender.close()


def run_one_bounded(kwargs: dict, task_timeout_seconds: float) -> dict:
    """Hard-bound an entire generated-SQL task, including a stalled model HTTP request."""
    if task_timeout_seconds <= 0:
        return run_one(**kwargs)
    context = mp.get_context("fork" if sys.platform != "win32" else "spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_run_one_worker, args=(sender, kwargs))
    started = time.monotonic()
    process.start()
    sender.close()
    try:
        if receiver.poll(task_timeout_seconds):
            record = receiver.recv()
            if "worker_error" not in record:
                return record
            error = record["worker_error"]
        else:
            error = f"task exceeded {task_timeout_seconds:g}s"
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        receiver.close()
    return {
        "example_index": kwargs["example_index"],
        "dataset_index": kwargs["ex"].get("index"),
        "instance_id": kwargs["ex"].get("instance_id"),
        "db_id": kwargs["ex"]["db_id"],
        "question": kwargs["ex"]["question"],
        "gold_sql": task_gold_sql(kwargs["ex"]),
        "correct": False,
        "failure_type": "task_timeout",
        "error": error,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=1034)
    parser.add_argument("--tasks-json", help="adapter-exported DatasetTask JSON or JSONL; defaults to Spider dev")
    parser.add_argument("--start-index", type=int, default=0,
                        help="first source index to evaluate; useful for bounded resume windows")
    parser.add_argument("--end-index", type=int,
                        help="exclusive source-index bound; defaults to the selected dataset size")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--execution-timeout-seconds", type=float, default=5.0,
                        help="per-query SQLite VM deadline for model-generated SQL; <=0 disables it")
    parser.add_argument(
        "--denotation-comparison", choices=DENOTATION_COMPARISONS,
        default="strict-multiset",
        help="result comparison contract; use bird-set for literature-comparable BIRD EX",
    )
    parser.add_argument("--task-timeout-seconds", type=float, default=0.0,
                        help="hard per-example deadline; records task_timeout and continues when positive")
    args = parser.parse_args()

    if args.tasks_json:
        indexed_dev = list(enumerate(load_tasks_json(args.tasks_json)[:args.n]))
        dataset = os.path.basename(args.tasks_json)
    else:
        with open(os.path.join(SPIDER, "dev.json")) as f:
            indexed_dev = list(enumerate(json.load(f)[:args.n]))
        dataset = "spider_dev"
    if args.start_index < 0:
        parser.error("--start-index must be non-negative")
    end_index = args.end_index if args.end_index is not None else len(indexed_dev)
    if end_index < args.start_index:
        parser.error("--end-index must be at least --start-index")
    indexed_window = [
        (index, ex) for index, ex in indexed_dev
        if args.start_index <= index < end_index
    ]

    writer = ArtifactWriter(args.result_dir, {
        "runner": "direct_sql",
        "dataset": dataset,
        "model": args.model,
        "base_url": args.base_url,
        "dev_size": len(indexed_dev),
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "predicted_sql_execution_timeout_seconds": args.execution_timeout_seconds,
        "denotation_comparison": args.denotation_comparison,
        "task_timeout_seconds": args.task_timeout_seconds,
        "enable_thinking": os.environ.get("EVAL_ENABLE_THINKING"),
        "execution_feedback": False,
        "system_prompt": SYSTEM_PROMPT,
    }, args.resume)
    pending = [(i, ex) for i, ex in indexed_window if i not in writer.completed]

    if args.task_timeout_seconds > 0:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(run_one_bounded, {
                    "ex": ex,
                    "example_index": i,
                    "base_url": args.base_url,
                    "model": args.model,
                    "max_tokens": args.max_tokens,
                    "execution_timeout_seconds": args.execution_timeout_seconds,
                    "denotation_comparison": args.denotation_comparison,
                }, args.task_timeout_seconds)
                for i, ex in pending
            ]
            for position, future in enumerate(as_completed(futures), 1):
                record = future.result()
                writer.append(record)
                flag = "OK" if record["correct"] else record["failure_type"]
                print(f"[{position}/{len(pending)}] {flag} q{record['example_index']} "
                      f"{record['question'][:65]}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    run_one,
                    ex,
                    i,
                    args.base_url,
                    args.model,
                    args.max_tokens,
                    args.execution_timeout_seconds,
                    args.denotation_comparison,
                )
                for i, ex in pending
            ]
            for position, future in enumerate(as_completed(futures), 1):
                record = future.result()
                writer.append(record)
                flag = "OK" if record["correct"] else record["failure_type"]
                print(f"[{position}/{len(pending)}] {flag} q{record['example_index']} "
                      f"{record['question'][:65]}", flush=True)

    summary = writer.summarize()
    print(f"EXEC-ACC {summary['correct']}/{summary['total']} "
          f"({100 * summary['accuracy']:.1f}%)")
    print(f"-> {writer.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
