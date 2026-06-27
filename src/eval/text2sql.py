#!/usr/bin/env python3
"""Zero-shot direct-SQL baseline with complete model I/O and split artifacts."""
from __future__ import annotations

import argparse
import json
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
from protocol import rows_equal                               # noqa: E402
from rollout import ContextOverflowError, chat, db_path, overview  # noqa: E402

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


def run_one(ex: dict, example_index: int, base_url: str, model: str, max_tokens: int = 512) -> dict:
    h = Harness(db_path(ex["db_id"]))
    h.conn.execute("PRAGMA query_only = ON")
    # Direct-SQL needs the FULL schema (columns/types/PK/FK) up front — the model cannot probe with
    # tools here. overview()/_catalog is the v2-ctx lazy catalog (names + row counts only), which
    # starves text-to-SQL and forces column hallucination; describe_table gives the complete schema.
    table_names = [t["table_name"] for t in overview(h)["tables"]]
    user_prompt = (
        "DATABASE SCHEMA\n"
        + json.dumps(h.describe_table(table_names), ensure_ascii=False, separators=(",", ":"))
        + "\n\nQUESTION\n"
        + ex["question"]
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    started = time.time()
    record = {
        "example_index": example_index,
        "db_id": ex["db_id"],
        "question": ex["question"],
        "gold_sql": ex["query"],
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
    try:
        predicted_rows = h.gold(sql)
        gold_rows = h.gold(ex["query"])
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
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=1034)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    writer = ArtifactWriter(args.result_dir, {
        "runner": "direct_sql",
        "model": args.model,
        "base_url": args.base_url,
        "dev_size": args.n,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "enable_thinking": os.environ.get("EVAL_ENABLE_THINKING"),
        "execution_feedback": False,
        "system_prompt": SYSTEM_PROMPT,
    }, args.resume)
    indexed_dev = list(enumerate(json.load(open(os.path.join(SPIDER, "dev.json")))[:args.n]))
    pending = [(i, ex) for i, ex in indexed_dev if i not in writer.completed]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, ex, i, args.base_url, args.model, args.max_tokens) for i, ex in pending]
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
