#!/usr/bin/env python3
"""Direct-SQL pass@k baseline with full model I/O and per-sample execution results."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "harness"))
sys.path.insert(0, os.path.join(ROOT, "src", "sft"))

from artifacts import ArtifactWriter  # noqa: E402
from executor import Harness  # noqa: E402
from passk import attach_passk_fields, parse_pass_k, write_passk_summary  # noqa: E402
from protocol import rows_equal  # noqa: E402
from rollout import ChatAPIError, ContextOverflowError, db_path, is_context_overflow, overview  # noqa: E402
from text2sql import SYSTEM_PROMPT, extract_sql  # noqa: E402

SPIDER = os.path.join(ROOT, "data", "spider_data")
DEFAULT_PASS_K = (2, 4, 8, 16, 32)


def chat_n(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    retries: int,
) -> list[str]:
    payload = {
        "model": model,
        "messages": messages,
        "n": n,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    think = os.environ.get("EVAL_ENABLE_THINKING")
    if think is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": think == "1"}

    transient_attempts = 0
    while True:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                data = json.loads(response.read())
            return [choice["message"]["content"] for choice in data.get("choices", [])]
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            if is_context_overflow(body_text):
                raise ContextOverflowError(f"HTTP {exc.code}: {body_text}", status=exc.code, body=body_text) from exc
            if exc.code >= 500 and transient_attempts < retries:
                transient_attempts += 1
                time.sleep(min(2 ** transient_attempts, 8))
                continue
            raise ChatAPIError(f"HTTP {exc.code}: {body_text}", status=exc.code, body=body_text) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if transient_attempts < retries:
                transient_attempts += 1
                time.sleep(min(2 ** transient_attempts, 8))
                continue
            raise ChatAPIError(f"{type(exc).__name__}: {exc}") from exc


def score_sample(h: Harness, output: str, gold_rows: list) -> dict:
    sample = {
        "model_output": output,
        "predicted_sql": None,
        "correct": False,
        "failure_type": None,
    }
    sql = extract_sql(output)
    sample["predicted_sql"] = sql
    if sql is None:
        sample["failure_type"] = "no_sql"
        return sample
    try:
        predicted_rows = h.gold(sql)
    except Exception as exc:  # noqa: BLE001
        sample["failure_type"] = "execution_error"
        sample["error"] = f"{type(exc).__name__}: {exc}"
        return sample
    sample["predicted_row_count"] = len(predicted_rows)
    sample["predicted_sample"] = [list(row) for row in predicted_rows[:10]]
    sample["correct"] = rows_equal(predicted_rows, gold_rows)
    if not sample["correct"]:
        sample["failure_type"] = "wrong_result"
    return sample


def run_one(
    ex: dict,
    example_index: int,
    base_url: str,
    model: str,
    *,
    n_samples: int,
    pass_k: tuple[int, ...],
    temperature: float,
    top_p: float,
    max_tokens: int,
    api_retries: int,
) -> dict:
    started = time.time()
    h = Harness(db_path(ex["db_id"]))
    h.conn.execute("PRAGMA query_only = ON")
    table_names = [table["table_name"] for table in overview(h)["tables"]]
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
    record = {
        "example_index": example_index,
        "db_id": ex["db_id"],
        "question": ex["question"],
        "gold_sql": ex["query"],
        "model_input": messages,
        "n_samples": n_samples,
        "pass_k": list(pass_k),
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "correct": False,
        "failure_type": None,
    }
    try:
        outputs = chat_n(
            base_url,
            model,
            messages,
            n=n_samples,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            retries=api_retries,
        )
        gold_rows = h.gold(ex["query"])
    except ContextOverflowError as exc:
        record["failure_type"] = "context_overflow"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = round(time.time() - started, 3)
        record["samples"] = []
        return record
    except Exception as exc:  # noqa: BLE001
        record["failure_type"] = "api_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["elapsed_seconds"] = round(time.time() - started, 3)
        record["samples"] = []
        return record

    samples = [score_sample(h, output, gold_rows) for output in outputs]
    record["samples"] = samples
    record["gold_row_count"] = len(gold_rows)
    record["gold_sample"] = [list(row) for row in gold_rows[:10]]
    attach_passk_fields(record, samples, pass_k)
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
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--pass-k", default="2,4,8,16,32")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--api-retries", type=int, default=3)
    args = parser.parse_args()

    try:
        pass_k = parse_pass_k(args.pass_k, n_samples=args.n_samples)
    except ValueError as exc:
        parser.error(str(exc))

    writer = ArtifactWriter(args.result_dir, {
        "runner": "direct_sql_passk",
        "model": args.model,
        "base_url": args.base_url,
        "dev_size": args.n,
        "n_samples": args.n_samples,
        "pass_k": list(pass_k),
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "enable_thinking": os.environ.get("EVAL_ENABLE_THINKING"),
        "execution_feedback": False,
        "system_prompt": SYSTEM_PROMPT,
    }, args.resume)
    indexed_dev = list(enumerate(json.load(open(os.path.join(SPIDER, "dev.json")))[:args.n]))
    pending = [(i, ex) for i, ex in indexed_dev if i not in writer.completed]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_one,
                ex,
                i,
                args.base_url,
                args.model,
                n_samples=args.n_samples,
                pass_k=pass_k,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                api_retries=args.api_retries,
            )
            for i, ex in pending
        ]
        for position, future in enumerate(as_completed(futures), 1):
            record = future.result()
            writer.append(record)
            flag = "OK" if record["correct"] else record["failure_type"]
            print(
                f"[{position}/{len(pending)}] {flag} q{record['example_index']} "
                f"correct_samples={record.get('sample_correct_count', 0)} "
                f"{record['question'][:60]}",
                flush=True,
            )
            write_passk_summary(writer, pass_k)

    summary = write_passk_summary(writer, pass_k)
    print(json.dumps(summary["pass_at"], ensure_ascii=False, indent=2))
    print(f"-> {writer.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
