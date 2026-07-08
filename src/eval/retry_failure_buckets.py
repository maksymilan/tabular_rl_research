#!/usr/bin/env python3
"""Build retry index files for rollout failures that may be fixed by relaxed runtime limits."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


DEFAULT_BUCKETS = ("context_overflow", "execution_error", "max_steps")


def load_records(result_dir: Path) -> list[dict]:
    failure_path = result_dir / "failure.jsonl"
    if not failure_path.exists():
        raise FileNotFoundError(f"missing failure file: {failure_path}")
    with failure_path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def last_error(record: dict) -> tuple[str | None, str]:
    turns = record.get("turns") or []
    last = turns[-1] if turns else {}
    parsed = last.get("parsed") or {}
    tool = parsed.get("tool")
    message = str(
        last.get("execution_error")
        or last.get("api_error")
        or record.get("fail")
        or ""
    )
    return tool, " ".join(message.split())


def classify_execution_error(message: str) -> str:
    if "KeyError: 'value'" in message:
        return "condition missing value"
    if "KeyError: 'column'" in message:
        return "condition missing column"
    if "KeyError: 'distinct'" in message:
        return "unsupported distinct argument"
    if "return_columns" in message:
        return "condition_filter unexpected return_columns"
    if "Could not decode to UTF-8" in message:
        return "sqlite utf8 decode"
    if "set_op" in message and "aligned columns" in message:
        return "set_op column alignment"
    if "ScalarGroundingError" in message or "value_ref must cite a scalar" in message:
        return "value_ref non-scalar"
    if "in_table" in message and "must resolve to one column" in message:
        return "in_table shape"
    if "unknown plan item" in message or "plan evidence references unknown step" in message:
        return "plan update invalid reference"
    if "unknown table" in message:
        return "unknown table/handle"
    if "no such column" in message or "ambiguous column" in message:
        return "column reference error"
    if "unrecognized token" in message or "near \"(\"" in message or "syntax error" in message:
        return "bad SQL expression"
    return message.split(":", 1)[0][:100] or "unknown execution error"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("result_dir", help="rollout result directory containing failure.jsonl")
    ap.add_argument("--buckets", nargs="*", default=list(DEFAULT_BUCKETS))
    ap.add_argument("--out-dir", default="",
                    help="where to write indices/report; default: <result_dir>/retry_buckets")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir) if args.out_dir else result_dir / "retry_buckets"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(result_dir)
    bucket_set = set(args.buckets)
    by_bucket: dict[str, list[dict]] = collections.defaultdict(list)
    failure_hist = collections.Counter()
    exec_clusters = collections.Counter()
    exec_examples: dict[str, dict] = {}

    for record in records:
        failure_type = record.get("failure_type") or record.get("fail") or "wrong_answer"
        failure_hist[failure_type] += 1
        if failure_type not in bucket_set:
            continue
        tool, message = last_error(record)
        item = {
            "example_index": record["example_index"],
            "db_id": record.get("db_id"),
            "question": record.get("question"),
            "failure_type": failure_type,
            "steps": record.get("steps"),
            "errors": record.get("errors"),
            "last_tool": tool,
            "last_error": message,
        }
        by_bucket[failure_type].append(item)
        if failure_type == "execution_error":
            cluster = classify_execution_error(message)
            exec_clusters[cluster] += 1
            exec_examples.setdefault(cluster, item)

    combined = []
    for bucket, items in sorted(by_bucket.items()):
        indices = sorted({int(item["example_index"]) for item in items})
        combined.extend(indices)
        (out_dir / f"{bucket}.indices.json").write_text(
            json.dumps({"bucket": bucket, "indices": indices}, indent=2) + "\n",
            encoding="utf-8",
        )
        (out_dir / f"{bucket}.examples.json").write_text(
            json.dumps(items, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    combined = sorted(set(combined))
    (out_dir / "combined.indices.json").write_text(
        json.dumps({"buckets": sorted(by_bucket), "indices": combined}, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "result_dir": str(result_dir),
        "failure_hist": dict(sorted(failure_hist.items())),
        "selected_buckets": {bucket: len(items) for bucket, items in sorted(by_bucket.items())},
        "combined_count": len(combined),
        "execution_error_clusters": {
            cluster: {
                "count": count,
                "example": exec_examples[cluster],
            }
            for cluster, count in exec_clusters.most_common()
        },
        "outputs": {
            "out_dir": str(out_dir),
            "combined_indices": str(out_dir / "combined.indices.json"),
        },
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
