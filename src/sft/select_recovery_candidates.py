#!/usr/bin/env python3
"""Select failed rollout samples that are worth repairing into recovery trajectories.

This is the first, deterministic step of the second-stage SFT data loop:

  TRAIN rollout/pass@k all.jsonl
  -> select legal or semantically useful failures
  -> external LLM repair loop (separate step)

The selector deliberately avoids pure API/context/protocol failures by default. Those failures are
important for operations, but they do not teach the model how to recover from table observations.
By default the selector refuses dev/eval artifacts: recovery training data must come from training
set rollouts, never Spider dev.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

BAD_ENV_FAILURES = {"api_error", "context_overflow"}
BAD_FORMAT_FAILURES = {"protocol_error"}
DEV_PATH_MARKERS = ("/dev", "_dev", "dev1034", "spider_dev")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                records.append(json.loads(line))
    return records


def sample_has_empty_result(sample: dict[str, Any]) -> bool:
    for turn in sample.get("turns") or []:
        output = turn.get("tool_output")
        if isinstance(output, dict) and output.get("row_count") == 0:
            return True
    return False


def sample_has_execution_error(sample: dict[str, Any]) -> bool:
    if sample.get("failure_type") == "execution_error":
        return True
    return any(turn.get("execution_error_type") == "execution_error"
               for turn in sample.get("turns") or [])


def sample_has_protocol_error(sample: dict[str, Any]) -> bool:
    if sample.get("failure_type") == "protocol_error":
        return True
    return any(turn.get("execution_error_type") == "protocol_error"
               for turn in sample.get("turns") or [])


def sample_is_environment_failure(sample: dict[str, Any]) -> bool:
    return sample.get("failure_type") in BAD_ENV_FAILURES


def classify_sample(record: dict[str, Any], sample: dict[str, Any]) -> str | None:
    """Return the recovery bucket, or None if the sample should not be repaired."""
    if sample.get("correct"):
        return None
    if sample_is_environment_failure(sample):
        return None
    if sample_has_protocol_error(sample):
        return None
    if sample_has_empty_result(sample):
        return "empty_result_recovery"
    if sample_has_execution_error(sample):
        return "execution_error_recovery"
    if sample.get("legal") and sample.get("failure_type") == "wrong_answer":
        if record.get("sample_correct_count", 0) > 0:
            return "mixed_success_wrong_answer"
        return "legal_wrong_answer"
    return None


def compact_sample(sample: dict[str, Any], *, keep_turns: bool) -> dict[str, Any]:
    out = {
        "sample_index": sample.get("sample_index"),
        "correct": bool(sample.get("correct")),
        "legal": bool(sample.get("legal")),
        "steps": sample.get("steps"),
        "errors": sample.get("errors"),
        "failure_type": sample.get("failure_type"),
    }
    if sample.get("error"):
        out["error"] = sample.get("error")
    if keep_turns:
        out["turns"] = sample.get("turns") or []
    return out


def successful_sample_summary(record: dict[str, Any]) -> dict[str, Any] | None:
    for sample in record.get("samples") or []:
        if sample.get("correct"):
            return compact_sample(sample, keep_turns=False)
    return None


def candidate_items(records: list[dict[str, Any]], *, keep_turns: bool) -> list[dict[str, Any]]:
    out = []
    for record in records:
        for sample in record.get("samples") or []:
            bucket = classify_sample(record, sample)
            if bucket is None:
                continue
            out.append({
                "example_index": record.get("example_index"),
                "db_id": record.get("db_id"),
                "question": record.get("question"),
                "gold_sql": record.get("gold_sql"),
                "bucket": bucket,
                "n_samples": record.get("n_samples", len(record.get("samples") or [])),
                "sample_correct_count": record.get("sample_correct_count", 0),
                "sample_legal_count": record.get("sample_legal_count", 0),
                "failed_sample": compact_sample(sample, keep_turns=keep_turns),
                "successful_sample": successful_sample_summary(record),
            })
    return out


def priority(candidate: dict[str, Any]) -> tuple[int, int, int]:
    order = {
        "mixed_success_wrong_answer": 0,
        "empty_result_recovery": 1,
        "execution_error_recovery": 2,
        "legal_wrong_answer": 3,
    }
    failed = candidate["failed_sample"]
    return (
        order.get(candidate["bucket"], 99),
        int(candidate.get("example_index") or 10**9),
        int(failed.get("sample_index") or 0),
    )


def select_diverse(candidates: list[dict[str, Any]], *, limit: int, per_db_cap: int) -> list[dict[str, Any]]:
    selected = []
    used_db: collections.Counter[str] = collections.Counter()
    for item in sorted(candidates, key=priority):
        if limit and len(selected) >= limit:
            break
        db_id = str(item.get("db_id"))
        if per_db_cap and used_db[db_id] >= per_db_cap:
            continue
        selected.append(item)
        used_db[db_id] += 1
    if limit and len(selected) < limit:
        seen = {(item.get("example_index"), item["failed_sample"].get("sample_index"))
                for item in selected}
        for item in sorted(candidates, key=priority):
            key = (item.get("example_index"), item["failed_sample"].get("sample_index"))
            if key in seen:
                continue
            selected.append(item)
            if len(selected) >= limit:
                break
    return selected


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as sink:
        for record in records:
            sink.write(json.dumps(record, ensure_ascii=False) + "\n")


def looks_like_dev_artifact(path: Path) -> bool:
    text = str(path).replace("\\", "/").lower()
    return any(marker in text for marker in DEV_PATH_MARKERS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="rollout/pass@k all.jsonl")
    parser.add_argument("--output", required=True, help="selected recovery candidates JSONL")
    parser.add_argument("--summary", default=None, help="summary JSON; default OUTPUT.summary.json")
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--per-db-cap", type=int, default=12)
    parser.add_argument("--no-turns", action="store_true",
                        help="omit failed_sample.turns for a compact planning-only candidate file")
    parser.add_argument("--allow-dev-analysis", action="store_true",
                        help="allow dev/eval artifacts for audit only; never use that output for SFT")
    args = parser.parse_args()

    input_path = Path(args.input)
    if looks_like_dev_artifact(input_path) and not args.allow_dev_analysis:
        parser.error(
            "input path looks like a dev/eval artifact. Recovery SFT candidates must come from "
            "TRAIN rollouts. Use --allow-dev-analysis only for temporary auditing."
        )

    records = load_jsonl(input_path)
    candidates = candidate_items(records, keep_turns=not args.no_turns)
    selected = select_diverse(candidates, limit=args.limit, per_db_cap=args.per_db_cap)

    output = Path(args.output)
    write_jsonl(output, selected)

    summary = {
        "input": args.input,
        "output": str(output),
        "allow_dev_analysis": args.allow_dev_analysis,
        "total_records": len(records),
        "candidate_samples": len(candidates),
        "selected_samples": len(selected),
        "limit": args.limit,
        "per_db_cap": args.per_db_cap,
        "bucket_counts": dict(collections.Counter(item["bucket"] for item in candidates)),
        "selected_bucket_counts": dict(collections.Counter(item["bucket"] for item in selected)),
        "selected_db_count": len({item.get("db_id") for item in selected}),
    }
    summary_path = Path(args.summary) if args.summary else output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"-> {output}")
    print(f"-> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
