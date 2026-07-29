#!/usr/bin/env python3
"""Paired prefix-20/prefix-50 audit for BIRD schema-context ablations."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def trajectory_id(example: dict) -> str:
    return str(example.get("example_id") or example.get("trajectory_id"))


def two_sided_sign_p(gains: int, regressions: int) -> float:
    discordant = gains + regressions
    if discordant == 0:
        return 1.0
    tail = min(gains, regressions)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1))
    return min(1.0, 2.0 * probability / (2 ** discordant))


def tool_histogram(records: list[dict]) -> Counter:
    histogram = Counter()
    for record in records:
        for turn in record.get("turns", []):
            parsed = turn.get("parsed")
            if isinstance(parsed, dict) and parsed.get("tool"):
                histogram[str(parsed["tool"])] += 1
    return histogram


def initial_user_chars(record: dict) -> int | None:
    turns = record.get("turns") or []
    model_input = turns[0].get("model_input") if turns else None
    if not isinstance(model_input, list):
        return None
    for message in model_input:
        if message.get("role") == "user":
            return len(str(message.get("content", "")))
    return None


def summarize(records: list[dict]) -> dict:
    tools = tool_histogram(records)
    total = len(records)
    chars = [
        value for value in (initial_user_chars(record) for record in records)
        if value is not None
    ]
    return {
        "total": total,
        "correct": sum(bool(record.get("correct")) for record in records),
        "accuracy": round(
            sum(bool(record.get("correct")) for record in records) / max(1, total),
            4,
        ),
        "legal": sum(bool(record.get("legal")) for record in records),
        "errors": sum(int(record.get("errors") or 0) for record in records),
        "mean_steps": round(
            sum(int(record.get("steps") or 0) for record in records) / max(1, total),
            3,
        ),
        "describe_table_calls": tools.get("describe_table", 0),
        "inspect_column_calls": tools.get("inspect_column", 0),
        "api_request_attempts": sum(
            int((record.get("usage") or {}).get("api_request_attempts") or 0)
            for record in records
        ),
        "total_tokens": sum(
            int((record.get("usage") or {}).get("total_tokens") or 0)
            for record in records
        ),
        "mean_initial_user_chars": (
            round(sum(chars) / len(chars), 1) if chars else None
        ),
        "failure_types": dict(
            Counter(
                str(record.get("failure_type"))
                for record in records
                if record.get("failure_type")
            )
        ),
    }


def paired(candidate: dict[str, dict], baseline: dict[str, dict], ids: list[str]) -> dict:
    gains = [
        item for item in ids
        if candidate[item].get("correct") and not baseline[item].get("correct")
    ]
    regressions = [
        item for item in ids
        if baseline[item].get("correct") and not candidate[item].get("correct")
    ]
    return {
        "gains": gains,
        "regressions": regressions,
        "gain_count": len(gains),
        "regression_count": len(regressions),
        "net": len(gains) - len(regressions),
        "exact_two_sided_p": round(
            two_sided_sign_p(len(gains), len(regressions)),
            6,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-file", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=all.jsonl; repeat for every candidate",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    examples = read_jsonl(Path(args.examples_file))[:50]
    ids50 = [trajectory_id(example) for example in examples]
    ids20 = ids50[:20]
    if any(not item or item == "None" for item in ids50):
        raise ValueError("examples file lacks stable example_id values")

    baseline_records = {
        str(record["trajectory_id"]): record
        for record in read_jsonl(Path(args.baseline))
    }
    if set(ids50) - set(baseline_records):
        raise ValueError("baseline is missing frozen first-50 task IDs")

    runs = {}
    for item in args.run:
        label, separator, value = item.partition("=")
        if not separator or not label or not value:
            raise ValueError(f"--run must be LABEL=PATH: {item}")
        records = {
            str(record["trajectory_id"]): record
            for record in read_jsonl(Path(value))
        }
        missing = set(ids50) - set(records)
        if missing:
            raise ValueError(f"{label} is missing {len(missing)} frozen task IDs")
        runs[label] = records

    result = {
        "cohort": {
            "examples_file": args.examples_file,
            "prefix20_ids": ids20,
            "prefix50_count": len(ids50),
        },
        "baseline": {
            "prefix20": summarize([baseline_records[item] for item in ids20]),
            "prefix50": summarize([baseline_records[item] for item in ids50]),
        },
        "runs": {},
    }
    for label, records in runs.items():
        result["runs"][label] = {
            "prefix20": summarize([records[item] for item in ids20]),
            "prefix50": summarize([records[item] for item in ids50]),
            "paired20": paired(records, baseline_records, ids20),
            "paired50": paired(records, baseline_records, ids50),
        }

    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
