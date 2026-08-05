#!/usr/bin/env python3
"""Extract model-authored reasoning from failed trajectory records.

The generated files intentionally omit tool calls, tool/environment outputs,
errors, resident state, predicted answers, gold answers, and gold SQL.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                records.append(record)
    return records


def authored_reasoning(turn: dict[str, Any]) -> str:
    provider_reasoning = turn.get("provider_reasoning_content")
    parsed = turn.get("parsed")
    parsed_reasoning = parsed.get("think") if isinstance(parsed, dict) else None

    provider_text = (
        provider_reasoning.strip()
        if isinstance(provider_reasoning, str)
        else ""
    )
    parsed_text = parsed_reasoning.strip() if isinstance(parsed_reasoning, str) else ""
    if provider_text and parsed_text and provider_text != parsed_text:
        raise ValueError(
            "provider_reasoning_content and parsed.think disagree for "
            f"turn_index={turn.get('turn_index')}"
        )
    reasoning = provider_text or parsed_text
    if not reasoning:
        raise ValueError(f"missing authored reasoning for turn_index={turn.get('turn_index')}")
    return reasoning


def extract_failure(record: dict[str, Any]) -> dict[str, Any]:
    turns = record.get("turns")
    if not isinstance(turns, list) or not turns:
        raise ValueError(f"{record.get('trajectory_id')}: missing turns")

    reasoning_turns = []
    for position, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict):
            raise ValueError(
                f"{record.get('trajectory_id')}: turn {position} is not an object"
            )
        turn_index = turn.get("turn_index")
        reasoning_turns.append(
            {
                "turn_index": (
                    int(turn_index) + 1
                    if isinstance(turn_index, int)
                    else position
                ),
                "reasoning": authored_reasoning(turn),
            }
        )

    return {
        "example_index": record.get("example_index"),
        "trajectory_id": record.get("trajectory_id"),
        "question": record.get("question"),
        "reasoning_turns": reasoning_turns,
    }


def write_jsonl(path: Path, failures: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for failure in failures:
            target.write(json.dumps(failure, ensure_ascii=False) + "\n")


def write_markdown(path: Path, failures: list[dict[str, Any]]) -> None:
    lines = [
        "# Version24 fixed-200 失败轨迹：模型推理内容",
        "",
        (
            "范围：version24 fixed-200 的 55 条失败轨迹。"
            "正文只保留任务定位信息与模型逐轮推理，不包含工具调用、"
            "工具或环境返回、错误反馈、resident state、预测答案、金标答案或 gold SQL。"
        ),
        "",
    ]
    for failure_number, failure in enumerate(failures, start=1):
        lines.extend(
            [
                f"## {failure_number}. {failure['trajectory_id']}",
                "",
                f"题目：{failure['question']}",
                "",
            ]
        )
        for turn in failure["reasoning_turns"]:
            lines.extend(
                [
                    f"### 推理 {turn['turn_index']}",
                    "",
                    turn["reasoning"],
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-records", type=int)
    parser.add_argument("--expected-failures", type=int)
    args = parser.parse_args()

    records = read_jsonl(args.input)
    if args.expected_records is not None and len(records) != args.expected_records:
        raise ValueError(
            f"expected {args.expected_records} input records, found {len(records)}"
        )

    failed_records = [record for record in records if record.get("correct") is False]
    if args.expected_failures is not None and len(failed_records) != args.expected_failures:
        raise ValueError(
            f"expected {args.expected_failures} failures, found {len(failed_records)}"
        )

    failures = [extract_failure(record) for record in failed_records]
    failures.sort(
        key=lambda item: (
            item["example_index"]
            if isinstance(item["example_index"], int)
            else float("inf"),
            str(item["trajectory_id"]),
        )
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "version24_fixed200_failures55_reasoning_only.jsonl"
    markdown_path = args.output_dir / "version24_fixed200_failures55_reasoning_only.md"
    write_jsonl(jsonl_path, failures)
    write_markdown(markdown_path, failures)

    reasoning_turns = sum(len(failure["reasoning_turns"]) for failure in failures)
    print(
        json.dumps(
            {
                "input_records": len(records),
                "correct_records": len(records) - len(failed_records),
                "failed_records": len(failed_records),
                "reasoning_turns": reasoning_turns,
                "jsonl": str(jsonl_path),
                "markdown": str(markdown_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
