#!/usr/bin/env python3
"""Teacher-forced next-action evaluation for step-level ShareGPT datasets.

This measures whether a model can predict a legal next action from an unseen harness state. It is
deliberately separate from closed-loop execution accuracy: exact teacher-action agreement is a
strict imitation metric, while rollout.py remains the end-to-end behavioral test.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src" / "sft"), str(ROOT / "src" / "harness")]

from protocol import ProtocolError, parse_assistant_strict  # noqa: E402
from rollout import chat  # noqa: E402


ROLE_MAP = {"human": "user", "gpt": "assistant"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def model_messages(record: dict, system_override: str | None = None) -> tuple[list[dict], str]:
    conversations = record.get("conversations") or []
    if not conversations or conversations[-1].get("from") != "gpt":
        raise ValueError("record must end with the supervised gpt target")
    messages = [{"role": "system", "content": system_override or record["system"]}]
    for message in conversations[:-1]:
        role = ROLE_MAP.get(message.get("from"))
        if role is None:
            raise ValueError(f"unsupported ShareGPT role: {message.get('from')!r}")
        messages.append({"role": role, "content": message["value"]})
    return messages, conversations[-1]["value"]


def evaluate_one(
    record: dict,
    index: int,
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    api_retries: int,
    system_override: str | None,
) -> dict:
    messages, target = model_messages(record, system_override)
    metadata = record.get("metadata") or {}
    result = {
        "index": index,
        "record_id": metadata.get("record_id"),
        "source_episode_id": metadata.get("source_episode_id"),
        "source_step_id": metadata.get("source_step_id"),
        "feedback_recovery": bool(metadata.get("feedback_recovery")),
        "model": model,
        "target": target,
    }
    try:
        _, gold_tool, gold_args = parse_assistant_strict(target)
    except ProtocolError as exc:
        raise ValueError(f"invalid target {result['record_id']}: {exc}") from exc
    result["target_tool"] = gold_tool
    retry_stats: dict = {}
    try:
        output = chat(
            base_url,
            model,
            messages,
            max_tokens=max_tokens,
            retries=api_retries,
            retry_stats=retry_stats,
        )
        result["output"] = output
        result["api_retry_stats"] = retry_stats
    except Exception as exc:  # keep every failed request auditable
        result.update({
            "valid": False,
            "tool_match": False,
            "arguments_match": False,
            "action_match": False,
            "error": f"api: {type(exc).__name__}: {exc}",
        })
        return result
    try:
        _, pred_tool, pred_args = parse_assistant_strict(output)
        result.update({
            "valid": True,
            "predicted_tool": pred_tool,
            "predicted_arguments": pred_args,
            "tool_match": pred_tool == gold_tool,
            "arguments_match": pred_args == gold_args,
            "action_match": pred_tool == gold_tool and pred_args == gold_args,
        })
    except ProtocolError as exc:
        result.update({
            "valid": False,
            "tool_match": False,
            "arguments_match": False,
            "action_match": False,
            "error": f"protocol: {exc}",
        })
    return result


def rate(rows: list[dict], field: str) -> float:
    return sum(bool(row.get(field)) for row in rows) / max(1, len(rows))


def summarize(rows: list[dict], *, dataset: Path, model: str) -> dict:
    by_position: dict[str, list[dict]] = defaultdict(list)
    by_tool: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        step = row.get("source_step_id") or ""
        by_position["first" if step == "step_1" else "later"].append(row)
        by_tool[row.get("target_tool") or "unknown"].append(row)

    def metrics(group: list[dict]) -> dict:
        return {
            "records": len(group),
            "valid": sum(bool(row.get("valid")) for row in group),
            "valid_rate": rate(group, "valid"),
            "tool_match": sum(bool(row.get("tool_match")) for row in group),
            "tool_match_rate": rate(group, "tool_match"),
            "arguments_match": sum(bool(row.get("arguments_match")) for row in group),
            "arguments_match_rate": rate(group, "arguments_match"),
            "action_match": sum(bool(row.get("action_match")) for row in group),
            "action_match_rate": rate(group, "action_match"),
        }

    return {
        "dataset": str(dataset),
        "model": model,
        **metrics(rows),
        "by_position": {name: metrics(group) for name, group in sorted(by_position.items())},
        "by_target_tool": {name: metrics(group) for name, group in sorted(by_tool.items())},
        "predicted_tool_histogram": dict(Counter(
            row.get("predicted_tool") or "<invalid>" for row in rows
        )),
        "note": (
            "Exact action match compares tool name and JSON arguments; equivalent alternative legal "
            "actions can score as mismatches. Closed-loop execution accuracy is the decisive metric."
        ),
    }


def safe_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model).strip("_") or "model"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", action="append", required=True, help="repeat for matched models")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--exclude-episodes-from",
        type=Path,
        default=None,
        help="trajectory JSONL whose trajectory_id values must be excluded from this evaluation",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument(
        "--system-prompt-manifest",
        type=Path,
        default=None,
        help="reuse the exact system_prompt stored by a historical rollout",
    )
    args = parser.parse_args()

    records = read_jsonl(args.input)
    if args.exclude_episodes_from:
        excluded = {
            row.get("trajectory_id")
            for row in read_jsonl(args.exclude_episodes_from)
            if row.get("trajectory_id")
        }
        records = [
            record for record in records
            if (record.get("metadata") or {}).get("source_episode_id") not in excluded
        ]
    if args.limit > 0:
        records = records[:args.limit]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    system_override = None
    if args.system_prompt_manifest:
        manifest = json.loads(args.system_prompt_manifest.read_text(encoding="utf-8"))
        system_override = manifest.get("system_prompt")
        if not isinstance(system_override, str) or not system_override.strip():
            parser.error("--system-prompt-manifest does not contain a non-empty system_prompt")

    for model in args.model:
        rows: list[dict] = []
        kwargs = {
            "base_url": args.base_url,
            "model": model,
            "max_tokens": args.max_tokens,
            "api_retries": args.api_retries,
            "system_override": system_override,
        }
        if args.workers > 1:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(evaluate_one, record, index, **kwargs): index
                    for index, record in enumerate(records)
                }
                for future in as_completed(futures):
                    rows.append(future.result())
        else:
            rows = [evaluate_one(record, index, **kwargs) for index, record in enumerate(records)]
        rows.sort(key=lambda row: row["index"])

        model_dir = args.output_dir / safe_name(model)
        model_dir.mkdir(parents=True, exist_ok=True)
        with (model_dir / "all.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary = summarize(rows, dataset=args.input, model=model)
        (model_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
