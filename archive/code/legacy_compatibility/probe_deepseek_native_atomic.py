#!/usr/bin/env python3
"""Run one isolated BIRD task through DeepSeek's native function-call carrier.

This is a diagnostic only.  It does not select or promote a protocol version and its output is
explicitly ineligible for SFT export.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]
for relative in ("src", "src/eval", "src/harness", "src/sft"):
    sys.path.insert(0, str(ROOT / relative))

from tool_modules.native_tool_bundle.provider_tools import (  # noqa: E402
    NATIVE_ASSISTANT_CARRIER,
    DeepSeekNativeAtomicDriver,
    native_system_prompt,
    native_tools_sha256,
)
from provider_client import load_api_config  # noqa: E402
from protocol import PROTOCOL_VERSION, get_system_prompt, rolling_system_prompt  # noqa: E402
from rollout import load_tasks_json, run_live, task_gold_sql  # noqa: E402


DEFAULT_TASKS = ROOT / "data/eval_inputs/bird_train_sft1_protocol_terminal10.jsonl"


def _official_deepseek_base(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == "api.deepseek.com"
        and parsed.path.rstrip("/") in {"", "/v1"}
        and not parsed.query
        and not parsed.fragment
    )


def _default_output() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return ROOT / f"data/results/deepseek_native_atomic_probe_{stamp}.json"


def _token_totals(events: list[dict]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for event in events:
        usage = event.get("usage")
        if not isinstance(usage, dict):
            continue
        for key, value in usage.items():
            if isinstance(value, int) and not isinstance(value, bool):
                totals[key] = totals.get(key, 0) + value
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe the atomic harness with official DeepSeek native function calls."
    )
    parser.add_argument("--tasks-json", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--api-retries", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    api_key, base_url = load_api_config()
    if not api_key:
        parser.error("api.md must define API_KEY")
    if not _official_deepseek_base(base_url):
        parser.error(
            "this diagnostic accepts only the official https://api.deepseek.com base URL"
        )
    tasks = load_tasks_json(str(args.tasks_json))
    if args.index < 0 or args.index >= len(tasks):
        parser.error(f"--index must be between 0 and {len(tasks) - 1}")
    example = tasks[args.index]

    system = rolling_system_prompt(native_system_prompt(get_system_prompt()))
    driver = DeepSeekNativeAtomicDriver(api_key=api_key, base_url=base_url)
    record = run_live(
        example,
        args.index,
        base_url,
        args.model,
        system,
        max_steps=args.max_steps,
        max_tokens=args.max_tokens,
        api_retries=args.api_retries,
        context_mode="rolling-legal-history",
        history_turns=4,
        compact_history_observations=True,
        denotation_comparison="bird-set",
        chat_fn=driver,
    )

    gold_sql = task_gold_sql(example) or ""
    serialized_inputs = json.dumps(
        [turn.get("model_input") for turn in record.get("turns", [])],
        ensure_ascii=False,
        sort_keys=True,
    )
    record.update(
        {
            "assistant_carrier": NATIVE_ASSISTANT_CARRIER,
            "canonical_protocol_version": PROTOCOL_VERSION,
            "diagnostic_only": True,
            "sft_export_eligible": False,
            "provider": "deepseek-official",
            "provider_base_url": base_url,
            "native_tools_sha256": native_tools_sha256(),
            "native_tool_events": driver.events,
            "native_token_totals": _token_totals(driver.events),
            "gold_sql_hidden_from_model_input": bool(gold_sql) and gold_sql not in serialized_inputs,
        }
    )
    if not record["gold_sql_hidden_from_model_input"]:
        raise RuntimeError("gold SQL unexpectedly appeared in model-visible input")

    output = (args.out or _default_output()).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "output": str(output),
        "db_id": record.get("db_id"),
        "model": args.model,
        "correct": record.get("correct"),
        "legal": record.get("legal"),
        "steps": record.get("steps"),
        "errors": record.get("errors"),
        "failure_type": record.get("failure_type"),
        "tool_sequence": [event.get("tool") for event in driver.events],
        "token_totals": record["native_token_totals"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if record.get("legal") else 1


if __name__ == "__main__":
    raise SystemExit(main())
