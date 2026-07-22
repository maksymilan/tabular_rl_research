#!/usr/bin/env python3
"""Verify AIHubMix reasoning-details continuation without changing rollout behavior."""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from provider_client import load_api_config  # noqa: E402


def request(base_url: str, api_key: str, payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:800]}") from exc


def message_summary(message: dict) -> dict:
    details = message.get("reasoning_details")
    return {
        "message_fields": sorted(message),
        "content_chars": len(str(message.get("content") or "")),
        "reasoning_content_chars": len(str(message.get("reasoning_content") or "")),
        "reasoning_details_present": isinstance(details, dict),
        "reasoning_details_keys": sorted(details) if isinstance(details, dict) else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    api_key, base_url = load_api_config()
    if not api_key or not base_url:
        parser.error("api.md must define API_KEY and BASE_URL")

    system = (
        "You are a two-turn table agent. Keep visible content to one short tool intent. "
        "Put the detailed reason in the provider reasoning channel."
    )
    first_messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "Inspect the Inventory table before deciding how to filter it."},
    ]
    first = request(base_url, api_key, {
        "model": args.model,
        "messages": first_messages,
        "temperature": 0,
        "max_tokens": 2048,
    })
    first_message = first["choices"][0]["message"]
    continuation = {"role": "assistant", "content": first_message.get("content") or ""}
    if isinstance(first_message.get("reasoning_details"), dict):
        continuation["reasoning_details"] = first_message["reasoning_details"]

    second_messages = first_messages + [
        continuation,
        {
            "role": "user",
            "content": (
                "TOOL RESULT: Inventory has columns id, item_name, category. "
                "Continue from the preceding reasoning and choose the single next action."
            ),
        },
    ]
    second = request(base_url, api_key, {
        "model": args.model,
        "messages": second_messages,
        "temperature": 0,
        "max_tokens": 2048,
    })
    second_message = second["choices"][0]["message"]

    report = {
        "model": args.model,
        "transport": "aihubmix_openai_chat_completions",
        "first": message_summary(first_message),
        "continuation_sent_reasoning_details": "reasoning_details" in continuation,
        "second_request_accepted": True,
        "second": message_summary(second_message),
        "usage": {"first": first.get("usage", {}), "second": second.get("usage", {})},
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
