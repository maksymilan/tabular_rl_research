#!/usr/bin/env python3
"""Use a local Qwen3 model to clean reasoning without changing the action.

This is a diagnostic editor, not a trajectory generator.  The model receives
only the current visible user context, the source reasoning, and the exact next
action.  It rewrites the reasoning to retain decision-relevant state analysis
and justification while removing repetition, abandoned branches, unrelated
speculation, and protocol meta-discussion.  Gold and future feedback are never
loaded.  Output records are diagnostic-only until human review and a separate
dataset projection/audit are completed.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Protocol

from build_checkpoint_relalg_qwen3_projection_v3 import TARGET_RE, read_jsonl, repeated_ngram_ratio, sha256


POLICY_VERSION = "checkpoint-relalg-qwen3-reasoning-semantic-clean-v2-context-isolated"
SYSTEM_PROMPT = """You edit reasoning targets for a small tool-use model.

You are given two quoted data fields: ORIGINAL_REASONING and EXACT_NEXT_ACTION. The full episode context is deliberately not included because this is an editing task, not a new attempt to solve the task.
Rewrite ORIGINAL_REASONING into a coherent decision rationale for EXACT_NEXT_ACTION.

Keep every key fact, calculation, uncertainty, error correction, and dependency needed to understand the current state and why this exact next action is appropriate. Complex decisions may remain detailed; there is no sentence limit. Use a direct operational style: begin from the established current state, name only facts that affect this action, and stop once the action choice is justified.

Remove repetition, self-talk, abandoned alternatives, speculative branches that do not affect the action, plans beyond the current action, restating the whole question, generic descriptions of what a tool does, and discussion of prompts, validators, schemas-as-instructions, or output formatting. Mention a later dependency only when it is necessary to explain why the current action must happen now. Do not turn an uncertainty from ORIGINAL_REASONING into a guarantee.

Do not reconstruct or restate the missing task. Do not solve the task again. Do not change or propose a different action. Do not add a table, column, value, row count, result, or fact that is absent from the quoted fields. Do not use gold answers or future observations. Do not mention this editing request. If ORIGINAL_REASONING is already concise and decision-relevant, preserve its substance instead of expanding it. Write the rationale in its main language.

Return exactly one JSON object whose only key is think and whose value is your rewritten rationale. Never output a placeholder such as "cleaned reasoning". The value must be non-empty prose without think tags, Markdown fences, or an action JSON object."""

FEW_SHOT_MESSAGES = (
    {
        "role": "user",
        "content": json.dumps(
            {
                "ORIGINAL_REASONING": (
                    "We need answer. Need inspect schema. Maybe orders has customer_id and customers has id. "
                    "Use describe_table first, then we can plan the join and later answer."
                ),
                "EXACT_NEXT_ACTION": {
                    "tool": "describe_table",
                    "arguments": {"tables": ["orders", "customers"]},
                },
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "think": (
                    "The columns and key relationship of orders and customers are not yet known, "
                    "so inspect both schemas before constructing the join."
                )
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": json.dumps(
            {
                "ORIGINAL_REASONING": (
                    "The previous shape failed because the alias Sub-product is not a valid simple "
                    "identifier. The source column e.Sub-product is present. We could try another alias, "
                    "but the tool allows us to omit as and preserve the logical name. Then later answer."
                ),
                "EXACT_NEXT_ACTION": {
                    "tool": "shape_rows",
                    "arguments": {
                        "table": "filter_003",
                        "outputs": [{"column": "e.Sub-product"}],
                    },
                },
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "think": (
                    "The explicit alias failed its identifier constraint, while filter_003 already "
                    "contains e.Sub-product. Omit as so shape_rows preserves that logical column name."
                )
            },
            ensure_ascii=False,
        ),
    },
)

FORBIDDEN_META = (
    "gold sql",
    "gold answer",
    "reference answer",
    "future feedback",
    "future observation",
    "system prompt",
    "editing request",
    "as an ai",
)


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_target(row: dict[str, Any], *, record_id: str) -> tuple[str, str, dict[str, Any]]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError(f"{record_id}: conversations are incomplete")
    target = conversations[-1]
    value = target.get("value") if isinstance(target, dict) else None
    match = TARGET_RE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"{record_id}: target is not inline think + JSON")
    reasoning = match.group("reasoning").strip()
    action_text = match.group("action")
    action = json.loads(action_text)
    if not reasoning or not isinstance(action, dict) or set(action) != {"tool", "arguments"}:
        raise ValueError(f"{record_id}: target envelope is invalid")
    return reasoning, action_text, action


def action_anchors(action: dict[str, Any]) -> tuple[str, ...]:
    result = {str(action.get("tool", "")).lower().replace("_", " ")}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, str):
            normalized = " ".join(value.lower().split())
            if len(normalized) >= 2:
                result.add(normalized)

    visit(action.get("arguments"))
    return tuple(sorted((item for item in result if item), key=len, reverse=True))


def contains_anchor(text: str, action: dict[str, Any]) -> bool:
    normalized = " ".join(text.lower().replace("_", " ").split())
    return any(anchor.replace("_", " ") in normalized for anchor in action_anchors(action))


def repeated_sentence(text: str) -> bool:
    sentences = [
        " ".join(item.lower().split())
        for item in re.split(r"(?<=[.!?。！？])\s+", text.strip())
        if len(" ".join(item.split())) >= 24
    ]
    return len(sentences) != len(set(sentences))


def decode_editor_response(raw: str) -> str:
    start = len(raw) - len(raw.lstrip())
    try:
        value, end = json.JSONDecoder().raw_decode(raw, start)
    except json.JSONDecodeError as exc:
        raise ValueError("editor response is not JSON") from exc
    if raw[end:].strip() or not isinstance(value, dict) or set(value) != {"think"}:
        raise ValueError("editor response must contain exactly think")
    think = value["think"]
    if not isinstance(think, str) or not think.strip():
        raise ValueError("editor think is empty")
    return think.strip()


def validate_cleaned_reasoning(
    cleaned: str,
    source: str,
    action: dict[str, Any],
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    if "<think>" in cleaned or "</think>" in cleaned or "```" in cleaned:
        raise ValueError("cleaned reasoning contains a forbidden carrier")
    if repeated_sentence(cleaned):
        raise ValueError("cleaned reasoning repeats a sentence")
    repeat_ratio = repeated_ngram_ratio(cleaned)
    if repeat_ratio >= 0.12:
        raise ValueError("cleaned reasoning is repetitive")
    lowered = cleaned.lower()
    if any(item in lowered for item in FORBIDDEN_META):
        raise ValueError("cleaned reasoning contains forbidden meta-discussion")
    if not contains_anchor(cleaned, action):
        raise ValueError("cleaned reasoning is not grounded to the exact action")
    source_tokens = len(tokenizer.encode(source, add_special_tokens=False))
    cleaned_tokens = len(tokenizer.encode(cleaned, add_special_tokens=False))
    if cleaned_tokens < 4:
        raise ValueError("cleaned reasoning is too short to be meaningful")
    # This is a cleaner, not an expansion model. A small boundary allowance is
    # useful for rephrasing already-short records; longer source reasoning must
    # never grow.
    allowed = source_tokens + 16 if source_tokens < 128 else source_tokens
    if cleaned_tokens > allowed:
        raise ValueError("cleaned reasoning is longer than the source")
    return {
        "source_tokens": source_tokens,
        "cleaned_tokens": cleaned_tokens,
        "token_reduction": 1 - cleaned_tokens / source_tokens,
        "repeated_8gram_ratio": repeat_ratio,
        "action_anchor_present": True,
    }


def select_stratified(
    rows: list[dict[str, Any]],
    indexes: list[dict[str, Any]],
    n: int,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(rows) != len(indexes):
        raise ValueError("canonical/index counts differ")
    groups: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for row, item in zip(rows, indexes, strict=True):
        record_id = (row.get("metadata") or {}).get("record_id")
        if item.get("record_id") != record_id:
            raise ValueError("canonical/index order differs")
        groups[str(item.get("tool_name"))].append((row, item))
    tools = sorted(groups)
    if n < len(tools):
        raise ValueError("sample size must cover every tool")
    quotas = {tool: n // len(tools) for tool in tools}
    for tool in sorted(tools, key=lambda name: (-len(groups[name]), name))[: n % len(tools)]:
        quotas[tool] += 1

    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for tool in tools:
        candidates = sorted(
            groups[tool],
            key=lambda pair: (
                int(pair[1].get("reasoning_characters", 0)),
                str(pair[1].get("record_id")),
            ),
        )
        quota = min(quotas[tool], len(candidates))
        if quota == 1:
            positions = [len(candidates) // 2]
        else:
            positions = [round(i * (len(candidates) - 1) / (quota - 1)) for i in range(quota)]
        selected.extend(candidates[position] for position in positions)

    # Ensure recovery behavior is represented without changing per-tool counts.
    chosen_ids = {item[1]["record_id"] for item in selected}
    recovery = [
        pair
        for pair in zip(rows, indexes, strict=True)
        if pair[1].get("feedback_recovery") is True and pair[1]["record_id"] not in chosen_ids
    ]
    recovery.sort(key=lambda pair: str(pair[1]["record_id"]))
    target_recovery = min(10, len(recovery))
    current_recovery = sum(item[1].get("feedback_recovery") is True for item in selected)
    for replacement in recovery[: max(0, target_recovery - current_recovery)]:
        tool = str(replacement[1].get("tool_name"))
        replace_at = next(
            (
                index
                for index in range(len(selected) - 1, -1, -1)
                if selected[index][1].get("tool_name") == tool
                and selected[index][1].get("feedback_recovery") is not True
            ),
            None,
        )
        if replace_at is not None:
            selected[replace_at] = replacement
    selected.sort(key=lambda pair: (int(pair[1].get("source_task_position", 0)), str(pair[1]["record_id"])))
    if len(selected) != n or len({pair[1]["record_id"] for pair in selected}) != n:
        raise ValueError("stratified selection is incomplete or duplicated")
    return selected


def request_payload(
    model: str,
    original_reasoning: str,
    action: dict[str, Any],
    max_tokens: int,
) -> dict[str, Any]:
    quoted = {
        "ORIGINAL_REASONING": original_reasoning,
        "EXACT_NEXT_ACTION": action,
    }
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            *FEW_SHOT_MESSAGES,
            {"role": "user", "content": json.dumps(quoted, ensure_ascii=False)},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "chat_template_kwargs": {"enable_thinking": False},
    }


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"local editor request failed: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise RuntimeError("local editor response is not an object")
    return value


async def edit_one(
    semaphore: asyncio.Semaphore,
    endpoint: str,
    model: str,
    row: dict[str, Any],
    item: dict[str, Any],
    tokenizer: Tokenizer,
    max_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    record_id = str(item["record_id"])
    source_reasoning, action_text, action = parse_target(row, record_id=record_id)
    current_context = str(row["conversations"][-2]["value"])
    payload = request_payload(model, source_reasoning, action, max_tokens)
    started = time.monotonic()
    async with semaphore:
        response = await asyncio.to_thread(post_json, endpoint, payload, timeout)
    elapsed = time.monotonic() - started
    choices = response.get("choices")
    if response.get("model") != model:
        raise ValueError(f"{record_id}: editor response model identity mismatch")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError(f"{record_id}: editor response choices are invalid")
    if choices[0].get("finish_reason") != "stop":
        raise ValueError(f"{record_id}: editor response did not finish cleanly")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"{record_id}: editor response content is missing")
    cleaned = decode_editor_response(content)
    quality = validate_cleaned_reasoning(cleaned, source_reasoning, action, tokenizer)
    return {
        "schema_version": "checkpoint-relalg-qwen3-reasoning-clean-pilot-record-v1",
        "policy_version": POLICY_VERSION,
        "record_id": record_id,
        "source_episode_id": item.get("source_episode_id"),
        "source_task_position": item.get("source_task_position"),
        "source_turn_index": item.get("source_turn_index"),
        "tool_name": item.get("tool_name"),
        "feedback_recovery": item.get("feedback_recovery"),
        "source_reasoning": source_reasoning,
        "cleaned_reasoning": cleaned,
        "exact_action_text": action_text,
        "exact_action_sha256": digest_text(action_text),
        "source_reasoning_sha256": digest_text(source_reasoning),
        "cleaned_reasoning_sha256": digest_text(cleaned),
        "current_context_sha256": digest_text(current_context),
        "quality": quality,
        "request": {
            "endpoint": endpoint,
            "model": model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "enable_thinking": False,
            "generation_context_policy": "original-reasoning-plus-exact-action-only-v1",
            "system_prompt_sha256": digest_text(SYSTEM_PROMPT),
        },
        "response": {
            "model": response.get("model"),
            "finish_reason": choices[0].get("finish_reason"),
            "usage": response.get("usage"),
            "elapsed_seconds": elapsed,
        },
        "sft_admission": False,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    canonical = args.source_canonical.resolve()
    index_path = args.source_index.resolve()
    rows = read_jsonl(canonical)
    indexes = read_jsonl(index_path)
    selected = select_stratified(rows, indexes, args.n)
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()), trust_remote_code=True)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(
            edit_one(
                semaphore,
                args.endpoint,
                args.model,
                row,
                item,
                tokenizer,
                args.max_tokens,
                args.timeout,
            )
        )
        for row, item in selected
    ]
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for task, (_, item) in zip(tasks, selected, strict=True):
        try:
            results.append(await task)
        except Exception as exc:  # diagnostic failures are retained, never repaired silently
            failures.append(
                {
                    "record_id": item.get("record_id"),
                    "tool_name": item.get("tool_name"),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    if args.output.exists() or args.summary.exists():
        raise FileExistsError(args.output if args.output.exists() else args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    source_tokens = [int(item["quality"]["source_tokens"]) for item in results]
    cleaned_tokens = [int(item["quality"]["cleaned_tokens"]) for item in results]
    summary = {
        "schema_version": "checkpoint-relalg-qwen3-reasoning-clean-pilot-summary-v1",
        "policy_version": POLICY_VERSION,
        "source_canonical": str(canonical),
        "source_canonical_sha256": sha256(canonical),
        "source_index": str(index_path),
        "source_index_sha256": sha256(index_path),
        "selection_policy": "tool-and-reasoning-length-stratified-with-recovery-v1",
        "requested_records": args.n,
        "successful_records": len(results),
        "failed_records": len(failures),
        "failures": failures,
        "tool_hist": dict(Counter(str(item["tool_name"]) for item in results).most_common()),
        "feedback_recovery_records": sum(item.get("feedback_recovery") is True for item in results),
        "source_reasoning_tokens_total": sum(source_tokens),
        "cleaned_reasoning_tokens_total": sum(cleaned_tokens),
        "token_reduction": 1 - sum(cleaned_tokens) / sum(source_tokens) if source_tokens else 0.0,
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
        "sft_admission": False,
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-canonical", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8766/v1/chat/completions")
    parser.add_argument("--model", default="qwen3-atomic-v24-projection-v3-base")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 24 or args.n < 1 or args.max_tokens < 64:
        raise ValueError("invalid pilot limits")
    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["failed_records"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
