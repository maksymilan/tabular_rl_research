#!/usr/bin/env python3
"""Regenerate one clean coherent rationale from a causal prefix and fixed action.

The editor never receives the source reasoning.  It sees only the exact causal
context that preceded the authored action plus that immutable action.  This
prevents pathological source reasoning from being copied while preserving the
original action and all Harness causality.  The editor prompt is separate from
the student runtime prompt, and the official DeepSeek request disables model
thinking so only the formal JSON result is admitted.
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

from build_checkpoint_relalg_qwen3_projection_v3 import read_jsonl, repeated_ngram_ratio, sha256
from provider_client import load_api_config
from rewrite_checkpoint_relalg_reasoning_with_qwen3 import parse_target
from tool_modules.checkpoint_relalg.protocol import (
    ATOMIC_OPERATOR_PROFILE_FROZEN_V24,
    get_tool_definitions,
    tool_schema_hash,
)
from tool_modules.checkpoint_relalg.qwen3_carrier import qwen3_student_prompt_sha256


OFFICIAL_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
POLICY_VERSION = "checkpoint-relalg-deepseek-coherent-reasoning-regeneration-v3"
GENERATION_CONTEXT_POLICY = "causal-context-current-contract-fixed-action-no-source-reasoning-v2"
ATOMIC_TOOLS = frozenset(
    {
        "answer",
        "describe_table",
        "filter_rows",
        "group_aggregate",
        "inspect_column",
        "join",
        "rank_select",
        "read_rows",
        "scalar_compute",
        "set_operation",
        "shape_rows",
    }
)

TEACHER_SYSTEM_PROMPT = """You are a reasoning-target editor for a database tool-use student.

This teacher/editor prompt is not the student's runtime prompt. You receive the exact causal context immediately before one already-authored action and the immutable exact action. You do not receive the original reasoning, gold SQL, gold answers, future feedback, or later actions.

Write one coherent natural rationale for this action. It may use one or more paragraphs and has no sentence-count requirement, but its reasoning flow must contain all three of these ideas in order: analyze only the latest Harness feedback that matters now; use that feedback together with the current environment and final goal to decide what this step must accomplish; explain why the exact current tool and its important arguments implement that decision. If this is the first action, replace feedback analysis with the specific information that remains unknown.

Make those ideas read as one decision, not three filled-in template fields. Start from the concrete consequence of the feedback, not phrases such as "the latest feedback indicates", "the current goal is", or "this tool is the correct choice". Do not restate the whole question. Explain a tool through the exact relation transformation or observation being requested, not through a generic tool definition. End after explaining the immediate output or observation of the current action; do not preview a later filter, join, aggregate, answer, or other next step. This student profile has no checkpoint tools, so never call an ordinary tool result, artifact, or step a checkpoint. Preserve necessary calculations, constraints, uncertainty, and error corrections, but remove self-talk, repetition, abandoned alternatives, prompt/schema instructions, and future plans. Never invent an identifier, value, row count, result, or guarantee. Do not change or second-guess the action. Do not mention editing, training, gold data, benchmarks, or hidden reasoning.

Good style example: "filter_003 now contains the qualifying transactions but not customer attributes. Joining it to customers on CustomerID attaches the required name and email columns while retaining only matched customers, so join uses that equality edge and the two existing relations." This is one connected rationale, not labeled fields, a task restatement, a tool definition, or a multi-step plan.

Return one JSON object with exactly one key, think, whose value is the complete coherent rationale. Do not label or mechanically enumerate the three ideas. Do not return think tags, Markdown, an action object, commentary, or placeholder text."""


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def split_causal_context(current_context: str) -> tuple[str, str]:
    marker = "\n\nQUESTION\n"
    if marker not in current_context:
        return json.dumps({"status": "no_previous_tool_feedback"}), current_context
    prefix, suffix = current_context.split(marker, 1)
    if "checkpoint_relalg_tool_result" not in prefix:
        return json.dumps({"status": "no_previous_tool_feedback"}), current_context
    return prefix.strip(), "QUESTION\n" + suffix


def request_payload(
    model: str,
    current_context: str,
    action: dict[str, Any],
    max_tokens: int,
    thinking_mode: str = "disabled",
) -> dict[str, Any]:
    if thinking_mode not in {"enabled", "disabled"}:
        raise ValueError("invalid DeepSeek thinking mode")
    feedback, goal_and_environment = split_causal_context(current_context)
    definitions = get_tool_definitions(
        "atomic", atomic_operator_profile=ATOMIC_OPERATOR_PROFILE_FROZEN_V24
    )
    matches = [
        definition
        for definition in definitions
        if definition.get("function", {}).get("name") == action.get("tool")
    ]
    if len(matches) != 1:
        raise ValueError("exact action tool is absent from the current student contract")
    user = {
        "LATEST_HARNESS_FEEDBACK": feedback,
        "CURRENT_GOAL_AND_ENVIRONMENT": goal_and_environment,
        "CURRENT_STUDENT_TOOL_CONTRACT": {
            "scheme": "checkpoint-relalg",
            "mode": "atomic",
            "atomic_operator_profile": ATOMIC_OPERATOR_PROFILE_FROZEN_V24,
            "tool_schema_sha256": tool_schema_hash(
                "atomic", ATOMIC_OPERATOR_PROFILE_FROZEN_V24
            ),
            "exact_tool_definition": matches[0],
        },
        "EXACT_NEXT_ACTION": action,
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ],
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "thinking": {"type": thinking_mode},
    }
    if thinking_mode == "enabled":
        payload["reasoning_effort"] = "high"
    return payload


def post_deepseek_json(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float,
    *,
    retries: int = 3,
) -> tuple[dict[str, Any], int]:
    """Post one formal-output request without exposing credentials or error bodies."""

    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read().decode("utf-8"))
            if not isinstance(value, dict):
                raise RuntimeError("DeepSeek response root is not an object")
            return value, attempt
        except urllib.error.HTTPError as exc:
            retryable = exc.code in {408, 409, 425, 429} or 500 <= exc.code <= 599
            if not retryable or attempt == retries:
                raise RuntimeError(f"DeepSeek HTTP {exc.code}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries:
                raise RuntimeError(
                    f"DeepSeek request failed: {type(exc).__name__}"
                ) from exc
        time.sleep(min(2 ** (attempt - 1), 4))
    raise AssertionError("unreachable")


def verify_deepseek_model(
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
) -> None:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            value = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("DeepSeek model verification failed") from exc
    advertised = {
        item.get("id")
        for item in value.get("data", [])
        if isinstance(item, dict)
    } if isinstance(value, dict) else set()
    if model not in advertised:
        raise RuntimeError("requested DeepSeek model is not advertised")


def decode_response(content: str) -> str:
    start = len(content) - len(content.lstrip())
    try:
        value, end = json.JSONDecoder().raw_decode(content, start)
    except json.JSONDecodeError as exc:
        raise ValueError("teacher response is not JSON") from exc
    if content[end:].strip() or not isinstance(value, dict) or set(value) != {"think"}:
        raise ValueError("teacher response keys or trailing content are invalid")
    think = value["think"]
    if not isinstance(think, str) or not think.strip():
        raise ValueError("teacher response think is empty")
    return think.strip()


def _sentences(text: str) -> list[str]:
    return [
        " ".join(item.lower().split())
        for item in re.split(r"(?<=[.!?。！？])\s+|\n+", text.strip())
        if len(" ".join(item.split())) >= 20
    ]


def _feedback_anchors(current_context: str) -> set[str]:
    feedback, _ = split_causal_context(current_context)
    if "no_previous_tool_feedback" in feedback:
        return set()
    candidates = set(
        re.findall(
            r"\b(?:[A-Za-z][A-Za-z0-9]*_[0-9]{3}|[A-Za-z][A-Za-z0-9_]*_error|invalid_[A-Za-z0-9_]+|unknown_[A-Za-z0-9_]+|result_too_large)\b",
            feedback,
        )
    )
    return {item.lower() for item in candidates}


def validate_reasoning(
    combined: str,
    source_reasoning: str,
    current_context: str,
    action: dict[str, Any],
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    lowered = combined.lower()
    if any(token in combined for token in ("<think>", "</think>", "```")):
        raise ValueError("structured reasoning contains a forbidden carrier")
    forbidden = ("gold sql", "gold answer", "benchmark", "editing", "training data", "hidden reasoning")
    if any(item in lowered for item in forbidden):
        raise ValueError("structured reasoning contains forbidden meta-discussion")
    boilerplate = (
        "the latest harness feedback indicates",
        "the latest feedback indicates",
        "the current goal is",
        "the exact tool to use is",
        "the tool is the correct choice",
        "tool is the appropriate choice",
    )
    if any(item in lowered for item in boilerplate):
        raise ValueError("structured reasoning uses a templated three-part narration")
    if "checkpoint" in lowered:
        raise ValueError("checkpoint-disabled reasoning invents checkpoint state")
    future_plan = re.search(
        r"\b(?:next|later|subsequent)\s+(?:tool|step|action|filter|join|aggregate|answer)\b"
        r"|\b(?:will|can|to)\s+(?:then|later)\s+(?:filter|join|aggregate|answer|call|use)\b",
        lowered,
    )
    if future_plan:
        raise ValueError("structured reasoning previews a future action")
    sentences = _sentences(combined)
    if len(sentences) != len(set(sentences)):
        raise ValueError("structured reasoning repeats a sentence")
    repeat_ratio = repeated_ngram_ratio(combined)
    if repeat_ratio >= 0.08:
        raise ValueError("structured reasoning repeats 8-grams")
    current_tool = str(action["tool"])
    normalized = lowered.replace("_", " ")
    if current_tool.replace("_", " ") not in normalized:
        raise ValueError("tool rationale does not name the exact tool")
    other_tools = []
    for tool in ATOMIC_TOOLS - {current_tool}:
        name = re.escape(tool.replace("_", " "))
        if re.search(rf"\b(?:then\s+)?(?:use|call|invoke)\s+(?:the\s+)?{name}\b", normalized):
            other_tools.append(tool)
    if other_tools:
        raise ValueError("structured reasoning proposes another tool")
    feedback_anchors = _feedback_anchors(current_context)
    if feedback_anchors and not any(anchor in lowered for anchor in feedback_anchors):
        raise ValueError("reasoning does not analyze the latest Harness feedback")
    action_strings = {
        item.lower()
        for item in re.findall(r'"([^"\\]{2,})"', json.dumps(action, ensure_ascii=False))
        if item not in {"tool", "arguments", "op", "left", "right", "column", "value"}
    }
    if action_strings and not any(item in lowered for item in action_strings):
        raise ValueError("reasoning does not ground the current decision in action arguments")
    source_tokens = len(tokenizer.encode(source_reasoning, add_special_tokens=False))
    regenerated_tokens = len(tokenizer.encode(combined, add_special_tokens=False))
    if regenerated_tokens < 12:
        raise ValueError("structured reasoning is too short")
    if regenerated_tokens > 512:
        raise ValueError("regenerated reasoning is unnecessarily long")
    grounding = (current_context + "\n" + json.dumps(action, ensure_ascii=False)).lower()
    specific = set(re.findall(r"\b(?:[A-Za-z][A-Za-z0-9]*[_.][A-Za-z0-9_.-]+|\d+(?:\.\d+)?)\b", combined))
    unsupported = sorted(item for item in specific if item.lower() not in grounding)
    if unsupported:
        raise ValueError("structured reasoning adds unsupported specific tokens")
    return {
        "source_tokens": source_tokens,
        "regenerated_tokens": regenerated_tokens,
        "repeated_8gram_ratio": repeat_ratio,
        "other_tool_mentions": [],
        "unsupported_specific_tokens": [],
        "feedback_anchor_required": bool(feedback_anchors),
        "feedback_anchor_present": not feedback_anchors
        or any(anchor in lowered for anchor in feedback_anchors),
        "decision_action_anchor_present": True,
        "exact_tool_named": True,
    }


async def edit_record(
    semaphore: asyncio.Semaphore,
    endpoint: str,
    api_key: str,
    model: str,
    row: dict[str, Any],
    item: dict[str, Any],
    tokenizer: Tokenizer,
    max_tokens: int,
    timeout: float,
    thinking_mode: str,
) -> dict[str, Any]:
    record_id = str(item["record_id"])
    source_reasoning, action_text, action = parse_target(row, record_id=record_id)
    current_context = str(row["conversations"][-2]["value"])
    payload = request_payload(
        model,
        current_context,
        action,
        max_tokens,
        thinking_mode,
    )
    started = time.monotonic()
    async with semaphore:
        response, request_attempts = await asyncio.to_thread(
            post_deepseek_json,
            endpoint,
            payload,
            api_key,
            timeout,
        )
    elapsed = time.monotonic() - started
    if response.get("model") != model:
        raise ValueError(f"{record_id}: teacher model identity mismatch")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or choices[0].get("finish_reason") != "stop":
        raise ValueError(f"{record_id}: teacher response did not finish cleanly")
    message = choices[0].get("message") or {}
    provider_reasoning = message.get("reasoning_content")
    if thinking_mode == "disabled" and provider_reasoning not in (None, ""):
        raise ValueError(f"{record_id}: disabled teacher returned reasoning_content")
    if thinking_mode == "enabled" and (
        not isinstance(provider_reasoning, str) or not provider_reasoning.strip()
    ):
        raise ValueError(f"{record_id}: enabled teacher omitted reasoning_content")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError(f"{record_id}: teacher formal output is missing")
    reasoning = decode_response(content)
    quality = validate_reasoning(reasoning, source_reasoning, current_context, action, tokenizer)
    return {
        "schema_version": "checkpoint-relalg-deepseek-coherent-reasoning-record-v3",
        "policy_version": POLICY_VERSION,
        "record_id": record_id,
        "source_episode_id": item["source_episode_id"],
        "source_turn_index": item["source_turn_index"],
        "tool_name": item["tool_name"],
        "feedback_recovery": item.get("feedback_recovery"),
        "source_reasoning_sha256": digest_text(source_reasoning),
        "rendered_reasoning": reasoning,
        "exact_action_text": action_text,
        "exact_action_sha256": digest_text(action_text),
        "current_context_sha256": digest_text(current_context),
        "quality": quality,
        "teacher_request": {
            "endpoint": endpoint,
            "model": model,
            "temperature_sent": False,
            "max_tokens": max_tokens,
            "thinking": {"type": thinking_mode},
            "reasoning_effort_sent": thinking_mode == "enabled",
            "generation_context_policy": GENERATION_CONTEXT_POLICY,
            "teacher_prompt_sha256": digest_text(TEACHER_SYSTEM_PROMPT),
            "student_prompt_was_sent_to_teacher": False,
            "student_atomic_operator_profile": ATOMIC_OPERATOR_PROFILE_FROZEN_V24,
            "student_tool_schema_sha256": tool_schema_hash(
                "atomic", ATOMIC_OPERATOR_PROFILE_FROZEN_V24
            ),
            "student_prompt_sha256": qwen3_student_prompt_sha256(),
        },
        "teacher_response": {
            "model": response.get("model"),
            "finish_reason": choices[0].get("finish_reason"),
            "reasoning_content_present": provider_reasoning not in (None, ""),
            "reasoning_content_discarded": True,
            "reasoning_content_sha256": (
                digest_text(provider_reasoning)
                if isinstance(provider_reasoning, str) and provider_reasoning
                else None
            ),
            "reasoning_content_characters": (
                len(provider_reasoning) if isinstance(provider_reasoning, str) else 0
            ),
            "usage": response.get("usage"),
            "request_attempts": request_attempts,
            "elapsed_seconds": elapsed,
        },
        "student_action_changed": False,
        "sft_admission": False,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    rows = read_jsonl(args.source_canonical.resolve())
    indexes = read_jsonl(args.source_index.resolve())
    if len(rows) != len(indexes):
        raise ValueError("canonical/index counts differ")
    selected: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    wanted = set(args.episode_id)
    for row, item in zip(rows, indexes, strict=True):
        if item.get("atomic_operator_profile") != ATOMIC_OPERATOR_PROFILE_FROZEN_V24:
            raise ValueError("source record is not bound to the current student tool profile")
        if item.get("source_episode_id") in wanted:
            selected[str(item["source_episode_id"])].append((row, item))
    if set(selected) != wanted:
        raise ValueError("one or more requested episodes are absent")
    for values in selected.values():
        values.sort(key=lambda pair: int(pair[1]["source_turn_index"]))

    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer.resolve()), trust_remote_code=True)
    api_key, base_url = load_api_config(args.api_config.resolve())
    if not api_key or base_url.rstrip("/") != OFFICIAL_DEEPSEEK_BASE_URL:
        raise ValueError("external regeneration requires the official DeepSeek API config")
    verify_deepseek_model(base_url, api_key, args.model, args.timeout)
    endpoint = base_url.rstrip("/") + "/chat/completions"
    semaphore = asyncio.Semaphore(args.concurrency)

    async def process_episode(episode_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for row, item in selected[episode_id]:
            try:
                accepted.append(
                    await edit_record(
                        semaphore,
                        endpoint,
                        api_key,
                        args.model,
                        row,
                        item,
                        tokenizer,
                        args.max_tokens,
                        args.timeout,
                        args.thinking,
                    )
                )
            except Exception as exc:
                rejected.append(
                    {
                        "record_id": item.get("record_id"),
                        "source_episode_id": episode_id,
                        "source_turn_index": item.get("source_turn_index"),
                        "tool_name": item.get("tool_name"),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
        if len({item["rendered_reasoning"] for item in accepted}) != len(accepted):
            raise ValueError(f"{episode_id}: duplicate reasoning across turns")
        return accepted, rejected

    episode_results = await asyncio.gather(*(process_episode(episode_id) for episode_id in args.episode_id))
    accepted = [item for episode, _ in episode_results for item in episode]
    rejected = [item for _, failures in episode_results for item in failures]
    if args.output.exists() or args.summary.exists():
        raise FileExistsError(args.output if args.output.exists() else args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in accepted:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": "checkpoint-relalg-deepseek-coherent-reasoning-summary-v3",
        "policy_version": POLICY_VERSION,
        "teacher_prompt_sha256": digest_text(TEACHER_SYSTEM_PROMPT),
        "student_prompt_changed": False,
        "student_atomic_operator_profile": ATOMIC_OPERATOR_PROFILE_FROZEN_V24,
        "student_tool_schema_sha256": tool_schema_hash(
            "atomic", ATOMIC_OPERATOR_PROFILE_FROZEN_V24
        ),
        "student_prompt_sha256": qwen3_student_prompt_sha256(),
        "generation_context_policy": GENERATION_CONTEXT_POLICY,
        "teacher_thinking_mode": args.thinking,
        "episode_ids": args.episode_id,
        "requested_records": sum(len(values) for values in selected.values()),
        "accepted_records": len(accepted),
        "rejected_records": len(rejected),
        "rejections": rejected,
        "tool_hist": dict(Counter(item["tool_name"] for item in accepted).most_common()),
        "source_tokens_for_diagnostic_comparison": sum(
            item["quality"]["source_tokens"] for item in accepted
        ),
        "regenerated_tokens": sum(
            item["quality"]["regenerated_tokens"] for item in accepted
        ),
        "hidden_teacher_reasoning_records": sum(
            item["teacher_response"]["reasoning_content_present"] for item in accepted
        ),
        "teacher_reasoning_admitted_records": 0,
        "student_action_changed_records": sum(item["student_action_changed"] for item in accepted),
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
    parser.add_argument("--episode-id", action="append", required=True)
    parser.add_argument("--api-config", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="enabled")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 24 or args.max_tokens < 128:
        raise ValueError("invalid editor limits")
    result = asyncio.run(run(args))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["rejected_records"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
