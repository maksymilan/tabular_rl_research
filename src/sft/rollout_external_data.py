#!/usr/bin/env python3
"""Generate SFT trajectories by closed-loop external-LLM rollout.

Unlike the compiler-backed enrichment pipeline, this script lets an external model solve the
question through the current model<->harness protocol. Each model turn must emit exactly one tool
call; the harness executes it and feeds back either a success observation or an actionable error
message. A trajectory is kept for SFT only when the final answer_from_context is execution-scored
against the gold SQL as correct.

This produces data whose successful traces can include realistic execution-error recovery turns,
instead of only polished gold-SQL decompositions.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "sft"))

from fill_think import load_api  # noqa: E402
from provider_adapter import (  # noqa: E402
    adapt_provider_response,
    provider_default_max_tokens,
    provider_instruction,
)
from rollout import (  # noqa: E402
    ChatAPIError,
    ContextOverflowError,
    db_path,
    execute_tool,
    format_tool_error,
    is_context_overflow,
    new_ctx,
    overview,
    score,
    task_db_path,
    task_gold_sql,
)
from executor import Harness  # noqa: E402
from protocol import (  # noqa: E402
    ProtocolError,
    assistant_message,
    first_user_message,
    get_system_prompt,
    model_context_messages,
    parse_assistant_strict,
    protocol_hash,
    rolling_legal_history_messages,
    rolling_system_prompt,
    state_context_message,
    tool_output_message,
)

SPIDER = ROOT / "data" / "spider_data"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_ERRORS_PER_TYPE = 3
DEFAULT_MAX_STEPS = 30
DEFAULT_MAX_TOKENS = 1024
MIN_CONTEXT_RETRY_TOKENS = 256
DATA_GENERATION_SUFFIX = (
    "\n\nDATA GENERATION STRICTNESS\n"
    "ONE REQUEST = ONE ACTION. Emit exactly one non-empty <think> block and exactly one "
    "<tool_call> block. Immediately STOP after that closing </tool_call>: never emit a second "
    "<think>, a second tool call, a numbered plan of calls, or a complete multi-step solution in "
    "one response. The harness will execute only this one action and return a fresh state before "
    "you choose the next action. Your <think> block must be non-empty on every turn. Put the reason inside <think> tags, "
    "not as plain text before the tool call. The reason should be specific to the current question, "
    "visible schema/observations, and the next tool arguments. After the first turn, do not restate "
    "the original user question; continue from the current environment state or error feedback. "
    "Do not call read_subtable again for the same table, columns, and limit if that read is already "
    "present in CURRENT ENVIRONMENT STATE. If a plan item has no evidence yet, omit the evidence "
    "field or set it to null; never use an empty string for evidence."
)

def compact_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def protocol_failure_type(exc: ProtocolError) -> str:
    text = str(exc).lower()
    if any(marker in text for marker in (
        "arguments", "unexpected", "requires", "missing required", "must be", "must contain",
    )):
        return "argument_validation_error"
    return "protocol_error"


def error_event(action_index: int, error_type: str, message: str,
                state_before: dict, state_after: dict | None = None,
                attempted_tool: str | None = None,
                attempted_arguments: dict | None = None) -> dict:
    """Audit a rejected model action without making it an SFT supervision target."""
    state_after = state_before if state_after is None else state_after
    event = {
        "action_index": action_index,
        "step_id": f"step_{action_index}",
        "error_type": error_type,
        "message": message,
        "state_before_hash": __import__("hashlib").sha256(compact_json(state_before).encode()).hexdigest(),
        "state_after_hash": __import__("hashlib").sha256(compact_json(state_after).encode()).hexdigest(),
    }
    if attempted_tool:
        event["attempted_tool"] = attempted_tool
        event["attempted_arguments"] = attempted_arguments or {}
    return event


def add_usage(target: collections.Counter, usage: dict | None, prefix: str = "") -> None:
    """Accumulate numeric usage fields; flatten provider-specific nested detail objects."""
    if not isinstance(usage, dict):
        return
    for key, value in usage.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, (int, float)):
            target[name] += value
        elif isinstance(value, dict):
            add_usage(target, value, name)


def request_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: int,
) -> tuple[str, dict, str]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        error_cls = ContextOverflowError if is_context_overflow(detail) else ChatAPIError
        raise error_cls(f"HTTP {exc.code}: {detail[:1200]}", status=exc.code, body=detail) from exc
    except urllib.error.URLError as exc:
        raise ChatAPIError(f"URL error: {exc}") from exc
    choice = data["choices"][0]
    message = choice["message"]
    usage = dict(data.get("usage") or {})
    # A successful HTTP response can still be length-truncated. This is audit metadata, not a retry.
    usage["api_finish_reason"] = choice.get("finish_reason")
    return message.get("content") or "", usage, message.get("reasoning_content") or ""


def chat_with_retries(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int,
    timeout: int,
    retries: int,
) -> tuple[str, dict, str]:
    last: Exception | None = None
    budget = max_tokens
    transport_retries = 0
    context_retries = 0
    for attempt in range(max(1, retries)):
        try:
            text, usage, reasoning = request_chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                max_tokens=budget,
                timeout=timeout,
            )
            usage = dict(usage or {})
            # These are client request retries, deliberately separate from environment error events.
            usage["api_request_attempts"] = attempt + 1
            usage["api_transport_retries"] = transport_retries
            usage["api_context_retries"] = context_retries
            return text, usage, reasoning
        except ContextOverflowError:
            if budget <= MIN_CONTEXT_RETRY_TOKENS:
                raise
            budget = max(MIN_CONTEXT_RETRY_TOKENS, budget // 2)
            context_retries += 1
        except Exception as exc:  # noqa: BLE001 - API surfaces many transient transport errors
            last = exc
            if attempt + 1 < retries:
                transport_retries += 1
                time.sleep(min(2 ** attempt, 8))
    if last:
        raise last
    raise ChatAPIError("chat request failed without an exception")


def split_path(split: str) -> Path:
    if split == "train":
        return SPIDER / "train_spider.json"
    if split == "dev":
        return SPIDER / "dev.json"
    raise ValueError(f"unsupported split: {split}")


def load_examples(args: argparse.Namespace) -> list[tuple[int, dict]]:
    if args.examples_file:
        path = Path(args.examples_file)
        if path.suffix == ".jsonl":
            examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            examples = payload.get("examples", payload) if isinstance(payload, dict) else payload
        if not isinstance(examples, list):
            raise ValueError(f"{args.examples_file}: expected JSONL records, a JSON list, or {{'examples': [...]}}")
        indexed = [(int(ex.get("example_index", i)), ex) for i, ex in enumerate(examples)]
    else:
        examples = json.loads(split_path(args.split).read_text(encoding="utf-8"))
        indexed = list(enumerate(examples))

    if args.start:
        indexed = indexed[args.start:]
    if args.limit:
        indexed = indexed[: args.limit]
    return [(i, ex) for i, ex in indexed if Path(task_db_path(ex)).exists()]


def trajectory_id(split: str, example_index: int, ex: dict) -> str:
    return str(ex.get("trajectory_id") or ex.get("example_id") or f"rollout_{split}_{example_index}")


def observation_for_error(step_id: str, error_type: str, message: str) -> str:
    return compact_json({
        "step_id": step_id,
        "status": "error",
        "error": {"type": error_type, "message": message},
    })


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that treats punctuation followed by a closing quote as a boundary."""
    parts: list[str] = []
    start = 0
    i = 0
    while i < len(text):
        if text[i] in ".!?":
            end = i + 1
            while end < len(text) and text[end] in "\"'”’)]}":
                end += 1
            if end >= len(text) or text[end].isspace():
                piece = text[start:end].strip()
                if piece:
                    parts.append(piece)
                while end < len(text) and text[end].isspace():
                    end += 1
                start = end
                i = end
                continue
        i += 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def normalize_reasoning_text(text: str) -> str:
    text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
    return " ".join(text.split())


def brief_reasoning(text: str, limit: int = 360) -> str:
    text = normalize_reasoning_text(text)
    if len(text) <= limit:
        return text
    parts = split_sentences(text)
    kept = []
    total = 0
    for part in parts:
        if total + len(part) + 1 > limit:
            break
        kept.append(part)
        total += len(part) + 1
    return " ".join(kept).strip() or text[:limit].rstrip() + "..."


QUESTION_RESTATEMENT = re.compile(
    r"^\s*(?:"
    r"the\s+(?:question|user)\s+(?:asks|wants|is\s+asking)|"
    r"we\s+need\s+to\s+(?:answer|find|list|show|compute|determine)|"
    r"i\s+need\s+to\s+(?:answer|find|list|show|compute|determine)\s+(?:the\s+)?(?:question|user)"
    r")\b",
    re.I,
)


def strip_question_restatements(text: str, *, min_keep: int = 40) -> str:
    """Remove leading sentence(s) that only restate the prompt, preserving action-specific logic."""
    sentences = split_sentences(normalize_reasoning_text(text))
    while len(sentences) > 1 and QUESTION_RESTATEMENT.search(sentences[0]):
        candidate = " ".join(sentences[1:]).strip()
        if len(candidate) < min_keep:
            break
        sentences = sentences[1:]
    return " ".join(sentences).strip() or text


def prepare_think(text: str, *, source: str, limit: int = 360) -> str:
    text = brief_reasoning(text, limit=limit * 2)
    if source in {"tagged", "pre_tool_text", "reasoning_content"}:
        text = strip_question_restatements(text)
    return brief_reasoning(text, limit=limit)


def recover_think(
    raw_text: str,
    parsed_think: str,
    tool: str,
    args: dict,
    reasoning_content: str = "",
) -> tuple[str, str]:
    """Use the model's own pre-tool prose when it omitted <think> tags."""
    if parsed_think and parsed_think.strip():
        return prepare_think(parsed_think, source="tagged"), "tagged"
    prefix = raw_text.split("<tool_call>", 1)[0]
    prefix = re.sub(r"</?think>", "", prefix, flags=re.I).strip()
    prefix = re.sub(r"^```(?:json)?|```$", "", prefix, flags=re.I | re.M).strip()
    if prefix:
        return prepare_think(prefix, source="pre_tool_text"), "pre_tool_text"
    if tool == "answer_from_context" and isinstance(args, dict) and str(args.get("reason", "")).strip():
        return prepare_think(str(args["reason"]), source="answer_reason"), "answer_reason"
    if reasoning_content and reasoning_content.strip():
        return prepare_think(reasoning_content, source="reasoning_content"), "reasoning_content"
    return (
        f"I will call {tool} because it is the next necessary operation for this question.",
        "fallback_template",
    )


def run_rollout(
    *,
    example_index: int,
    ex: dict,
    split: str,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    max_steps: int,
    max_tokens: int,
    api_retries: int,
    api_timeout: int,
    max_errors_per_type: int,
    table_output_rows: int,
    context_mode: str,
    history_turns: int,
    rolling_prompt_variant: str,
) -> dict:
    task_path = task_db_path(ex)
    gold_sql = task_gold_sql(ex)
    if not gold_sql:
        raise ValueError(f"task has no gold SQL: {ex.get('db_id')} / {ex.get('question')}")
    h = Harness(task_path)
    dataset_overview = overview(h)
    external_knowledge = ex.get("external_knowledge") or None
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_user_message(dataset_overview, ex["question"])},
    ]
    created: set[str] = set()
    ctx = new_ctx(dataset_overview)
    last_error: dict | None = None
    turns: list[dict] = []
    steps: list[dict] = []
    errors = action_count = 0
    error_counts = collections.Counter()
    error_events: list[dict] = []
    legal_history: list[dict] = []
    successful_tool_steps = 0
    usage = collections.Counter()
    started = time.time()
    rec = {
        "example_index": example_index,
        "trajectory_id": trajectory_id(split, example_index, ex),
        "db_id": ex["db_id"],
        "question": ex["question"],
        "gold_sql": gold_sql,
        "difficulty": (ex.get("metadata") or {}).get("difficulty_proxy"),
        "correct": False,
        "legal": False,
        "steps": 0,
        "errors": 0,
        "failure_type": None,
        "fail": None,
        "turns": turns,
        "error_events": error_events,
        "outcome": None,
    }

    while action_count < max_steps:
        action_count += 1
        state_before = ctx["environment"].snapshot()
        if context_mode == "rolling-legal-history":
            model_input = rolling_legal_history_messages(
                system_prompt,
                dataset_overview,
                ex["question"],
                state_before,
                last_error,
                external_knowledge,
                legal_history,
                history_turns,
            )
        else:
            model_input = model_context_messages(
                system_prompt,
                dataset_overview,
                ex["question"],
                state_before,
                last_error,
                external_knowledge,
            )
        turn = {"turn_index": len(turns), "model_input": deepcopy(model_input)}
        try:
            text, call_usage, reasoning_content = chat_with_retries(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=model_input,
                max_tokens=max_tokens,
                timeout=api_timeout,
                retries=api_retries,
            )
            add_usage(usage, call_usage)
            turn["api_finish_reason"] = call_usage.get("api_finish_reason")
        except ContextOverflowError as exc:
            rec.update({
                "failure_type": "context_overflow",
                "fail": f"api: {type(exc).__name__}: {exc}",
                "steps": action_count - 1,
                "errors": errors,
            })
            turn["api_error"] = rec["fail"]
            turn["api_error_type"] = "context_overflow"
            turns.append(turn)
            break
        except Exception as exc:  # noqa: BLE001
            rec.update({
                "failure_type": "api_error",
                "fail": f"api: {type(exc).__name__}: {exc}",
                "steps": action_count - 1,
                "errors": errors,
            })
            turn["api_error"] = rec["fail"]
            turn["api_error_type"] = "api_error"
            turns.append(turn)
            break

        raw_model_output = text
        text, adapter_record = adapt_provider_response(model, raw_model_output, reasoning_content)
        turn["raw_model_output"] = raw_model_output
        turn["response_adapter"] = adapter_record
        turn["model_output"] = text
        if reasoning_content:
            turn["provider_reasoning_content"] = reasoning_content
        messages.append({"role": "assistant", "content": text})
        try:
            think, tool, args = parse_assistant_strict(text)
            think_source = "model"
            turn["parsed"] = {"think": think, "tool": tool, "arguments": args}
            turn["think_source"] = think_source
            turn["feedback_recovery"] = bool(last_error)
            turn["recovered_from_error_type"] = (last_error or {}).get("error", {}).get("type")
            if tool == "answer_from_context":
                rec["legal"] = True
                rec["steps"] = action_count
                rec["errors"] = errors
                correct, pred_sample, gold_sample = score(h, gold_sql, args, created)
                rec["correct"] = correct
                rec["pred_sample"] = pred_sample
                rec["gold_sample"] = gold_sample
                if not correct:
                    rec["failure_type"] = "wrong_answer"
                else:
                    rec["outcome"] = "recovered_success" if error_events else "clean_success"
                turns.append(turn)
                steps.append({
                    "step_id": f"step_{action_count}",
                    "think": think,
                    "think_source": think_source,
                    "tool_call": {"tool": tool, "arguments": args},
                    "tool_output": {"final_answer": args.get("answer")},
                    "environment_state_before": state_before,
                    "environment_state": ctx["environment"].snapshot(),
                    "last_tool_error_before": last_error,
                    "feedback_recovery": bool(last_error),
                    "recovered_from_error_type": (last_error or {}).get("error", {}).get("type"),
                })
                break

            step_id = f"step_{action_count}"
            out, table_name = execute_tool(
                h,
                tool,
                args,
                ctx,
                step_id,
                table_output_rows=table_output_rows,
            )
            turn["tool_output"] = out
            steps.append({
                "step_id": step_id,
                "think": think,
                "think_source": think_source,
                "tool_call": {"tool": tool, "arguments": args},
                "tool_output": out,
                "environment_state_before": state_before,
                "environment_state": ctx["environment"].snapshot(),
                "last_tool_error_before": last_error,
                "feedback_recovery": bool(last_error),
                "recovered_from_error_type": (last_error or {}).get("error", {}).get("type"),
            })
            turns.append(turn)
            last_error = None
            successful_tool_steps += 1
            legal_history.append({
                "assistant": text,
                "observation": tool_output_message(step_id, out),
            })
            if table_name:
                created.add(table_name)
            messages.append({"role": "user", "content": tool_output_message(step_id, out)})
        except (ProtocolError, Exception) as exc:  # noqa: BLE001
            errors += 1
            parsed = turn.get("parsed") or {}
            state_after = ctx["environment"].snapshot()
            error_type = protocol_failure_type(exc) if isinstance(exc, ProtocolError) else "execution_error"
            if error_type == "execution_error" and compact_json(state_after) != compact_json(state_before):
                error_type = "nonrecoverable_execution_error"
            message = format_tool_error(exc, h, parsed.get("tool"), parsed.get("arguments"))
            turn["execution_error"] = message
            turn["execution_error_type"] = error_type
            event = error_event(
                action_count,
                error_type,
                message,
                state_before,
                state_after,
                parsed.get("tool"),
                parsed.get("arguments"),
            )
            turn["error_event"] = event
            error_events.append(event)
            turns.append(turn)

            step_id = f"step_{action_count}"
            if error_type == "nonrecoverable_execution_error":
                rec.update({
                    "failure_type": error_type,
                    "fail": message,
                    "steps": action_count,
                    "errors": errors,
                })
                break
            error_counts[error_type] += 1
            if error_counts[error_type] >= max_errors_per_type:
                rec.update({
                    "failure_type": error_type,
                    "fail": f"aborted after {error_counts[error_type]} {error_type} events: {message}",
                    "steps": action_count,
                    "errors": errors,
                })
                break
            last_error = {
                "step_id": step_id,
                "status": "error",
                "error": {"type": error_type, "message": message},
            }
            messages.append({"role": "user", "content": observation_for_error(step_id, error_type, message)})
            continue
    else:
        rec.update({
            "failure_type": "max_steps",
            "fail": "max_steps",
            "steps": action_count,
            "errors": errors,
        })

    rec["final_messages"] = messages
    rec["elapsed_seconds"] = round(time.time() - started, 3)
    rec["usage"] = dict(usage)
    if rec["correct"]:
        adapter_turns = [turn.get("response_adapter") or {} for turn in turns]
        rec["trajectory"] = {
            "trajectory_id": rec["trajectory_id"],
            "schema_version": "v4-external-rollout",
            "source": {
                "dataset": ex.get("dataset", "spider"),
                "split": ex.get("split", split),
                "example_id": ex.get("example_id"),
                "db_id": ex["db_id"],
                "db_path": task_path,
                "external_knowledge": external_knowledge,
                "gold_sql": gold_sql,
            },
            "question": ex["question"],
            "difficulty": rec["difficulty"],
            "label_status": "verified",
            "initial_state": {"dataset_overview": dataset_overview},
            "steps": steps,
            "rollout_generation": {
                "method": "external_llm_closed_loop",
                "model": model,
                "protocol_hash": protocol_hash(system_prompt),
                "context_mode": context_mode,
                "history_turns": history_turns,
                "rolling_prompt_variant": rolling_prompt_variant,
                "sft_export_eligible": context_mode == "state-only",
                "error_actions_are_sft_targets": False,
                "errors": errors,
                "successful_tool_steps": successful_tool_steps,
                "action_count": action_count,
                "error_events": error_events,
                "error_counts": dict(error_counts),
                "provider_adapter": {
                    "applied_actions": sum(bool(adapter.get("applied")) for adapter in adapter_turns),
                    "unusable_actions": sum(
                        adapter.get("name") != "none" and not adapter.get("applied")
                        for adapter in adapter_turns
                    ),
                    "names": sorted({adapter.get("name") for adapter in adapter_turns if adapter.get("name")}),
                },
                "outcome": rec["outcome"],
                "elapsed_seconds": rec["elapsed_seconds"],
                "usage": dict(usage),
            },
        }
    return rec


def read_completed(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    # New pass@k-style generation marks an example complete only after either a
                    # correct attempt or after exhausting attempts_per_example. Older all.jsonl
                    # files did not carry this field; treat those records as completed so --resume
                    # remains backward-compatible.
                    if record.get("example_complete", "legacy") in (True, "legacy"):
                        done.add(record.get("trajectory_id"))
    return done


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def audit_summary(path: Path) -> dict:
    """Aggregate all persisted attempts so a resumed run has an honest manifest."""
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    final_by_trajectory: dict[str, dict] = {}
    for record in records:
        trajectory_id = str(record.get("trajectory_id"))
        previous = final_by_trajectory.get(trajectory_id)
        # A higher attempt index is a declared whole-episode retry and is eligible for success@k.
        # A duplicate write of the same attempt must never overwrite its first observed outcome.
        if previous is None or int(record.get("attempt_index", 1)) > int(previous.get("attempt_index", 1)):
            final_by_trajectory[trajectory_id] = record
    finals = list(final_by_trajectory.values())
    raw_counts = collections.Counter()
    final_counts = collections.Counter()
    usage_total = collections.Counter()
    for record in records:
        raw_counts["total"] += 1
        raw_counts["correct" if record.get("correct") else "failed"] += 1
        if record.get("legal"):
            raw_counts["legal"] += 1
        if record.get("failure_type"):
            raw_counts[f"failure:{record['failure_type']}"] += 1
        add_usage(usage_total, record.get("usage") or {})
    for record in finals:
        final_counts["examples"] += 1
        final_counts["examples_correct" if record.get("correct") else "examples_failed"] += 1
        if record.get("failure_type"):
            final_counts[f"final_failure:{record['failure_type']}"] += 1
    return {
        "raw_attempt_records": len(records),
        "unique_examples": len(finals),
        "duplicate_attempt_records": len(records) - len(finals),
        "counts": dict(raw_counts + final_counts),
        "usage_total": dict(usage_total),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--examples-file", default="",
                        help="optional JSON from build_rollout_examples.py; overrides --split source")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", required=True, help="verified trajectory JSONL output")
    parser.add_argument("--failures-out", default="", help="default: OUT.failures.jsonl")
    parser.add_argument("--all-out", default="", help="default: OUT.all.jsonl")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-errors-per-type", type=int,
                        default=DEFAULT_MAX_ERRORS_PER_TYPE,
                        help="terminate only after this many recoverable errors of one class")
    parser.add_argument("--attempts-per-example", type=int, default=1,
                        help="whole-trajectory attempts per example; stop early once verifier-correct")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="provider default when omitted; explicit value always wins")
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--table-output-rows", type=int, default=0)
    parser.add_argument("--context-mode", choices=["state-only", "rolling-legal-history"],
                        default="state-only")
    parser.add_argument("--history-turns", type=int, default=4,
                        help="number of successful assistant/tool pairs to retain; 0 keeps all")
    parser.add_argument("--rolling-prompt-variant", choices=["full", "compact"], default="full",
                        help="rolling-only prompt ablation; full preserves existing runs")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_tokens is None:
        args.max_tokens = provider_default_max_tokens(args.model, DEFAULT_MAX_TOKENS)

    api_key, base_url = load_api()
    if not api_key or not base_url:
        parser.error("api.md must define API_KEY and BASE_URL")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failure_path = Path(args.failures_out) if args.failures_out else out_path.with_suffix(".failures.jsonl")
    all_path = Path(args.all_out) if args.all_out else out_path.with_suffix(".all.jsonl")
    manifest_path = out_path.with_suffix(".manifest.json")

    examples = load_examples(args)
    completed = read_completed(all_path) if args.resume else set()
    work = [
        (i, ex) for i, ex in examples
        if trajectory_id(args.split, i, ex) not in completed
    ]
    base_system_prompt = get_system_prompt()
    if args.context_mode == "rolling-legal-history":
        base_system_prompt = rolling_system_prompt(
            base_system_prompt,
            compact=args.rolling_prompt_variant == "compact",
        )
    system_prompt = base_system_prompt + DATA_GENERATION_SUFFIX + provider_instruction(args.model)
    started = time.time()
    counts = collections.Counter()
    tool_hist = collections.Counter()
    error_turn_hist = collections.Counter()
    usage_total = collections.Counter()

    def process_once(item: tuple[int, dict], attempt_index: int) -> dict:
        i, ex = item
        rec = run_rollout(
            example_index=i,
            ex=ex,
            split=args.split,
            base_url=base_url,
            api_key=api_key,
            model=args.model,
            system_prompt=system_prompt,
            max_steps=args.max_steps,
            max_tokens=args.max_tokens,
            api_retries=args.api_retries,
            api_timeout=args.api_timeout,
            max_errors_per_type=args.max_errors_per_type,
            table_output_rows=args.table_output_rows,
            context_mode=args.context_mode,
            history_turns=args.history_turns,
            rolling_prompt_variant=args.rolling_prompt_variant,
        )
        rec["attempt_index"] = attempt_index
        rec["attempts_per_example"] = max(1, args.attempts_per_example)
        return rec

    def process(item: tuple[int, dict]) -> list[dict]:
        attempts = []
        max_attempts = max(1, args.attempts_per_example)
        for attempt_index in range(1, max_attempts + 1):
            rec = process_once(item, attempt_index)
            rec["example_complete"] = bool(rec.get("correct")) or attempt_index == max_attempts
            attempts.append(rec)
            if rec.get("correct"):
                break
        return attempts

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(process, item) for item in work]
        for done, future in enumerate(as_completed(futures), 1):
            attempts = future.result()
            final = attempts[-1]
            counts["examples"] += 1
            counts["examples_correct" if final.get("correct") else "examples_failed"] += 1
            counts["attempts_total"] += len(attempts)
            for rec in attempts:
                counts["total"] += 1
                counts["correct" if rec.get("correct") else "failed"] += 1
                if rec.get("legal"):
                    counts["legal"] += 1
                if rec.get("failure_type"):
                    counts[f"failure:{rec['failure_type']}"] += 1
                add_usage(usage_total, rec.get("usage") or {})
                if rec.get("correct"):
                    traj = rec["trajectory"]
                    traj["rollout_generation"]["attempt_index"] = rec.get("attempt_index")
                    traj["rollout_generation"]["attempts_per_example"] = rec.get("attempts_per_example")
                    append_jsonl(out_path, traj)
                    for step in traj.get("steps", []):
                        tool_hist[step["tool_call"]["tool"]] += 1
                        if step.get("think_source"):
                            counts[f"think_source:{step['think_source']}"] += 1
                        if step.get("tool_status") == "error":
                            error_turn_hist[step["tool_call"]["tool"]] += 1
                else:
                    append_jsonl(failure_path, rec)
                append_jsonl(all_path, {
                    k: v for k, v in rec.items()
                    if k != "trajectory"
                })
            status = "OK " if final.get("correct") else "ERR"
            print(
                f"[{done}/{len(work)}] {status} attempts={len(attempts)} "
                f"steps={final.get('steps')} errs={final.get('errors')} "
                f"type={final.get('failure_type')} {final.get('trajectory_id')}"
            )

    persisted = audit_summary(all_path)
    manifest = {
        "generator": "src/sft/rollout_external_data.py",
        "method": "external_llm_closed_loop",
        "model": args.model,
        "split": args.split,
        "examples_file": args.examples_file or None,
        "source_count": len(examples),
        "attempted_this_run": len(work),
        "resume": args.resume,
        "output": str(out_path),
        "failures_output": str(failure_path),
        "all_output": str(all_path),
        "protocol_hash": protocol_hash(system_prompt),
        "max_steps": args.max_steps,
        "max_errors_per_type": args.max_errors_per_type,
        "attempts_per_example": max(1, args.attempts_per_example),
        "max_tokens": args.max_tokens,
        "table_output_rows": args.table_output_rows,
        "context_mode": args.context_mode,
        "history_turns": args.history_turns,
        "rolling_prompt_variant": args.rolling_prompt_variant,
        "sft_export_eligible": args.context_mode == "state-only",
        "teacher_parser": "strict_no_repair",
        "error_actions_are_sft_targets": False,
        "api_transport_retries_per_request": args.api_retries,
        "counts": persisted["counts"],
        "raw_attempt_records": persisted["raw_attempt_records"],
        "unique_examples": persisted["unique_examples"],
        "duplicate_attempt_records": persisted["duplicate_attempt_records"],
        "tool_hist": dict(tool_hist.most_common()),
        "error_turn_tool_hist": dict(error_turn_hist.most_common()),
        "usage_total": persisted["usage_total"],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if persisted["counts"].get("total"):
        manifest["accuracy"] = persisted["counts"].get("correct", 0) / persisted["counts"]["total"]
        manifest["legal_rate"] = persisted["counts"].get("legal", 0) / persisted["counts"]["total"]
        manifest["example_success_rate"] = (
            persisted["counts"].get("examples_correct", 0) /
            max(1, persisted["counts"].get("examples", 0))
        )
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
