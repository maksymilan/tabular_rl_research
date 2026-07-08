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
)
from executor import Harness  # noqa: E402
from protocol import (  # noqa: E402
    ProtocolError,
    assistant_message,
    first_user_message,
    get_system_prompt,
    parse_assistant,
    protocol_hash,
    tool_output_message,
    with_environment_state,
)

SPIDER = ROOT / "data" / "spider_data"
DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_MAX_CONSECUTIVE_ERRORS = 3
DEFAULT_MAX_STEPS = 30
DEFAULT_MAX_TOKENS = 1024
MIN_CONTEXT_RETRY_TOKENS = 256
DATA_GENERATION_SUFFIX = (
    "\n\nDATA GENERATION STRICTNESS\n"
    "Your <think> block must be non-empty on every turn. Put the reason inside <think> tags, "
    "not as plain text before the tool call. The reason should be specific to the current question, "
    "visible schema/observations, and the next tool arguments. After the first turn, do not restate "
    "the original user question; continue from the latest observation or error feedback."
)


def compact_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


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
    message = data["choices"][0]["message"]
    return message.get("content") or "", data.get("usage", {}), message.get("reasoning_content") or ""


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
    for attempt in range(max(1, retries)):
        try:
            return request_chat(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=messages,
                max_tokens=budget,
                timeout=timeout,
            )
        except ContextOverflowError:
            if budget <= MIN_CONTEXT_RETRY_TOKENS:
                raise
            budget = max(MIN_CONTEXT_RETRY_TOKENS, budget // 2)
        except Exception as exc:  # noqa: BLE001 - API surfaces many transient transport errors
            last = exc
            if attempt + 1 < retries:
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
        payload = json.loads(Path(args.examples_file).read_text(encoding="utf-8"))
        examples = payload.get("examples")
        if not isinstance(examples, list):
            raise ValueError(f"{args.examples_file}: expected an object with an examples list")
        indexed = [(int(ex.get("example_index", i)), ex) for i, ex in enumerate(examples)]
    else:
        examples = json.loads(split_path(args.split).read_text(encoding="utf-8"))
        indexed = list(enumerate(examples))

    if args.start:
        indexed = indexed[args.start:]
    if args.limit:
        indexed = indexed[: args.limit]
    return [(i, ex) for i, ex in indexed if Path(db_path(ex["db_id"])).exists()]


def trajectory_id(split: str, example_index: int, ex: dict) -> str:
    return str(ex.get("trajectory_id") or f"spider_{split}_{example_index}")


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
    max_consecutive_errors: int,
    table_output_rows: int,
    include_error_turns: bool,
) -> dict:
    h = Harness(db_path(ex["db_id"]))
    dataset_overview = overview(h)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_user_message(dataset_overview, ex["question"])},
    ]
    created: set[str] = set()
    ctx = new_ctx(dataset_overview)
    turns: list[dict] = []
    steps: list[dict] = []
    errors = consecutive = 0
    successful_tool_steps = 0
    usage = collections.Counter()
    started = time.time()
    rec = {
        "example_index": example_index,
        "trajectory_id": trajectory_id(split, example_index, ex),
        "db_id": ex["db_id"],
        "question": ex["question"],
        "gold_sql": ex["query"],
        "correct": False,
        "legal": False,
        "steps": 0,
        "errors": 0,
        "failure_type": None,
        "fail": None,
        "turns": turns,
    }

    while successful_tool_steps < max_steps:
        model_input = with_environment_state(messages, ctx["environment"].snapshot())
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
        except ContextOverflowError as exc:
            rec.update({
                "failure_type": "context_overflow",
                "fail": f"api: {type(exc).__name__}: {exc}",
                "steps": successful_tool_steps,
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
                "steps": successful_tool_steps,
                "errors": errors,
            })
            turn["api_error"] = rec["fail"]
            turn["api_error_type"] = "api_error"
            turns.append(turn)
            break

        turn["model_output"] = text
        if reasoning_content:
            turn["reasoning_content"] = reasoning_content
        messages.append({"role": "assistant", "content": text})
        try:
            think, tool, args = parse_assistant(text)
            think, think_source = recover_think(text, think, tool, args, reasoning_content)
            turn["parsed"] = {"think": think, "tool": tool, "arguments": args}
            turn["think_source"] = think_source
            if tool == "answer_from_context":
                rec["legal"] = True
                rec["steps"] = successful_tool_steps + 1
                rec["errors"] = errors
                correct, pred_sample, gold_sample = score(h, ex["query"], args, created)
                rec["correct"] = correct
                rec["pred_sample"] = pred_sample
                rec["gold_sample"] = gold_sample
                if not correct:
                    rec["failure_type"] = "wrong_answer"
                turns.append(turn)
                steps.append({
                    "step_id": f"step_{successful_tool_steps + 1}",
                    "think": think,
                    "think_source": think_source,
                    "tool_call": {"tool": tool, "arguments": args},
                    "tool_output": {"final_answer": args.get("answer")},
                })
                break

            step_id = f"step_{successful_tool_steps + 1}"
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
            })
            turns.append(turn)
            consecutive = 0
            successful_tool_steps += 1
            if table_name:
                created.add(table_name)
            messages.append({"role": "user", "content": tool_output_message(step_id, out)})
        except (ProtocolError, Exception) as exc:  # noqa: BLE001
            errors += 1
            consecutive += 1
            parsed = turn.get("parsed") or {}
            error_type = "protocol_error" if isinstance(exc, ProtocolError) else "execution_error"
            message = format_tool_error(exc, h, parsed.get("tool"), parsed.get("arguments"))
            turn["execution_error"] = message
            turn["execution_error_type"] = error_type
            turns.append(turn)

            step_id = f"step_{successful_tool_steps + 1}"
            # Protocol errors are bad assistant demonstrations; do not train on malformed output.
            if include_error_turns and error_type == "execution_error" and parsed:
                steps.append({
                    "step_id": step_id,
                    "think": parsed.get("think", ""),
                    "think_source": turn.get("think_source", "unknown"),
                    "tool_call": {
                        "tool": parsed.get("tool"),
                        "arguments": parsed.get("arguments", {}),
                    },
                    "tool_status": "error",
                    "tool_output": {"error": {"type": error_type, "message": message}},
                })
            if consecutive >= max_consecutive_errors:
                rec.update({
                    "failure_type": error_type,
                    "fail": f"aborted after {consecutive} consecutive errors: {message}",
                    "steps": successful_tool_steps,
                    "errors": errors,
                })
                break
            messages.append({"role": "user", "content": observation_for_error(step_id, error_type, message)})
            continue
    else:
        rec.update({
            "failure_type": "max_steps",
            "fail": "max_steps",
            "steps": successful_tool_steps,
            "errors": errors,
        })

    rec["final_messages"] = messages
    rec["elapsed_seconds"] = round(time.time() - started, 3)
    rec["usage"] = dict(usage)
    if rec["correct"]:
        rec["trajectory"] = {
            "trajectory_id": rec["trajectory_id"],
            "schema_version": "v4-external-rollout",
            "source": {
                "dataset": "spider",
                "split": split,
                "db_id": ex["db_id"],
                "gold_sql": ex["query"],
            },
            "question": ex["question"],
            "label_status": "verified",
            "initial_state": {"dataset_overview": dataset_overview},
            "steps": steps,
            "rollout_generation": {
                "method": "external_llm_closed_loop",
                "model": model,
                "protocol_hash": protocol_hash(),
                "include_error_turns": include_error_turns,
                "errors": errors,
                "successful_tool_steps": successful_tool_steps,
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
    parser.add_argument("--max-consecutive-errors", type=int,
                        default=DEFAULT_MAX_CONSECUTIVE_ERRORS)
    parser.add_argument("--attempts-per-example", type=int, default=1,
                        help="whole-trajectory attempts per example; stop early once verifier-correct")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--table-output-rows", type=int, default=0)
    parser.add_argument("--drop-error-turns", action="store_true",
                        help="do not keep parsed execution-error calls as SFT turns")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

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
    system_prompt = get_system_prompt() + DATA_GENERATION_SUFFIX
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
            max_consecutive_errors=args.max_consecutive_errors,
            table_output_rows=args.table_output_rows,
            include_error_turns=not args.drop_error_turns,
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
        "protocol_hash": protocol_hash(),
        "max_steps": args.max_steps,
        "max_consecutive_errors": args.max_consecutive_errors,
        "attempts_per_example": max(1, args.attempts_per_example),
        "max_tokens": args.max_tokens,
        "table_output_rows": args.table_output_rows,
        "include_error_turns": not args.drop_error_turns,
        "counts": dict(counts),
        "tool_hist": dict(tool_hist.most_common()),
        "error_turn_tool_hist": dict(error_turn_hist.most_common()),
        "usage_total": dict(usage_total),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if counts["total"]:
        manifest["accuracy"] = counts["correct"] / counts["total"]
        manifest["legal_rate"] = counts["legal"] / counts["total"]
        manifest["example_success_rate"] = counts["examples_correct"] / max(1, counts["examples"])
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
