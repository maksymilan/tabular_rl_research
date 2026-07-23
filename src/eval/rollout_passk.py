#!/usr/bin/env python3
"""Closed-loop tool-use pass@k evaluation.

Each example is rolled out independently `n_samples` times with sampling enabled. A question is
pass@k-correct if any of the first k trajectories reaches a legal answer whose denotation matches
the gold SQL result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import BoundedSemaphore
from typing import Literal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "harness"))
sys.path.insert(0, os.path.join(ROOT, "src", "sft"))

from artifacts import ArtifactWriter  # noqa: E402
from denotation import add_denotation_comparison_argument  # noqa: E402
from executor import Harness  # noqa: E402
from passk import attach_passk_fields, parse_pass_k, write_passk_summary  # noqa: E402
from protocol import (  # noqa: E402
    ProtocolError,
    assistant_message,
    first_user_message,
    get_system_prompt,
    model_context_messages,
    parse_assistant_strict,
    rolling_legal_history_messages,
    rolling_system_prompt,
    state_context_message,
    tool_error_message,
    tool_output_message,
)
from rollout import (  # noqa: E402
    ChatAPIError,
    ContextOverflowError,
    DEFAULT_MAX_TOKENS,
    MAX_ERRORS_PER_TYPE,
    MIN_CONTEXT_RETRY_TOKENS,
    execute_tool,
    format_tool_error,
    is_context_overflow,
    new_ctx,
    overview,
    score,
    protocol_failure_type,
    state_digest,
    task_db_path,
    task_gold_sql,
)

SPIDER = os.path.join(ROOT, "data", "spider_data")


def load_indexed_examples(path: str | None, n: int | None) -> tuple[list[tuple[int, dict]], str]:
    """Load eval examples.

    Default is Spider dev for held-out evaluation. For training-set recovery data generation, pass
    an explicit --examples-json produced from train_spider.json. Each item may carry example_index
    and trajectory_id; example_index is used for resume/artifact names.
    """
    if path:
        with open(path, encoding="utf-8") as source:
            if path.endswith(".jsonl"):
                payload = [json.loads(line) for line in source if line.strip()]
            else:
                payload = json.load(source)
        examples = payload.get("examples", payload) if isinstance(payload, dict) else payload
        if not isinstance(examples, list):
            raise ValueError("--examples-json must be a JSON list or {'examples': [...]}")
        if n is not None:
            examples = examples[:n]
        out = []
        for local_index, ex in enumerate(examples):
            if not isinstance(ex, dict):
                raise ValueError(f"--examples-json item {local_index} is not an object")
            example_index = int(ex.get("example_index", local_index))
            out.append((example_index, ex))
        return out, path
    dev = json.load(open(os.path.join(SPIDER, "dev.json"), encoding="utf-8"))
    examples = dev[: n if n is not None else len(dev)]
    return list(enumerate(examples)), "data/spider_data/dev.json"


def chat_sample(
    base_url: str,
    model: str,
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    retries: int,
    min_context_retry_tokens: int = MIN_CONTEXT_RETRY_TOKENS,
    retry_stats: dict | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }
    think = os.environ.get("EVAL_ENABLE_THINKING")
    if think is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": think == "1"}

    current_max_tokens = max_tokens
    transient_attempts = 0
    context_retries = 0
    while True:
        payload["max_tokens"] = current_max_tokens
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as response:
                data = json.loads(response.read())
            if retry_stats is not None:
                retry_stats.update({
                    "api_request_attempts": transient_attempts + context_retries + 1,
                    "api_transport_retries": transient_attempts,
                    "api_context_retries": context_retries,
                })
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            body_text = exc.read().decode("utf-8", errors="replace")
            if is_context_overflow(body_text):
                if current_max_tokens > min_context_retry_tokens:
                    current_max_tokens = max(
                        min_context_retry_tokens,
                        current_max_tokens // 2,
                    )
                    context_retries += 1
                    continue
                raise ContextOverflowError(
                    f"HTTP {exc.code}: {body_text}", status=exc.code, body=body_text
                ) from exc
            if exc.code >= 500 and transient_attempts < retries:
                transient_attempts += 1
                time.sleep(min(2 ** transient_attempts, 8))
                continue
            raise ChatAPIError(f"HTTP {exc.code}: {body_text}", status=exc.code, body=body_text) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if transient_attempts < retries:
                transient_attempts += 1
                time.sleep(min(2 ** transient_attempts, 8))
                continue
            raise ChatAPIError(f"{type(exc).__name__}: {exc}") from exc


def run_sample(
    ex: dict,
    sample_index: int,
    base_url: str,
    model: str,
    system: str,
    *,
    request_semaphore: BoundedSemaphore | None = None,
    max_steps: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    api_retries: int,
    context_mode: str,
    history_turns: int,
    compact_history_observations: bool,
    denotation_comparison: str,
) -> dict:
    gold_sql = task_gold_sql(ex)
    if not gold_sql:
        raise ValueError(f"task has no gold SQL: {ex.get('db_id')} / {ex.get('question')}")
    h = Harness(task_db_path(ex))
    ov = overview(h)
    external_knowledge = ex.get("external_knowledge")
    legal_history: list[dict] = []
    created: set[str] = set()
    ctx = new_ctx(ov)
    last_error: dict | None = None
    action_count = errors = 0
    error_counts: dict[str, int] = {}
    error_events: list[dict] = []
    turns = []
    started = time.time()
    rec = {
        "sample_index": sample_index,
        "denotation_comparison": denotation_comparison,
        "correct": False,
        "legal": False,
        "steps": 0,
        "errors": 0,
        "failure_type": None,
        "turns": turns,
        "error_events": error_events,
        "outcome": None,
        "api_transport_retries": 0,
        "api_context_retries": 0,
    }
    while action_count < max_steps:
        action_count += 1
        state_before = ctx["environment"].snapshot()
        if context_mode == "rolling-legal-history":
            model_input = rolling_legal_history_messages(
                system,
                ov,
                ex["question"],
                state_before,
                last_error,
                external_knowledge,
                legal_history,
                history_turns,
                compact_observations=compact_history_observations,
            )
        else:
            model_input = model_context_messages(
                system, ov, ex["question"], state_before, last_error, external_knowledge,
            )
        turn = {"turn_index": len(turns), "model_input": deepcopy(model_input)}
        try:
            retry_stats: dict = {}
            if request_semaphore is None:
                text = chat_sample(
                    base_url,
                    model,
                    model_input,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    retries=api_retries,
                    retry_stats=retry_stats,
                )
            else:
                with request_semaphore:
                    text = chat_sample(
                        base_url,
                        model,
                        model_input,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        retries=api_retries,
                        retry_stats=retry_stats,
                    )
        except ContextOverflowError as exc:
            rec["failure_type"] = "context_overflow"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            break
        except Exception as exc:  # noqa: BLE001
            rec["failure_type"] = "api_error"
            rec["error"] = f"{type(exc).__name__}: {exc}"
            break

        turn["model_output"] = text
        turn["api_retry_stats"] = retry_stats
        rec["api_transport_retries"] += retry_stats["api_transport_retries"]
        rec["api_context_retries"] += retry_stats["api_context_retries"]
        try:
            think, tool, args = parse_assistant_strict(text)
            turn["parsed"] = {"think": think, "tool": tool, "arguments": args}
            turn["feedback_recovery"] = bool(last_error)
            turn["recovered_from_error_type"] = (last_error or {}).get("error", {}).get("type")
            if tool == "answer_from_context":
                rec["legal"] = True
                rec["steps"] = action_count
                rec["errors"] = errors
                rec["correct"], rec["pred_sample"], rec["gold_sample"] = score(
                    h, gold_sql, args, created, denotation_comparison
                )
                if not rec["correct"]:
                    rec["failure_type"] = "wrong_answer"
                else:
                    rec["outcome"] = "recovered_success" if error_events else "clean_success"
                turns.append(turn)
                rec["elapsed_seconds"] = round(time.time() - started, 3)
                return rec

            step_id = f"step_{action_count}"
            out, table_name = execute_tool(h, tool, args, ctx, step_id)
            turn["tool_output"] = out
        except (ProtocolError, Exception) as exc:  # noqa: BLE001
            errors += 1
            parsed = turn.get("parsed") or {}
            error = format_tool_error(exc, h, parsed.get("tool"), parsed.get("arguments"))
            state_after = ctx["environment"].snapshot()
            error_type = protocol_failure_type(exc) if isinstance(exc, ProtocolError) else "execution_error"
            if error_type == "execution_error" and state_digest(state_after) != state_digest(state_before):
                error_type = "nonrecoverable_execution_error"
            turn["execution_error"] = error
            turn["execution_error_type"] = error_type
            event = {
                "action_index": action_count,
                "step_id": f"step_{action_count}",
                "error_type": error_type,
                "message": error,
                "state_before_hash": state_digest(state_before),
                "state_after_hash": state_digest(state_after),
            }
            if parsed.get("tool"):
                event["attempted_tool"] = parsed["tool"]
                event["attempted_arguments"] = parsed.get("arguments") or {}
            turn["error_event"] = event
            error_events.append(event)
            turns.append(turn)
            if error_type == "nonrecoverable_execution_error":
                rec["failure_type"] = error_type
                rec["error"] = error
                rec["errors"] = errors
                rec["steps"] = action_count
                rec["elapsed_seconds"] = round(time.time() - started, 3)
                return rec
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            if error_counts[error_type] >= MAX_ERRORS_PER_TYPE:
                rec["failure_type"] = error_type
                rec["error"] = f"aborted after {error_counts[error_type]} {error_type} events: {error}"
                rec["errors"] = errors
                rec["steps"] = action_count
                rec["elapsed_seconds"] = round(time.time() - started, 3)
                return rec
            last_error = {
                "step_id": f"step_{action_count}",
                "status": "error",
                "error": {"type": error_type, "message": error},
            }
            continue

        turns.append(turn)
        last_error = None
        if table_name:
            created.add(table_name)
        legal_history.append({
            "assistant": assistant_message(think, tool, args),
            "observation": tool_output_message(step_id, out),
        })

    if rec["failure_type"] is None:
        rec["failure_type"] = "max_steps"
    rec["steps"] = action_count
    rec["errors"] = errors
    rec["elapsed_seconds"] = round(time.time() - started, 3)
    return rec


SampleDetail = Literal["full", "compact", "none"]


def stored_sample(sample: dict, detail: SampleDetail) -> dict:
    """Return the artifact representation for one pass@k sample."""
    if detail == "full":
        return sample
    record = {key: value for key, value in sample.items() if key != "turns"}
    if detail == "none":
        record["turn_count"] = len(sample.get("turns") or [])
        return record
    turns = []
    for turn in sample.get("turns") or []:
        slim_turn = {key: value for key, value in turn.items() if key != "model_input"}
        turns.append(slim_turn)
    record["turns"] = turns
    return record


def run_one(
    ex: dict,
    example_index: int,
    base_url: str,
    model: str,
    system: str,
    *,
    n_samples: int,
    sample_workers: int,
    first_sample_workers: int,
    stop_on_success: bool,
    pass_k: tuple[int, ...],
    request_semaphore: BoundedSemaphore | None,
    sample_detail: SampleDetail,
    max_steps: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    api_retries: int,
    context_mode: str,
    history_turns: int,
    compact_history_observations: bool,
    denotation_comparison: str,
) -> dict:
    gold_sql = task_gold_sql(ex)
    if not gold_sql:
        raise ValueError(f"task has no gold SQL: {ex.get('db_id')} / {ex.get('question')}")
    initial_harness = Harness(task_db_path(ex))
    initial_overview = overview(initial_harness)
    initial_state = new_ctx(initial_overview)["environment"].snapshot()
    if context_mode == "rolling-legal-history":
        initial_input = rolling_legal_history_messages(
            system, initial_overview, ex["question"], initial_state, None,
            ex.get("external_knowledge"), [], history_turns,
            compact_observations=compact_history_observations,
        )
    else:
        initial_input = model_context_messages(
            system, initial_overview, ex["question"], initial_state, None,
            ex.get("external_knowledge"),
        )
    started = time.time()
    record = {
        "example_index": example_index,
        "trajectory_id": ex.get("trajectory_id"),
        "dataset_split": ex.get("dataset_split") or ex.get("split"),
        "db_id": ex["db_id"],
        "question": ex["question"],
        "gold_sql": gold_sql,
        "initial_model_input": initial_input,
        "n_samples": n_samples,
        "sample_workers": sample_workers,
        "stop_on_success": stop_on_success,
        "pass_k": list(pass_k),
        "temperature": temperature,
        "top_p": top_p,
        "max_steps": max_steps,
        "max_tokens": max_tokens,
        "denotation_comparison": denotation_comparison,
        "samples": [],
        "correct": False,
        "failure_type": None,
    }
    sample_kwargs = {
        "request_semaphore": request_semaphore,
        "max_steps": max_steps,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "api_retries": api_retries,
        "context_mode": context_mode,
        "history_turns": history_turns,
        "compact_history_observations": compact_history_observations,
        "denotation_comparison": denotation_comparison,
    }
    if stop_on_success:
        samples = []
        start = 0
        first_chunk = min(n_samples, first_sample_workers or sample_workers)
        while start < n_samples:
            chunk_size = first_chunk if start == 0 else sample_workers
            end = min(n_samples, start + chunk_size)
            pending: list[dict | None] = [None] * (end - start)
            with ThreadPoolExecutor(max_workers=max(1, end - start)) as pool:
                futures = {
                    pool.submit(run_sample, ex, sample_index, base_url, model, system, **sample_kwargs): sample_index
                    for sample_index in range(start, end)
                }
                for future in as_completed(futures):
                    sample_index = futures[future]
                    pending[sample_index - start] = future.result()
            samples.extend(sample for sample in pending if sample is not None)
            if any(sample and sample["correct"] for sample in pending):
                break
            start = end
    else:
        pending: list[dict | None] = [None] * n_samples
        with ThreadPoolExecutor(max_workers=sample_workers) as pool:
            futures = {
                pool.submit(run_sample, ex, sample_index, base_url, model, system, **sample_kwargs): sample_index
                for sample_index in range(n_samples)
            }
            for future in as_completed(futures):
                pending[futures[future]] = future.result()
        samples = [sample for sample in pending if sample is not None]
    samples = [stored_sample(sample, sample_detail) for sample in samples]
    attach_passk_fields(record, samples, pass_k)
    record["attempted_samples"] = len(samples)
    record["elapsed_seconds"] = round(time.time() - started, 3)
    return record


def fewshot_text(trajectory_ids: list[str]) -> str:
    if not trajectory_ids:
        return ""
    wanted = set(trajectory_ids)
    found = {}
    with open(os.path.join(ROOT, "data", "trajectories", "spider_train_v2.jsonl")) as source:
        for line in source:
            trajectory = json.loads(line)
            if trajectory["trajectory_id"] in wanted:
                found[trajectory["trajectory_id"]] = trajectory
    missing = [trajectory_id for trajectory_id in trajectory_ids if trajectory_id not in found]
    if missing:
        raise ValueError(f"few-shot trajectories not found: {missing}")
    blocks = []
    for trajectory_id in trajectory_ids:
        trajectory = found[trajectory_id]
        overview_payload = trajectory["initial_state"]["dataset_overview"]
        lines = [f"USER: {first_user_message(overview_payload, trajectory['question'])}"]
        for index, step in enumerate(trajectory["steps"]):
            if index > 0:
                previous = trajectory["steps"][index - 1]
                lines.append(
                    "USER: "
                    + state_context_message(step.get("environment_state_before", previous.get("environment_state")))
                )
            tool_call = step["tool_call"]
            lines.append(
                "ASSISTANT: "
                + assistant_message(step.get("think", ""), tool_call["tool"], tool_call["arguments"])
            )
        blocks.append("\n".join(lines))
    return "\n\nEXAMPLE SESSIONS\n" + "\n\n---\n\n".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=None,
                        help="number of examples; default = all examples in the selected input")
    parser.add_argument("--examples-json", default=None,
                        help=("explicit JSON examples to run instead of Spider dev; use this for "
                              "training-subset recovery rollouts"))
    parser.add_argument("--allow-eval-tasks", action="store_true",
                        help="explicitly allow dev/eval DatasetTask inputs; outputs are evaluation-only")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--sample-workers", type=int, default=1,
                        help="parallel rollouts per question; pass@k is computed in sample_index order")
    parser.add_argument("--first-sample-workers", type=int, default=0,
                        help=("initial sample chunk per question when --stop-on-success is active; "
                              "use 1 to screen pass@1 before spending more samples"))
    parser.add_argument("--max-inflight-requests", type=int, default=0,
                        help="global cap on simultaneous chat completion requests; 0 means uncapped")
    parser.add_argument("--stop-on-success", action="store_true",
                        help="sample sequentially and stop a question after its first correct trajectory")
    parser.add_argument("--pass-k", default="2,4,8,16,32")
    parser.add_argument("--sample-detail", choices=("full", "compact", "none"), default="full",
                        help="artifact detail for per-sample turns; none is fastest for pass@k screening")
    parser.add_argument("--summary-every", type=int, default=1,
                        help="rewrite summary.json every N completed examples; 0 disables interim summaries")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--few-shot", type=int, default=0)
    parser.add_argument(
        "--context-mode", choices=["state-only", "rolling-legal-history"], default="state-only",
    )
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument(
        "--rolling-prompt-variant", choices=["full", "compact"], default="full",
    )
    parser.add_argument(
        "--rolling-observation-style", choices=["resident", "full"], default="resident",
    )
    add_denotation_comparison_argument(parser)
    args = parser.parse_args()

    try:
        pass_k = parse_pass_k(args.pass_k, n_samples=args.n_samples)
    except ValueError as exc:
        parser.error(str(exc))
    if args.sample_workers <= 0:
        parser.error("--sample-workers must be positive")
    if args.first_sample_workers < 0:
        parser.error("--first-sample-workers must be non-negative")
    if args.first_sample_workers > args.sample_workers:
        parser.error("--first-sample-workers must be <= --sample-workers")
    if args.max_inflight_requests < 0:
        parser.error("--max-inflight-requests must be non-negative")
    if args.summary_every < 0:
        parser.error("--summary-every must be non-negative")
    if args.history_turns < 0:
        parser.error("--history-turns must be non-negative")
    if args.context_mode == "rolling-legal-history" and args.few_shot:
        parser.error("few-shot examples are not supported with rolling history")
    is_eval_tasks = bool(args.examples_json and "dev" in args.examples_json.lower())
    if is_eval_tasks and not args.allow_eval_tasks:
        parser.error("--examples-json appears to be a dev/eval file; pass --allow-eval-tasks explicitly")

    prompt_variant = os.environ.get("EVAL_SYSTEM_PROMPT_VARIANT") or "default"
    system = get_system_prompt()
    if args.context_mode == "rolling-legal-history":
        system = rolling_system_prompt(
            system,
            compact=args.rolling_prompt_variant == "compact",
        )
    if args.few_shot:
        from rollout import DEFAULT_FEWSHOT_IDS

        system += fewshot_text(DEFAULT_FEWSHOT_IDS[:args.few_shot])

    manifest = {
        "runner": "tool_rollout_passk",
        "model": args.model,
        "base_url": args.base_url,
        "dataset": args.examples_json or "data/spider_data/dev.json",
        "requested_size": args.n,
        "n_samples": args.n_samples,
        "sample_workers": args.sample_workers,
        "first_sample_workers": args.first_sample_workers,
        "max_inflight_requests": args.max_inflight_requests,
        "stop_on_success": args.stop_on_success,
        "pass_k": list(pass_k),
        "sample_detail": args.sample_detail,
        "summary_every": args.summary_every,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "max_steps": args.max_steps,
        "api_retries": args.api_retries,
        "few_shot": args.few_shot,
        "enable_thinking": os.environ.get("EVAL_ENABLE_THINKING"),
        "system_prompt_variant": prompt_variant,
        "system_prompt": system,
        "context_mode": args.context_mode,
        "history_turns": args.history_turns,
        "rolling_prompt_variant": args.rolling_prompt_variant,
        "rolling_observation_style": args.rolling_observation_style,
        "denotation_comparison": args.denotation_comparison,
    }
    if is_eval_tasks:
        manifest.update({"dataset_purpose": "evaluation", "sft_export_eligible": False})
    writer = ArtifactWriter(args.result_dir, manifest, args.resume)

    indexed_examples, source_name = load_indexed_examples(args.examples_json, args.n)
    pending = [(index, example) for index, example in indexed_examples if index not in writer.completed]
    print(f"loaded {len(indexed_examples)} examples from {source_name}; pending {len(pending)}")

    request_semaphore = (
        BoundedSemaphore(args.max_inflight_requests)
        if args.max_inflight_requests > 0
        else None
    )
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_one,
                example,
                index,
                args.base_url,
                args.model,
                system,
                n_samples=args.n_samples,
                sample_workers=args.sample_workers,
                first_sample_workers=args.first_sample_workers,
                stop_on_success=args.stop_on_success,
                pass_k=pass_k,
                request_semaphore=request_semaphore,
                sample_detail=args.sample_detail,
                context_mode=args.context_mode,
                history_turns=args.history_turns,
                compact_history_observations=args.rolling_observation_style == "resident",
                max_steps=args.max_steps,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                api_retries=args.api_retries,
                denotation_comparison=args.denotation_comparison,
            )
            for index, example in pending
        ]
        for position, future in enumerate(as_completed(futures), 1):
            record = future.result()
            writer.append(record)
            flag = "OK" if record["correct"] else record["failure_type"]
            print(
                f"[{position}/{len(pending)}] {flag} q{record['example_index']} "
                f"correct_samples={record.get('sample_correct_count', 0)} "
                f"attempted={record.get('attempted_samples', 0)} "
                f"legal_samples={record.get('sample_legal_count', 0)} "
                f"{record['question'][:60]}",
                flush=True,
            )
            if args.summary_every and position % args.summary_every == 0:
                write_passk_summary(writer, pass_k, include_legal=True)

    summary = write_passk_summary(writer, pass_k, include_legal=True)
    print(json.dumps(summary["pass_at"], ensure_ascii=False, indent=2))
    print(f"-> {writer.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
