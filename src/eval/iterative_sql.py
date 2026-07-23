#!/usr/bin/env python3
"""Evaluate an external model through a minimal iterative-SQL feedback interface.

This is an interface ablation, not a model-visible extension of the typed tool protocol. The model
starts from the same lazy BIRD catalog and may execute read-only SQLite statements to inspect schema,
probe data, and debug errors before submitting one final SQL query for hidden denotation scoring.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "sft"))

from artifacts import ArtifactWriter  # noqa: E402
from denotation import add_denotation_comparison_argument, compare_denotations  # noqa: E402
from executor import Harness  # noqa: E402
from generate_teacher_rollouts import add_usage, chat_with_retries  # noqa: E402
from protocol import ProtocolError  # noqa: E402
from provider_adapter import (  # noqa: E402
    adapt_provider_response,
    is_deepseek_split_model,
    provider_default_max_tokens,
    provider_instruction,
    provider_request_messages,
    provider_request_options,
    provider_rejection_message,
)
from provider_client import load_api_config  # noqa: E402
from rollout import (  # noqa: E402
    ChatAPIError,
    ContextOverflowError,
    overview,
    task_db_path,
    task_gold_sql,
)
from text2sql import execute_predicted_sql  # noqa: E402


SQL_TOOLS = frozenset({"execute_sql", "submit_sql"})
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
READ_ONLY_RE = re.compile(r"^(?:SELECT|WITH|PRAGMA|EXPLAIN\s+QUERY\s+PLAN)\b", re.I)
SUBMIT_RE = re.compile(r"^(?:SELECT|WITH)\b", re.I)

SYSTEM_PROMPT = """You are a SQLite data-analysis agent. Solve the user's question by iteratively
executing read-only SQL against the provided database and using the real results or errors to improve
the next query. The opening catalog contains table names, row counts, and foreign-key relations but
not full columns. Inspect schemas with execute_sql using PRAGMA table_info('TableName') or SQLite
catalog queries before relying on column names. You may also execute bounded SELECT queries to inspect
values and test joins. The harness preserves successful query results in SQL WORKSPACE and returns the
latest rejected action in LAST SQL ERROR.

TOOLS
execute_sql(sql) executes one read-only SELECT, WITH, PRAGMA, or EXPLAIN QUERY PLAN statement and
returns columns plus at most 20 rows. Use it for schema inspection, data exploration, and validating a
candidate query.
submit_sql(sql) is terminal. It accepts exactly one read-only SELECT or WITH query. Submit only after
the query has been executed successfully and you believe its complete result answers the question.
The hidden correctness judge is never shown to you.

RULES
1. Each turn output exactly one non-empty <think> block followed by exactly one
   <tool_call>{"tool":"...","arguments":{"sql":"..."}}</tool_call> block and nothing else.
2. Make one atomic call per turn. Never emit multiple SQL actions or multiple tool_call blocks.
3. Do not use INSERT, UPDATE, DELETE, CREATE, DROP, ALTER, ATTACH, or multiple SQL statements.
4. Use only information in the question, external knowledge, catalog, SQL WORKSPACE, and LAST SQL
   ERROR. Do not assume hidden columns or values.
5. An execution error is feedback: correct the SQL in the next turn rather than restarting.
"""
SQL_CANONICAL_RESPONSE_RULE = """1. Each turn output exactly one non-empty <think> block followed by exactly one
   <tool_call>{"tool":"...","arguments":{"sql":"..."}}</tool_call> block and nothing else."""
SQL_SPLIT_RESPONSE_RULE = """1. Produce exactly one SQL tool action per turn using the provider-specific response
   envelope at the end of this prompt."""


def build_system_prompt(model: str) -> str:
    """Build a provider-aware prompt containing only this ablation's SQL tools."""
    sql_example = (
        '{"tool":"execute_sql","arguments":'
        '{"sql":"PRAGMA table_info(Orders)"}}'
    )
    if not is_deepseek_split_model(model):
        return SYSTEM_PROMPT
    if SQL_CANONICAL_RESPONSE_RULE not in SYSTEM_PROMPT:
        raise ValueError("iterative-SQL response rule drifted")
    prompt = SYSTEM_PROMPT.replace(
        SQL_CANONICAL_RESPONSE_RULE,
        SQL_SPLIT_RESPONSE_RULE,
        1,
    )
    return prompt + provider_instruction(model, example_visible_content=sql_example)


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def parse_sql_action_strict(text: str) -> tuple[str, str, dict]:
    payloads = TOOL_CALL_RE.findall(text)
    if len(payloads) != 1:
        raise ProtocolError(
            "expected exactly one complete <tool_call>{...}</tool_call> block; "
            f"received {len(payloads)}"
        )
    try:
        call = json.loads(payloads[0])
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"tool_call is not valid JSON: {exc}") from exc
    if not isinstance(call, dict) or set(call) != {"tool", "arguments"}:
        raise ProtocolError('tool_call must contain exactly "tool" and "arguments" keys')
    tool = call.get("tool")
    arguments = call.get("arguments")
    if tool not in SQL_TOOLS:
        raise ProtocolError(f"unknown tool {tool!r}; legal tools: {sorted(SQL_TOOLS)}")
    if not isinstance(arguments, dict) or set(arguments) != {"sql"}:
        raise ProtocolError(f'{tool}: arguments must contain exactly one "sql" field')
    if not isinstance(arguments["sql"], str) or not arguments["sql"].strip():
        raise ProtocolError(f"{tool}: sql must be a non-empty string")
    think_blocks = THINK_RE.findall(text)
    if len(think_blocks) != 1 or not think_blocks[0].strip():
        raise ProtocolError("expected exactly one non-empty <think>...</think> block")
    return think_blocks[0].strip(), tool, {"sql": arguments["sql"].strip()}


def validate_read_only_sql(sql: str, *, terminal: bool = False) -> None:
    statement = sql.strip().rstrip(";").strip()
    matcher = SUBMIT_RE if terminal else READ_ONLY_RE
    if not matcher.match(statement):
        allowed = "SELECT or WITH" if terminal else "SELECT, WITH, PRAGMA, or EXPLAIN QUERY PLAN"
        raise ValueError(f"only one read-only {allowed} statement is allowed")
    if ";" in statement:
        raise ValueError("multiple SQL statements are not allowed")


def execute_preview(h: Harness, sql: str, *, limit: int, timeout_seconds: float) -> dict:
    validate_read_only_sql(sql)
    deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None
    if deadline is not None:
        h.conn.set_progress_handler(lambda: 1 if time.monotonic() >= deadline else 0, 10_000)
    try:
        cursor = h.conn.execute(sql)
        columns = [item[0] for item in (cursor.description or [])]
        rows = cursor.fetchmany(limit + 1)
    finally:
        h.conn.set_progress_handler(None, 0)
    truncated = len(rows) > limit
    shown = rows[:limit]
    return {
        "sql": sql,
        "columns": columns,
        "rows": [list(row) for row in shown],
        "returned_rows": len(shown),
        "rows_truncated": truncated,
    }


def task_prompt(catalog: dict, ex: dict) -> str:
    text = (
        "DATABASE CATALOG\n" + compact_json(catalog)
        + "\n\nQUESTION\n" + ex["question"]
    )
    if ex.get("external_knowledge"):
        text += "\n\nEXTERNAL KNOWLEDGE\n" + str(ex["external_knowledge"])
    return text


def state_prompt(workspace: list[dict], last_error: dict | None) -> str:
    text = "CURRENT SQL WORKSPACE\n" + compact_json({"successful_queries": workspace[-8:]})
    if last_error:
        text += "\n\nLAST SQL ERROR\n" + compact_json(last_error)
    return text


def context_messages(
    system_prompt: str,
    catalog: dict,
    ex: dict,
    workspace: list[dict],
    last_error: dict | None,
    legal_history: list[dict],
    history_turns: int,
) -> list[dict]:
    retained = legal_history[-history_turns:] if history_turns > 0 else legal_history
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task_prompt(catalog, ex)},
    ]
    for item in retained:
        messages.extend([
            {"role": "assistant", "content": item["assistant"]},
            {"role": "user", "content": item["observation"]},
        ])
    messages[-1]["content"] += "\n\n" + state_prompt(workspace, last_error)
    return messages


def error_type(exc: Exception) -> str:
    if isinstance(exc, ProtocolError):
        text = str(exc).lower()
        if "arguments" in text or "sql must" in text:
            return "argument_validation_error"
        return "protocol_error"
    return "execution_error"


def run_one(
    *,
    ex: dict,
    example_index: int,
    base_url: str,
    api_key: str,
    model: str,
    max_steps: int,
    max_errors_per_type: int,
    max_tokens: int,
    api_retries: int,
    api_timeout: int,
    execution_timeout_seconds: float,
    preview_rows: int,
    history_turns: int,
    denotation_comparison: str,
) -> dict:
    started = time.time()
    h = Harness(task_db_path(ex))
    h.conn.execute("PRAGMA query_only = ON")
    catalog = overview(h)
    gold_sql = task_gold_sql(ex)
    system_prompt = build_system_prompt(model)
    workspace: list[dict] = []
    legal_history: list[dict] = []
    last_error = None
    turns: list[dict] = []
    error_events: list[dict] = []
    counts = collections.Counter()
    usage = collections.Counter()
    record = {
        "example_index": example_index,
        "instance_id": ex.get("instance_id") or ex.get("example_id"),
        "db_id": ex["db_id"],
        "difficulty": (ex.get("metadata") or {}).get("difficulty_proxy"),
        "question": ex["question"],
        "gold_sql": gold_sql,
        "correct": False,
        "legal": False,
        "failure_type": None,
        "turns": turns,
        "error_events": error_events,
        "denotation_comparison": denotation_comparison,
    }

    for action_index in range(1, max_steps + 1):
        model_input = context_messages(
            system_prompt, catalog, ex, workspace, last_error, legal_history, history_turns
        )
        model_input = provider_request_messages(model, model_input)
        turn = {
            "turn_index": action_index - 1,
            "model_input": deepcopy(model_input),
            "provider_request_options": provider_request_options(model),
        }
        try:
            content, call_usage, reasoning = chat_with_retries(
                base_url=base_url,
                api_key=api_key,
                model=model,
                messages=model_input,
                max_tokens=max_tokens,
                timeout=api_timeout,
                retries=api_retries,
            )
        except ContextOverflowError as exc:
            turn["api_error"] = f"{type(exc).__name__}: {exc}"
            turns.append(turn)
            record["failure_type"] = "context_overflow"
            break
        except ChatAPIError as exc:
            turn["api_error"] = f"{type(exc).__name__}: {exc}"
            turns.append(turn)
            record["failure_type"] = "api_error"
            break
        except Exception as exc:  # provider clients may surface transport-specific exceptions
            turn["api_error"] = f"{type(exc).__name__}: {exc}"
            turns.append(turn)
            record["failure_type"] = "api_error"
            break

        try:
            add_usage(usage, call_usage)
            turn["api_finish_reason"] = call_usage.get("api_finish_reason")
            if call_usage.get("provider_response_metadata"):
                turn["provider_response_metadata"] = deepcopy(
                    call_usage["provider_response_metadata"]
                )
            adapted, adapter = adapt_provider_response(model, content, reasoning)
            turn.update({
                "raw_model_output": content,
                "provider_reasoning_content": reasoning,
                "response_adapter": adapter,
                "model_output": adapted,
            })
            rejection = provider_rejection_message(adapter)
            if rejection:
                raise ProtocolError(rejection)
            think, tool, arguments = parse_sql_action_strict(adapted)
            turn["parsed"] = {"think": think, "tool": tool, "arguments": arguments}
            turn["feedback_recovery"] = bool(last_error)
            turn["recovered_from_error_type"] = (
                (last_error or {}).get("error", {}).get("type")
            )
            sql = arguments["sql"]

            if tool == "submit_sql":
                validate_read_only_sql(sql, terminal=True)
                if sql not in {item["sql"] for item in workspace}:
                    raise ProtocolError(
                        "submit_sql arguments.sql must exactly match a previously successful "
                        "execute_sql query"
                    )
                predicted = execute_predicted_sql(h, sql, execution_timeout_seconds)
                gold = h.gold(gold_sql)
                record.update({
                    "legal": True,
                    "correct": compare_denotations(predicted, gold, denotation_comparison),
                    "predicted_sql": sql,
                    "predicted_row_count": len(predicted),
                    "predicted_sample": [list(row) for row in predicted[:10]],
                    "gold_row_count": len(gold),
                    "gold_sample": [list(row) for row in gold[:10]],
                })
                if not record["correct"]:
                    record["failure_type"] = "wrong_answer"
                record["outcome"] = (
                    "recovered_success" if record["correct"] and error_events
                    else "clean_success" if record["correct"] else None
                )
                turns.append(turn)
                break

            output = execute_preview(
                h, sql, limit=preview_rows, timeout_seconds=execution_timeout_seconds
            )
            workspace.append({"step_id": f"step_{action_index}", **output})
            observation = compact_json({
                "step_id": f"step_{action_index}", "status": "success", "output": output
            })
            turn["tool_output"] = output
            turns.append(turn)
            legal_history.append({"assistant": adapted, "observation": observation})
            last_error = None
        except Exception as exc:  # semantic and protocol errors are same-episode feedback
            kind = error_type(exc)
            counts[kind] += 1
            event = {
                "action_index": action_index,
                "step_id": f"step_{action_index}",
                "error_type": kind,
                "message": f"{type(exc).__name__}: {exc}",
            }
            if turn.get("parsed"):
                event["attempted_tool"] = turn["parsed"]["tool"]
                event["attempted_arguments"] = turn["parsed"]["arguments"]
            error_events.append(event)
            turn["error_event"] = event
            turns.append(turn)
            last_error = {
                "step_id": event["step_id"],
                "status": "error",
                "error": {"type": kind, "message": event["message"]},
            }
            if counts[kind] >= max_errors_per_type:
                record["failure_type"] = kind
                break
    else:
        record["failure_type"] = "max_steps"

    if not record["correct"] and record["failure_type"] is None:
        record["failure_type"] = "max_steps"
    record["steps"] = len(turns)
    record["errors"] = len(error_events)
    record["usage"] = dict(usage)
    record["elapsed_seconds"] = round(time.time() - started, 3)
    return record


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-json", required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-errors-per-type", type=int, default=3)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument("--api-timeout", type=int, default=300)
    parser.add_argument("--execution-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--preview-rows", type=int, default=20)
    parser.add_argument("--history-turns", type=int, default=4)
    add_denotation_comparison_argument(parser)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.max_tokens is None:
        args.max_tokens = provider_default_max_tokens(args.model, 1024)

    tasks_path = Path(args.tasks_json)
    tasks = [json.loads(line) for line in tasks_path.read_text(encoding="utf-8").splitlines()
             if line.strip()][:args.n]
    system_prompt = build_system_prompt(args.model)
    api_key, base_url = load_api_config()
    if not api_key or not base_url:
        parser.error("api.md must define API_KEY and BASE_URL")

    manifest = {
        "runner": "iterative_sql_feedback",
        "interface": "execute_sql_submit_sql_v1",
        "tasks_json": str(tasks_path),
        "tasks_sha256": file_sha256(tasks_path),
        "task_count": len(tasks),
        "model": args.model,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
        "max_steps": args.max_steps,
        "max_errors_per_type": args.max_errors_per_type,
        "max_tokens": args.max_tokens,
        "api_retries": args.api_retries,
        "execution_timeout_seconds": args.execution_timeout_seconds,
        "preview_rows": args.preview_rows,
        "history_turns": args.history_turns,
        "denotation_comparison": args.denotation_comparison,
        "strict_parser": True,
        "parser_repair": False,
        "gold_visible_to_model": False,
    }
    writer = ArtifactWriter(args.result_dir, manifest, resume=args.resume)
    work = [ex for ex in tasks if int(ex["example_index"]) not in writer.completed]
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                run_one,
                ex=ex,
                example_index=int(ex["example_index"]),
                base_url=base_url,
                api_key=api_key,
                model=args.model,
                max_steps=args.max_steps,
                max_errors_per_type=args.max_errors_per_type,
                max_tokens=args.max_tokens,
                api_retries=args.api_retries,
                api_timeout=args.api_timeout,
                execution_timeout_seconds=args.execution_timeout_seconds,
                preview_rows=args.preview_rows,
                history_turns=args.history_turns,
                denotation_comparison=args.denotation_comparison,
            ): ex for ex in work
        }
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            writer.append(record)
            print(
                f"[{done}/{len(work)}] {'OK' if record['correct'] else 'ERR'} "
                f"steps={record['steps']} errors={record['errors']} "
                f"type={record.get('failure_type')} {record.get('instance_id')}",
                flush=True,
            )
    print(json.dumps(writer.summarize(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
