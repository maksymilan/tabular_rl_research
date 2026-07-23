#!/usr/bin/env python3
"""Closed-loop rollout runner: a model converses with the harness through the shared protocol
(src/sft/protocol.py) until answer_from_context; the answer is scored against the gold
SQL executed on the real DB. One runner serves three purposes: (a) few-shot baseline eval of
the base model, (b) post-SFT eval, (c) later, the RL environment interaction loop.

Modes
  live   : .venv/bin/python src/eval/rollout.py --base-url http://127.0.0.1:8000/v1 \
               --model Qwen/Qwen2.5-3B-Instruct [--n 50] [--few-shot 2] \
               [--result-dir data/results/run_name]
           (vLLM on the GPU server; reach it from the Mac via `ssh -L 8000:127.0.0.1:8000 NewGNN`)
  replay : .venv/bin/python src/eval/rollout.py --replay 50
           No model: re-executes recorded dev trajectories through the same loop machinery.
           Verified data must score ~100% — this validates executor wiring + scoring.

Scoring: prefer the rows of the evidence table the model cites when that table already matches
the gold answer. If the cited table is broader than the final answer, fall back to the explicit
`answer` field so a correct scalar/projection is not marked wrong just because the evidence table
contains extra columns.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from copy import deepcopy

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "harness"))
sys.path.insert(0, os.path.join(ROOT, "src", "sft"))

from executor import Harness                                    # noqa: E402
from environment_state import EnvironmentState                  # noqa: E402
from plan import resolve_cond, _value_ref_ids                  # noqa: E402
from scalar_grounding import extract_scalar                    # noqa: E402
from provenance import build_grounding_references, build_references  # noqa: E402
from emitter import _catalog                                   # noqa: E402
from artifacts import ArtifactWriter                           # noqa: E402
from protocol import (ACCEPTED_TOOLS, ProtocolError, get_system_prompt,  # noqa: E402
                      assistant_message, first_user_message, parse_assistant_strict,
                      model_context_messages, protocol_hash, rows_equal, tool_error_message,
                      rolling_legal_history_messages, rolling_system_prompt,
                      state_context_message, tool_output_message)

SPIDER = os.path.join(ROOT, "data", "spider_data")
MAX_ERRORS_PER_TYPE = 3
DEFAULT_FEWSHOT_IDS = ["spider_train_0", "spider_train_1"]
DEFAULT_MAX_TOKENS = 768
MIN_CONTEXT_RETRY_TOKENS = 128


class ChatAPIError(RuntimeError):
    """OpenAI-compatible chat endpoint failure with enough detail for eval accounting."""

    def __init__(self, message: str, *, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class ContextOverflowError(ChatAPIError):
    """The request exceeds the serving context window, even after reducing output tokens."""


def is_context_overflow(text: str) -> bool:
    lowered = text.lower()
    return (
        "maximum context length" in lowered
        or "context length" in lowered and "max_tokens" in lowered
        or "too many tokens" in lowered
    )


def db_path(db_id: str) -> str:
    return os.path.join(SPIDER, "database", db_id, f"{db_id}.sqlite")


def task_db_path(ex: dict) -> str:
    """Use an adapter-provided SQLite path, with Spider as the legacy default."""
    return str(ex.get("db_path") or db_path(ex["db_id"]))


def task_gold_sql(ex: dict) -> str | None:
    return ex.get("gold_sql") or ex.get("query")


def protocol_failure_type(exc: ProtocolError) -> str:
    text = str(exc).lower()
    if any(marker in text for marker in (
        "arguments", "unexpected", "requires", "missing required", "must be", "must contain",
    )):
        return "argument_validation_error"
    return "protocol_error"


def state_digest(state: dict) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(state, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def load_tasks_json(path: str) -> list[dict]:
    """Load common DatasetTask JSON/JSONL records emitted by a dataset adapter."""
    with open(path) as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"--tasks-json must contain a JSON array or JSONL records: {path}")
    return data


def overview(h: Harness) -> dict:
    # V2-ctx: the opening overview is the lazy catalog (names + row_counts + FK relations, no columns)
    # — the SAME renderer the emitter uses, so eval matches training. describe_table acquires schema.
    return _catalog(h)


def new_ctx(catalog: dict | None = None) -> dict:
    """Online harness state threaded across one trajectory's actions (history + provenance)."""
    return {"history": {}, "handle_to_step": {}, "environment": EnvironmentState(catalog)}


def _preview_table_rows(h: Harness, table: str, limit: int) -> list[list]:
    """Return a bounded inline preview for a table handle without materializing the full table."""
    if limit <= 0:
        return []
    rows = h.conn.execute(f"SELECT * FROM {h._src(table)} LIMIT {int(limit)}").fetchall()
    return [list(r) for r in rows]


def _table_from_step(ctx: dict, ref):
    if not isinstance(ref, str):
        return ref
    record = ctx.get("history", {}).get(ref)
    if not isinstance(record, dict):
        return ref
    output = record.get("output") or {}
    return output.get("table") or ref


def _normalize_table_refs(args, ctx: dict, parent_key: str | None = None):
    table_keys = {"table", "left", "right", "in_table"}
    if isinstance(args, list):
        return [_normalize_table_refs(item, ctx, parent_key) for item in args]
    if isinstance(args, dict):
        if parent_key in {"left", "right"} and set(args) == {"table"}:
            return _table_from_step(ctx, args["table"])
        return {key: _normalize_table_refs(value, ctx, key) for key, value in args.items()}
    if parent_key in table_keys:
        return _table_from_step(ctx, args)
    return args


def execute_tool(h: Harness, tool: str, args: dict, ctx: dict, step_id: str,
                 table_output_rows: int = 0):
    """Run one tool call, threading online provenance in `ctx`. Returns (output, created|None).

    V2b: a predicate's `value_ref` cites the producing step_id directly (no add_to_memory); the
    harness grounds the scalar from `ctx["history"]` with strict validation. An illegal value_ref
    (unknown step / non-scalar source) raises ScalarGroundingError -> surfaced as an execution_error."""
    if tool not in ACCEPTED_TOOLS or tool == "answer_from_context":
        raise ProtocolError(f"tool {tool!r} not executable here")
    args = _normalize_table_refs(args, ctx)

    def resolve_step(ref):
        if ref in ctx["handle_to_step"]:
            return ctx["handle_to_step"][ref]
        if ref in ctx["history"]:            # already a step_id (a value_ref)
            return ref
        return None
    references = build_references(tool, args, resolve_step)
    references.extend(build_grounding_references(tool, args, ctx["history"]))

    if tool == "plan":
        output = ctx["environment"].apply_plan_ops(args.get("ops"), step_id, ctx["history"])
        ctx["history"][step_id] = {"tool": tool, "arguments": args, "output": output, "references": references}
        return output, None

    if tool in ("describe_table", "inspect_column", "read_subtable"):   # read-only perception; no table
        out = getattr(h, tool)(**args)
        output = out if isinstance(out, dict) else {"rows": [list(r) for r in out], "row_count": len(out)}
        history_record = {"tool": tool, "arguments": args, "output": output, "references": references}
        if tool == "read_subtable":
            table = args.get("table")
            requested_columns = args.get("columns")
            history_record["observed_columns"] = (
                list(requested_columns) if requested_columns else list(h._cols(table))
            )
        ctx["history"][step_id] = history_record
        ctx["environment"].apply_tool_result(tool, args, output, step_id)
        return output, None

    exec_args = dict(args)
    if tool == "condition_filter":
        # resolve each value_ref (a producing step_id) to its grounded scalar; a bad reference raises
        values = {ref: extract_scalar(ctx["history"], ref)
                  for ref in _value_ref_ids(exec_args.get("conditions"))}
        exec_args["conditions"] = resolve_cond(exec_args.get("conditions"), {}, values)
    out = getattr(h, tool)(**exec_args)
    if isinstance(out, dict) and "table_name" in out:
        output = {"table": out["table_name"], "kind": out["kind"],     # V2-ctx: metadata-only handle
                  "columns": out["columns"], "row_count": out["row_count"]}
        preview_limit = max(0, int(table_output_rows or 0))
        if preview_limit:
            rows = _preview_table_rows(h, out["table_name"], preview_limit)
            output["rows"] = rows
            output["preview_limit"] = preview_limit
            output["rows_truncated"] = out["row_count"] > len(rows)
        elif out["row_count"] == 1 and len(out["columns"]) == 1:       # scalar-shaped result keeps its cell
            output["rows"] = _preview_table_rows(h, out["table_name"], 1)
        ctx["handle_to_step"][out["table_name"]] = step_id
        created = out["table_name"]
    else:
        rows = out if isinstance(out, list) else [(out,)]
        output = {"result_sample": [list(r) for r in rows[:5]], "row_count": len(rows)}
        created = None
    ctx["history"][step_id] = {"tool": tool, "arguments": args, "output": output, "references": references}
    ctx["environment"].apply_tool_result(tool, args, output, step_id)
    return output, created


def _evidence_table(evidence) -> str | None:
    if isinstance(evidence, str):
        return evidence
    if isinstance(evidence, dict):
        return evidence.get("table")
    return None


def _gold_width(gold: list) -> int | None:
    if not gold:
        return None
    first = gold[0]
    try:
        return len(first)
    except TypeError:
        return 1


def _dedupe_row_candidates(candidates: list[list]) -> list[list]:
    seen = set()
    out = []
    for rows in candidates:
        try:
            key = json.dumps(rows, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            key = repr(rows)
        if key in seen:
            continue
        seen.add(key)
        out.append(rows)
    return out


def answer_row_candidates(answer, gold: list | None = None) -> list[list]:
    """Return plausible row-shaped interpretations of a model-authored final answer.

    The model often writes compact scalar answers as `151` or `[151]`, while SQL gold rows are
    represented as `[[151]]`. For single-column multi-row answers it may write `["a", "b"]`.
    Keep evidence-table scoring strict, but make this explicit-answer fallback tolerant.
    """
    width = _gold_width(gold or [])
    candidates: list[list] = []
    if answer is None:
        return [[]]
    if isinstance(answer, (str, int, float, bool)):
        return [[[answer]]]
    if isinstance(answer, tuple):
        answer = list(answer)
    if not isinstance(answer, list):
        return [[[answer]]]
    if not answer:
        return [[]]

    if all(isinstance(item, (list, tuple)) for item in answer):
        candidates.append([list(row) for row in answer])
    elif any(isinstance(item, (list, tuple)) for item in answer):
        candidates.append([list(item) if isinstance(item, (list, tuple)) else [item] for item in answer])
    else:
        if width == 1 or width is None:
            candidates.append([[item] for item in answer])
        if width and len(answer) == width:
            candidates.append([answer])
        if len(answer) == 1:
            candidates.append([[answer[0]]])
        candidates.append([answer])
    return _dedupe_row_candidates(candidates)


def _rows_equal_safe(pred, gold) -> bool:
    try:
        return rows_equal(pred, gold)
    except Exception:
        return False


def projected_row_candidates(rows, gold: list | None = None, max_combinations: int = 5000) -> list[list]:
    """Permute same-width evidence columns to tolerate answer-column order differences.

    This is only used after exact evidence matching and explicit-answer matching fail. It must not
    drop columns from a broader evidence table: answer_from_context cites the table as the final
    answer, so a table with extra helper columns is not an exact final answer.
    """
    width = _gold_width(gold or [])
    if width is None or width <= 0 or not rows:
        return []
    try:
        row_width = len(rows[0])
    except TypeError:
        return []
    if row_width != width:
        return []

    candidates: list[list] = []
    for count, cols in enumerate(itertools.permutations(range(row_width), width), 1):
        if count > max_combinations:
            break
        try:
            candidates.append([[row[i] for i in cols] for row in rows])
        except (IndexError, TypeError):
            continue
    return _dedupe_row_candidates(candidates)


def score(h: Harness, gold_sql: str, answer_args: dict, created: set) -> tuple[bool, list, list]:
    """Score only the exact harness-owned table cited by the terminal tool call.

    Model-authored ``answer`` values and inferred column permutations are deliberately ignored.
    This makes the metric a test of the executed tool trajectory's output rather than a hybrid
    tool/direct-answer metric.
    """
    gold = h.gold(gold_sql)
    ev = _evidence_table(answer_args.get("evidence"))
    if not ev or ev not in created:
        return False, [], gold[:5]
    try:
        evidence_rows = h.rows(ev)
    except Exception:
        return False, [], gold[:5]
    return _rows_equal_safe(evidence_rows, gold), evidence_rows[:5], gold[:5]


def _condition_table_refs(cond) -> list[str]:
    if isinstance(cond, list):
        refs = []
        for item in cond:
            refs.extend(_condition_table_refs(item))
        return refs
    if not isinstance(cond, dict):
        return []
    refs = []
    if isinstance(cond.get("in_table"), str):
        refs.append(cond["in_table"])
    for key in ("and", "or"):
        for item in cond.get(key, []) or []:
            refs.extend(_condition_table_refs(item))
    if "not" in cond:
        refs.extend(_condition_table_refs(cond["not"]))
    return refs


def _arg_table_refs(tool: str | None, args: dict | None) -> list[str]:
    if not isinstance(args, dict):
        return []
    refs = []
    for key in ("table", "left", "right"):
        if isinstance(args.get(key), str):
            refs.append(args[key])
    if isinstance(args.get("tables"), list):
        refs.extend(item for item in args["tables"] if isinstance(item, str))
    refs.extend(_condition_table_refs(args.get("conditions")))
    return refs


def _safe_cols(h: Harness, table: str) -> list[str]:
    try:
        return h._cols(table)  # noqa: SLF001 - diagnostic-only, same harness boundary
    except Exception:
        return []


def _join_identifier_hint(h: Harness, args: dict | None) -> str | None:
    if not isinstance(args, dict) or not isinstance(args.get("tables"), list):
        return None
    tables = [table for table in args["tables"] if isinstance(table, str)]
    if len(tables) < 2:
        return None
    prefixes = args.get("prefixes")
    if isinstance(prefixes, list) and prefixes and isinstance(prefixes[0], str) and prefixes[0]:
        first_cols = _safe_cols(h, tables[0])
        first_example = f"{prefixes[0]}__{first_cols[0]}" if first_cols else f"{prefixes[0]}__<column>"
        return (
            "join_tables.on uses model-facing column identifiers, never SQL aliases like L./R. "
            "or table-qualified names. With prefixes, the first edge's left key may use the declared "
            f"prefix form {first_example!r}; its right key remains a bare column of {tables[1]!r}. "
            "Later left keys use the already-produced prefix__column names."
        )
    return (
        "join_tables.on uses bare column names from the listed tables; never write SQL aliases "
        "like L./R. or table-qualified names such as table.column."
    )


def format_tool_error(exc: Exception, h: Harness, tool: str | None, args: dict | None) -> str:
    """Attach compact, actionable environment hints to tool/protocol feedback."""
    base = f"{type(exc).__name__}: {exc}"
    hints = []
    text = base.lower()
    valid = h.available_tables() if hasattr(h, "available_tables") else sorted(getattr(h, "views", {}))
    if "unknown table" in text:
        hints.append(f"valid table handles are {valid}")
        if re.search(r"unknown table: T\d+", base):
            hints.append("T1/T2/etc. are column prefixes, not table handles; use the produced handle such as join_001/project_002")
    if "no such column" in text or "ambiguous column" in text:
        refs = []
        for table in _arg_table_refs(tool, args):
            cols = _safe_cols(h, table)
            if cols:
                refs.append({"table": table, "columns": cols})
        if refs:
            hints.append(f"available columns for referenced tables: {refs}")
        if tool == "join_tables":
            join_hint = _join_identifier_hint(h, args)
            if join_hint:
                hints.append(join_hint)
    if "set_op" in base or "selects to the left and right" in text or "aligned columns" in text:
        refs = []
        for table in _arg_table_refs(tool, args):
            cols = _safe_cols(h, table)
            if cols:
                refs.append({"table": table, "columns": cols})
        if refs:
            hints.append(f"set_op requires both sides to have the same output columns; current columns: {refs}")
    if "scalargroundingerror" in text or "not scalar" in text:
        hints.append("value_ref must cite a scalar-producing step; use in_table for membership against a table")
    if 'near "*"' in text:
        hints.append('condition_filter cannot filter column "*"; use project(table, expressions=[...]) or conditions=null/no-op')
    if hints:
        return base + " | " + " | ".join(hints)
    return base


# ---------------- live mode ----------------
def _chat_once(base_url: str, model: str, messages: list[dict], max_tokens: int) -> str:
    payload = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    # Qwen3.5 is a thinking model. EVAL_ENABLE_THINKING=0 disables chain-of-thought via the chat
    # template so the baseline is directly comparable to the non-thinking Qwen2.5 runs and stays
    # within max_tokens; =1 forces it on; unset leaves the model/template default.
    think = os.environ.get("EVAL_ENABLE_THINKING")
    if think is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": think == "1"}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read())["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise ChatAPIError(f"HTTP {e.code}: {body}", status=e.code, body=body) from e


def chat(
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    retries: int = 2,
    min_context_retry_tokens: int = MIN_CONTEXT_RETRY_TOKENS,
    retry_stats: dict | None = None,
) -> str:
    """Call the model with retry semantics that do not conflate infra failures with model quality.

    Context-window errors are deterministic for a given prompt/output budget, so retrying the same
    request is useless. Instead, progressively shrink the requested output budget; if even the
    minimum budget overflows, surface `ContextOverflowError` so the evaluator can count it
    separately from protocol/execution failures.
    """
    current_max_tokens = max_tokens
    transient_attempts = 0
    context_retries = 0
    last_error: Exception | None = None
    while True:
        try:
            text = _chat_once(base_url, model, messages, current_max_tokens)
            if retry_stats is not None:
                retry_stats.update({
                    "api_request_attempts": transient_attempts + context_retries + 1,
                    "api_transport_retries": transient_attempts,
                    "api_context_retries": context_retries,
                })
            return text
        except ChatAPIError as e:
            last_error = e
            if is_context_overflow(str(e)) or is_context_overflow(e.body):
                if current_max_tokens > min_context_retry_tokens:
                    current_max_tokens = max(
                        min_context_retry_tokens,
                        current_max_tokens // 2,
                    )
                    context_retries += 1
                    continue
                raise ContextOverflowError(str(e), status=e.status, body=e.body) from e
            if e.status is not None and e.status >= 500 and transient_attempts < retries:
                transient_attempts += 1
                time.sleep(min(2 ** transient_attempts, 8))
                continue
            raise
        except (TimeoutError, urllib.error.URLError) as e:
            last_error = e
            if transient_attempts < retries:
                transient_attempts += 1
                time.sleep(min(2 ** transient_attempts, 8))
                continue
            raise ChatAPIError(f"{type(e).__name__}: {e}") from e
    raise ChatAPIError(f"chat failed: {last_error}")


def fewshot_text(trajectory_ids: list[str]) -> str:
    """Render fixed train trajectories as example transcripts appended to the system prompt."""
    if not trajectory_ids:
        return ""
    wanted = set(trajectory_ids)
    found = {}
    with open(os.path.join(ROOT, "data", "trajectories", "spider_train_v2.jsonl")) as f:
        for line in f:
            t = json.loads(line)
            if t["trajectory_id"] in wanted:
                found[t["trajectory_id"]] = t
    missing = [trajectory_id for trajectory_id in trajectory_ids if trajectory_id not in found]
    if missing:
        raise ValueError(f"few-shot trajectories not found: {missing}")
    blocks = []
    for trajectory_id in trajectory_ids:
        t = found[trajectory_id]
        ov = t["initial_state"]["dataset_overview"]
        lines = [f"USER: {first_user_message(ov, t['question'])}"]
        for i, s in enumerate(t["steps"]):
            if i > 0:
                prev = t["steps"][i - 1]
                lines.append(
                    "USER: "
                    + state_context_message(s.get("environment_state_before", prev.get("environment_state")))
                )
            tc = s["tool_call"]
            lines.append(
                f"ASSISTANT: "
                f"{assistant_message(s.get('think', ''), tc['tool'], tc['arguments'])}"
            )
        blocks.append("\n".join(lines))
    return "\n\nEXAMPLE SESSIONS\n" + "\n\n---\n\n".join(blocks)


def run_live(
    ex: dict,
    example_index: int,
    base_url: str,
    model: str,
    system: str,
    max_steps: int,
    max_tokens: int,
    api_retries: int,
    max_errors_per_type: int = MAX_ERRORS_PER_TYPE,
    table_output_rows: int = 0,
    context_mode: str = "state-only",
    history_turns: int = 4,
    compact_history_observations: bool = True,
) -> dict:
    task_path = task_db_path(ex)
    gold_sql = task_gold_sql(ex)
    if not gold_sql:
        raise ValueError(f"task has no gold SQL for execution scoring: {ex.get('db_id')} / {ex.get('question')}")
    h = Harness(task_path)
    ov = overview(h)
    created: set[str] = set()
    ctx = new_ctx(ov)
    last_error: dict | None = None
    legal_history: list[dict] = []
    external_knowledge = ex.get("external_knowledge")
    if context_mode == "rolling-legal-history":
        initial_messages = rolling_legal_history_messages(
            system, ov, ex["question"], ctx["environment"].snapshot(), None,
            external_knowledge, legal_history, history_turns,
            compact_observations=compact_history_observations,
        )
    else:
        initial_messages = model_context_messages(
            system, ov, ex["question"], ctx["environment"].snapshot(), None, external_knowledge,
        )
    messages = deepcopy(initial_messages)
    action_count = errors = 0
    error_counts: dict[str, int] = {}
    error_events: list[dict] = []
    text = ""
    turns = []
    started = time.time()
    rec = {
        "example_index": example_index,
        "db_id": ex["db_id"],
        "question": ex["question"],
        "gold_sql": gold_sql,
        "initial_model_input": initial_messages,
        "turns": turns,
        "correct": False,
        "legal": False,
        "steps": 0,
        "errors": 0,
        "failure_type": None,
        "fail": None,
        "outcome": None,
        "error_events": error_events,
        "api_transport_retries": 0,
        "api_context_retries": 0,
    }

    while action_count < max_steps:
        action_count += 1
        state_before = ctx["environment"].snapshot()
        if context_mode == "rolling-legal-history":
            model_input = rolling_legal_history_messages(
                system, ov, ex["question"], state_before, last_error,
                external_knowledge, legal_history, history_turns,
                compact_observations=compact_history_observations,
            )
        else:
            model_input = model_context_messages(
                system, ov, ex["question"], state_before, last_error, external_knowledge,
            )
        turn = {"turn_index": len(turns), "model_input": deepcopy(model_input)}
        try:
            retry_stats: dict = {}
            text = chat(base_url, model, model_input, max_tokens=max_tokens, retries=api_retries,
                        retry_stats=retry_stats)
            turn["api_retry_stats"] = retry_stats
        except ContextOverflowError as e:
            rec["failure_type"] = "context_overflow"
            rec["fail"] = f"api: {type(e).__name__}: {e}"
            rec["steps"] = action_count - 1
            rec["errors"] = errors
            turn["api_error"] = rec["fail"]
            turn["api_error_type"] = "context_overflow"
            turns.append(turn)
            rec["final_messages"] = messages
            rec["elapsed_seconds"] = round(time.time() - started, 3)
            return rec
        except Exception as e:
            rec["failure_type"] = "api_error"
            rec["fail"] = f"api: {type(e).__name__}: {e}"
            rec["steps"] = action_count - 1
            rec["errors"] = errors
            turn["api_error"] = rec["fail"]
            turn["api_error_type"] = "api_error"
            turns.append(turn)
            rec["final_messages"] = messages
            rec["elapsed_seconds"] = round(time.time() - started, 3)
            return rec
        turn["model_output"] = text
        rec["api_transport_retries"] += turn["api_retry_stats"]["api_transport_retries"]
        rec["api_context_retries"] += turn["api_retry_stats"]["api_context_retries"]
        messages.append({"role": "assistant", "content": text})
        try:
            think, tool, args = parse_assistant_strict(text)
            turn["parsed"] = {"think": think, "tool": tool, "arguments": args}
            turn["feedback_recovery"] = bool(last_error)
            turn["recovered_from_error_type"] = (last_error or {}).get("error", {}).get("type")
            if tool == "answer_from_context":
                rec["legal"] = True
                rec["steps"] = action_count
                rec["errors"] = errors
                rec["correct"], rec["pred_sample"], rec["gold_sample"] = score(h, gold_sql, args, created)
                if not rec["correct"]:
                    rec["failure_type"] = "wrong_answer"
                else:
                    rec["outcome"] = "recovered_success" if error_events else "clean_success"
                turns.append(turn)
                rec["final_messages"] = messages
                rec["elapsed_seconds"] = round(time.time() - started, 3)
                return rec
            step_id = f"step_{action_count}"
            out, tname = execute_tool(h, tool, args, ctx, step_id, table_output_rows=table_output_rows)
            turn["tool_output"] = out
        except (ProtocolError, Exception) as e:  # noqa: BLE001 — every failure becomes feedback
            errors += 1
            parsed = turn.get("parsed") or {}
            error = format_tool_error(e, h, parsed.get("tool"), parsed.get("arguments"))
            state_after = ctx["environment"].snapshot()
            error_type = protocol_failure_type(e) if isinstance(e, ProtocolError) else "execution_error"
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
                rec["fail"] = error
                rec["errors"] = errors
                rec["steps"] = action_count
                rec["final_messages"] = messages
                rec["elapsed_seconds"] = round(time.time() - started, 3)
                return rec
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            if error_counts[error_type] >= max_errors_per_type:
                rec["failure_type"] = error_type
                rec["fail"] = f"aborted after {error_counts[error_type]} {error_type} events: {error}"
                rec["errors"] = errors
                rec["steps"] = action_count
                rec["final_messages"] = messages
                rec["elapsed_seconds"] = round(time.time() - started, 3)
                return rec
            last_error = {
                "step_id": f"step_{action_count}",
                "status": "error",
                "error": {"type": turn["execution_error_type"], "message": error},
            }
            messages.append({"role": "user", "content": tool_error_message(
                last_error["step_id"],
                turn["execution_error_type"],
                error,
            )})
            continue
        turns.append(turn)
        last_error = None
        if tname:
            created.add(tname)
        observation = tool_output_message(step_id, out)
        legal_history.append({"assistant": text, "observation": observation})
        messages.append({"role": "user", "content": observation})

    rec["failure_type"] = "max_steps"
    rec["fail"] = "max_steps"
    rec["steps"] = action_count
    rec["errors"] = errors
    rec["final_messages"] = messages
    rec["elapsed_seconds"] = round(time.time() - started, 3)
    return rec


# ---------------- replay mode (no model) ----------------
def run_replay(n: int, path: str = "") -> int:
    path = path or os.path.join(ROOT, "data", "trajectories", "spider_dev_v2ctx.jsonl")
    total = ok_exec = ok_score = 0
    with open(path) as f:
        for line in f:
            if total >= n:
                break
            t = json.loads(line)
            total += 1
            h = Harness(db_path(t["source"]["db_id"]))
            created: set[str] = set()
            ctx = new_ctx(t.get("initial_state", {}).get("dataset_overview"))
            try:
                for s in t["steps"][:-1]:
                    tc = s["tool_call"]
                    _, tname = execute_tool(h, tc["tool"], tc["arguments"], ctx, s["step_id"])
                    if tname:
                        created.add(tname)
                ok_exec += 1
            except Exception as e:
                print(f"  REPLAY EXEC FAIL {t['trajectory_id']}: {type(e).__name__}: {e}")
                continue
            args = t["steps"][-1]["tool_call"]["arguments"]
            correct, _, _ = score(h, t["source"]["gold_sql"], args, created)
            ok_score += correct
            if not correct:
                print(f"  REPLAY SCORE FAIL {t['trajectory_id']}")
    print(f"replay: {total} trajectories | executed {ok_exec} | scored correct {ok_score} "
          f"({100 * ok_score / max(1, total):.1f}%)  — expect ~100% on verified data")
    return 0 if ok_score == total else 1


def load_indices_file(path: str) -> set[int]:
    """Load dev example indices from JSON or plain text.

    Accepted shapes:
      - [1, 2, 3]
      - {"indices": [1, 2, 3]}
      - JSONL records with example_index/index
      - plain text with one integer per line
    """
    if not path:
        return set()
    raw = open(path, encoding="utf-8").read().strip()
    if not raw:
        return set()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        indices = set()
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            indices.add(int(line.split()[0]))
        return indices
    if isinstance(payload, dict):
        payload = payload.get("indices", payload.get("example_indices", []))
    if isinstance(payload, list):
        indices = set()
        for item in payload:
            if isinstance(item, int):
                indices.add(item)
            elif isinstance(item, dict):
                if "example_index" in item:
                    indices.add(int(item["example_index"]))
                elif "index" in item:
                    indices.add(int(item["index"]))
            else:
                indices.add(int(item))
        return indices
    raise ValueError(f"unsupported indices file shape: {path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", type=int, default=0, help="replay N dev trajectories (no model)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="")
    ap.add_argument("--n", type=int, default=50, help="number of dev questions (live mode)")
    ap.add_argument("--few-shot", type=int, default=0)
    ap.add_argument("--few-shot-ids", nargs="*", default=DEFAULT_FEWSHOT_IDS)
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                    help="per-turn generation budget; smaller values leave more room for tool context")
    ap.add_argument("--api-retries", type=int, default=2,
                    help="retry count for transient API errors; context overflow uses adaptive token shrink")
    ap.add_argument("--max-errors-per-type", type=int, default=MAX_ERRORS_PER_TYPE,
                    help="abort after this many recoverable errors of one class in one trajectory")
    ap.add_argument("--table-output-rows", type=int, default=0,
                    help="include up to N rows in each table-producing tool observation (0 = metadata only)")
    ap.add_argument("--context-mode", choices=["state-only", "rolling-legal-history"],
                    default="state-only")
    ap.add_argument("--history-turns", type=int, default=4,
                    help="successful assistant/tool pairs retained in rolling mode; 0 keeps all")
    ap.add_argument("--rolling-prompt-variant", choices=["full", "compact"], default="full",
                    help="rolling-only prompt ablation; full preserves existing runs")
    ap.add_argument(
        "--rolling-observation-style",
        choices=["resident", "full"],
        default="resident",
        help="resident is current R2; full is evaluation-only compatibility for pre-R2 adapters",
    )
    ap.add_argument(
        "--system-prompt-manifest",
        default="",
        help="evaluation-only: reuse the exact system_prompt stored in a historical run manifest",
    )
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent questions (vLLM batches requests; each worker owns its Harness/sqlite)")
    ap.add_argument("--indices-file", default="",
                    help="optional JSON/text file of dev example indices to run instead of the first --n")
    ap.add_argument("--tasks-json", default="",
                    help="optional common DatasetTask JSON/JSONL file; enables BIRD/other SQLite adapters")
    ap.add_argument("--result-dir", default="")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    if args.replay:
        return run_replay(args.replay)

    selected_indices = load_indices_file(args.indices_file)
    if args.tasks_json:
        task_examples = load_tasks_json(args.tasks_json)
        if selected_indices:
            indexed_dev = [(i, task_examples[i]) for i in sorted(selected_indices)
                           if 0 <= i < len(task_examples)]
        else:
            indexed_dev = list(enumerate(task_examples[: args.n]))
    else:
        dev_examples = json.load(open(os.path.join(SPIDER, "dev.json")))
        if selected_indices:
            indexed_dev = [(i, dev_examples[i]) for i in sorted(selected_indices)
                           if 0 <= i < len(dev_examples)]
        else:
            indexed_dev = list(enumerate(dev_examples[: args.n]))
    indexed_dev = [(i, ex) for i, ex in indexed_dev if os.path.exists(task_db_path(ex))]
    if args.history_turns < 0:
        ap.error("--history-turns must be non-negative")
    if args.context_mode == "rolling-legal-history" and args.few_shot:
        ap.error("few-shot examples are not supported with rolling history")
    fewshot_ids = args.few_shot_ids[:args.few_shot] if args.few_shot else []
    prompt_variant = os.environ.get("EVAL_SYSTEM_PROMPT_VARIANT") or "default"
    if args.system_prompt_manifest:
        with open(args.system_prompt_manifest, encoding="utf-8") as handle:
            system = json.load(handle).get("system_prompt")
        if not isinstance(system, str) or not system.strip():
            ap.error("--system-prompt-manifest does not contain a non-empty system_prompt")
    else:
        system = get_system_prompt()
        if args.context_mode == "rolling-legal-history":
            system = rolling_system_prompt(
                system,
                compact=args.rolling_prompt_variant == "compact",
            )
    system += fewshot_text(fewshot_ids)
    writer = None
    if args.result_dir:
        writer = ArtifactWriter(args.result_dir, {
            "runner": "tool_rollout",
            "dataset": args.tasks_json or "data/spider_data/dev.json",
            "model": args.model,
            "base_url": args.base_url,
            "dev_size": args.n,
            "indices_file": args.indices_file or None,
            "selected_indices": sorted(selected_indices) if selected_indices else None,
            "few_shot_ids": fewshot_ids,
            "max_steps": args.max_steps,
            "max_errors_per_type": args.max_errors_per_type,
            "parser": "strict_no_repair",
            "api_transport_retries_per_request": args.api_retries,
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "api_retries": args.api_retries,
            "table_output_rows": args.table_output_rows,
            "context_mode": args.context_mode,
            "history_turns": args.history_turns,
            "rolling_prompt_variant": args.rolling_prompt_variant,
            "rolling_observation_style": args.rolling_observation_style,
            "system_prompt_manifest": args.system_prompt_manifest or None,
            "min_context_retry_tokens": MIN_CONTEXT_RETRY_TOKENS,
            "system_prompt_variant": prompt_variant,
            "system_prompt": system,
            "protocol_hash": protocol_hash(system),
        }, args.resume)
        indexed_dev = [(i, ex) for i, ex in indexed_dev if i not in writer.completed]
    results = []
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(
                run_live,
                ex,
                i,
                args.base_url,
                args.model,
                system,
                args.max_steps,
                args.max_tokens,
                args.api_retries,
                args.max_errors_per_type,
                args.table_output_rows,
                args.context_mode,
                args.history_turns,
                args.rolling_observation_style == "resident",
            )
                    for i, ex in indexed_dev]
            for fut in as_completed(futs):
                r = fut.result()
                results.append(r)
                if writer:
                    writer.append(r)
                flag = "OK " if r["correct"] else ("LEG" if r["legal"] else "ERR")
                print(f"[{len(results)}/{len(indexed_dev)}] {flag} steps={r['steps']} errs={r['errors']} {r['question'][:60]}")
    else:
        for position, (i, ex) in enumerate(indexed_dev, 1):
            r = run_live(
                ex,
                i,
                args.base_url,
                args.model,
                system,
                args.max_steps,
                args.max_tokens,
                args.api_retries,
                args.max_errors_per_type,
                args.table_output_rows,
                args.context_mode,
                args.history_turns,
                args.rolling_observation_style == "resident",
            )
            results.append(r)
            if writer:
                writer.append(r)
            flag = "OK " if r["correct"] else ("LEG" if r["legal"] else "ERR")
            print(f"[{position}/{len(indexed_dev)}] {flag} steps={r['steps']} errs={r['errors']} {ex['question'][:60]}")

    summary = writer.summarize() if writer else {
        "total": len(results),
        "correct": sum(r["correct"] for r in results),
        "accuracy": sum(r["correct"] for r in results) / max(1, len(results)),
    }
    print(f"\nEXEC-ACC {summary['correct']}/{summary['total']} "
          f"({100 * summary['accuracy']:.1f}%)")
    if writer:
        print(f"-> {writer.path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
