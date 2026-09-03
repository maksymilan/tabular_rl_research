#!/usr/bin/env python3
"""Audit model-visible behavior in a direct-SQL-search discovery artifact.

The report is deliberately label-blind: it classifies only public actions, public feedback,
provider-authored reasoning markers, and serialized model-context size.  It never emits SQL text,
database rows, questions, external knowledge, or verifier reference data.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "src"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from tool_modules.direct_sql_search.protocol import (  # noqa: E402
    DIRECT_SQL_SEARCH_INTERFACE,
    DIRECT_SQL_SEARCH_PROTOCOL_VERSION_V1,
)
from tool_modules.sql_common.runner import (  # noqa: E402
    build_system_prompt,
    compact_direct_sql_observation,
    compact_prior_direct_sql_assistant,
    state_prompt,
)
from provider_adapter import provider_request_messages  # noqa: E402


UNCERTAINTY_RE = re.compile(
    r"\b(?:maybe|perhaps|likely|probably|could|might|not sure|need (?:to )?decide|"
    r"uncertain|infer|guess|hmm)\b",
    re.I,
)
OUTPUT_AUDIT_RE = re.compile(
    r"\b(?:exact output|output (?:shape|column)|requested (?:field|column|row)|"
    r"helper field|column order|representation|grain|population)\b",
    re.I,
)
DISTINCT_DEBATE_RE = re.compile(r"\bdistinct\b", re.I)


def compact_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def canonical_sql(sql: object) -> str | None:
    if not isinstance(sql, str):
        return None
    return sql.strip().rstrip(";").strip()


def _load_records(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _last_user_content(model_input: object) -> str:
    if not isinstance(model_input, list):
        return ""
    for message in reversed(model_input):
        if isinstance(message, dict) and message.get("role") == "user":
            content = message.get("content")
            return content if isinstance(content, str) else ""
    return ""


def audit_record(record: dict) -> dict:
    seen_actions: collections.Counter[tuple[str, str]] = collections.Counter()
    seen_inspection_sql: collections.Counter[str] = collections.Counter()
    successful_sql: list[str] = []
    prior_signature: tuple[str, str] | None = None
    adjacent_repeats = 0
    repeated_actions = 0
    repeated_inspections = 0
    search_calls = 0
    schema_calls = 0
    data_inspections = 0
    final_calls = 0
    uncertainty_turns = 0
    output_audit_turns = 0
    distinct_debate_turns = 0
    model_input_chars: list[int] = []
    last_user_chars: list[int] = []
    prior_sql_occurrences: list[int] = []
    error_feedback_with_attempt = 0
    error_feedback_without_attempt = 0
    counterfactual_v2_chars: list[int] = []
    legal_history: list[dict] = []
    workspace_v2: list[dict] = []
    simulated_last_error: dict | None = None
    simulate_v2_context = (
        record.get("protocol_version") == DIRECT_SQL_SEARCH_PROTOCOL_VERSION_V1
    )
    initial_user = ""
    v2_system = ""
    if simulate_v2_context:
        first_model_input = (record.get("turns") or [{}])[0].get("model_input") or []
        initial_user = _last_user_content(first_model_input)
        initial_user = initial_user.split("\n\nCURRENT SQL WORKSPACE", 1)[0]
        v2_system = build_system_prompt(
            "deepseek-v4-flash",
            context_profile=record.get("context_profile") or "lazy-catalog-v1",
            interface=DIRECT_SQL_SEARCH_INTERFACE,
        )

    for turn in record.get("turns", []):
        reasoning = turn.get("provider_reasoning_content") or ""
        if isinstance(reasoning, str):
            uncertainty_turns += bool(UNCERTAINTY_RE.search(reasoning))
            output_audit_turns += bool(OUTPUT_AUDIT_RE.search(reasoning))
            distinct_debate_turns += bool(
                DISTINCT_DEBATE_RE.search(reasoning) and UNCERTAINTY_RE.search(reasoning)
            )

        model_input = turn.get("model_input")
        if isinstance(model_input, list):
            serialized = compact_json(model_input)
            model_input_chars.append(len(serialized))
            last_user = _last_user_content(model_input)
            last_user_chars.append(len(last_user))
            if successful_sql:
                prior_sql_occurrences.append(sum(last_user.count(sql) for sql in successful_sql))

        if simulate_v2_context:
            retained = legal_history[-4:]
            simulated_messages = [
                {"role": "system", "content": v2_system},
                {"role": "user", "content": initial_user},
            ]
            for index, item in enumerate(retained):
                assistant = item["assistant"]
                if index < len(retained) - 1:
                    assistant = compact_prior_direct_sql_assistant(assistant)
                simulated_messages.extend([
                    {"role": "assistant", "content": assistant},
                    {
                        "role": "user",
                        "content": compact_direct_sql_observation(item["observation"]),
                    },
                ])
            simulated_messages[-1]["content"] += "\n\n" + state_prompt(
                workspace_v2,
                simulated_last_error,
                interface=DIRECT_SQL_SEARCH_INTERFACE,
            )
            simulated_messages = provider_request_messages(
                "deepseek-v4-flash",
                simulated_messages,
            )
            counterfactual_v2_chars.append(len(compact_json(simulated_messages)))

        parsed = turn.get("parsed")
        if not isinstance(parsed, dict):
            if isinstance(turn.get("error_event"), dict):
                event = turn["error_event"]
                simulated_last_error = {
                    "step_id": event.get("step_id"),
                    "status": "error",
                    "error": {
                        "type": event.get("error_type"),
                        "code": "provider_action_carrier_rejected",
                        "details": {"successful_state_changed": False},
                    },
                }
            continue
        tool = parsed.get("tool")
        arguments = parsed.get("arguments")
        if not isinstance(tool, str) or not isinstance(arguments, dict):
            continue
        signature = (tool, compact_json(arguments))
        if seen_actions[signature]:
            repeated_actions += 1
        seen_actions[signature] += 1
        if prior_signature == signature:
            adjacent_repeats += 1
        prior_signature = signature

        if tool == "search_values":
            search_calls += 1
        elif tool == "execute_sql":
            sql = canonical_sql(arguments.get("sql"))
            mode = arguments.get("mode")
            if mode == "final":
                final_calls += 1
            elif sql is not None:
                if seen_inspection_sql[sql]:
                    repeated_inspections += 1
                seen_inspection_sql[sql] += 1
                if re.match(r"^(?:PRAGMA|EXPLAIN\s+QUERY\s+PLAN)\b", sql, re.I):
                    schema_calls += 1
                else:
                    data_inspections += 1
                if turn.get("tool_output") is not None:
                    successful_sql.append(sql)

        tool_output = turn.get("tool_output")
        if isinstance(tool_output, dict) and not turn.get("error_event"):
            step_id = f"step_{int(turn.get('turn_index', 0)) + 1}"
            entry = {
                "step_id": step_id,
                "tool": tool,
                **({"mode": "inspect"} if tool == "execute_sql" else {}),
                **tool_output,
            }
            workspace_v2.append(entry)
            observation = compact_json({
                "step_id": step_id,
                "status": "success",
                "tool": tool,
                "output": tool_output,
            })
            legal_history.append({
                "assistant": turn.get("model_output") or "",
                "observation": observation,
            })
            simulated_last_error = None

        error_event = turn.get("error_event")
        if isinstance(error_event, dict):
            if error_event.get("attempted_tool") and error_event.get("attempted_arguments"):
                error_feedback_with_attempt += 1
            else:
                error_feedback_without_attempt += 1

    return {
        "instance_id": record.get("instance_id"),
        "correct": bool(record.get("correct")),
        "legal": bool(record.get("legal")),
        "failure_type": record.get("failure_type"),
        "steps": record.get("steps", len(record.get("turns", []))),
        "search_calls": search_calls,
        "schema_calls": schema_calls,
        "data_inspections": data_inspections,
        "final_calls": final_calls,
        "unique_inspection_sql": len(seen_inspection_sql),
        "repeated_actions": repeated_actions,
        "repeated_inspections": repeated_inspections,
        "adjacent_repeats": adjacent_repeats,
        "uncertainty_turns": uncertainty_turns,
        "output_audit_turns": output_audit_turns,
        "distinct_debate_turns": distinct_debate_turns,
        "mean_model_input_chars": round(sum(model_input_chars) / len(model_input_chars), 1)
        if model_input_chars else 0,
        "max_model_input_chars": max(model_input_chars, default=0),
        "mean_counterfactual_v2_model_input_chars": round(
            sum(counterfactual_v2_chars) / len(counterfactual_v2_chars), 1
        ) if counterfactual_v2_chars else 0,
        "max_counterfactual_v2_model_input_chars": max(
            counterfactual_v2_chars, default=0
        ),
        "counterfactual_v2_context_applicable": simulate_v2_context,
        "mean_last_user_chars": round(sum(last_user_chars) / len(last_user_chars), 1)
        if last_user_chars else 0,
        "mean_prior_sql_occurrences_in_last_user": round(
            sum(prior_sql_occurrences) / len(prior_sql_occurrences), 2
        ) if prior_sql_occurrences else 0,
        "error_feedback_with_attempt_in_audit": error_feedback_with_attempt,
        "error_feedback_without_attempt_in_audit": error_feedback_without_attempt,
    }


def summarize(rows: list[dict]) -> dict:
    totals = collections.Counter()
    for row in rows:
        for key in (
            "steps", "search_calls", "schema_calls", "data_inspections", "final_calls",
            "unique_inspection_sql", "repeated_actions", "repeated_inspections",
            "adjacent_repeats", "uncertainty_turns", "output_audit_turns",
            "distinct_debate_turns", "error_feedback_with_attempt_in_audit",
            "error_feedback_without_attempt_in_audit",
        ):
            totals[key] += int(row[key])
    n = len(rows)
    counterfactual_rows = [
        row for row in rows if row["counterfactual_v2_context_applicable"]
    ]
    return {
        "episodes": n,
        "correct": sum(bool(row["correct"]) for row in rows),
        "legal": sum(bool(row["legal"]) for row in rows),
        "failures": dict(collections.Counter(row["failure_type"] for row in rows if not row["correct"])),
        "totals": dict(totals),
        "episodes_with_repeated_actions": sum(row["repeated_actions"] > 0 for row in rows),
        "episodes_with_adjacent_repeats": sum(row["adjacent_repeats"] > 0 for row in rows),
        "episodes_with_distinct_debate": sum(row["distinct_debate_turns"] > 0 for row in rows),
        "mean_model_input_chars": round(
            sum(float(row["mean_model_input_chars"]) for row in rows) / n, 1
        ) if n else 0,
        "max_model_input_chars": max((int(row["max_model_input_chars"]) for row in rows), default=0),
        "mean_counterfactual_v2_model_input_chars": round(
            sum(
                float(row["mean_counterfactual_v2_model_input_chars"])
                for row in counterfactual_rows
            ) / len(counterfactual_rows),
            1,
        ) if counterfactual_rows else 0,
        "max_counterfactual_v2_model_input_chars": max(
            (
                int(row["max_counterfactual_v2_model_input_chars"])
                for row in counterfactual_rows
            ),
            default=0,
        ),
        "counterfactual_v2_context_applicable_episodes": len(counterfactual_rows),
        "mean_prior_sql_occurrences_in_last_user": round(
            sum(float(row["mean_prior_sql_occurrences_in_last_user"]) for row in rows) / n, 2
        ) if n else 0,
        "per_episode": rows,
        "information_boundary": {
            "emits_sql": False,
            "emits_rows": False,
            "emits_question": False,
            "emits_gold_or_reference_values": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()
    report = summarize([audit_record(record) for record in _load_records(Path(args.input))])
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
