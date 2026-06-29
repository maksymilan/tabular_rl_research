#!/usr/bin/env python3
"""Transparent pilot rewards for table-tool RL.

These rewards are deliberately simple and auditable. They are meant for the first
RL smoke tests and reward-shaping ablations, not as a final paper reward.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


DEFAULT_WEIGHTS = {
    "terminal_correct": 1.0,
    "terminal_wrong": -0.2,
    "legal_answer": 0.1,
    "valid_tool_call": 0.01,
    "tool_error": -0.08,
    "protocol_or_api_failure": -0.3,
    "repeat_call": -0.05,
    "read_evidence_before_answer": 0.05,
    "max_steps": -0.2,
    "step_cost": -0.005,
}


@dataclass
class RewardBreakdown:
    reward: float = 0.0
    components: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.reward += value
        self.components[name] = self.components.get(name, 0.0) + value


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def short_hash(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:12]


def parsed_tool(turn: dict) -> tuple[str | None, dict]:
    parsed = turn.get("parsed") or {}
    tool = parsed.get("tool")
    args = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {}
    return tool, args


def tool_call_signature(turn: dict) -> str | None:
    tool, args = parsed_tool(turn)
    if not tool:
        return None
    return short_hash({"tool": tool, "arguments": args})


def is_successful_tool_turn(turn: dict) -> bool:
    tool, _ = parsed_tool(turn)
    return bool(tool and tool != "answer_from_context" and "tool_output" in turn)


def iter_episode_records(record: dict):
    """Yield (sample_id, episode_record) from either normal rollout or pass@k records."""
    samples = record.get("samples")
    if isinstance(samples, list) and samples:
        for sample in samples:
            merged = {
                "example_index": record.get("example_index"),
                "db_id": record.get("db_id"),
                "question": record.get("question"),
                "gold_sql": record.get("gold_sql"),
                **sample,
            }
            yield f"q{record.get('example_index')}_sample{sample.get('sample_index')}", merged
    else:
        yield f"q{record.get('example_index')}", record


def score_episode(record: dict, weights: dict[str, float] | None = None) -> RewardBreakdown:
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    out = RewardBreakdown()
    turns = record.get("turns") or []
    seen_calls: set[str] = set()
    repeated = 0
    valid_tool_calls = 0
    tool_errors = 0
    last_non_answer_tool = None

    for turn in turns:
        tool, _ = parsed_tool(turn)
        if tool and tool != "answer_from_context":
            last_non_answer_tool = tool
        if is_successful_tool_turn(turn):
            valid_tool_calls += 1
            out.add("valid_tool_call", weights["valid_tool_call"])
            signature = tool_call_signature(turn)
            if signature in seen_calls:
                repeated += 1
                out.add("repeat_call", weights["repeat_call"])
            elif signature:
                seen_calls.add(signature)
        if turn.get("execution_error") or turn.get("api_error"):
            tool_errors += 1
            out.add("tool_error", weights["tool_error"])

    steps = int(record.get("steps") or valid_tool_calls)
    if steps:
        out.add("step_cost", weights["step_cost"] * steps)

    if record.get("legal"):
        out.add("legal_answer", weights["legal_answer"])
    if record.get("correct"):
        out.add("terminal_correct", weights["terminal_correct"])
    else:
        failure_type = record.get("failure_type") or record.get("fail") or "wrong_answer"
        if failure_type in {"protocol_error", "api_error", "context_overflow", "execution_error"}:
            out.add("protocol_or_api_failure", weights["protocol_or_api_failure"])
        elif failure_type == "max_steps":
            out.add("max_steps", weights["max_steps"])
        else:
            out.add("terminal_wrong", weights["terminal_wrong"])

    if record.get("legal") and last_non_answer_tool == "read_subtable":
        out.add("read_evidence_before_answer", weights["read_evidence_before_answer"])

    out.reward = round(out.reward, 6)
    out.components = {key: round(value, 6) for key, value in sorted(out.components.items())}
    out.diagnostics = {
        "correct": bool(record.get("correct")),
        "legal": bool(record.get("legal")),
        "failure_type": record.get("failure_type"),
        "steps": steps,
        "turns": len(turns),
        "valid_tool_calls": valid_tool_calls,
        "tool_errors": tool_errors,
        "repeat_calls": repeated,
        "last_non_answer_tool": last_non_answer_tool,
    }
    return out
