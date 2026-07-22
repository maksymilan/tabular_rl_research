#!/usr/bin/env python3
"""Reusable closed-loop table-tool environment for RL experiments.

The trainer owns model generation. This module owns everything after a model turn:
parsing, tool execution, observation construction, scoring, and episode bookkeeping.
It intentionally reuses the exact eval protocol/harness functions so SFT, eval, and RL
do not drift.
"""
from __future__ import annotations

import json
import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src", "eval"))
sys.path.insert(0, os.path.join(ROOT, "src", "harness"))
sys.path.insert(0, os.path.join(ROOT, "src", "sft"))

from executor import Harness  # noqa: E402
from protocol import (  # noqa: E402
    ProtocolError,
    SYSTEM_PROMPT,
    first_user_message,
    model_context_messages,
    parse_assistant_strict,
    rolling_legal_history_messages,
    tool_error_message,
    tool_output_message,
)
from rollout import (  # noqa: E402
    MAX_ERRORS_PER_TYPE,
    execute_tool,
    new_ctx,
    overview,
    protocol_failure_type,
    score,
    state_digest,
    task_db_path,
    task_gold_sql,
)


@dataclass
class EnvStep:
    """Result of applying one model output to the harness."""

    done: bool
    observation: str | None
    turn: dict[str, Any]
    correct: bool = False
    legal: bool = False
    failure_type: str | None = None


class ToolUseEnv:
    """One dataset-adapter task episode with state-only model context and harness state."""

    def __init__(
        self,
        example: dict,
        *,
        example_index: int | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_steps: int = 20,
        max_errors_per_type: int = MAX_ERRORS_PER_TYPE,
        context_mode: str = "state-only",
        history_turns: int = 4,
        compact_observations: bool = False,
    ):
        self.example = example
        self.example_index = example_index
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_errors_per_type = max_errors_per_type
        self.context_mode = context_mode
        self.history_turns = history_turns
        self.compact_observations = compact_observations
        self.reset()

    def reset(self) -> list[dict]:
        self.harness = Harness(task_db_path(self.example))
        self.overview = overview(self.harness)
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": first_user_message(
                    self.overview,
                    self.example["question"],
                    self.example.get("external_knowledge"),
                ),
            },
        ]
        self.initial_messages = deepcopy(self.messages)
        self.created: set[str] = set()
        self.ctx = new_ctx(self.overview)
        self.last_error: dict | None = None
        self.steps = 0
        self.errors = 0
        self.error_counts: dict[str, int] = {}
        self.error_events: list[dict[str, Any]] = []
        self.legal_history: list[dict[str, str]] = []
        self.turns: list[dict] = []
        self.done = False
        self.correct = False
        self.legal = False
        self.failure_type: str | None = None
        self.started = time.time()
        return deepcopy(self.messages)

    def model_messages(self) -> list[dict]:
        if self.context_mode == "rolling-legal-history":
            return rolling_legal_history_messages(
                self.system_prompt,
                self.overview,
                self.example["question"],
                self.ctx["environment"].snapshot(),
                self.last_error,
                self.example.get("external_knowledge"),
                self.legal_history,
                self.history_turns,
                compact_observations=self.compact_observations,
            )
        if self.context_mode != "state-only":
            raise ValueError(f"unsupported context_mode: {self.context_mode}")
        return model_context_messages(
            self.system_prompt,
            self.overview,
            self.example["question"],
            self.ctx["environment"].snapshot(),
            self.last_error,
            self.example.get("external_knowledge"),
        )

    def record(self) -> dict:
        return {
            "example_index": self.example_index,
            "db_id": self.example["db_id"],
            "question": self.example["question"],
            "gold_sql": task_gold_sql(self.example),
            "initial_model_input": self.initial_messages,
            "turns": self.turns,
            "correct": self.correct,
            "legal": self.legal,
            "steps": self.steps,
            "errors": self.errors,
            "failure_type": self.failure_type,
            "error_events": deepcopy(self.error_events),
            "final_messages": deepcopy(self.messages),
            "elapsed_seconds": round(time.time() - self.started, 3),
        }

    def apply_model_output(self, text: str) -> EnvStep:
        """Apply one assistant message, append feedback, and return the environment transition."""
        if self.done:
            raise RuntimeError("episode is already done; call reset() before stepping again")

        self.steps += 1
        state_before = self.ctx["environment"].snapshot()
        step_id = f"step_{self.steps}"
        turn = {
            "turn_index": len(self.turns),
            "model_input": self.model_messages(),
            "feedback_recovery": bool(self.last_error),
            "recovered_from_error_type": (self.last_error or {}).get("error", {}).get("type"),
        }
        turn["model_output"] = text
        self.messages.append({"role": "assistant", "content": text})

        try:
            think, tool, args = parse_assistant_strict(text)
            turn["parsed"] = {"think": think, "tool": tool, "arguments": args}
            if tool == "answer_from_context":
                self.correct, turn["pred_sample"], turn["gold_sample"] = score(
                    self.harness, task_gold_sql(self.example), args, self.created
                )
                self.legal = True
                if not self.correct:
                    self.failure_type = "wrong_answer"
                self.done = True
                self.turns.append(turn)
                return EnvStep(
                    done=True,
                    observation=None,
                    turn=turn,
                    correct=self.correct,
                    legal=True,
                    failure_type=self.failure_type,
                )

            output, table_name = execute_tool(self.harness, tool, args, self.ctx, step_id)
            turn["tool_output"] = output
            self.turns.append(turn)
            self.last_error = None
            if table_name:
                self.created.add(table_name)
            observation = tool_output_message(
                step_id,
                output,
            )
            self.legal_history.append({"assistant": text, "observation": observation})
            self.messages.append({"role": "user", "content": observation})
            if self.steps >= self.max_steps:
                self.done = True
                self.failure_type = "max_steps"
            return EnvStep(done=self.done, observation=observation, turn=turn)

        except Exception as exc:  # noqa: BLE001 - every model action becomes an audited transition
            self.errors += 1
            parsed = turn.get("parsed") or {}
            error_type = protocol_failure_type(exc) if isinstance(exc, ProtocolError) else "execution_error"
            error = f"{type(exc).__name__}: {exc}"
            state_after = self.ctx["environment"].snapshot()
            if error_type == "execution_error" and state_digest(state_after) != state_digest(state_before):
                error_type = "nonrecoverable_execution_error"
            turn["execution_error"] = error
            turn["execution_error_type"] = error_type
            event = {
                "action_index": self.steps,
                "step_id": step_id,
                "error_type": error_type,
                "message": error,
                "state_before_hash": state_digest(state_before),
                "state_after_hash": state_digest(state_after),
            }
            if parsed.get("tool"):
                event["attempted_tool"] = parsed["tool"]
                event["attempted_arguments"] = parsed.get("arguments") or {}
            turn["error_event"] = event
            self.error_events.append(event)
            self.turns.append(turn)
            observation = tool_error_message(step_id, error_type, error)
            self.last_error = json.loads(observation)
            if error_type == "nonrecoverable_execution_error":
                self.done = True
                self.failure_type = error_type
                return EnvStep(
                    done=True,
                    observation=observation,
                    turn=turn,
                    failure_type=error_type,
                )
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            if self.error_counts[error_type] >= self.max_errors_per_type:
                self.done = True
                self.failure_type = error_type
                return EnvStep(
                    done=True,
                    observation=observation,
                    turn=turn,
                    failure_type=error_type,
                )
            if self.steps >= self.max_steps:
                self.done = True
                self.failure_type = "max_steps"
                return EnvStep(
                    done=True,
                    observation=observation,
                    turn=turn,
                    failure_type=self.failure_type,
                )
            self.messages.append({"role": "user", "content": observation})
            return EnvStep(done=False, observation=observation, turn=turn, failure_type=error_type)
