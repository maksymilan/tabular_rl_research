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
    parse_assistant,
    tool_output_message,
)
from rollout import (  # noqa: E402
    MAX_CONSECUTIVE_ERRORS,
    db_path,
    execute_tool,
    new_ctx,
    overview,
    score,
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
    """One Spider question episode with model-visible messages and harness state."""

    def __init__(
        self,
        example: dict,
        *,
        example_index: int | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_steps: int = 20,
        max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS,
    ):
        self.example = example
        self.example_index = example_index
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.max_consecutive_errors = max_consecutive_errors
        self.reset()

    def reset(self) -> list[dict]:
        self.harness = Harness(db_path(self.example["db_id"]))
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": first_user_message(overview(self.harness), self.example["question"]),
            },
        ]
        self.initial_messages = deepcopy(self.messages)
        self.created: set[str] = set()
        self.ctx = new_ctx()
        self.steps = 0
        self.errors = 0
        self.consecutive_errors = 0
        self.turns: list[dict] = []
        self.done = False
        self.correct = False
        self.legal = False
        self.failure_type: str | None = None
        self.started = time.time()
        return deepcopy(self.messages)

    def model_messages(self) -> list[dict]:
        return deepcopy(self.messages)

    def record(self) -> dict:
        return {
            "example_index": self.example_index,
            "db_id": self.example["db_id"],
            "question": self.example["question"],
            "gold_sql": self.example["query"],
            "initial_model_input": self.initial_messages,
            "turns": self.turns,
            "correct": self.correct,
            "legal": self.legal,
            "steps": self.steps,
            "errors": self.errors,
            "failure_type": self.failure_type,
            "final_messages": deepcopy(self.messages),
            "elapsed_seconds": round(time.time() - self.started, 3),
        }

    def apply_model_output(self, text: str) -> EnvStep:
        """Apply one assistant message, append feedback, and return the environment transition."""
        if self.done:
            raise RuntimeError("episode is already done; call reset() before stepping again")

        turn = {"turn_index": len(self.turns), "model_input": deepcopy(self.messages)}
        turn["model_output"] = text
        self.messages.append({"role": "assistant", "content": text})

        try:
            think, tool, args = parse_assistant(text)
            turn["parsed"] = {"think": think, "tool": tool, "arguments": args}
            if tool == "answer_from_context":
                self.legal = True
                self.steps += 1
                self.correct, turn["pred_sample"], turn["gold_sample"] = score(
                    self.harness, self.example["query"], args, self.created
                )
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

            step_id = f"step_{self.steps + 1}"
            output, table_name = execute_tool(self.harness, tool, args, self.ctx, step_id)
            turn["tool_output"] = output
            self.turns.append(turn)
            self.consecutive_errors = 0
            self.steps += 1
            if table_name:
                self.created.add(table_name)
            observation = tool_output_message(step_id, output)
            self.messages.append({"role": "user", "content": observation})
            if self.steps >= self.max_steps:
                self.done = True
                self.failure_type = "max_steps"
            return EnvStep(done=self.done, observation=observation, turn=turn)

        except Exception as exc:  # noqa: BLE001 - every tool/protocol failure becomes feedback
            self.errors += 1
            self.consecutive_errors += 1
            error_type = "protocol_error" if isinstance(exc, ProtocolError) else "execution_error"
            error = f"{type(exc).__name__}: {exc}"
            turn["execution_error"] = error
            turn["execution_error_type"] = error_type
            self.turns.append(turn)
            observation = json.dumps(
                {
                    "step_id": f"step_{self.steps + 1}",
                    "status": "error",
                    "error": {"type": error_type, "message": error},
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if self.consecutive_errors >= self.max_consecutive_errors:
                self.done = True
                self.failure_type = error_type
                return EnvStep(
                    done=True,
                    observation=observation,
                    turn=turn,
                    failure_type=error_type,
                )
            self.messages.append({"role": "user", "content": observation})
            return EnvStep(done=False, observation=observation, turn=turn, failure_type=error_type)
