"""Verl multi-turn adapter for the framework-neutral table-tool environment.

The loop preserves exact generated token ids, inserts harness observations with response_mask=0,
and assigns a single terminal reward: 1 for exact execution correctness, otherwise 0.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "rl"))
sys.path.insert(0, str(ROOT / "src" / "eval"))
sys.path.insert(0, str(ROOT / "src" / "harness"))
sys.path.insert(0, str(ROOT / "src" / "sft"))

from executor import Harness
from protocol import ProtocolError, parse_assistant, tool_output_message
from reward import terminal_result_reward
from rollout import MAX_CONSECUTIVE_ERRORS, db_path, execute_tool, new_ctx, overview, score

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopMetrics, AgentLoopOutput, register
from verl.utils.profiler import simple_timer


@register("table_tool_agent")
class TableToolAgentLoop(AgentLoopBase):
    """Run one raw-protocol table-agent episode and return a verl-compatible token trajectory."""

    def __init__(self, *args, max_steps: int = 20, max_consecutive_errors: int = MAX_CONSECUTIVE_ERRORS,
                 table_output_rows: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps
        self.max_consecutive_errors = max_consecutive_errors
        self.table_output_rows = table_output_rows

    async def _user_message_tokens(self, content: str) -> list[int]:
        tokens = await self.apply_chat_template(
            [{"role": "user", "content": content}], remove_system_prompt=True
        )
        return list(self.turn_separator) + tokens

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra = dict(kwargs.get("extra_info") or {})
        db_id = str(extra["db_id"])
        gold_sql = str(extra["gold_sql"])
        harness = Harness(db_path(db_id))
        ctx = new_ctx(overview(harness))
        created: set[str] = set()
        initial_prompt_ids = await self.apply_chat_template(list(kwargs["raw_prompt"]))
        runtime_ids = list(initial_prompt_ids)
        response_ids: list[int] = []
        response_mask: list[int] = []
        response_logprobs: list[float] = []
        have_rollout_logprobs = True
        metrics = AgentLoopMetrics()
        steps = errors = consecutive_errors = assistant_turns = user_turns = 0
        correct = legal = False
        failure_type: str | None = None
        request_id = uuid4().hex

        while len(response_ids) < self.rollout_config.response_length and steps < self.max_steps:
            with simple_timer("generate_sequences", metrics):
                generated = await self.server_manager.generate(
                    request_id=request_id,
                    prompt_ids=runtime_ids,
                    sampling_params=sampling_params,
                )
            assistant_ids = list(generated.token_ids)
            if not assistant_ids:
                failure_type = "empty_generation"
                break
            assistant_turns += 1
            runtime_ids.extend(assistant_ids)
            response_ids.extend(assistant_ids)
            response_mask.extend([1] * len(assistant_ids))
            if generated.log_probs:
                response_logprobs.extend(generated.log_probs)
            else:
                have_rollout_logprobs = False
            text = self.tokenizer.decode(assistant_ids, skip_special_tokens=True)

            try:
                _, tool, arguments = parse_assistant(text)
                if tool == "answer_from_context":
                    legal = True
                    steps += 1
                    correct, _, _ = score(harness, gold_sql, arguments, created)
                    failure_type = None if correct else "wrong_answer"
                    break

                step_id = f"step_{steps + 1}"
                with simple_timer("tool_calls", metrics):
                    output, table_name = execute_tool(
                        harness, tool, arguments, ctx, step_id, table_output_rows=self.table_output_rows
                    )
                if table_name:
                    created.add(table_name)
                steps += 1
                consecutive_errors = 0
                observation = tool_output_message(step_id, output)
            except Exception as exc:  # model errors are environment feedback, never shaped rewards
                errors += 1
                consecutive_errors += 1
                kind = "protocol_error" if isinstance(exc, ProtocolError) else "execution_error"
                observation = json.dumps({
                    "step_id": f"step_{steps + 1}",
                    "status": "error",
                    "error": {"type": kind, "message": f"{type(exc).__name__}: {exc}"},
                }, ensure_ascii=False, separators=(",", ":"))
                if consecutive_errors >= self.max_consecutive_errors:
                    failure_type = kind
                    break

            observation_ids = await self._user_message_tokens(observation)
            if len(response_ids) + len(observation_ids) > self.rollout_config.response_length:
                failure_type = "response_length"
                break
            runtime_ids.extend(observation_ids)
            response_ids.extend(observation_ids)
            response_mask.extend([0] * len(observation_ids))
            if have_rollout_logprobs:
                response_logprobs.extend([0.0] * len(observation_ids))
            user_turns += 1
        else:
            failure_type = failure_type or "max_steps"

        if not legal and failure_type is None:
            failure_type = "max_steps"
        reward = terminal_result_reward(correct)
        return AgentLoopOutput(
            prompt_ids=initial_prompt_ids,
            response_ids=response_ids[: self.rollout_config.response_length],
            response_mask=response_mask[: self.rollout_config.response_length],
            response_logprobs=(response_logprobs[: self.rollout_config.response_length]
                               if have_rollout_logprobs else None),
            reward_score=reward,
            num_turns=assistant_turns + user_turns,
            metrics=metrics,
            extra_fields={
                "table_rl": {
                    "correct": correct,
                    "legal": legal,
                    "failure_type": failure_type,
                    "steps": steps,
                    "errors": errors,
                    "reward": reward,
                    "db_id": db_id,
                    "example_index": extra.get("example_index"),
                }
            },
        )
