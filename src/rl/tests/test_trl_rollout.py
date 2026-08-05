from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "src" / "rl"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from frameworks.trl.rollout import (  # noqa: E402
    RolloutSettings,
    TableAgentRolloutCollector,
)


class _Tokenized:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return "|".join(message["content"] for message in messages)

    def __call__(self, text, add_special_tokens=False):
        return _Tokenized([len(text), 1])

    def decode(self, token_ids, skip_special_tokens=True):
        return f"response-{token_ids[0]}"


class _Env:
    def __init__(self):
        self.done = False
        self.failure_type = None
        self.turn_count = 0
        self.correct = False
        self.closed = False

    def model_messages(self):
        return [
            {"role": "system", "content": "system"},
            {"role": "user", "content": f"state-{self.turn_count}"},
        ]

    def apply_model_output(self, text):
        self.turn_count += 1
        if self.turn_count == 2:
            self.done = True
            self.correct = True

    def record(self):
        return {
            "example_index": 3,
            "correct": self.correct,
            "failure_type": self.failure_type,
            "turns": [],
        }

    def close(self):
        self.closed = True


class TrlRolloutTest(unittest.TestCase):
    def test_collector_keeps_one_exact_prefix_and_logprob_vector_per_turn(self):
        call_count = 0
        seen_request_keys = []

        def generate(prompts, request_keys):
            nonlocal call_count
            call_count += 1
            seen_request_keys.append(request_keys)
            token = 6 + call_count
            return {
                "prompt_ids": [[100 + call_count, 1] for _ in prompts],
                "completion_ids": [[token] for _ in prompts],
                "logprobs": [[[-0.25]] for _ in prompts],
                "logprob_token_ids": [[[token]] for _ in prompts],
            }

        env = _Env()
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "rollouts.jsonl"
            collector = TableAgentRolloutCollector(
                _Tokenizer(),
                RolloutSettings(
                    reward_mode="result-only",
                    max_steps=2,
                    max_new_tokens=8,
                    max_context_tokens=128,
                ),
                generate_batch_with_keys=generate,
                rollout_log_path=log_path,
            )
            item = {
                "environment": {
                    "example_index": 3,
                    "task_id": "task-3",
                    "db_id": "db",
                    "db_path": "/unused",
                    "question": "q",
                    "gold_sql": "select 1",
                    "external_knowledge": None,
                }
            }
            with patch(
                "frameworks.trl.rollout.create_tool_use_env",
                return_value=env,
            ):
                episodes = collector.collect([item], SimpleNamespace())

            self.assertEqual(call_count, 2)
            self.assertEqual(
                seen_request_keys,
                [[("task-3", 0, 0)], [("task-3", 0, 1)]],
            )
            self.assertEqual(len(episodes), 1)
            self.assertEqual(len(episodes[0].policy_turns), 2)
            self.assertEqual(
                [turn.response_ids for turn in episodes[0].policy_turns],
                [(7,), (8,)],
            )
            self.assertEqual(
                [turn.sampling_logprobs for turn in episodes[0].policy_turns],
                [(-0.25,), (-0.25,)],
            )
            self.assertTrue(episodes[0].sample.correct)
            self.assertTrue(
                episodes[0].sample.audit_record["rollout_tokenization_warning"]
            )
            self.assertTrue(log_path.is_file())
            self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
