from __future__ import annotations

import hashlib
import json
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

from rl.frameworks.trl.rollout import (  # noqa: E402
    RolloutSettings,
    TableAgentRolloutCollector,
)
from rl.frameworks.trl.transition_batch import build_transition_updates  # noqa: E402


class _Tokenized:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _Tokenizer:
    def __init__(self):
        self.template_calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_calls.append(kwargs)
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
    def test_render_binds_qwen_thinking_template_when_declared(self):
        tokenizer = _Tokenizer()
        collector = TableAgentRolloutCollector(
            tokenizer,
            RolloutSettings(enable_thinking=True),
        )
        collector._render([
            {"role": "system", "content": "system"},
            {"role": "user", "content": "question"},
        ])
        self.assertIs(tokenizer.template_calls[0]["enable_thinking"], True)
        self.assertNotIn("chat_template_kwargs", tokenizer.template_calls[0])

    def test_render_omits_qwen_thinking_template_when_unspecified(self):
        tokenizer = _Tokenizer()
        collector = TableAgentRolloutCollector(tokenizer, RolloutSettings())
        collector._render([{"role": "user", "content": "question"}])
        self.assertNotIn("enable_thinking", tokenizer.template_calls[0])

    def test_render_passes_qwen_thinking_false_at_top_level(self):
        tokenizer = _Tokenizer()
        collector = TableAgentRolloutCollector(
            tokenizer,
            RolloutSettings(enable_thinking=False),
        )
        collector._render([{"role": "user", "content": "question"}])
        self.assertIs(tokenizer.template_calls[0]["enable_thinking"], False)
        self.assertNotIn("chat_template_kwargs", tokenizer.template_calls[0])

    def test_generation_at_token_limit_is_audited_and_excluded(self):
        max_new_tokens = 3

        def generate(prompts, request_keys):
            del request_keys
            return {
                "prompt_ids": [[101, 1] for _ in prompts],
                "completion_ids": [[7, 8, 9] for _ in prompts],
                "logprobs": [[[-0.1], [-0.2], [-0.3]] for _ in prompts],
                "logprob_token_ids": [[[7], [8], [9]] for _ in prompts],
            }

        env = _Env()
        collector = TableAgentRolloutCollector(
            _Tokenizer(),
            RolloutSettings(
                reward_mode="result-only",
                max_steps=2,
                max_new_tokens=max_new_tokens,
                max_context_tokens=128,
            ),
            generate_batch_with_keys=generate,
        )
        item = {
            "prompt": [{"role": "system", "content": "pinned-system-prompt"}],
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
        ) as factory:
            episode = collector.collect([item], SimpleNamespace())[0]

        self.assertEqual(
            factory.call_args.kwargs["system_prompt"],
            "pinned-system-prompt",
        )
        self.assertEqual(env.turn_count, 0)
        self.assertEqual(episode.sample.failure_type, "generation_length")
        self.assertFalse(episode.sample.process_update)
        self.assertEqual(episode.sample.reward, 0.0)
        self.assertEqual(len(episode.policy_turns), 1)
        self.assertEqual(
            build_transition_updates([episode], reward_mode="result-only"),
            [],
        )
        self.assertEqual(
            episode.sample.audit_record["optimization_exclusion"],
            "nonsemantic_runtime_failure",
        )
        self.assertEqual(
            episode.sample.audit_record["generation_truncation"],
            {
                "schema_version": "vllm-generation-truncation-v2",
                "kind": "length",
                "detection": "max_new_tokens_reached",
                "turn_index": 0,
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "max_new_tokens": 3,
                "response_token_ids_sha256": hashlib.sha256(
                    json.dumps([7, 8, 9], separators=(",", ":")).encode("utf-8")
                ).hexdigest(),
                "finish_reason": None,
                "finish_reason_field": None,
                "stop_reason": None,
                "stop_reason_field": None,
            },
        )

    def test_explicit_finish_reason_overrides_token_count_fallback(self):
        stopped = TableAgentRolloutCollector._generation_truncation(
            {"finish_reasons": ["stop"]},
            row_index=0,
            row_count=1,
            prompt_tokens=2,
            response_token_ids=[7, 8, 9],
            completion_tokens=3,
            max_new_tokens=3,
            turn_index=0,
        )
        length = TableAgentRolloutCollector._generation_truncation(
            {"finish_reason": ["length"]},
            row_index=0,
            row_count=1,
            prompt_tokens=5,
            response_token_ids=[7, 8],
            completion_tokens=2,
            max_new_tokens=3,
            turn_index=4,
        )
        self.assertIsNone(stopped)
        self.assertEqual(length["detection"], "explicit_finish_reason")
        self.assertEqual(length["turn_index"], 4)
        self.assertEqual(length["prompt_tokens"], 5)

    def test_context_overflow_has_exact_budget_evidence_and_is_excluded(self):
        env = _Env()
        collector = TableAgentRolloutCollector(
            _Tokenizer(),
            RolloutSettings(
                reward_mode="result-only",
                max_steps=2,
                max_new_tokens=3,
                max_context_tokens=4,
            ),
        )
        item = {
            "prompt": [{"role": "system", "content": "pinned-system-prompt"}],
            "environment": {
                "example_index": 3,
                "task_id": "task-3",
                "db_id": "db",
                "db_path": "/unused",
                "question": "q",
                "gold_sql": "select 1",
                "external_knowledge": None,
            },
        }
        with patch(
            "frameworks.trl.rollout.create_tool_use_env",
            return_value=env,
        ):
            episode = collector.collect([item], SimpleNamespace())[0]

        self.assertEqual(env.turn_count, 0)
        self.assertEqual(episode.sample.failure_type, "context_overflow")
        self.assertFalse(episode.sample.process_update)
        self.assertEqual(episode.sample.reward, 0.0)
        self.assertEqual(episode.policy_turns, [])
        self.assertEqual(
            build_transition_updates([episode], reward_mode="result-only"),
            [],
        )
        self.assertEqual(
            episode.sample.audit_record["optimization_exclusion"],
            "nonsemantic_runtime_failure",
        )
        self.assertEqual(
            episode.sample.audit_record["context_overflow"],
            {
                "schema_version": "vllm-context-overflow-v1",
                "kind": "context_overflow",
                "detection": "prompt_plus_max_new_tokens_exceeds_context",
                "turn_index": 0,
                "prompt_tokens": 2,
                "max_new_tokens": 3,
                "max_context_tokens": 4,
                "required_tokens": 5,
            },
        )

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
                "prompt": [
                    {"role": "system", "content": "pinned-system-prompt"}
                ],
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
                episodes = collector.collect(
                    [item],
                    SimpleNamespace(
                        state=SimpleNamespace(global_step=3),
                        _step=7,
                        _last_loaded_step=3,
                    ),
                )

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
            self.assertEqual(
                episodes[0].sample.audit_record["policy_global_step"], 3
            )
            self.assertEqual(
                episodes[0].sample.audit_record["policy_micro_step"], 7
            )
            self.assertEqual(
                episodes[0].sample.audit_record["policy_synced_global_step"], 3
            )
            self.assertTrue(log_path.is_file())
            self.assertTrue(env.closed)


if __name__ == "__main__":
    unittest.main()
