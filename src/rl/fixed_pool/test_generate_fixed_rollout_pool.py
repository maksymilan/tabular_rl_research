from __future__ import annotations

import asyncio
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


class _SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _LoRARequest:
    def __init__(self, name, identifier, path):
        self.name = name
        self.identifier = identifier
        self.path = path


class _AsyncLLM:
    @classmethod
    def from_engine_args(cls, args):
        return cls()


@pytest.fixture
def generator_module(monkeypatch):
    """Import the GPU generator against schema-only CPU module doubles."""
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = object

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    vllm.LLM = object
    vllm.SamplingParams = _SamplingParams
    engine = types.ModuleType("vllm.engine")
    engine.__path__ = []
    arg_utils = types.ModuleType("vllm.engine.arg_utils")
    arg_utils.AsyncEngineArgs = _SamplingParams
    lora = types.ModuleType("vllm.lora")
    lora.__path__ = []
    lora_request = types.ModuleType("vllm.lora.request")
    lora_request.LoRARequest = _LoRARequest
    v1 = types.ModuleType("vllm.v1")
    v1.__path__ = []
    v1_engine = types.ModuleType("vllm.v1.engine")
    v1_engine.__path__ = []
    async_llm = types.ModuleType("vllm.v1.engine.async_llm")
    async_llm.AsyncLLM = _AsyncLLM

    modules = {
        "transformers": transformers,
        "vllm": vllm,
        "vllm.engine": engine,
        "vllm.engine.arg_utils": arg_utils,
        "vllm.lora": lora,
        "vllm.lora.request": lora_request,
        "vllm.v1": v1,
        "vllm.v1.engine": v1_engine,
        "vllm.v1.engine.async_llm": async_llm,
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    name = "src.rl.fixed_pool.generate_fixed_rollout_pool"
    sys.modules.pop(name, None)
    module = importlib.import_module(name)
    yield module
    sys.modules.pop(name, None)


def _candidate(logprob: float = -0.25) -> SimpleNamespace:
    return SimpleNamespace(logprob=logprob)


def _completion(
    token_ids: list[int],
    *,
    finish_reason: str | None,
    stop_reason: int | str | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        token_ids=token_ids,
        logprobs=[{token_id: _candidate()} for token_id in token_ids],
        finish_reason=finish_reason,
        stop_reason=stop_reason,
    )


def _request_output(
    token_ids: list[int],
    *,
    finish_reason: str | None,
    stop_reason: int | str | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        prompt_token_ids=[101, 102],
        outputs=[
            _completion(
                token_ids,
                finish_reason=finish_reason,
                stop_reason=stop_reason,
            )
        ],
        finished=True,
    )


def test_static_generator_emits_flat_finish_rows_compatible_with_collector(
    generator_module,
) -> None:
    outputs = [
        _request_output(
            [11, 12, 13],
            finish_reason="stop",
            stop_reason=13,
        ),
        _request_output(
            [21, 22],
            finish_reason="length",
            stop_reason=None,
        ),
    ]

    class FakeLLM:
        def generate(self, prompts, params, **kwargs):
            assert prompts == ["first", "second"]
            assert len(params) == 2
            assert all(param.kwargs["n"] == 1 for param in params)
            assert kwargs["use_tqdm"] is False
            return outputs

    generator = object.__new__(generator_module.VLLMLoRAGenerator)
    generator.llm = FakeLLM()
    generator.lora_request = object()
    generator.settings = {
        "temperature": 0.8,
        "top_p": 1.0,
        "max_tokens": 3,
        "seed": 7,
    }
    result = generator(
        ["first", "second"],
        [("task-a", 0, 0), ("task-b", 0, 0)],
    )

    assert result["finish_reasons"] == ["stop", "length"]
    assert result["stop_reasons"] == [13, None]
    stopped = generator_module.TableAgentRolloutCollector._generation_truncation(
        result,
        row_index=0,
        row_count=2,
        completion_tokens=3,
        max_new_tokens=3,
        turn_index=0,
    )
    length = generator_module.TableAgentRolloutCollector._generation_truncation(
        result,
        row_index=1,
        row_count=2,
        completion_tokens=2,
        max_new_tokens=3,
        turn_index=0,
    )
    assert stopped is None
    assert length["detection"] == "explicit_finish_reason"
    assert length["finish_reason"] == "length"


class _Tokenizer:
    def __init__(self):
        self.template_kwargs = []

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs.append(kwargs)
        return "rendered-prompt"

    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[101, 102])

    def decode(self, token_ids, skip_special_tokens=True):
        return "complete-action"


class _Env:
    def __init__(self):
        self.done = False
        self.failure_type = None
        self.correct = False
        self.legal = False
        self.applied = []
        self.closed = False

    def model_messages(self):
        return [{"role": "user", "content": "question"}]

    def apply_model_output(self, text):
        self.applied.append(text)
        self.done = True
        self.correct = True
        self.legal = True

    def record(self):
        return {
            "example_index": 7,
            "protocol_version": "version26",
            "protocol_hash": "4da19387399bd3a5",
            "correct": self.correct,
            "legal": self.legal,
            "failure_type": self.failure_type,
            "turns": [],
        }

    def close(self):
        self.closed = True


def _pool(generator_module, tokenizer: _Tokenizer):
    settings = generator_module.RolloutSettings(
        reward_mode="result-only",
        result_reward_profile="binary",
        max_steps=1,
        max_new_tokens=3,
        max_context_tokens=32,
        enable_thinking=True,
    )
    return generator_module.AsyncVLLMRolloutPool(
        tokenizer=tokenizer,
        settings=settings,
        model_path=Path("/model"),
        adapter_path=Path("/adapter"),
        seed=7,
        gpu_memory_utilization=0.5,
    )


def _metadata() -> dict:
    return {
        "task_id": "bird_train_00007",
        "example_index": 7,
        "db_id": "db",
        "db_path": "/unused.sqlite",
        "question": "question",
        "gold_sql": "select 1",
        "external_knowledge": None,
    }


def test_dynamic_explicit_stop_at_cap_executes_and_remains_eligible(
    generator_module, monkeypatch
) -> None:
    tokenizer = _Tokenizer()
    env = _Env()
    pool = _pool(generator_module, tokenizer)
    pool._generate_turn = AsyncMock(
        return_value=_request_output(
            [11, 12, 13],
            finish_reason="stop",
            stop_reason=13,
        )
    )
    monkeypatch.setattr(generator_module, "create_tool_use_env", lambda *a, **k: env)

    episode = asyncio.run(pool.collect_episode(_metadata(), 0))

    assert env.applied == ["complete-action"]
    assert env.closed
    assert episode.sample.correct
    assert episode.sample.reward == 1.0
    assert episode.sample.process_update
    assert "generation_truncation" not in episode.sample.audit_record
    assert "optimization_exclusion" not in episode.sample.audit_record
    assert episode.sample.audit_record["result_reward"]["profile"] == "binary"
    assert tokenizer.template_kwargs[0]["enable_thinking"] is True


def test_dynamic_explicit_length_is_retained_for_audit_but_excluded(
    generator_module, monkeypatch
) -> None:
    tokenizer = _Tokenizer()
    env = _Env()
    pool = _pool(generator_module, tokenizer)
    pool._generate_turn = AsyncMock(
        return_value=_request_output(
            [21, 22],
            finish_reason="length",
            stop_reason=None,
        )
    )
    monkeypatch.setattr(generator_module, "create_tool_use_env", lambda *a, **k: env)

    episode = asyncio.run(pool.collect_episode(_metadata(), 0))

    assert env.applied == []
    assert env.closed
    assert episode.sample.failure_type == "generation_length"
    assert episode.sample.reward == 0.0
    assert not episode.sample.process_update
    assert len(episode.policy_turns) == 1
    assert episode.sample.audit_record["optimization_exclusion"] == (
        "nonsemantic_runtime_failure"
    )
    truncation = episode.sample.audit_record["generation_truncation"]
    assert truncation["detection"] == "explicit_finish_reason"
    assert truncation["finish_reason"] == "length"
    assert truncation["completion_tokens"] == 2


def test_finalize_manifest_binds_thinking_protocol_and_binary_reward(
    generator_module, monkeypatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "pool"
    groups_dir = output_dir / "groups"
    groups_dir.mkdir(parents=True)
    task_id = "bird_train_00007"
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(
        json.dumps({"example_id": task_id, "example_index": 7}) + "\n",
        encoding="utf-8",
    )
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    rows = [
        {
            "schema_version": "table-agent-fixed-policy-episode-v1",
            "sequence": sample_index,
            "environment": {"task_id": task_id, "example_index": 7},
            "sample": {
                "correct": sample_index < 4,
                "reward": float(sample_index < 4),
                "audit_record": {
                    "result_reward": {
                        "profile": "binary",
                        "correct": sample_index < 4,
                        "value": float(sample_index < 4),
                    }
                },
            },
            "policy_turns": [],
        }
        for sample_index in range(8)
    ]
    (groups_dir / f"{task_id}.json").write_text(
        json.dumps(rows),
        encoding="utf-8",
    )

    runtime_prompt = "frozen-version26-runtime-prompt"
    monkeypatch.setattr(generator_module, "PROTOCOL_VERSION", "version26")
    monkeypatch.setattr(
        generator_module,
        "student_runtime_system_prompt",
        lambda **kwargs: runtime_prompt,
    )
    monkeypatch.setattr(
        generator_module,
        "protocol_hash",
        lambda prompt: "4da19387399bd3a5",
    )
    monkeypatch.setattr(generator_module, "tool_schema_hash", lambda: "tool-sha")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_fixed_rollout_pool.py",
            "--model-path",
            "/model",
            "--adapter-path",
            str(adapter),
            "--tasks",
            str(tasks),
            "--output-dir",
            str(output_dir),
            "--group-size",
            "8",
            "--enable-thinking",
            "--finalize-only",
        ],
    )

    generator_module.main()

    manifest = json.loads((output_dir / "manifest.pending.json").read_text())
    assert manifest["protocol_version"] == "version26"
    assert manifest["protocol_hash"] == "4da19387399bd3a5"
    assert manifest["student_prompt_sha256"] == generator_module.hashlib.sha256(
        runtime_prompt.encode()
    ).hexdigest()
    assert manifest["reward_mode"] == "result-only"
    assert manifest["result_reward_profile"] == "binary"
    assert manifest["enable_thinking"] is True
    assert manifest["group_size"] == 8
    assert manifest["trajectories"] == 8
