#!/usr/bin/env python3
"""Generate verifier-backed Action-DPO pairs by branching exact version26 prefixes.

The source of every anchor is one clean, correct trajectory from an immutable BIRD-train
SFT2 rollout pool.  The source prefix is replayed through the real harness and must reproduce
the recorded model input byte-for-byte.  Candidate next actions are then sampled from that one
prompt, executed in independent harnesses, and continued causally.  Gold SQL is used only inside
the harness terminal scorer and is removed from every emitted diagnostic artifact.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "src" / "rl"),
    str(ROOT / "src" / "rl" / "diagnostics"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from protocol import PROTOCOL_VERSION, student_runtime_system_prompt, tool_schema_hash  # noqa: E402
from rollout_scoring import episode_example  # noqa: E402
from tool_environment import create_tool_use_env  # noqa: E402


SHARED_REASON = "Compare the next legal tool action for the current database state."
NONSEMANTIC_FAILURES = {
    "api_error",
    "context_overflow",
    "generation_oom",
    "provider_error",
    "transport_error",
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.next.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.next.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def action_of(turn: dict[str, Any]) -> dict[str, Any] | None:
    parsed = turn.get("parsed") or {}
    tool = parsed.get("tool")
    arguments = parsed.get("arguments")
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        return None
    return {"tool": tool, "arguments": arguments}


def action_text(action: dict[str, Any]) -> str:
    return (
        f"<think>{SHARED_REASON}</think>\n"
        + json.dumps(action, ensure_ascii=False, separators=(",", ":"))
    )


def anchor_turn_indices(turn_count: int, anchors_per_trajectory: int) -> list[int]:
    """Choose deterministic early/middle/late decisions, including terminal when present."""
    if turn_count < 1:
        return []
    if anchors_per_trajectory < 1:
        raise ValueError("anchors_per_trajectory must be positive")
    if anchors_per_trajectory >= turn_count:
        return list(range(turn_count))
    if anchors_per_trajectory == 1:
        return [(turn_count - 1) // 2]
    indices = {
        round(position * (turn_count - 1) / (anchors_per_trajectory - 1))
        for position in range(anchors_per_trajectory)
    }
    return sorted(indices)


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "\0".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def classify_pair(positive: dict[str, Any], negative: dict[str, Any], turn_index: int) -> str:
    tools = {positive["tool"], negative["tool"]}
    if "answer_from_context" in tools:
        return "final_output_slot"
    if "extreme_value_select" in tools:
        return "ranking_tie_order"
    if tools & {"join_tables", "group_aggregate", "condition_filter", "set_op"}:
        return "population_grain_join"
    if turn_index == 0:
        return "normal_first_step"
    return "semantic_divergence"


def forbidden_visible_field(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"gold_sql", "gold_query", "reference_sql"}:
                return str(key)
            nested = forbidden_visible_field(child)
            if nested:
                return nested
    elif isinstance(value, list):
        for child in value:
            nested = forbidden_visible_field(child)
            if nested:
                return nested
    return None


def sanitized_turn(turn: dict[str, Any], *, anchor_turn_index: int) -> dict[str, Any]:
    result = {
        key: turn[key]
        for key in (
            "turn_index",
            "model_output",
            "parsed",
            "tool_output",
            "execution_error",
            "execution_error_type",
            "feedback_recovery",
            "recovered_from_error_type",
        )
        if key in turn
    }
    model_input = turn.get("model_input")
    if isinstance(model_input, list):
        result["model_input_sha256"] = sha256_json(model_input)
    result["relative_to_anchor"] = int(turn.get("turn_index", 0)) - anchor_turn_index
    return result


def source_rows(
    pool: Path,
    *,
    limit_trajectories: int,
    include_example_indices: set[int] | None = None,
) -> list[dict[str, Any]]:
    """Select one clean correct trajectory per question in stable order."""
    chosen: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(pool):
        environment = row.get("environment") or {}
        sample = row.get("sample") or {}
        audit = sample.get("audit_record") or {}
        if str(environment.get("dataset_split") or "train") != "train":
            continue
        if not sample.get("correct") or not audit.get("correct") or not audit.get("legal"):
            continue
        if int(audit.get("errors") or 0) != 0:
            continue
        if audit.get("protocol_version") != "version26":
            continue
        turns = audit.get("turns") or []
        if not turns or any(not isinstance(turn.get("model_output"), str) for turn in turns):
            continue
        example_index = int(environment["example_index"])
        if (
            include_example_indices is not None
            and example_index not in include_example_indices
        ):
            continue
        previous = chosen.get(example_index)
        sample_index = int(audit.get("sample_index") or 0)
        if previous is None or sample_index < int(
            previous["sample"]["audit_record"].get("sample_index") or 0
        ):
            chosen[example_index] = row
    rows = [chosen[key] for key in sorted(chosen)]
    return rows[:limit_trajectories] if limit_trajectories > 0 else rows


def build_plan(
    rows: list[dict[str, Any]],
    *,
    pool_sha256: str,
    anchors_per_trajectory: int,
    min_turn_index: int = 0,
) -> dict[str, Any]:
    anchors = []
    for row in rows:
        environment = row["environment"]
        audit = row["sample"]["audit_record"]
        sample_index = int(audit.get("sample_index") or 0)
        for turn_index in anchor_turn_indices(len(audit["turns"]), anchors_per_trajectory):
            if turn_index < min_turn_index:
                continue
            anchors.append(
                {
                    "anchor_id": (
                        f"e{int(environment['example_index']):05d}"
                        f"_s{sample_index:02d}_t{turn_index:02d}"
                    ),
                    "example_index": int(environment["example_index"]),
                    "sample_index": sample_index,
                    "turn_index": turn_index,
                }
            )
    return {
        "schema_version": "exact-prefix-branch-plan-v1",
        "split": "bird-train",
        "protocol_version": "version26",
        "source_pool_sha256": pool_sha256,
        "state_normalization": "none",
        "source_trajectories": len(rows),
        "anchors_per_trajectory": anchors_per_trajectory,
        "minimum_anchor_turn_index": min_turn_index,
        "anchors": anchors,
    }


def create_env(metadata: dict[str, Any], *, max_steps: int, history_turns: int):
    return create_tool_use_env(
        episode_example(metadata),
        tool_scheme="atomic",
        example_index=int(metadata["example_index"]),
        max_steps=max_steps,
        context_mode="rolling-legal-history",
        history_turns=history_turns,
        compact_observations=True,
        denotation_comparison="bird-set",
    )


def replay_prefix(
    row: dict[str, Any],
    turn_index: int,
    *,
    max_steps: int,
    history_turns: int,
):
    metadata = row["environment"]
    turns = row["sample"]["audit_record"]["turns"]
    env = create_env(metadata, max_steps=max_steps, history_turns=history_turns)
    try:
        for index in range(turn_index):
            current = env.model_messages()
            if current != turns[index]["model_input"]:
                raise ValueError(
                    f"source prefix drift before turn {index}: "
                    f"{sha256_json(current)} != {sha256_json(turns[index]['model_input'])}"
                )
            step = env.apply_model_output(turns[index]["model_output"])
            if step.done:
                raise ValueError(f"source prefix terminated before anchor turn {turn_index}")
        current = env.model_messages()
        expected = turns[turn_index]["model_input"]
        if current != expected:
            raise ValueError(
                f"anchor model input drift: {sha256_json(current)} != {sha256_json(expected)}"
            )
        return env
    except Exception:
        env.close()
        raise


def verify_source_suffix(
    row: dict[str, Any],
    turn_index: int,
    *,
    max_steps: int,
    history_turns: int,
) -> dict[str, Any]:
    env = replay_prefix(
        row,
        turn_index,
        max_steps=max_steps,
        history_turns=history_turns,
    )
    try:
        turns = row["sample"]["audit_record"]["turns"]
        for source_turn in turns[turn_index:]:
            if env.model_messages() != source_turn["model_input"]:
                raise ValueError("source suffix visible input did not replay exactly")
            env.apply_model_output(source_turn["model_output"])
        if not env.done or not env.correct or not env.legal:
            raise ValueError("source suffix did not replay to a correct legal terminal")
        return {
            "correct": True,
            "legal": True,
            "failure_type": None,
            "steps": env.steps,
            "errors": env.errors,
            "denotation_comparison": "bird-set",
        }
    finally:
        env.close()


class VLLMBranchGenerator:
    def __init__(
        self,
        *,
        model_path: Path,
        adapter_path: Path,
        max_context_tokens: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        gpu_memory_utilization: float,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        self.SamplingParams = SamplingParams
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(
            model=str(model_path),
            dtype="bfloat16",
            max_model_len=max_context_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=True,
            max_lora_rank=16,
            enable_prefix_caching=True,
            enforce_eager=True,
            trust_remote_code=True,
        )
        self.lora_request = LoRARequest("exp15-exact-prefix-branch", 1, str(adapter_path))
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def render(self, messages: list[dict[str, Any]]) -> tuple[str, list[int]]:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = list(self.tokenizer(text, add_special_tokens=False).input_ids)
        if len(ids) + self.max_new_tokens > self.max_context_tokens:
            raise ValueError(
                f"exact prefix exceeds context budget: {len(ids)} + "
                f"{self.max_new_tokens} > {self.max_context_tokens}"
            )
        return text, ids

    def generate(self, prompts: list[str], seeds: list[int]) -> list[dict[str, Any]]:
        if len(prompts) != len(seeds):
            raise ValueError("prompts and seeds must align")
        params = [
            self.SamplingParams(
                n=1,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_new_tokens,
                seed=seed,
            )
            for seed in seeds
        ]
        outputs = self.llm.generate(
            prompts,
            params,
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        result = []
        for output in outputs:
            completion = output.outputs[0]
            token_ids = list(completion.token_ids)
            result.append(
                {
                    "text": self.tokenizer.decode(token_ids, skip_special_tokens=True),
                    "prompt_ids_sha256": sha256_json(list(output.prompt_token_ids)),
                    "response_ids_sha256": sha256_json(token_ids),
                    "response_tokens": len(token_ids),
                    "finish_reason": completion.finish_reason,
                }
            )
        return result


class AsyncVLLMBranchGenerator:
    """Serve many independent anchors through one rolling vLLM request scheduler."""

    def __init__(
        self,
        *,
        model_path: Path,
        adapter_path: Path,
        max_context_tokens: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        gpu_memory_utilization: float,
        max_num_batched_tokens: int,
        max_num_seqs: int,
    ) -> None:
        from transformers import AutoTokenizer
        from vllm import SamplingParams
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.lora.request import LoRARequest
        from vllm.v1.engine.async_llm import AsyncLLM

        self.SamplingParams = SamplingParams
        self.AsyncEngineArgs = AsyncEngineArgs
        self.AsyncLLM = AsyncLLM
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model_path = model_path
        self.max_context_tokens = max_context_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_num_seqs = max_num_seqs
        self.engine = None
        self.lora_request = LoRARequest(
            "exp15-exact-prefix-branch-dynamic",
            1,
            str(adapter_path),
        )

    async def start(self) -> None:
        if self.engine is not None:
            raise RuntimeError("dynamic branch engine is already started")
        engine_args = self.AsyncEngineArgs(
            model=str(self.model_path),
            dtype="bfloat16",
            max_model_len=self.max_context_tokens,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_lora=True,
            max_lora_rank=16,
            enable_prefix_caching=True,
            enforce_eager=True,
            trust_remote_code=True,
            disable_log_stats=True,
        )
        self.engine = self.AsyncLLM.from_engine_args(engine_args)

    def close(self) -> None:
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

    def render(self, messages: list[dict[str, Any]]) -> tuple[str, list[int]]:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = list(self.tokenizer(text, add_special_tokens=False).input_ids)
        if len(ids) + self.max_new_tokens > self.max_context_tokens:
            raise ValueError(
                f"exact prefix exceeds context budget: {len(ids)} + "
                f"{self.max_new_tokens} > {self.max_context_tokens}"
            )
        return text, ids

    async def _generate_one(
        self,
        prompt: str,
        seed: int,
        request_id: str,
    ) -> dict[str, Any]:
        if self.engine is None:
            raise RuntimeError("dynamic branch engine is not started")
        params = self.SamplingParams(
            n=1,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
            seed=seed,
        )
        final_output = None
        async for output in self.engine.generate(
            prompt,
            params,
            request_id,
            lora_request=self.lora_request,
        ):
            final_output = output
        if final_output is None or not final_output.finished:
            raise RuntimeError(f"vLLM request did not finish: {request_id}")
        completion = final_output.outputs[0]
        token_ids = list(completion.token_ids)
        return {
            "text": self.tokenizer.decode(token_ids, skip_special_tokens=True),
            "prompt_ids_sha256": sha256_json(list(final_output.prompt_token_ids)),
            "response_ids_sha256": sha256_json(token_ids),
            "response_tokens": len(token_ids),
            "finish_reason": completion.finish_reason,
        }

    async def generate(
        self,
        prompts: list[str],
        seeds: list[int],
        request_ids: list[str],
    ) -> list[dict[str, Any]]:
        if not (len(prompts) == len(seeds) == len(request_ids)):
            raise ValueError("dynamic prompts, seeds, and request ids must align")
        return list(
            await asyncio.gather(
                *(
                    self._generate_one(prompt, seed, request_id)
                    for prompt, seed, request_id in zip(
                        prompts,
                        seeds,
                        request_ids,
                        strict=True,
                    )
                )
            )
        )


def inspect_candidate(
    row: dict[str, Any],
    turn_index: int,
    model_output: str,
    *,
    max_steps: int,
    history_turns: int,
) -> dict[str, Any]:
    env = replay_prefix(
        row,
        turn_index,
        max_steps=max_steps,
        history_turns=history_turns,
    )
    try:
        step = env.apply_model_output(model_output)
        return {
            "action": action_of(step.turn),
            "root_done": step.done,
            "root_correct": step.correct,
            "root_legal": step.legal,
            "root_failure_type": step.failure_type,
            "root_error_type": step.turn.get("execution_error_type"),
        }
    finally:
        env.close()


def continue_branches(
    generator: VLLMBranchGenerator,
    row: dict[str, Any],
    turn_index: int,
    candidates: list[dict[str, Any]],
    *,
    continuation_count: int,
    base_seed: int,
    anchor_id: str,
    max_steps: int,
    history_turns: int,
) -> list[dict[str, Any]]:
    branches = []
    for candidate_index, candidate in enumerate(candidates):
        for trial_index in range(continuation_count):
            env = replay_prefix(
                row,
                turn_index,
                max_steps=max_steps,
                history_turns=history_turns,
            )
            env.apply_model_output(candidate["model_output"])
            branches.append(
                {
                    "candidate_index": candidate_index,
                    "trial_index": trial_index,
                    "env": env,
                    "generated_turns": 0,
                    "tokenization_warning": False,
                }
            )
    try:
        while any(not branch["env"].done for branch in branches):
            active = []
            prompts = []
            prompt_id_rows = []
            seeds = []
            for branch in branches:
                if branch["env"].done:
                    continue
                try:
                    prompt, prompt_ids = generator.render(branch["env"].model_messages())
                except ValueError as exc:
                    if "exceeds context budget" not in str(exc):
                        raise
                    branch["env"].done = True
                    branch["env"].failure_type = "context_overflow"
                    branch["context_overflow"] = str(exc)
                    continue
                active.append(branch)
                prompts.append(prompt)
                prompt_id_rows.append(prompt_ids)
                seeds.append(
                    stable_seed(
                        base_seed,
                        anchor_id,
                        branch["candidate_index"],
                        branch["trial_index"],
                        branch["generated_turns"],
                    )
                )
            if not active:
                break
            outputs = generator.generate(prompts, seeds)
            for branch, prompt_ids, output in zip(active, prompt_id_rows, outputs, strict=True):
                if sha256_json(prompt_ids) != output["prompt_ids_sha256"]:
                    branch["tokenization_warning"] = True
                branch["env"].apply_model_output(output["text"])
                branch["generated_turns"] += 1

        results = []
        for branch in branches:
            env = branch["env"]
            record = env.record()
            results.append(
                {
                    "candidate_index": branch["candidate_index"],
                    "trial_index": branch["trial_index"],
                    "correct": bool(record["correct"]),
                    "legal": bool(record["legal"]),
                    "failure_type": record.get("failure_type"),
                    "steps": int(record["steps"]),
                    "errors": int(record["errors"]),
                    "generated_continuation_turns": branch["generated_turns"],
                    "tokenization_warning": branch["tokenization_warning"],
                    "context_overflow": branch.get("context_overflow"),
                    "turns_from_anchor": [
                        sanitized_turn(turn, anchor_turn_index=turn_index)
                        for turn in record["turns"][turn_index:]
                    ],
                }
            )
        return results
    finally:
        for branch in branches:
            branch["env"].close()


async def continue_branches_async(
    generator: AsyncVLLMBranchGenerator,
    row: dict[str, Any],
    turn_index: int,
    candidates: list[dict[str, Any]],
    *,
    continuation_count: int,
    base_seed: int,
    anchor_id: str,
    max_steps: int,
    history_turns: int,
) -> list[dict[str, Any]]:
    branches = []
    for candidate_index, candidate in enumerate(candidates):
        for trial_index in range(continuation_count):
            env = replay_prefix(
                row,
                turn_index,
                max_steps=max_steps,
                history_turns=history_turns,
            )
            env.apply_model_output(candidate["model_output"])
            branches.append(
                {
                    "candidate_index": candidate_index,
                    "trial_index": trial_index,
                    "env": env,
                    "generated_turns": 0,
                    "tokenization_warning": False,
                }
            )
    try:
        while any(not branch["env"].done for branch in branches):
            active = []
            prompts = []
            prompt_id_rows = []
            seeds = []
            request_ids = []
            for branch in branches:
                if branch["env"].done:
                    continue
                try:
                    prompt, prompt_ids = generator.render(branch["env"].model_messages())
                except ValueError as exc:
                    if "exceeds context budget" not in str(exc):
                        raise
                    branch["env"].done = True
                    branch["env"].failure_type = "context_overflow"
                    branch["context_overflow"] = str(exc)
                    continue
                active.append(branch)
                prompts.append(prompt)
                prompt_id_rows.append(prompt_ids)
                seeds.append(
                    stable_seed(
                        base_seed,
                        anchor_id,
                        branch["candidate_index"],
                        branch["trial_index"],
                        branch["generated_turns"],
                    )
                )
                request_ids.append(
                    "exp15-cont:"
                    f"{anchor_id}:{branch['candidate_index']}:"
                    f"{branch['trial_index']}:{branch['generated_turns']}"
                )
            if not active:
                break
            outputs = await generator.generate(prompts, seeds, request_ids)
            for branch, prompt_ids, output in zip(
                active,
                prompt_id_rows,
                outputs,
                strict=True,
            ):
                if sha256_json(prompt_ids) != output["prompt_ids_sha256"]:
                    branch["tokenization_warning"] = True
                branch["env"].apply_model_output(output["text"])
                branch["generated_turns"] += 1

        results = []
        for branch in branches:
            env = branch["env"]
            record = env.record()
            results.append(
                {
                    "candidate_index": branch["candidate_index"],
                    "trial_index": branch["trial_index"],
                    "correct": bool(record["correct"]),
                    "legal": bool(record["legal"]),
                    "failure_type": record.get("failure_type"),
                    "steps": int(record["steps"]),
                    "errors": int(record["errors"]),
                    "generated_continuation_turns": branch["generated_turns"],
                    "tokenization_warning": branch["tokenization_warning"],
                    "context_overflow": branch.get("context_overflow"),
                    "turns_from_anchor": [
                        sanitized_turn(turn, anchor_turn_index=turn_index)
                        for turn in record["turns"][turn_index:]
                    ],
                }
            )
        return results
    finally:
        for branch in branches:
            branch["env"].close()


def build_pairs(
    *,
    row: dict[str, Any],
    anchor_id: str,
    turn_index: int,
    state: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    branches: list[dict[str, Any]],
    source_replay: dict[str, Any],
    max_pairs_per_anchor: int,
) -> list[dict[str, Any]]:
    source_action = candidates[0]["action"]
    state_sha = sha256_json(state)
    by_candidate: dict[int, list[dict[str, Any]]] = {}
    for branch in branches:
        by_candidate.setdefault(int(branch["candidate_index"]), []).append(branch)
    negatives = []
    for candidate_index, candidate in enumerate(candidates[1:], start=1):
        trials = by_candidate.get(candidate_index, [])
        if not trials or candidate.get("action") is None:
            continue
        if any(trial.get("failure_type") in NONSEMANTIC_FAILURES for trial in trials):
            continue
        if any(trial["correct"] for trial in trials):
            continue
        if sha256_json(candidate["action"]) == sha256_json(source_action):
            continue
        negatives.append((candidate_index, candidate, trials))

    metadata = row["environment"]
    audit = row["sample"]["audit_record"]
    source = {
        "artifact": "immutable_frozen_source_pool",
        "example_index": int(metadata["example_index"]),
        "sample_index": int(audit.get("sample_index") or 0),
        "turn_index": turn_index,
        "trajectory_correct": True,
        "trajectory_legal": True,
        "feedback_recovery": False,
        "explicit_error": None,
    }
    pairs = []
    for candidate_index, candidate, trials in negatives[:max_pairs_per_anchor]:
        pair = {
            "schema_version": "fixed-prefix-action-pair-v1",
            "question_id": str(metadata["example_index"]),
            "example_index": int(metadata["example_index"]),
            "db_id": metadata["db_id"],
            "question": metadata["question"],
            "state": state,
            "state_sha256": state_sha,
            "state_normalization": "none",
            "positive_action": source_action,
            "negative_action": candidate["action"],
            "positive_scoring_response": action_text(source_action),
            "negative_scoring_response": action_text(candidate["action"]),
            "shared_reason": SHARED_REASON,
            "category": classify_pair(source_action, candidate["action"], turn_index),
            "positive_verified_by": {
                "real_harness_execution": True,
                "successful_suffix": True,
                "source_trajectory_replayed": True,
                "source_replay": source_replay,
                "source": source,
                "exact_prefix_replay": True,
                "branching_online_trials": len(by_candidate.get(0, [])),
                "branching_online_successes": sum(
                    bool(trial["correct"]) for trial in by_candidate.get(0, [])
                ),
            },
            "negative_observed_from": {
                "method": "same_exact_prefix_active_branch",
                "anchor_id": anchor_id,
                "candidate_index": candidate_index,
                "online_trials": len(trials),
                "online_successes": 0,
                "failure_types": dict(Counter(str(trial.get("failure_type")) for trial in trials)),
                "real_harness_execution": True,
            },
            "generation_method": "exact-prefix-active-branch-v1",
        }
        pair["pair_sha256"] = sha256_json(
            {
                "state": state_sha,
                "positive": source_action,
                "negative": candidate["action"],
            }
        )
        pairs.append(pair)
    return pairs


def process_anchor(
    generator: VLLMBranchGenerator,
    row: dict[str, Any],
    anchor: dict[str, Any],
    *,
    candidate_count: int,
    candidate_draws: int,
    continuation_count: int,
    max_pairs_per_anchor: int,
    base_seed: int,
    max_steps: int,
    history_turns: int,
) -> dict[str, Any]:
    turn_index = int(anchor["turn_index"])
    anchor_id = str(anchor["anchor_id"])
    source_turn = row["sample"]["audit_record"]["turns"][turn_index]
    state = source_turn["model_input"]
    forbidden = forbidden_visible_field(state)
    if forbidden:
        raise ValueError(f"model-visible anchor contains forbidden field: {forbidden}")
    source_action = action_of(source_turn)
    if source_action is None:
        raise ValueError("source anchor has no structured action")
    source_replay = verify_source_suffix(
        row,
        turn_index,
        max_steps=max_steps,
        history_turns=history_turns,
    )
    replayed = replay_prefix(
        row,
        turn_index,
        max_steps=max_steps,
        history_turns=history_turns,
    )
    try:
        prompt, prompt_ids = generator.render(replayed.model_messages())
    finally:
        replayed.close()

    draws = generator.generate(
        [prompt] * candidate_draws,
        [stable_seed(base_seed, anchor_id, "candidate", draw) for draw in range(candidate_draws)],
    )
    candidates = [
        {
            "candidate_index": 0,
            "origin": "recorded_correct_source_action",
            "model_output": source_turn["model_output"],
            "action": source_action,
            "action_sha256": sha256_json(source_action),
            "prompt_ids_sha256": sha256_json(prompt_ids),
        }
    ]
    seen = {sha256_json(source_action)}
    rejected_draws = []
    for draw_index, draw in enumerate(draws):
        inspection = inspect_candidate(
            row,
            turn_index,
            draw["text"],
            max_steps=max_steps,
            history_turns=history_turns,
        )
        action = inspection["action"]
        if action is None:
            rejected_draws.append(
                {
                    "draw_index": draw_index,
                    "reason": "unparsed_action",
                    "root_failure_type": inspection["root_failure_type"],
                    "response_ids_sha256": draw["response_ids_sha256"],
                }
            )
            continue
        action_sha = sha256_json(action)
        if action_sha in seen:
            rejected_draws.append(
                {
                    "draw_index": draw_index,
                    "reason": "duplicate_structured_action",
                    "action_sha256": action_sha,
                }
            )
            continue
        seen.add(action_sha)
        candidates.append(
            {
                "candidate_index": len(candidates),
                "origin": "sft2_same_prefix_sample",
                "draw_index": draw_index,
                "model_output": draw["text"],
                "action": action,
                "action_sha256": action_sha,
                "prompt_ids_sha256": draw["prompt_ids_sha256"],
                "response_ids_sha256": draw["response_ids_sha256"],
                "response_tokens": draw["response_tokens"],
                "finish_reason": draw["finish_reason"],
                **{key: value for key, value in inspection.items() if key != "action"},
            }
        )
        if len(candidates) >= candidate_count:
            break

    branches = continue_branches(
        generator,
        row,
        turn_index,
        candidates,
        continuation_count=continuation_count,
        base_seed=base_seed,
        anchor_id=anchor_id,
        max_steps=max_steps,
        history_turns=history_turns,
    )
    pairs = build_pairs(
        row=row,
        anchor_id=anchor_id,
        turn_index=turn_index,
        state=state,
        candidates=candidates,
        branches=branches,
        source_replay=source_replay,
        max_pairs_per_anchor=max_pairs_per_anchor,
    )
    return {
        "schema_version": "exact-prefix-branch-anchor-v1",
        "training_admission": "diagnostic_only_exp15_branching_pending_scale_gate",
        "sft_export_eligible": False,
        "anchor": anchor,
        "db_id": row["environment"]["db_id"],
        "question": row["environment"]["question"],
        "state": state,
        "state_sha256": sha256_json(state),
        "state_normalization": "none",
        "source_suffix_replay": source_replay,
        "candidate_draws_requested": candidate_draws,
        "candidates": candidates,
        "rejected_draws": rejected_draws,
        "continuation_count": continuation_count,
        "branches": branches,
        "pairs": pairs,
    }


async def process_anchor_async(
    generator: AsyncVLLMBranchGenerator,
    row: dict[str, Any],
    anchor: dict[str, Any],
    *,
    candidate_count: int,
    candidate_draws: int,
    continuation_count: int,
    max_pairs_per_anchor: int,
    base_seed: int,
    max_steps: int,
    history_turns: int,
) -> dict[str, Any]:
    turn_index = int(anchor["turn_index"])
    anchor_id = str(anchor["anchor_id"])
    source_turn = row["sample"]["audit_record"]["turns"][turn_index]
    state = source_turn["model_input"]
    forbidden = forbidden_visible_field(state)
    if forbidden:
        raise ValueError(f"model-visible anchor contains forbidden field: {forbidden}")
    source_action = action_of(source_turn)
    if source_action is None:
        raise ValueError("source anchor has no structured action")
    source_replay = verify_source_suffix(
        row,
        turn_index,
        max_steps=max_steps,
        history_turns=history_turns,
    )
    replayed = replay_prefix(
        row,
        turn_index,
        max_steps=max_steps,
        history_turns=history_turns,
    )
    try:
        prompt, prompt_ids = generator.render(replayed.model_messages())
    finally:
        replayed.close()

    draws = await generator.generate(
        [prompt] * candidate_draws,
        [
            stable_seed(base_seed, anchor_id, "candidate", draw)
            for draw in range(candidate_draws)
        ],
        [f"exp15-candidate:{anchor_id}:{draw}" for draw in range(candidate_draws)],
    )
    candidates = [
        {
            "candidate_index": 0,
            "origin": "recorded_correct_source_action",
            "model_output": source_turn["model_output"],
            "action": source_action,
            "action_sha256": sha256_json(source_action),
            "prompt_ids_sha256": sha256_json(prompt_ids),
        }
    ]
    seen = {sha256_json(source_action)}
    rejected_draws = []
    for draw_index, draw in enumerate(draws):
        inspection = inspect_candidate(
            row,
            turn_index,
            draw["text"],
            max_steps=max_steps,
            history_turns=history_turns,
        )
        action = inspection["action"]
        if action is None:
            rejected_draws.append(
                {
                    "draw_index": draw_index,
                    "reason": "unparsed_action",
                    "root_failure_type": inspection["root_failure_type"],
                    "response_ids_sha256": draw["response_ids_sha256"],
                }
            )
            continue
        action_sha = sha256_json(action)
        if action_sha in seen:
            rejected_draws.append(
                {
                    "draw_index": draw_index,
                    "reason": "duplicate_structured_action",
                    "action_sha256": action_sha,
                }
            )
            continue
        seen.add(action_sha)
        candidates.append(
            {
                "candidate_index": len(candidates),
                "origin": "sft2_same_prefix_sample",
                "draw_index": draw_index,
                "model_output": draw["text"],
                "action": action,
                "action_sha256": action_sha,
                "prompt_ids_sha256": draw["prompt_ids_sha256"],
                "response_ids_sha256": draw["response_ids_sha256"],
                "response_tokens": draw["response_tokens"],
                "finish_reason": draw["finish_reason"],
                **{key: value for key, value in inspection.items() if key != "action"},
            }
        )
        if len(candidates) >= candidate_count:
            break

    branches = await continue_branches_async(
        generator,
        row,
        turn_index,
        candidates,
        continuation_count=continuation_count,
        base_seed=base_seed,
        anchor_id=anchor_id,
        max_steps=max_steps,
        history_turns=history_turns,
    )
    pairs = build_pairs(
        row=row,
        anchor_id=anchor_id,
        turn_index=turn_index,
        state=state,
        candidates=candidates,
        branches=branches,
        source_replay=source_replay,
        max_pairs_per_anchor=max_pairs_per_anchor,
    )
    return {
        "schema_version": "exact-prefix-branch-anchor-v1",
        "training_admission": "diagnostic_only_exp15_branching_pending_scale_gate",
        "sft_export_eligible": False,
        "anchor": anchor,
        "db_id": row["environment"]["db_id"],
        "question": row["environment"]["question"],
        "state": state,
        "state_sha256": sha256_json(state),
        "state_normalization": "none",
        "source_suffix_replay": source_replay,
        "candidate_draws_requested": candidate_draws,
        "candidates": candidates,
        "rejected_draws": rejected_draws,
        "continuation_count": continuation_count,
        "branches": branches,
        "pairs": pairs,
    }


def validate_source_manifest(manifest_path: Path, pool_path: Path) -> tuple[dict[str, Any], str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") not in {"frozen_passed", "frozen_rank_ready"}:
        raise ValueError("source pool is not immutable and verifier-passed/rank-ready")
    if manifest.get("protocol_version") != "version26":
        raise ValueError("source pool is not version26")
    expected = (
        (manifest.get("files") or {}).get("validated_trajectories") or {}
    ).get("sha256")
    actual = sha256_file(pool_path)
    if expected != actual:
        raise ValueError(f"source pool hash mismatch: {actual} != {expected}")
    return manifest, actual


def finalize(
    output_dir: Path,
    plan: dict[str, Any],
    *,
    source_manifest: Path,
    source_pool: Path,
    model_path: Path,
    adapter_path: Path,
    settings: dict[str, Any],
    allow_partial: bool = False,
) -> dict[str, Any]:
    anchors = []
    for item in plan["anchors"]:
        path = output_dir / "anchors" / f"{item['anchor_id']}.json"
        if not path.is_file():
            if allow_partial:
                continue
            raise ValueError(f"cannot finalize; missing anchor artifact: {path}")
        anchors.append(json.loads(path.read_text(encoding="utf-8")))
    if not anchors:
        raise ValueError("cannot finalize; no completed anchor artifacts")
    pairs = []
    seen = set()
    for anchor in anchors:
        for pair in anchor["pairs"]:
            if pair["pair_sha256"] in seen:
                continue
            seen.add(pair["pair_sha256"])
            pairs.append(pair)
    pairs.sort(key=lambda row: (int(row["example_index"]), row["state_sha256"], row["pair_sha256"]))
    pairs_path = output_dir / "exact_prefix_pairs.jsonl"
    online_supported_pairs = [
        pair
        for pair in pairs
        if int(pair["positive_verified_by"]["branching_online_successes"]) >= 1
    ]
    online_consistent_pairs = [
        pair
        for pair in pairs
        if int(pair["positive_verified_by"]["branching_online_successes"])
        == int(pair["positive_verified_by"]["branching_online_trials"])
    ]
    online_supported_path = output_dir / "exact_prefix_pairs.online_supported.jsonl"
    online_consistent_path = output_dir / "exact_prefix_pairs.online_consistent.jsonl"
    anchors_path = output_dir / "anchor_summaries.jsonl"
    write_jsonl_atomic(pairs_path, pairs)
    write_jsonl_atomic(online_supported_path, online_supported_pairs)
    write_jsonl_atomic(online_consistent_path, online_consistent_pairs)
    write_jsonl_atomic(
        anchors_path,
        (
            {
                "anchor": anchor["anchor"],
                "state_sha256": anchor["state_sha256"],
                "candidates": len(anchor["candidates"]),
                "branches": len(anchor["branches"]),
                "correct_branches": sum(branch["correct"] for branch in anchor["branches"]),
                "pairs": len(anchor["pairs"]),
            }
            for anchor in anchors
        ),
    )
    continuation_count = int(settings["continuation_count"])
    manifest = {
        "schema_version": "exact-prefix-active-branch-dataset-v1",
        "status": (
            "generated_verified_diagnostic"
            if len(anchors) == len(plan["anchors"])
            else "partial_time_budget_verified_diagnostic"
        ),
        "training_admission": "diagnostic_only_exp15_branching_pending_scale_gate",
        "sft_export_eligible": False,
        "split": "bird-train",
        "tool_scheme": "atomic",
        "assistant_carrier": "think-json-v1",
        "protocol_version": "version26",
        "denotation_comparison": "bird-set",
        "state_normalization": "none",
        "same_visible_prefix_required": True,
        "prefix_source": "clean_correct_sft2_frozen_trajectory",
        "generation_method": "exact-prefix-active-branch-v1",
        "source_pool": str(source_pool),
        "source_pool_sha256": plan["source_pool_sha256"],
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256_file(source_manifest),
        "model_path": str(model_path),
        "adapter_path": str(adapter_path),
        "adapter_sha256": sha256_file(adapter_path / "adapter_model.safetensors"),
        "student_prompt_sha256": hashlib.sha256(
            student_runtime_system_prompt(
                context_mode="rolling-legal-history",
                compact=False,
            ).encode("utf-8")
        ).hexdigest(),
        "tool_schema_sha256": tool_schema_hash(),
        "source_trajectories": plan["source_trajectories"],
        "planned_anchors": len(plan["anchors"]),
        "attempted_anchors": len(anchors),
        "retained_pair_anchors": sum(bool(anchor["pairs"]) for anchor in anchors),
        "sampled_candidates": sum(len(anchor["candidates"]) - 1 for anchor in anchors),
        "executed_branches": sum(len(anchor["branches"]) for anchor in anchors),
        "correct_branches": sum(
            branch["correct"] for anchor in anchors for branch in anchor["branches"]
        ),
        "retained_pairs": len(pairs),
        "retained_questions": len({pair["question_id"] for pair in pairs}),
        "pair_views": {
            "exp15_faithful": {
                "definition": (
                    "recorded correct source suffix; negative succeeds "
                    f"0/{continuation_count} online"
                ),
                "pairs": len(pairs),
                "questions": len({pair["question_id"] for pair in pairs}),
            },
            "online_supported": {
                "definition": (
                    "exp15_faithful plus positive succeeds at least "
                    f"1/{continuation_count} online"
                ),
                "pairs": len(online_supported_pairs),
                "questions": len(
                    {pair["question_id"] for pair in online_supported_pairs}
                ),
            },
            "online_consistent": {
                "definition": (
                    "exp15_faithful plus positive succeeds "
                    f"{continuation_count}/{continuation_count} online"
                ),
                "pairs": len(online_consistent_pairs),
                "questions": len(
                    {pair["question_id"] for pair in online_consistent_pairs}
                ),
            },
        },
        "settings": settings,
        "files": {
            "plan": {"path": str(output_dir / "plan.json"), "sha256": sha256_file(output_dir / "plan.json")},
            "anchor_summaries": {"path": str(anchors_path), "sha256": sha256_file(anchors_path)},
            "pairs": {"path": str(pairs_path), "sha256": sha256_file(pairs_path)},
            "pairs_online_supported": {
                "path": str(online_supported_path),
                "sha256": sha256_file(online_supported_path),
            },
            "pairs_online_consistent": {
                "path": str(online_consistent_path),
                "sha256": sha256_file(online_consistent_path),
            },
        },
        "gold_sql_model_visible": False,
        "model_parameter_updates_during_generation": False,
    }
    if plan.get("question_filter"):
        manifest["question_filter"] = plan["question_filter"]
    write_json_atomic(output_dir / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-pool", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument(
        "--question-ids-dataset",
        type=Path,
        help=(
            "optional audited JSONL whose question_id/example_index values restrict the "
            "clean source trajectories used to build the immutable anchor plan"
        ),
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit-trajectories", type=int, default=8)
    parser.add_argument("--anchors-per-trajectory", type=int, default=3)
    parser.add_argument("--min-anchor-turn-index", type=int, default=0)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--candidate-draws", type=int, default=8)
    parser.add_argument("--continuation-count", type=int, default=2)
    parser.add_argument("--max-pairs-per-anchor", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1515)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--scheduler", choices=("static", "dynamic"), default="static")
    parser.add_argument(
        "--anchor-window",
        type=int,
        default=8,
        help="number of independent exact-prefix anchors kept active by dynamic scheduling",
    )
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--no-finalize", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--finalize-partial-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--replay-validate-only",
        action="store_true",
        help="replay every selected source prefix/suffix without loading the policy",
    )
    args = parser.parse_args()

    if PROTOCOL_VERSION != "version26":
        raise SystemExit(f"exact-prefix Exp15 branching requires version26, got {PROTOCOL_VERSION}")
    if not 0 <= args.shard_index < args.shard_count:
        raise SystemExit("shard-index must be in [0, shard-count)")
    if args.candidate_count < 2 or args.candidate_draws < args.candidate_count - 1:
        raise SystemExit("candidate settings cannot produce a positive/negative pair")
    if args.continuation_count < 1 or args.max_pairs_per_anchor < 1:
        raise SystemExit("continuation and pair counts must be positive")
    if args.min_anchor_turn_index < 0:
        raise SystemExit("min-anchor-turn-index must be non-negative")
    if args.anchor_window < 1 or args.max_num_batched_tokens < 1 or args.max_num_seqs < 1:
        raise SystemExit("dynamic scheduler bounds must be positive")
    exclusive_modes = sum(
        bool(value)
        for value in (
            args.plan_only,
            args.replay_validate_only,
            args.finalize_only,
            args.finalize_partial_only,
        )
    )
    if exclusive_modes > 1:
        raise SystemExit("plan/replay/finalize modes are mutually exclusive")

    source_manifest, pool_sha = validate_source_manifest(args.source_manifest, args.source_pool)
    include_example_indices = None
    question_filter = None
    if args.question_ids_dataset:
        filter_rows = list(load_jsonl(args.question_ids_dataset))
        include_example_indices = {
            int(row.get("example_index", row.get("question_id"))) for row in filter_rows
        }
        if not include_example_indices:
            raise SystemExit("question-ids-dataset contains no question ids")
        question_filter = {
            "dataset": str(args.question_ids_dataset),
            "dataset_sha256": sha256_file(args.question_ids_dataset),
            "questions": len(include_example_indices),
        }
    rows = source_rows(
        args.source_pool,
        limit_trajectories=args.limit_trajectories,
        include_example_indices=include_example_indices,
    )
    if not rows:
        raise SystemExit("source pool has no clean correct version26 BIRD-train trajectories")
    row_lookup = {
        (
            int(row["environment"]["example_index"]),
            int(row["sample"]["audit_record"].get("sample_index") or 0),
        ): row
        for row in rows
    }
    plan = build_plan(
        rows,
        pool_sha256=pool_sha,
        anchors_per_trajectory=args.anchors_per_trajectory,
        min_turn_index=args.min_anchor_turn_index,
    )
    if question_filter:
        plan["question_filter"] = question_filter
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = args.output_dir / "plan.json"
    if plan_path.is_file():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != plan:
            raise SystemExit("existing plan differs from requested immutable anchor plan")
    else:
        write_json_atomic(plan_path, plan)
    if args.plan_only:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return

    if args.replay_validate_only:
        validations = []
        for anchor in plan["anchors"]:
            row = row_lookup[(int(anchor["example_index"]), int(anchor["sample_index"]))]
            replay = verify_source_suffix(
                row,
                int(anchor["turn_index"]),
                max_steps=args.max_steps,
                history_turns=args.history_turns,
            )
            validations.append(
                {
                    "anchor_id": anchor["anchor_id"],
                    "state_sha256": sha256_json(
                        row["sample"]["audit_record"]["turns"][int(anchor["turn_index"])][
                            "model_input"
                        ]
                    ),
                    "source_suffix_replay": replay,
                }
            )
        audit = {
            "schema_version": "exact-prefix-source-replay-audit-v1",
            "status": "passed",
            "protocol_version": "version26",
            "denotation_comparison": "bird-set",
            "state_normalization": "none",
            "anchors": len(validations),
            "correct_legal_suffixes": sum(
                item["source_suffix_replay"]["correct"]
                and item["source_suffix_replay"]["legal"]
                for item in validations
            ),
            "validations": validations,
        }
        write_json_atomic(args.output_dir / "source_replay_audit.json", audit)
        print(json.dumps(audit, ensure_ascii=False, indent=2))
        return

    settings = {
        "anchors_per_trajectory": args.anchors_per_trajectory,
        "min_anchor_turn_index": args.min_anchor_turn_index,
        "candidate_count": args.candidate_count,
        "candidate_draws": args.candidate_draws,
        "continuation_count": args.continuation_count,
        "max_pairs_per_anchor": args.max_pairs_per_anchor,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_steps": args.max_steps,
        "max_new_tokens": args.max_new_tokens,
        "max_context_tokens": args.max_context_tokens,
        "history_turns": args.history_turns,
        "seed": args.seed,
        "scheduler": args.scheduler,
        "anchor_window": args.anchor_window if args.scheduler == "dynamic" else None,
        "max_num_batched_tokens": (
            args.max_num_batched_tokens if args.scheduler == "dynamic" else None
        ),
        "max_num_seqs": args.max_num_seqs if args.scheduler == "dynamic" else None,
    }
    if args.finalize_only or args.finalize_partial_only:
        manifest = finalize(
            args.output_dir,
            plan,
            source_manifest=args.source_manifest,
            source_pool=args.source_pool,
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            settings=settings,
            allow_partial=args.finalize_partial_only,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    anchors_dir = args.output_dir / "anchors"
    anchors_dir.mkdir(exist_ok=True)
    selected = [
        anchor
        for index, anchor in enumerate(plan["anchors"])
        if index % args.shard_count == args.shard_index
    ]
    pending = []
    for position, anchor in enumerate(selected, start=1):
        output_path = anchors_dir / f"{anchor['anchor_id']}.json"
        if output_path.is_file():
            print(json.dumps({"anchor": anchor["anchor_id"], "status": "existing"}), flush=True)
            continue
        pending.append((position, anchor, output_path))

    def report_result(position: int, anchor: dict[str, Any], result: dict[str, Any]) -> None:
        print(
            json.dumps(
                {
                    "anchor": anchor["anchor_id"],
                    "position": position,
                    "assigned": len(selected),
                    "candidates": len(result["candidates"]),
                    "branches": len(result["branches"]),
                    "correct_branches": sum(
                        branch["correct"] for branch in result["branches"]
                    ),
                    "pairs": len(result["pairs"]),
                    "status": "generated",
                    "scheduler": args.scheduler,
                }
            ),
            flush=True,
        )

    if pending and args.scheduler == "dynamic":
        async def run_dynamic() -> None:
            generator = AsyncVLLMBranchGenerator(
                model_path=args.model_path,
                adapter_path=args.adapter_path,
                max_context_tokens=args.max_context_tokens,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_num_batched_tokens=args.max_num_batched_tokens,
                max_num_seqs=args.max_num_seqs,
            )
            await generator.start()
            semaphore = asyncio.Semaphore(args.anchor_window)

            async def run_one(
                position: int,
                anchor: dict[str, Any],
                output_path: Path,
            ) -> None:
                async with semaphore:
                    row = row_lookup[
                        (int(anchor["example_index"]), int(anchor["sample_index"]))
                    ]
                    result = await process_anchor_async(
                        generator,
                        row,
                        anchor,
                        candidate_count=args.candidate_count,
                        candidate_draws=args.candidate_draws,
                        continuation_count=args.continuation_count,
                        max_pairs_per_anchor=args.max_pairs_per_anchor,
                        base_seed=args.seed,
                        max_steps=args.max_steps,
                        history_turns=args.history_turns,
                    )
                    write_json_atomic(output_path, result)
                    report_result(position, anchor, result)

            try:
                await asyncio.gather(
                    *(run_one(position, anchor, output_path) for position, anchor, output_path in pending)
                )
            finally:
                generator.close()

        asyncio.run(run_dynamic())
    elif pending:
        generator = VLLMBranchGenerator(
            model_path=args.model_path,
            adapter_path=args.adapter_path,
            max_context_tokens=args.max_context_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        for position, anchor, output_path in pending:
            row = row_lookup[(int(anchor["example_index"]), int(anchor["sample_index"]))]
            result = process_anchor(
                generator,
                row,
                anchor,
                candidate_count=args.candidate_count,
                candidate_draws=args.candidate_draws,
                continuation_count=args.continuation_count,
                max_pairs_per_anchor=args.max_pairs_per_anchor,
                base_seed=args.seed,
                max_steps=args.max_steps,
                history_turns=args.history_turns,
            )
            write_json_atomic(output_path, result)
            report_result(position, anchor, result)

    if args.no_finalize:
        print(
            json.dumps(
                {
                    "status": "worker_complete",
                    "shard_index": args.shard_index,
                    "shard_count": args.shard_count,
                    "assigned_anchors": len(selected),
                }
            ),
            flush=True,
        )
        return
    manifest = finalize(
        args.output_dir,
        plan,
        source_manifest=args.source_manifest,
        source_pool=args.source_pool,
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        settings=settings,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
