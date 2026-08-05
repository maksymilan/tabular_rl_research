#!/usr/bin/env python3
"""Batched vLLM rollouts that preserve one exact prefix per assistant transition."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from process_credit import ProcessRewardConfig
from rollout_scoring import episode_example, score_completed_rollout
from tool_environment import create_tool_use_env

from frameworks.trl.transition_batch import PolicyEpisode, PolicyTurn


GenerateBatch = Callable[[list[str]], dict[str, Any]]
GenerateBatchWithKeys = Callable[
    [list[str], list[tuple[str, int, int]]],
    dict[str, Any],
]


@dataclass(frozen=True)
class RolloutSettings:
    reward_mode: str = "result-only"
    process_admission_policy: str = "counterfactual-completeness"
    tool_scheme: str = "atomic"
    max_steps: int = 30
    max_batch_calls: int = 5
    max_new_tokens: int = 1024
    max_context_tokens: int = 8192
    context_mode: str = "rolling-legal-history"
    history_turns: int = 4
    denotation_comparison: str = "bird-set"
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0

    def validate(self) -> None:
        if self.reward_mode not in {"result-only", "process"}:
            raise ValueError(f"unsupported reward mode: {self.reward_mode}")
        if self.denotation_comparison != "bird-set":
            raise ValueError("active RL requires denotation_comparison='bird-set'")
        if self.max_steps < 1 or self.max_new_tokens < 1:
            raise ValueError("rollout step and token budgets must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


class TableAgentRolloutCollector:
    """Run independent harness episodes and retain policy logprobs for every turn."""

    def __init__(
        self,
        tokenizer,
        settings: RolloutSettings,
        *,
        process_config: ProcessRewardConfig | None = None,
        counterfactual_suites: dict[str, Any] | None = None,
        generate_batch: GenerateBatch | None = None,
        generate_batch_with_keys: GenerateBatchWithKeys | None = None,
        rollout_log_path: Path | None = None,
    ):
        settings.validate()
        if settings.reward_mode == "process" and process_config is None:
            raise ValueError("process rollout requires a ProcessRewardConfig")
        self.tokenizer = tokenizer
        self.settings = settings
        self.process_config = process_config
        self.counterfactual_suites = counterfactual_suites or {}
        self.generate_batch_override = generate_batch
        self.generate_batch_with_keys_override = generate_batch_with_keys
        self.rollout_log_path = rollout_log_path

    def _render(self, messages: list[dict[str, str]]) -> tuple[str, list[int]]:
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return text, list(ids)

    def _generate_with_trainer(self, prompts: list[str], trainer) -> dict[str, Any]:
        generation = trainer.vllm_generation
        if generation.mode != "server":
            raise RuntimeError(
                "the transition rollout adapter currently requires TRL vLLM server mode"
            )
        return generation.vllm_client.generate(
            prompts=prompts,
            n=1,
            repetition_penalty=self.settings.repetition_penalty,
            temperature=self.settings.temperature,
            top_p=self.settings.top_p,
            top_k=self.settings.top_k,
            min_p=self.settings.min_p,
            max_tokens=self.settings.max_new_tokens,
            logprobs=0,
        )

    @staticmethod
    def _sampled_logprobs(
        response_ids: Sequence[int],
        logprobs: Sequence[Any],
        logprob_token_ids: Sequence[Any] | None,
    ) -> tuple[float, ...]:
        if len(response_ids) != len(logprobs):
            raise RuntimeError(
                "vLLM response ids and logprobs do not align: "
                f"{len(response_ids)} != {len(logprobs)}"
            )
        values = []
        for index, (token_id, token_logprobs) in enumerate(
            zip(response_ids, logprobs, strict=True)
        ):
            if not token_logprobs:
                raise RuntimeError(f"vLLM omitted the sampled logprob at token {index}")
            if logprob_token_ids is not None:
                token_candidates = logprob_token_ids[index]
                if not token_candidates or int(token_candidates[0]) != int(token_id):
                    raise RuntimeError(
                        "vLLM sampled-token identity does not align with completion ids"
                    )
            values.append(float(token_logprobs[0]))
        return tuple(values)

    def _write_rollouts(self, episodes: Sequence[PolicyEpisode]) -> None:
        if self.rollout_log_path is None:
            return
        self.rollout_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.rollout_log_path.open("a", encoding="utf-8") as handle:
            for episode in episodes:
                handle.write(
                    json.dumps(
                        episode.sample.audit_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def collect(self, inputs: list[dict[str, Any]], trainer) -> list[PolicyEpisode]:
        """Collect one episode per repeated GRPO input row."""
        settings = self.settings
        sample_counts: Counter[int] = Counter()
        envs = []
        metadata_rows = []
        sample_indices = []
        policy_turns: list[list[PolicyTurn]] = []
        scored_turns: list[list[tuple[list[int], list[int]]]] = []
        tokenization_warnings: list[bool] = []

        for item in inputs:
            metadata = dict(item["environment"])
            example_index = int(metadata["example_index"])
            sample_index = sample_counts[example_index]
            sample_counts[example_index] += 1
            envs.append(
                create_tool_use_env(
                    episode_example(metadata),
                    tool_scheme=settings.tool_scheme,
                    example_index=example_index,
                    max_steps=settings.max_steps,
                    max_batch_calls=settings.max_batch_calls,
                    context_mode=settings.context_mode,
                    history_turns=settings.history_turns,
                    compact_observations=True,
                    denotation_comparison=settings.denotation_comparison,
                )
            )
            metadata_rows.append(metadata)
            sample_indices.append(sample_index)
            policy_turns.append([])
            scored_turns.append([])
            tokenization_warnings.append(False)

        try:
            while any(not env.done for env in envs):
                active_indices = []
                active_prompts = []
                local_prompt_ids = []
                for env_index, env in enumerate(envs):
                    if env.done:
                        continue
                    prompt_text, prompt_ids = self._render(env.model_messages())
                    if len(prompt_ids) + settings.max_new_tokens > settings.max_context_tokens:
                        env.done = True
                        env.failure_type = "context_overflow"
                        continue
                    active_indices.append(env_index)
                    active_prompts.append(prompt_text)
                    local_prompt_ids.append(prompt_ids)
                if not active_indices:
                    break

                request_keys = [
                    (
                        str(metadata_rows[env_index]["task_id"]),
                        int(sample_indices[env_index]),
                        len(policy_turns[env_index]),
                    )
                    for env_index in active_indices
                ]
                if self.generate_batch_with_keys_override is not None:
                    output = self.generate_batch_with_keys_override(
                        active_prompts,
                        request_keys,
                    )
                elif self.generate_batch_override is not None:
                    output = self.generate_batch_override(active_prompts)
                else:
                    output = self._generate_with_trainer(active_prompts, trainer)
                prompt_id_rows = output["prompt_ids"]
                response_id_rows = output["completion_ids"]
                logprob_rows = output["logprobs"]
                token_id_rows = output.get("logprob_token_ids")
                if not (
                    len(prompt_id_rows)
                    == len(response_id_rows)
                    == len(logprob_rows)
                    == len(active_indices)
                ):
                    raise RuntimeError("vLLM batch output does not align with active environments")

                for row_index, env_index in enumerate(active_indices):
                    env = envs[env_index]
                    response_ids = list(response_id_rows[row_index])
                    if not response_ids:
                        env.done = True
                        env.failure_type = "generation_oom"
                        continue
                    server_prompt_ids = list(prompt_id_rows[row_index])
                    # Tokenization should be identical. Keep the actual server prefix either way,
                    # because it is the distribution under which vLLM sampled the action.
                    if server_prompt_ids != local_prompt_ids[row_index]:
                        tokenization_warnings[env_index] = True
                    token_candidates = (
                        token_id_rows[row_index] if token_id_rows is not None else None
                    )
                    sampled_logprobs = self._sampled_logprobs(
                        response_ids,
                        logprob_rows[row_index],
                        token_candidates,
                    )
                    policy_turn = PolicyTurn(
                        prompt_ids=tuple(server_prompt_ids),
                        response_ids=tuple(response_ids),
                        sampling_logprobs=sampled_logprobs,
                    )
                    policy_turn.validate()
                    text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    policy_turns[env_index].append(policy_turn)
                    scored_turns[env_index].append(
                        (server_prompt_ids, response_ids)
                    )
                    env.apply_model_output(text)

            episodes = []
            for env_index, env in enumerate(envs):
                metadata = metadata_rows[env_index]
                suite = self.counterfactual_suites.get(str(metadata["task_id"]))
                sample = score_completed_rollout(
                    env,
                    scored_turns[env_index],
                    metadata,
                    sample_index=sample_indices[env_index],
                    reward_mode=settings.reward_mode,
                    process_config=self.process_config,
                    process_admission_policy=settings.process_admission_policy,
                    denotation_comparison=settings.denotation_comparison,
                    counterfactual_suite=suite,
                )
                if tokenization_warnings[env_index]:
                    sample.audit_record["rollout_tokenization_warning"] = True
                episode = PolicyEpisode(
                    sample=sample,
                    policy_turns=policy_turns[env_index],
                )
                episode.validate()
                episodes.append(episode)
            self._write_rollouts(episodes)
            return episodes
        finally:
            for env in envs:
                env.close()
