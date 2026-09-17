#!/usr/bin/env python3
"""Batched vLLM rollouts that preserve one exact prefix per assistant transition."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

from protocol import PROTOCOL_VERSION
import rollout as evaluator_runtime

if PROTOCOL_VERSION != "version26":
    raise ImportError(
        "the active RL mainline is pinned to the frozen version26 protocol; "
        f"received {PROTOCOL_VERSION!r}"
    )

from rl.runtime.tool_environment_v26 import (  # noqa: E402
    TOOL_EXECUTION_TIMEOUT_SECONDS,
    create_tool_use_env,
)

TOOL_ENVIRONMENT_FACTORY_MODULE = create_tool_use_env.__module__

from rl.runtime.rollout_scoring import episode_example, score_completed_rollout
from rl.runtime.terminal_reward import RESULT_REWARD_PROFILES
from rl.frameworks.trl.transition_batch import PolicyEpisode, PolicyTurn

if TYPE_CHECKING:
    from rl.objectives.process_credit import ProcessRewardConfig


GenerateBatch = Callable[[list[str]], dict[str, Any]]
GenerateBatchWithKeys = Callable[
    [list[str], list[tuple[str, int, int]]],
    dict[str, Any],
]


_LENGTH_FINISH_REASONS = frozenset(
    {"length", "max_length", "max_tokens", "max_new_tokens"}
)


@dataclass(frozen=True)
class RolloutSettings:
    reward_mode: str = "result-only"
    result_reward_profile: str = "binary"
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
    enable_thinking: bool | None = None
    tool_execution_timeout_seconds: float = TOOL_EXECUTION_TIMEOUT_SECONDS
    adaptive_group_size_max: int | None = None

    def validate(self) -> None:
        if self.reward_mode not in {"result-only", "process"}:
            raise ValueError(f"unsupported reward mode: {self.reward_mode}")
        if self.result_reward_profile not in RESULT_REWARD_PROFILES:
            raise ValueError(
                f"unsupported result reward profile: {self.result_reward_profile}"
            )
        if self.denotation_comparison != "bird-set":
            raise ValueError("active RL requires denotation_comparison='bird-set'")
        if self.max_steps < 1 or self.max_new_tokens < 1:
            raise ValueError("rollout step and token budgets must be positive")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.tool_execution_timeout_seconds <= 0:
            raise ValueError("tool_execution_timeout_seconds must be positive")
        if self.adaptive_group_size_max is not None:
            if self.adaptive_group_size_max < 2:
                raise ValueError("adaptive_group_size_max must be >= 2")


def adaptive_extension_counts(
    *,
    outcomes: Mapping[int, tuple[int, int]],
    initial_group_size: int,
    max_group_size: int,
) -> dict[int, int]:
    """Extra samples to draw for prompts whose current group carries no signal.

    ``outcomes`` maps an example index to ``(samples, correct)``.  A group is
    extended while it is all-correct or all-wrong, in steps of the initial group
    size, and never beyond ``max_group_size``.  Mixed groups are left untouched,
    which is what turns wasted rollouts into GRPO contrast.
    """
    if initial_group_size < 2:
        raise ValueError("initial_group_size must be >= 2")
    if max_group_size < initial_group_size:
        raise ValueError("max_group_size must be >= initial_group_size")
    targets: dict[int, int] = {}
    for example_index, (samples, correct) in outcomes.items():
        if samples < initial_group_size:
            continue
        if correct not in (0, samples):
            continue
        room = max_group_size - samples
        if room <= 0:
            continue
        targets[example_index] = min(initial_group_size, room)
    return targets


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
        self.last_adaptive_stats: dict[str, Any] = {}

    def _render(self, messages: list[dict[str, str]]) -> tuple[str, list[int]]:
        template_kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.settings.enable_thinking is not None:
            # Hugging Face forwards extra top-level kwargs into the Jinja chat
            # template.  Nesting this under ``chat_template_kwargs`` is the TRL
            # client API shape, not the tokenizer API shape, and silently leaves
            # Qwen3 on its default thinking mode.
            template_kwargs["enable_thinking"] = self.settings.enable_thinking
        text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        return text, list(ids)

    @staticmethod
    def _optional_generation_value(
        output: dict[str, Any],
        keys: Sequence[str],
        *,
        row_index: int,
        row_count: int,
    ) -> tuple[Any, str | None]:
        """Read an optional row field without requiring a patched TRL server.

        TRL 0.29's vLLM client exposes token ids and logprobs but drops vLLM's
        finish metadata.  Test/offline generators and future clients may expose
        either singular or plural field names, so accept both while validating
        their batch alignment when present.
        """
        for key in keys:
            if key not in output or output[key] is None:
                continue
            rows = output[key]
            if isinstance(rows, (str, int, float, bool)):
                if row_count != 1:
                    raise RuntimeError(
                        f"vLLM scalar {key} cannot describe a {row_count}-row batch"
                    )
                return rows, key
            if not isinstance(rows, Sequence) or len(rows) != row_count:
                raise RuntimeError(
                    f"vLLM {key} does not align with active environments"
                )
            value = rows[row_index]
            # Some adapters preserve vLLM's n-completion dimension.  This
            # collector always requests n=1, so unwrap exactly that shape.
            if isinstance(value, Sequence) and not isinstance(value, str):
                if len(value) != 1:
                    raise RuntimeError(
                        f"vLLM {key} row must contain exactly one completion"
                    )
                value = value[0]
            return value, key
        return None, None

    @classmethod
    def _generation_truncation(
        cls,
        output: dict[str, Any],
        *,
        row_index: int,
        row_count: int,
        completion_tokens: int,
        max_new_tokens: int,
        turn_index: int,
        prompt_tokens: int | None = None,
        response_token_ids: Sequence[int] | None = None,
    ) -> dict[str, Any] | None:
        """Return fail-closed length metadata for one generated completion."""
        finish_reason, finish_reason_field = cls._optional_generation_value(
            output,
            ("finish_reasons", "finish_reason"),
            row_index=row_index,
            row_count=row_count,
        )
        stop_reason, stop_reason_field = cls._optional_generation_value(
            output,
            ("stop_reasons", "stop_reason"),
            row_index=row_index,
            row_count=row_count,
        )
        normalized_finish_reason = (
            finish_reason.strip().lower() if isinstance(finish_reason, str) else None
        )
        normalized_stop_reason = (
            stop_reason.strip().lower() if isinstance(stop_reason, str) else None
        )

        if normalized_finish_reason in _LENGTH_FINISH_REASONS:
            detection = "explicit_finish_reason"
        elif finish_reason is not None:
            # An explicit non-length finish reason is stronger evidence than the
            # token-count fallback, including the rare case where EOS lands on
            # the final allowed token.
            return None
        elif normalized_stop_reason in _LENGTH_FINISH_REASONS:
            detection = "explicit_stop_reason"
        elif stop_reason is not None:
            return None
        elif completion_tokens >= max_new_tokens:
            # This is the compatibility path for the pinned TRL 0.29 schema,
            # which does not return CompletionOutput.finish_reason.
            detection = "max_new_tokens_reached"
        else:
            return None

        evidence = {
            "schema_version": "vllm-generation-truncation-v1",
            "kind": "length",
            "detection": detection,
            "turn_index": turn_index,
            "completion_tokens": completion_tokens,
            "max_new_tokens": max_new_tokens,
            "finish_reason": finish_reason,
            "finish_reason_field": finish_reason_field,
            "stop_reason": stop_reason,
            "stop_reason_field": stop_reason_field,
        }
        if (prompt_tokens is None) != (response_token_ids is None):
            raise RuntimeError(
                "generation truncation requires prompt tokens and response ids together"
            )
        if prompt_tokens is not None and response_token_ids is not None:
            evidence.update(
                {
                    "schema_version": "vllm-generation-truncation-v2",
                    "prompt_tokens": int(prompt_tokens),
                    "response_token_ids_sha256": hashlib.sha256(
                        json.dumps(
                            [int(token_id) for token_id in response_token_ids],
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            )
        return evidence

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
        """Collect GRPO groups, extending zero-signal groups up to a fixed cap.

        The first pass is exactly the historical behaviour (one episode per
        repeated input row).  With ``adaptive_group_size_max`` set, every prompt
        whose group is still all-correct or all-wrong is sampled again in steps
        of the initial group size, up to the cap; mixed groups are never
        extended.  All samples of one update come from the same synced policy, so
        the merged group stays on-policy.
        """
        example_of = lambda item: int(item["environment"]["example_index"])  # noqa: E731
        initial_sizes = Counter(example_of(item) for item in inputs)
        initial_group_size = max(initial_sizes.values()) if initial_sizes else 0
        representative: dict[int, dict[str, Any]] = {}
        for item in inputs:
            representative.setdefault(example_of(item), item)

        episodes = self._collect_pass(inputs, trainer)
        samples = Counter()
        correct = Counter()
        for episode in episodes:
            example_index = int(episode.sample.audit_record["example_index"])
            samples[example_index] += 1
            correct[example_index] += int(bool(episode.sample.correct))

        stats: dict[str, Any] = {
            "initial_group_size": initial_group_size,
            "max_group_size": self.settings.adaptive_group_size_max,
            "prompts": len(samples),
            "rounds": 0,
            "extended_prompts": 0,
            "extra_episodes": 0,
            "activated_groups": 0,
            "remaining_zero_signal_at_cap": 0,
            "final_group_size_histogram": {},
        }
        max_group_size = self.settings.adaptive_group_size_max
        if max_group_size is None or initial_group_size < 2 or max_group_size <= initial_group_size:
            self.last_adaptive_stats = stats
            return episodes

        max_rounds = (max_group_size - initial_group_size) // initial_group_size + 1
        while stats["rounds"] < max_rounds:
            outcomes = {
                example_index: (samples[example_index], correct[example_index])
                for example_index in samples
            }
            targets = adaptive_extension_counts(
                outcomes=outcomes,
                initial_group_size=initial_group_size,
                max_group_size=max_group_size,
            )
            if not targets:
                break
            offsets = {
                example_index: samples[example_index] for example_index in targets
            }
            extra_inputs: list[dict[str, Any]] = []
            for example_index, extra in sorted(targets.items()):
                extra_inputs.extend([representative[example_index]] * int(extra))
            new_episodes = self._collect_pass(
                extra_inputs, trainer, sample_index_offsets=offsets
            )
            for episode in new_episodes:
                example_index = int(episode.sample.audit_record["example_index"])
                was_mixed = 0 < correct[example_index] < samples[example_index]
                samples[example_index] += 1
                correct[example_index] += int(bool(episode.sample.correct))
                if not was_mixed and 0 < correct[example_index] < samples[example_index]:
                    stats["activated_groups"] += 1
            episodes.extend(new_episodes)
            stats["rounds"] += 1
            stats["extended_prompts"] += len(targets)
            stats["extra_episodes"] += len(new_episodes)

        histogram = Counter(samples[example_index] for example_index in samples)
        stats["final_group_size_histogram"] = {
            str(size): histogram[size] for size in sorted(histogram)
        }
        stats["remaining_zero_signal_at_cap"] = sum(
            1
            for example_index in samples
            if samples[example_index] >= max_group_size
            and correct[example_index] in (0, samples[example_index])
        )
        self.last_adaptive_stats = stats
        return episodes

    def _collect_pass(
        self,
        inputs: list[dict[str, Any]],
        trainer,
        *,
        sample_index_offsets: Mapping[int, int] | None = None,
    ) -> list[PolicyEpisode]:
        """Collect one episode per repeated GRPO input row."""
        settings = self.settings
        sample_counts: Counter[int] = Counter()
        envs = []
        metadata_rows = []
        sample_indices = []
        policy_turns: list[list[PolicyTurn]] = []
        scored_turns: list[list[tuple[list[int], list[int]]]] = []
        tokenization_warnings: list[bool] = []
        generation_truncations: list[dict[str, Any] | None] = []
        context_overflows: list[dict[str, Any] | None] = []

        for item in inputs:
            metadata = dict(item["environment"])
            prompt_messages = item.get("prompt") or []
            if (
                not prompt_messages
                or prompt_messages[0].get("role") != "system"
                or not isinstance(prompt_messages[0].get("content"), str)
            ):
                raise ValueError("RL task record is missing its pinned system prompt")
            system_prompt = prompt_messages[0]["content"]
            example_index = int(metadata["example_index"])
            sample_index = (sample_index_offsets or {}).get(example_index, 0) + sample_counts[
                example_index
            ]
            sample_counts[example_index] += 1
            envs.append(
                create_tool_use_env(
                    episode_example(metadata),
                    tool_scheme=settings.tool_scheme,
                    example_index=example_index,
                    system_prompt=system_prompt,
                    max_steps=settings.max_steps,
                    max_batch_calls=settings.max_batch_calls,
                    context_mode=settings.context_mode,
                    history_turns=settings.history_turns,
                    compact_observations=True,
                    denotation_comparison=settings.denotation_comparison,
                    tool_execution_timeout_seconds=(
                        settings.tool_execution_timeout_seconds
                    ),
                )
            )
            metadata_rows.append(metadata)
            sample_indices.append(sample_index)
            policy_turns.append([])
            scored_turns.append([])
            tokenization_warnings.append(False)
            generation_truncations.append(None)
            context_overflows.append(None)

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
                        context_overflows[env_index] = {
                            "schema_version": "vllm-context-overflow-v1",
                            "kind": "context_overflow",
                            "detection": (
                                "prompt_plus_max_new_tokens_exceeds_context"
                            ),
                            "turn_index": len(policy_turns[env_index]),
                            "prompt_tokens": len(prompt_ids),
                            "max_new_tokens": settings.max_new_tokens,
                            "max_context_tokens": settings.max_context_tokens,
                            "required_tokens": (
                                len(prompt_ids) + settings.max_new_tokens
                            ),
                        }
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
                    truncation = self._generation_truncation(
                        output,
                        row_index=row_index,
                        row_count=len(active_indices),
                        prompt_tokens=len(server_prompt_ids),
                        response_token_ids=response_ids,
                        completion_tokens=len(response_ids),
                        max_new_tokens=settings.max_new_tokens,
                        turn_index=len(policy_turns[env_index]) - 1,
                    )
                    if truncation is not None:
                        # A length-limited carrier is not a complete authored
                        # action.  Retain its exact policy evidence for audit, but
                        # neither execute it nor let any part of the episode enter
                        # policy optimization.
                        generation_truncations[env_index] = truncation
                        env.done = True
                        env.failure_type = "generation_length"
                        continue
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
                    result_reward_profile=settings.result_reward_profile,
                    generation_truncation=generation_truncations[env_index],
                )
                if context_overflows[env_index] is not None:
                    sample.audit_record["context_overflow"] = dict(
                        context_overflows[env_index]
                    )
                # Persist the exact online-policy generation that authored this
                # trajectory. Chronological row counts can reconstruct it, but the
                # explicit binding makes weight-refresh audits fail closed instead of
                # relying on an implicit Trainer batching convention.
                trainer_state = getattr(trainer, "state", None)
                sample.audit_record["policy_global_step"] = int(
                    getattr(trainer_state, "global_step", 0)
                )
                sample.audit_record["policy_micro_step"] = int(
                    getattr(trainer, "_step", 0)
                )
                sample.audit_record["policy_synced_global_step"] = int(
                    getattr(trainer, "_last_loaded_step", -1)
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
