"""Causal batched teacher repairs from four predeclared exact prefixes."""
from __future__ import annotations

import hashlib
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from src.rl.distillation.repair_plan import (
    RepairCandidate,
    RepairSelection,
    parallel_repair_anchor_turns,
    select_verified_repair,
)


def stable_seed(base_seed: int, *parts: Any) -> int:
    payload = "\0".join([str(base_seed), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "big") & 0x7FFFFFFF


def nonterminal_turn_count(audit_turns: Sequence[Mapping[str, Any]]) -> int:
    """Count turns before the first authored terminal action, if one exists."""
    for index, turn in enumerate(audit_turns):
        parsed = turn.get("parsed") or {}
        if parsed.get("tool") == "answer_from_context":
            return index
    return len(audit_turns)


@dataclass(frozen=True)
class GeneratedRepairBranch:
    teacher: str
    anchor_turn: int
    trial_index: int
    correct: bool
    legal: bool
    failure_type: str | None
    steps: int
    errors: int
    tokenization_warning: bool
    policy_turns: tuple[Any, ...]


@dataclass(frozen=True)
class VerifiedRepairGroup:
    candidate: RepairCandidate
    chosen_branches: tuple[GeneratedRepairBranch, ...]


@dataclass(frozen=True)
class VerifiedRepairBatch:
    anchors: tuple[int, ...]
    selection: RepairSelection
    chosen_branch: GeneratedRepairBranch | None
    branches: tuple[GeneratedRepairBranch, ...]
    strong_groups: tuple[VerifiedRepairGroup, ...]


def select_strong_repair_groups(
    branches: Sequence[GeneratedRepairBranch],
    *,
    minimum_strong_successes: int = 2,
    max_correct_branches_per_group: int = 2,
) -> tuple[VerifiedRepairGroup, ...]:
    """Retain every verifier-stable teacher-anchor group for task-balanced DPO."""
    if minimum_strong_successes < 2:
        raise ValueError("a stable repair group must require at least two successes")
    if max_correct_branches_per_group < 1:
        raise ValueError("a repair group must retain at least one correct branch")
    grouped: dict[tuple[str, int], list[GeneratedRepairBranch]] = defaultdict(list)
    for branch in branches:
        grouped[(branch.teacher, branch.anchor_turn)].append(branch)
    output = []
    for (teacher, anchor_turn), values in grouped.items():
        correct = [value for value in values if value.correct and value.legal]
        if len(correct) < minimum_strong_successes:
            continue
        candidate = RepairCandidate(
            teacher=teacher,
            anchor_turn=anchor_turn,
            trials=len(values),
            correct=len(correct),
            legal=sum(value.legal for value in values),
            total_errors=sum(value.errors for value in values),
            mean_steps=sum(value.steps for value in values) / len(values),
        )
        chosen = tuple(
            sorted(correct, key=lambda value: (value.errors, value.steps, value.trial_index))[
                :max_correct_branches_per_group
            ]
        )
        output.append(VerifiedRepairGroup(candidate=candidate, chosen_branches=chosen))
    return tuple(
        sorted(
            output,
            key=lambda value: (
                -value.candidate.anchor_turn,
                value.candidate.teacher,
            ),
        )
    )


def _adapter_name(backend, teacher: str) -> str:
    if teacher == "sft2":
        return backend.SFT2
    if teacher == "exp15":
        return backend.EXP15
    raise ValueError(f"unsupported teacher: {teacher}")


def _release_cuda_cache(backend) -> None:
    cuda = getattr(backend.torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()


def _generate_with_oom_backoff(
    backend,
    adapter: str,
    prompts: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Retry only the physical generation batch after a CUDA OOM.

    Repair branches, exact prefixes, seeds, and their declared launch order are
    fixed before this helper is called.  Splitting therefore changes only the
    number of rows resident in one GPU forward.  A single-row OOM is surfaced
    because there is no semantics-preserving physical split left to attempt.
    """
    if len(prompts) != len(seeds) or not prompts:
        raise ValueError("repair prompts and seeds must be non-empty and aligned")
    oom_type = getattr(backend.torch, "OutOfMemoryError", None)
    if oom_type is None:
        return backend.generate(
            adapter,
            list(prompts),
            seeds=list(seeds),
            include_sampled_logprobs=False,
        )
    retry_smaller = False
    try:
        return backend.generate(
            adapter,
            list(prompts),
            seeds=list(seeds),
            include_sampled_logprobs=False,
        )
    except oom_type as error:
        if len(prompts) == 1:
            error.add_note(
                "repair generation already reached physical microbatch size 1"
            )
            raise
        retry_smaller = True
    if not retry_smaller:  # pragma: no cover - defensive for unusual exception types
        raise RuntimeError("unreachable repair generation retry state")
    _release_cuda_cache(backend)
    midpoint = len(prompts) // 2
    warnings.warn(
        "CUDA OOM in repair generation; retrying the same predeclared branches "
        f"as physical microbatches {midpoint}+{len(prompts) - midpoint}",
        RuntimeWarning,
        stacklevel=2,
    )
    left = _generate_with_oom_backoff(
        backend,
        adapter,
        prompts[:midpoint],
        seeds[:midpoint],
    )
    right = _generate_with_oom_backoff(
        backend,
        adapter,
        prompts[midpoint:],
        seeds[midpoint:],
    )
    keys = ("prompt_ids", "completion_ids", "logprobs", "logprob_token_ids")
    return {key: [*left[key], *right[key]] for key in keys}


def generate_parallel_repairs(
    backend,
    source_row: dict[str, Any],
    *,
    teachers: Sequence[str],
    trials_per_anchor: int,
    base_seed: int,
    max_steps: int,
    history_turns: int,
    generation_batch_size: int = 8,
    max_correct_branches_per_group: int = 2,
) -> VerifiedRepairBatch:
    """Launch every teacher × anchor × trial before selecting a repair.

    All harnesses are independent and all four anchor depths are live together.
    A single GPU still switches teacher adapters serially, but each teacher's
    active environments are generated as one batch; no anchor result controls
    whether another anchor is attempted.
    """
    from src.rl.action_dpo.generate_exact_prefix_branches import replay_prefix
    from src.rl.frameworks.trl.transition_batch import PolicyTurn

    if trials_per_anchor < 2:
        raise ValueError("verified repair requires at least two trials per anchor")
    if generation_batch_size < 1:
        raise ValueError("repair generation batch size must be positive")
    audit_turns = source_row["sample"]["audit_record"]["turns"]
    if not audit_turns:
        return VerifiedRepairBatch((0,), RepairSelection("none", None), None, (), ())
    anchors = parallel_repair_anchor_turns(nonterminal_turn_count(audit_turns))
    branches: list[dict[str, Any]] = []
    for teacher in teachers:
        for anchor_turn in anchors:
            for trial_index in range(trials_per_anchor):
                env = replay_prefix(
                    source_row,
                    anchor_turn,
                    max_steps=max_steps,
                    history_turns=history_turns,
                )
                branches.append(
                    {
                        "teacher": teacher,
                        "adapter": _adapter_name(backend, teacher),
                        "anchor_turn": anchor_turn,
                        "trial_index": trial_index,
                        "env": env,
                        "policy_turns": [],
                        "generation_turn": 0,
                        "tokenization_warning": False,
                    }
                )

    try:
        while any(not branch["env"].done for branch in branches):
            by_teacher: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for branch in branches:
                if not branch["env"].done:
                    by_teacher[branch["teacher"]].append(branch)
            if not by_teacher:
                break
            for teacher in sorted(by_teacher):
                active = []
                prompts = []
                local_prompt_ids = []
                seeds = []
                for branch in by_teacher[teacher]:
                    try:
                        prompt, prompt_ids = backend.render(branch["env"].model_messages())
                    except ValueError as exc:
                        if "exceeds context budget" not in str(exc):
                            raise
                        branch["env"].done = True
                        branch["env"].failure_type = "context_overflow"
                        continue
                    active.append(branch)
                    prompts.append(prompt)
                    local_prompt_ids.append(prompt_ids)
                    seeds.append(
                        stable_seed(
                            base_seed,
                            teacher,
                            branch["anchor_turn"],
                            branch["trial_index"],
                            branch["generation_turn"],
                        )
                    )
                if not active:
                    continue
                # Every branch is declared before generation and attempted in
                # this round.  Chunking is only a physical GPU microbatch
                # bound; no result controls whether another anchor is launched
                # or changes its deterministic seed.
                for start in range(0, len(active), generation_batch_size):
                    end = start + generation_batch_size
                    output = _generate_with_oom_backoff(
                        backend,
                        _adapter_name(backend, teacher),
                        prompts[start:end],
                        seeds=seeds[start:end],
                    )
                    rows = zip(
                        active[start:end],
                        local_prompt_ids[start:end],
                        output["prompt_ids"],
                        output["completion_ids"],
                        output["logprobs"],
                        output["logprob_token_ids"],
                        strict=True,
                    )
                    for branch, local_ids, prompt_ids, response_ids, logprobs, candidates in rows:
                        if list(local_ids) != list(prompt_ids):
                            branch["tokenization_warning"] = True
                        sampled = []
                        for token_id, token_logps, token_candidates in zip(
                            response_ids, logprobs, candidates, strict=True
                        ):
                            if not token_logps or int(token_candidates[0]) != int(token_id):
                                raise RuntimeError("teacher generation omitted sampled-token evidence")
                            sampled.append(float(token_logps[0]))
                        turn = PolicyTurn(
                            prompt_ids=tuple(int(value) for value in prompt_ids),
                            response_ids=tuple(int(value) for value in response_ids),
                            sampling_logprobs=tuple(sampled),
                        )
                        turn.validate()
                        branch["policy_turns"].append(turn)
                        text = backend.tokenizer.decode(response_ids, skip_special_tokens=True)
                        branch["env"].apply_model_output(text)
                        branch["generation_turn"] += 1

        completed = []
        for branch in branches:
            record = branch["env"].record()
            completed.append(
                GeneratedRepairBranch(
                    teacher=branch["teacher"],
                    anchor_turn=int(branch["anchor_turn"]),
                    trial_index=int(branch["trial_index"]),
                    correct=bool(record["correct"]),
                    legal=bool(record["legal"]),
                    failure_type=record.get("failure_type"),
                    steps=int(record["steps"]),
                    errors=int(record["errors"]),
                    tokenization_warning=bool(branch["tokenization_warning"]),
                    policy_turns=tuple(branch["policy_turns"]),
                )
            )

        grouped: dict[tuple[str, int], list[GeneratedRepairBranch]] = defaultdict(list)
        for branch in completed:
            grouped[(branch.teacher, branch.anchor_turn)].append(branch)
        candidates = []
        for (teacher, anchor_turn), values in grouped.items():
            candidates.append(
                RepairCandidate(
                    teacher=teacher,
                    anchor_turn=anchor_turn,
                    trials=len(values),
                    correct=sum(value.correct for value in values),
                    legal=sum(value.legal for value in values),
                    total_errors=sum(value.errors for value in values),
                    mean_steps=sum(value.steps for value in values) / len(values),
                )
            )
        selection = select_verified_repair(candidates)
        chosen = None
        if selection.candidate is not None:
            matches = [
                branch
                for branch in completed
                if branch.teacher == selection.candidate.teacher
                and branch.anchor_turn == selection.candidate.anchor_turn
                and branch.correct
                and branch.legal
            ]
            if matches:
                chosen = min(matches, key=lambda value: (value.errors, value.steps, value.trial_index))
        strong_groups = select_strong_repair_groups(
            completed,
            max_correct_branches_per_group=max_correct_branches_per_group,
        )
        return VerifiedRepairBatch(
            anchors,
            selection,
            chosen,
            tuple(completed),
            strong_groups,
        )
    finally:
        for branch in branches:
            branch["env"].close()
