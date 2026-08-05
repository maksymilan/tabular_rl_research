from __future__ import annotations

from src.rl.distillation.repair_rollout import (
    GeneratedRepairBranch,
    _generate_with_oom_backoff,
    nonterminal_turn_count,
    select_strong_repair_groups,
    stable_seed,
)


class _FakeCuda:
    def __init__(self):
        self.empty_cache_calls = 0

    @staticmethod
    def is_available():
        return True

    def empty_cache(self):
        self.empty_cache_calls += 1


class _FakeTorch:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self):
        self.cuda = _FakeCuda()


class _OOMUntilSingleBackend:
    def __init__(self):
        self.torch = _FakeTorch()
        self.batch_sizes = []

    def generate(self, adapter, prompts, *, seeds, include_sampled_logprobs):
        assert adapter == "teacher"
        assert include_sampled_logprobs is False
        self.batch_sizes.append(len(prompts))
        if len(prompts) > 1:
            raise self.torch.OutOfMemoryError("synthetic OOM")
        seed = int(seeds[0])
        return {
            "prompt_ids": [[seed]],
            "completion_ids": [[seed + 1]],
            "logprobs": [[[0.0]]],
            "logprob_token_ids": [[[seed + 1]]],
        }


def test_repair_generation_oom_backoff_preserves_row_order_and_seeds():
    backend = _OOMUntilSingleBackend()
    with __import__("pytest").warns(RuntimeWarning, match="physical microbatches"):
        output = _generate_with_oom_backoff(
            backend,
            "teacher",
            ["p0", "p1", "p2", "p3"],
            [10, 20, 30, 40],
        )
    assert output["prompt_ids"] == [[10], [20], [30], [40]]
    assert output["completion_ids"] == [[11], [21], [31], [41]]
    assert backend.batch_sizes == [4, 2, 1, 1, 2, 1, 1]
    assert backend.torch.cuda.empty_cache_calls == 3


def test_terminal_turn_is_not_part_of_fraction_denominator():
    turns = [
        {"parsed": {"tool": "describe_table"}},
        {"parsed": {"tool": "project"}},
        {"parsed": {"tool": "answer_from_context"}},
    ]
    assert nonterminal_turn_count(turns) == 2


def test_max_step_failure_counts_all_model_turns():
    assert nonterminal_turn_count([{"parsed": {}}, {"parsed": {"tool": "project"}}]) == 2


def test_repair_seeds_are_stable_and_branch_specific():
    first = stable_seed(101, "sft2", 4, 0, 0)
    assert first == stable_seed(101, "sft2", 4, 0, 0)
    assert first != stable_seed(101, "sft2", 4, 1, 0)


def _branch(teacher, anchor, trial, *, correct, errors=0, steps=3):
    return GeneratedRepairBranch(
        teacher=teacher,
        anchor_turn=anchor,
        trial_index=trial,
        correct=correct,
        legal=True,
        failure_type=None if correct else "wrong_answer",
        steps=steps,
        errors=errors,
        tokenization_warning=False,
        policy_turns=(),
    )


def test_all_strong_groups_are_retained_with_at_most_two_branches_each():
    branches = [
        _branch("sft2", 6, 0, correct=True, errors=1),
        _branch("sft2", 6, 1, correct=True, errors=0),
        _branch("sft2", 6, 2, correct=True, errors=2),
        _branch("sft2", 6, 3, correct=False),
        _branch("exp15", 4, 0, correct=True),
        _branch("exp15", 4, 1, correct=True),
        _branch("exp15", 4, 2, correct=False),
        _branch("exp15", 4, 3, correct=False),
        _branch("exp15", 2, 0, correct=True),
        _branch("exp15", 2, 1, correct=False),
    ]
    groups = select_strong_repair_groups(branches)
    assert [(g.candidate.teacher, g.candidate.anchor_turn) for g in groups] == [
        ("sft2", 6),
        ("exp15", 4),
    ]
    assert [branch.trial_index for branch in groups[0].chosen_branches] == [1, 0]
    assert all(len(group.chosen_branches) == 2 for group in groups)
