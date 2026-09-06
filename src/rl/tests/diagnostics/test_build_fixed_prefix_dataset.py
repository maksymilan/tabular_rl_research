from __future__ import annotations

from rl.scenarios.diagnostics.build_fixed_prefix_dataset import (
    ASSISTANT_REASONING_PLACEHOLDER,
    build_candidates,
    collect_occurrences,
    normalize_assistant_reasoning,
    select_candidates,
)

import json


def test_normalize_assistant_reasoning_preserves_actions_and_harness_messages() -> None:
    first = [
        {"role": "system", "content": "contract"},
        {"role": "assistant", "content": '<think>reason A</think>\n{"tool":"x"}'},
        {"role": "user", "content": "observation"},
    ]
    second = [
        {"role": "system", "content": "contract"},
        {"role": "assistant", "content": '<think>reason B</think>\n{"tool":"x"}'},
        {"role": "user", "content": "observation"},
    ]

    normalized_first = normalize_assistant_reasoning(first)
    normalized_second = normalize_assistant_reasoning(second)

    assert normalized_first == normalized_second
    assert normalized_first[1]["content"] == (
        f"<think>{ASSISTANT_REASONING_PLACEHOLDER}</think>\n" '{"tool":"x"}'
    )
    assert normalized_first[2] == first[2]
    assert first[1]["content"] == '<think>reason A</think>\n{"tool":"x"}'


def test_normalize_assistant_reasoning_does_not_merge_different_actions() -> None:
    first = [{"role": "assistant", "content": '<think>a</think>\n{"tool":"x"}'}]
    second = [{"role": "assistant", "content": '<think>b</think>\n{"tool":"y"}'}]

    assert normalize_assistant_reasoning(first) != normalize_assistant_reasoning(second)


def test_builds_pair_only_for_exact_same_visible_prefix(tmp_path) -> None:
    shared_state = [{"role": "user", "content": "same"}]
    other_state = [{"role": "user", "content": "different"}]
    row = {
        "example_index": 4,
        "db_id": "db",
        "question": "q",
        "samples": [
            {
                "sample_index": 0,
                "correct": True,
                "legal": True,
                "turns": [
                    {
                        "turn_index": 0,
                        "model_input": shared_state,
                        "parsed": {"tool": "describe_table", "arguments": {"table": "a"}},
                    }
                ],
            },
            {
                "sample_index": 1,
                "correct": False,
                "legal": True,
                "turns": [
                    {
                        "turn_index": 0,
                        "model_input": shared_state,
                        "parsed": {"tool": "describe_table", "arguments": {"table": "b"}},
                    },
                    {
                        "turn_index": 1,
                        "model_input": other_state,
                        "parsed": {"tool": "project", "arguments": {"table": "x"}},
                    },
                ],
            },
        ],
    }
    artifact = tmp_path / "all.jsonl"
    artifact.write_text(json.dumps(row) + "\n")

    candidates = build_candidates(collect_occurrences([artifact]))

    assert len(candidates) == 1
    assert candidates[0]["category"] == "normal_first_step"
    assert candidates[0]["positive_action"]["arguments"]["table"] == "a"
    assert candidates[0]["negative_action"]["arguments"]["table"] == "b"
    assert candidates[0]["positive_verified_by"]["real_harness_execution"] is True


def test_quota_selection_reports_shortfalls_without_fabricating_pairs() -> None:
    candidates = [
        {
            "question_id": "1",
            "category": "semantic_divergence",
            "pair_sha256": "a",
        }
    ]
    selected, shortfalls = select_candidates(
        candidates,
        {"semantic_divergence": 2, "error_recovery": 1},
        max_per_question=4,
        seed=101,
    )

    assert len(selected) == 1
    assert shortfalls == {"semantic_divergence": 1, "error_recovery": 1}
