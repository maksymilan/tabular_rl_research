from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_experiment_eval_metrics import build_metrics


def test_build_metrics_counts_passk_actions_and_logprob_coverage() -> None:
    records = [
        {
            "example_index": 1,
            "db_id": "db",
            "question": "q",
            "pass_at": {"1": False, "4": True},
            "sample_correct_count": 1,
            "sample_legal_count": 1,
            "samples": [
                {
                    "sample_index": 0,
                    "correct": False,
                    "legal": False,
                    "steps": 2,
                    "errors": 1,
                    "failure_type": "max_steps",
                    "generation_stats": {
                        "trajectory_action_logprob": -5.0,
                        "mean_token_surprisal": 1.0,
                        "mean_token_entropy_lower_bound": 0.5,
                    },
                    "turns": [
                        {"turn_index": 0, "parsed": {"tool": "describe_table", "arguments": {}}},
                        {"turn_index": 1, "execution_error_type": "protocol_error"},
                    ],
                },
                {
                    "sample_index": 1,
                    "correct": True,
                    "legal": True,
                    "steps": 1,
                    "errors": 0,
                    "failure_type": None,
                    "generation_stats": {
                        "trajectory_action_logprob": -2.0,
                        "mean_token_surprisal": 0.5,
                        "mean_token_entropy_lower_bound": 0.25,
                    },
                    "turns": [
                        {"turn_index": 0, "parsed": {"tool": "answer_from_context", "arguments": {}}},
                    ],
                },
            ],
        }
    ]

    metrics = build_metrics(records)

    assert metrics["accuracy"]["pass@1"]["correct"] == 0
    assert metrics["accuracy"]["pass@4"]["correct"] == 1
    assert metrics["valid_rate"]["all_samples"]["rate"] == 0.5
    assert metrics["avg_steps"]["all_samples"] == 1.5
    assert metrics["action_distribution"]["counts"] == {
        "answer_from_context": 1,
        "describe_table": 1,
    }
    assert metrics["trajectory_entropy"]["coverage"]["rate"] == 1.0
    assert len(metrics["trajectory"]) == 2
    assert metrics["sample0_at_1"] == {"correct": 0, "total": 1, "rate": 0.0}
    assert metrics["trajectory_accuracy"] == {
        "correct": 1,
        "total": 2,
        "rate": 0.5,
    }
    assert metrics["correct_count_distribution"]["1"]["questions"] == 1
    assert metrics["failure_types"]["all_samples"] == {"correct": 1, "max_steps": 1}
    assert metrics["greedy_at_1"]["available"] is False


def test_build_metrics_marks_strict_single_sample_decoding_as_greedy() -> None:
    records = [
        {
            "example_index": 7,
            "n_samples": 1,
            "temperature": 0,
            "top_p": 1,
            "max_steps": 30,
            "protocol_version": "version36",
            "denotation_comparison": "bird-set",
            "pass_at": {"1": True},
            "samples": [
                {
                    "sample_index": 0,
                    "correct": True,
                    "legal": True,
                    "steps": 3,
                    "failure_type": None,
                    "turns": [],
                }
            ],
        }
    ]

    metrics = build_metrics(records)

    assert metrics["evaluation_protocol"]["mode"] == "greedy"
    assert metrics["greedy_at_1"] == {
        "correct": 1,
        "total": 1,
        "rate": 1.0,
        "available": True,
    }
