#!/usr/bin/env python3
"""Audit process-credit assignment over full-sample pass@k training artifacts."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

from rl.runtime.external_failure_adapter import normalize_failure_record
from rl.objectives.process_credit import ProcessRewardConfig, score_rollout_trajectory


INFRASTRUCTURE_FAILURE_TYPES = {
    "api_error",
    "context_overflow",
    "generation_oom",
    "incomplete_api_response",
    "provider_carrier_error",
    "task_timeout",
    "transport_error",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_examples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("examples", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a JSON list or an examples list")
    return rows


def load_config(path: Path) -> ProcessRewardConfig:
    values = {
        key: value
        for key, value in json.loads(path.read_text(encoding="utf-8")).items()
        if not key.startswith("_")
    }
    config = ProcessRewardConfig(**values)
    config.validate()
    return config


def sample_has_infrastructure_failure(sample: dict[str, Any]) -> bool:
    return (
        sample.get("failure_type") in INFRASTRUCTURE_FAILURE_TYPES
        or any(bool(turn.get("api_error")) for turn in sample.get("turns") or [])
    )


def task_record(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "example_id": f"bird_train_{int(task['example_index']):05d}",
        "dataset": "bird-sql",
        "split": "train",
        "db_id": task["db_id"],
        "db_path": task["db_path"],
        "question": task["question"],
        "gold_sql": task.get("gold_sql") or task.get("query"),
        "external_knowledge": task.get("external_knowledge"),
        "denotation_comparison": "bird-set",
    }


def expected_positive(step: dict[str, Any], config: ProcessRewardConfig) -> float:
    feature = step["features"]
    return (
        config.w_terminal_correct * float(bool(feature.get("is_terminal")))
        + config.w_back_slice * float(feature.get("back_slice", 0.0))
        + config.w_new_evidence * float(feature.get("new_used_evidence", 0.0))
        + config.w_search_reduction * float(feature.get("search_reduction", 0.0))
        + config.w_feedback_response * float(feature.get("feedback_response", 0.0))
        + config.w_target_potential * float(feature.get("target_potential_delta", 0.0))
    )


def expected_local_penalty(step: dict[str, Any], config: ProcessRewardConfig) -> float:
    feature = step["features"]
    return (
        config.lambda_tool_error * float(feature.get("tool_error", 0.0))
        + config.lambda_adjacent_repeat * float(bool(feature.get("adjacent_repeat")))
        + config.lambda_repeat_without_feedback
        * float(feature.get("repeat_without_feedback", 0.0))
        + config.lambda_legal_no_state_change
        * float(feature.get("legal_no_state_change", 0.0))
        + config.lambda_ignored_feedback * float(feature.get("ignored_feedback", 0.0))
        + config.lambda_unsupported_guess * float(feature.get("unsupported_guess", 0.0))
        + config.lambda_empty_result * float(feature.get("empty_result_penalty", 0.0))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passk-all", type=Path, required=True)
    parser.add_argument("--tasks-json", type=Path, required=True)
    parser.add_argument("--selected-json", type=Path, required=True)
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = load_config(args.config_json)
    task_map = {
        int(task["example_index"]): task for task in read_examples(args.tasks_json)
    }
    selected_ids = {
        int(task["example_index"]) for task in read_examples(args.selected_json)
    }
    rows = [
        row for row in read_jsonl(args.passk_all)
        if int(row["example_index"]) in selected_ids
    ]
    if {int(row["example_index"]) for row in rows} != selected_ids:
        raise ValueError("selected task IDs and pass@k rows do not match")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = args.output_dir / "normalized_trajectories.jsonl"
    scored_path = args.output_dir / "scored_trajectories.jsonl"
    excluded_path = args.output_dir / "excluded_infrastructure.jsonl"

    normalized_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    recorded_correct: dict[str, bool] = {}
    exclusion_types: collections.Counter[str] = collections.Counter()
    for row in rows:
        example_index = int(row["example_index"])
        task = task_map[example_index]
        if not Path(task["db_path"]).is_file():
            raise ValueError(f"missing task database for example_index={example_index}")
        for sample in row.get("samples") or []:
            sample_index = int(sample["sample_index"])
            trajectory_id = f"bird_train_{example_index:05d}_passk8_sample_{sample_index}"
            if sample_has_infrastructure_failure(sample):
                failure_type = str(sample.get("failure_type") or "api_error")
                exclusion_types[failure_type] += 1
                excluded_rows.append({
                    "trajectory_id": trajectory_id,
                    "example_index": example_index,
                    "sample_index": sample_index,
                    "reason": "infrastructure_failure",
                    "failure_type": failure_type,
                })
                continue
            converted, reason = normalize_failure_record(
                {**sample, "trajectory_id": trajectory_id},
                task_record(task),
            )
            if converted is None:
                exclusion_types[str(reason)] += 1
                excluded_rows.append({
                    "trajectory_id": trajectory_id,
                    "example_index": example_index,
                    "sample_index": sample_index,
                    "reason": reason,
                    "failure_type": sample.get("failure_type"),
                })
                continue
            normalized_rows.append(converted)
            recorded_correct[trajectory_id] = bool(sample.get("correct"))

    positive_formula_exact = True
    penalty_formula_exact = True
    reward_sign_valid = True
    replay_agreement = True
    feature_hits: collections.Counter[str] = collections.Counter()
    step_signs: collections.Counter[str] = collections.Counter()
    for trajectory in normalized_rows:
        result = score_rollout_trajectory(
            trajectory,
            config,
            denotation_comparison="bird-set",
        )
        payload = result.to_dict()
        scored_rows.append(payload)
        replay_agreement &= result.correct == recorded_correct[result.trajectory_id]
        reward_sign_valid &= (
            result.total_reward > 0 if result.correct else result.total_reward <= 0
        )
        for step in payload["steps"]:
            positive_formula_exact &= abs(
                float(step["g_positive"]) - expected_positive(step, config)
            ) < 1e-8
            penalty_formula_exact &= (
                abs(float(step["p_local"]) - expected_local_penalty(step, config)) < 1e-8
                and abs(float(step["p_outcome"])) < 1e-8
            )
            feature = step["features"]
            for name in (
                "back_slice",
                "search_reduction",
                "tool_error",
                "adjacent_repeat",
                "repeated_call",
                "legal_no_state_change",
            ):
                feature_hits[name] += float(feature.get(name, 0.0)) > 0
            if step["reward"] > 0:
                step_signs["positive"] += 1
            elif step["reward"] < 0:
                step_signs["negative"] += 1
            else:
                step_signs["zero"] += 1

    normalized_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in normalized_rows),
        encoding="utf-8",
    )
    scored_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored_rows),
        encoding="utf-8",
    )
    excluded_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in excluded_rows),
        encoding="utf-8",
    )

    correct_rows = [row for row in scored_rows if row["correct"]]
    deterministic_complete = sum(
        bool(row["diagnostics"].get("deterministic_grounding_complete"))
        for row in correct_rows
    )
    nonadjacent_repeats = sum(
        bool(step["features"].get("repeated_call"))
        and not bool(step["features"].get("adjacent_repeat"))
        for row in scored_rows
        for step in row["steps"]
    )
    nonadjacent_repeat_penalty_leaks = sum(
        bool(step["features"].get("repeated_call"))
        and not bool(step["features"].get("adjacent_repeat"))
        and not bool(step["features"].get("tool_error"))
        and not bool(step["features"].get("legal_no_state_change"))
        and float(step["p_local"]) != 0.0
        for row in scored_rows
        for step in row["steps"]
    )
    summary = {
        "passk_input": str(args.passk_all.resolve()),
        "task_count": len(rows),
        "attempted_samples": sum(len(row.get("samples") or []) for row in rows),
        "scored_samples": len(scored_rows),
        "excluded_infrastructure_samples": len(excluded_rows),
        "exclusion_types": dict(sorted(exclusion_types.items())),
        "recorded_correct": sum(recorded_correct.values()),
        "replay_correct": len(correct_rows),
        "recorded_replay_correctness_agreement": replay_agreement,
        "config": config.__dict__,
        "feature_hits": dict(sorted(feature_hits.items())),
        "step_reward_signs": dict(sorted(step_signs.items())),
        "nonadjacent_repeats": nonadjacent_repeats,
        "nonadjacent_repeat_penalty_leaks": nonadjacent_repeat_penalty_leaks,
        "correct_deterministic_grounding_complete": deterministic_complete,
        "correct_deterministic_grounding_total": len(correct_rows),
        "gates": {
            "positive_formula_exact": positive_formula_exact,
            "penalty_formula_exact": penalty_formula_exact,
            "reward_sign_valid": reward_sign_valid,
            "recorded_replay_correctness_agreement": replay_agreement,
            "nonadjacent_repeat_penalty_zero": nonadjacent_repeat_penalty_leaks == 0,
            "deterministic_grounding_complete": (
                bool(correct_rows) and deterministic_complete == len(correct_rows)
            ),
            "independent_grounding_edge_precision": False,
            "counterfactual_suite_quality": False,
        },
        "training_ready": False,
        "training_ready_reason": (
            "independent grounding-edge precision and task-keyed counterfactual-suite "
            "quality gates have not yet passed for this cohort"
        ),
        "normalized_output": str(normalized_path.resolve()),
        "scored_output": str(scored_path.resolve()),
        "excluded_output": str(excluded_path.resolve()),
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: value
        for key, value in summary.items()
        if key not in {"config", "normalized_output", "scored_output", "excluded_output"}
    }, ensure_ascii=False, indent=2))
    required = (
        positive_formula_exact
        and penalty_formula_exact
        and reward_sign_valid
        and replay_agreement
        and nonadjacent_repeat_penalty_leaks == 0
    )
    return 0 if required else 1


if __name__ == "__main__":
    raise SystemExit(main())
