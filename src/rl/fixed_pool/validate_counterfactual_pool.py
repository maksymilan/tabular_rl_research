#!/usr/bin/env python3
"""Attach deterministic process credit and counterfactual admission to a fixed pool."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT / "src" / "rl"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from counterfactual_suite import load_counterfactual_suite_manifest  # noqa: E402
from external_failure_adapter import normalize_failure_record  # noqa: E402
from frameworks.trl.fixed_rollout_pool import (  # noqa: E402
    deserialize_episode,
    load_rows,
    write_rows_atomic,
)
from frameworks.trl.transition_batch import build_transition_updates  # noqa: E402
from process_credit import ProcessRewardConfig, score_rollout_trajectory  # noqa: E402
from trajectory_replay import evaluate_counterfactual_suite  # noqa: E402


def load_process_config(path: Path) -> ProcessRewardConfig:
    values = {
        key: value
        for key, value in json.loads(path.read_text()).items()
        if not key.startswith("_")
    }
    config = ProcessRewardConfig(**values)
    config.validate()
    return config


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    write_rows_atomic(path, rows)


def validate_one_row(
    row: dict[str, Any],
    *,
    config: ProcessRewardConfig,
    manifest,
    admission_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None, list[str]]:
    """Score/replay one immutable trajectory without shared mutable state."""
    metadata = row["environment"]
    record = row["sample"]["audit_record"]
    normalized, exclusion = normalize_failure_record(
        record,
        {
            "example_id": record["trajectory_id"],
            "dataset": "bird-sql",
            "split": "train",
            "db_id": metadata["db_id"],
            "db_path": metadata.get("db_path"),
            "question": metadata["question"],
            "gold_sql": metadata["gold_sql"],
            "external_knowledge": metadata.get("external_knowledge"),
            "denotation_comparison": "bird-set",
        },
    )
    updated = json.loads(json.dumps(row))
    exclusions = []
    if normalized is None:
        updated["sample"].update(
            {"reward": 0.0, "step_rewards": [], "process_update": False}
        )
        updated["sample"]["audit_record"]["process_reward_exclusion"] = exclusion
        exclusions.append(str(exclusion))
        return updated, [], None, exclusions

    reward = score_rollout_trajectory(
        normalized,
        config,
        denotation_comparison="bird-set",
    )
    step_rewards = [float(step.reward) for step in reward.steps]
    if len(step_rewards) != len(row["policy_turns"]):
        raise RuntimeError(
            f"trajectory {record['trajectory_id']} has {len(row['policy_turns'])} policy "
            f"turns but {len(step_rewards)} process steps"
        )
    process_update = bool(reward.process_update)
    scalar_reward = float(reward.total_reward)
    updated_audit = updated["sample"]["audit_record"]
    updated_audit["process_reward"] = reward.to_dict()
    updated_audit["process_admission_policy"] = admission_mode
    counterfactual_row = None
    if reward.correct and process_update and admission_mode not in {
        "rank-only",
        "dense-outcome",
    }:
        try:
            suite = manifest.suite_for(metadata)
        except ValueError as exc:
            suite = None
            counterfactual = {
                "passed": False,
                "reason": f"suite_unavailable:{exc}",
                "informative_databases": 0,
            }
        if suite is not None:
            completeness = evaluate_counterfactual_suite(
                normalized,
                suite.database_paths,
                min_informative_databases=suite.min_informative_databases,
                denotation_comparison="bird-set",
            )
            counterfactual = completeness.to_dict()
        counterfactual_row = {
            "sequence": row["sequence"],
            "trajectory_id": record["trajectory_id"],
            "task_id": metadata["task_id"],
            "passed": bool(counterfactual["passed"]),
            "result": counterfactual,
        }
        updated_audit["counterfactual_completeness"] = counterfactual
        if not counterfactual["passed"]:
            process_update = False
            scalar_reward = 0.0
            reason = f"counterfactual_completeness:{counterfactual['reason']}"
            updated_audit["process_reward_exclusion"] = reason
            exclusions.append(reason)
    updated["sample"].update(
        {
            "reward": scalar_reward,
            "step_rewards": step_rewards,
            "process_update": process_update,
        }
    )
    feature_rows = []
    for step_index, step in enumerate(reward.steps):
        payload = step.to_dict()
        feature_rows.append(
            {
                "sequence": row["sequence"],
                "trajectory_id": record["trajectory_id"],
                "task_id": metadata["task_id"],
                "example_index": metadata["example_index"],
                "step_index": step_index,
                **payload,
            }
        )
    return updated, feature_rows, counterfactual_row, exclusions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", required=True, type=Path)
    parser.add_argument("--process-reward-config", required=True, type=Path)
    parser.add_argument("--counterfactual-manifest", type=Path)
    parser.add_argument(
        "--admission-mode",
        choices=("rank-only", "dense-outcome", "process-screened", "process-required"),
        default="process-required",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise SystemExit("--workers must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "validated": args.output_dir / "validated_trajectories.jsonl",
        "features": args.output_dir / "process_features.jsonl",
        "counterfactual": args.output_dir / "counterfactual_results.jsonl",
        "transitions": args.output_dir / "transitions.jsonl",
        "summary": args.output_dir / "validation_summary.json",
    }
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError("refusing to overwrite an existing fixed-pool validation artifact")

    config = load_process_config(args.process_reward_config)
    if args.admission_mode in {"rank-only", "dense-outcome"}:
        if args.counterfactual_manifest is not None:
            raise SystemExit(
                f"{args.admission_mode} validation must not consume a counterfactual manifest"
            )
        manifest = None
    else:
        if args.counterfactual_manifest is None:
            raise SystemExit(f"{args.admission_mode} requires --counterfactual-manifest")
        statuses = (
            ("passed", "screening_partial")
            if args.admission_mode == "process-screened"
            else ("passed",)
        )
        manifest = load_counterfactual_suite_manifest(
            args.counterfactual_manifest,
            allowed_quality_statuses=statuses,
        )
    rows = load_rows(args.trajectories)
    validated_rows = []
    feature_rows = []
    counterfactual_rows = []
    excluded = defaultdict(int)
    validate = partial(
        validate_one_row,
        config=config,
        manifest=manifest,
        admission_mode=args.admission_mode,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for updated, features, counterfactual, reasons in executor.map(validate, rows):
            validated_rows.append(updated)
            feature_rows.extend(features)
            if counterfactual is not None:
                counterfactual_rows.append(counterfactual)
            for reason in reasons:
                excluded[reason] += 1

    episodes_by_example = defaultdict(list)
    for row in validated_rows:
        episode = deserialize_episode(row)
        episodes_by_example[int(row["environment"]["example_index"])].append(episode)
    transition_rows = []
    for example_index in sorted(episodes_by_example):
        episodes = sorted(
            episodes_by_example[example_index],
            key=lambda episode: int(episode.sample.audit_record["sample_index"]),
        )
        for update in build_transition_updates(episodes, reward_mode="process"):
            transition_rows.append(
                {
                    "example_index": update.example_index,
                    "trajectory_id": update.trajectory_id,
                    "turn_index": update.turn_index,
                    "prompt_ids": list(update.prompt_ids),
                    "response_ids": list(update.response_ids),
                    "sampling_logprobs": list(update.sampling_logprobs),
                    "advantage": update.advantage,
                    "trajectory_correct": update.trajectory_correct,
                    "legal_success": update.legal_success,
                    "local_penalty": update.local_penalty,
                }
            )

    write_jsonl(outputs["validated"], validated_rows)
    write_jsonl(outputs["features"], feature_rows)
    write_jsonl(outputs["counterfactual"], counterfactual_rows)
    write_jsonl(outputs["transitions"], transition_rows)
    summary = {
        "schema_version": "fixed-pool-process-validation-v1",
        "admission_mode": args.admission_mode,
        "trajectories": len(validated_rows),
        "correct_trajectories": sum(row["sample"]["correct"] for row in validated_rows),
        "process_update_trajectories": sum(
            row["sample"]["process_update"] for row in validated_rows
        ),
        "process_steps": len(feature_rows),
        "transitions": len(transition_rows),
        "counterfactual_correct_trajectories": len(counterfactual_rows),
        "counterfactual_passed": sum(row["passed"] for row in counterfactual_rows),
        "counterfactual_screened": sum(not row["passed"] for row in counterfactual_rows),
        "excluded": dict(sorted(excluded.items())),
    }
    outputs["summary"].write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    if (
        args.admission_mode == "process-required"
        and summary["counterfactual_passed"] != summary["correct_trajectories"]
    ):
        raise SystemExit("not every correct fixed-pool trajectory passed counterfactual replay")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
