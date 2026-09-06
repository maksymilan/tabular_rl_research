#!/usr/bin/env python3
"""Audit tool-conditioned reward bias on frozen Atomic version26 rollouts.

This diagnostic deliberately does not train or alter the policy.  It replays recorded rollouts,
uses hidden gold SQL only on the reward side, and compares two allocations over the same
Harness-derived features:

* ``current_multisignal``: the repository's current B/E/search/feedback/target-potential reward;
* ``causal_state_delta``: fixed episode credit allocated only to first gold-state progress that is
  also on the successful terminal dependency/evidence slice.  Failed trajectories retain only a
  small bounded progress term and deterministic local/outcome penalties.

Per-tool raw means are descriptive, not a balancing target.  The diagnostic also subtracts a
state/opportunity baseline that excludes the selected tool.  This tests whether a tool still has a
systematic residual after controlling for correctness, legality, terminal status, remaining target
potential, and relative episode depth.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import copy
import hashlib
import json
import math
import signal
import sqlite3
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src" / "rl")]

import rl.objectives.process_credit as process_credit_module  # noqa: E402
from rl.runtime.external_failure_adapter import normalize_failure_record  # noqa: E402
from rl.objectives.process_credit import (  # noqa: E402
    ProcessRewardConfig,
    StepFeature,
    allocate_process_rewards,
    replay_step_features,
)


INFRASTRUCTURE_FAILURE_TYPES = frozenset(
    {
        "api_error",
        "context_overflow",
        "context_length_exceeded",
        "generation_oom",
        "incomplete_api_response",
        "provider_carrier_error",
        "task_timeout",
        "transport_error",
    }
)


@dataclass(frozen=True)
class CausalStateDeltaAllocation:
    rewards: tuple[float, ...]
    positive_allocations: tuple[float, ...]
    negative_allocations: tuple[float, ...]
    raw_progress: tuple[float, ...]
    eligible_progress: tuple[float, ...]
    positive_mass: float
    raw_penalty_mass: float
    capped_penalty: float
    total_reward: float
    process_update: bool


class EpisodeReplayTimeout(TimeoutError):
    """One reward-side replay exceeded its explicit wall-clock budget."""


@contextmanager
def episode_time_limit(seconds: float | None):
    if seconds is None:
        yield
        return
    if seconds <= 0:
        raise ValueError("episode timeout must be positive")

    def handle_timeout(_signum, _frame):
        raise EpisodeReplayTimeout(f"episode replay exceeded {seconds:g} seconds")

    previous_handler = signal.signal(signal.SIGALRM, handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)


@contextmanager
def replay_time_limit(seconds: float | None):
    """Bound both Python work and long-running SQLite virtual-machine execution."""
    if seconds is None:
        yield
        return
    deadline = time.monotonic() + seconds
    interrupted = {"value": False}
    original_harness = process_credit_module.Harness

    class DeadlineHarness(original_harness):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            def stop_after_deadline() -> int:
                expired = time.monotonic() >= deadline
                interrupted["value"] = interrupted["value"] or expired
                return int(expired)

            self.conn.set_progress_handler(stop_after_deadline, 10_000)

    process_credit_module.Harness = DeadlineHarness
    try:
        with episode_time_limit(seconds + 1.0):
            try:
                yield
            except sqlite3.OperationalError as exc:
                if interrupted["value"] and "interrupt" in str(exc).casefold():
                    raise EpisodeReplayTimeout(
                        f"episode SQLite replay exceeded {seconds:g} seconds"
                    ) from exc
                raise
    finally:
        process_credit_module.Harness = original_harness


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_process_config(path: Path) -> ProcessRewardConfig:
    values = {
        key: value
        for key, value in json.loads(path.read_text(encoding="utf-8")).items()
        if not key.startswith("_")
    }
    config = ProcessRewardConfig(**values)
    config.validate()
    return config


def _local_penalty(feature: StepFeature, config: ProcessRewardConfig) -> float:
    return (
        config.lambda_tool_error * max(feature.tool_error, float(bool(feature.error_type)))
        + config.lambda_adjacent_repeat * float(feature.adjacent_repeat)
        + config.lambda_repeat_without_feedback * feature.repeat_without_feedback
        + config.lambda_legal_no_state_change * feature.legal_no_state_change
        + config.lambda_ignored_feedback * feature.ignored_feedback
        + config.lambda_unsupported_guess * feature.unsupported_guess
        + config.lambda_empty_result * feature.empty_result_penalty
    )


def allocate_causal_state_delta(
    features: list[StepFeature],
    *,
    correct: bool,
    config: ProcessRewardConfig,
) -> CausalStateDeltaAllocation:
    """Allocate fixed-mass credit from tool-independent state progress.

    A successful step receives positive credit only when its hidden target-state delta is supported
    by the terminal backward/evidence slice.  This prevents an irrelevant gold-table inspection
    from earning reward merely because it happens to match one gold SQL label.  On failures the
    causal slice is unavailable, so bounded target progress is retained only through ``eta``.
    """
    if not features:
        raise ValueError("cannot allocate reward over an empty trajectory")
    config.validate()

    working = copy.deepcopy(features)
    for feature in working:
        feature.tool_error = max(feature.tool_error, float(bool(feature.error_type)))
        feature.terminal_failure = float(not correct and feature is working[-1])
        feature.target_potential_delta = (
            config.omega_target_table * feature.target_table_delta
            + config.omega_target_column * feature.target_column_delta
            + config.omega_target_row * feature.target_row_delta
        )

    raw_progress = [max(0.0, feature.target_potential_delta) for feature in working]
    eligible_progress = [
        progress
        * float(
            not correct
            or feature.back_slice > 0
            or feature.new_used_evidence > 0
            or feature.is_terminal
        )
        for feature, progress in zip(working, raw_progress, strict=True)
    ]
    positive_mass = sum(eligible_progress) if correct else 0.0
    positive_allocations = (
        [value / positive_mass for value in eligible_progress]
        if positive_mass > 0
        else [0.0] * len(working)
    )

    local_penalties = [_local_penalty(feature, config) for feature in working]
    outcome_penalties = [
        config.lambda_terminal_failure * feature.terminal_failure
        for feature in working
    ]
    penalties = [
        local + outcome
        for local, outcome in zip(local_penalties, outcome_penalties, strict=True)
    ]
    raw_penalty_mass = sum(penalties)
    capped_penalty = min(config.penalty_cap, raw_penalty_mass)
    negative_allocations = (
        [value / raw_penalty_mass for value in penalties]
        if raw_penalty_mass > 0
        else [0.0] * len(working)
    )
    rewards = [
        float(correct) * positive
        + float(not correct) * config.eta_failure_progress * progress
        - capped_penalty * negative
        for progress, positive, negative in zip(
            raw_progress,
            positive_allocations,
            negative_allocations,
            strict=True,
        )
    ]
    total_reward = sum(rewards)
    process_update = bool(not correct or positive_mass > 0)

    if correct and process_update and total_reward <= 0:
        raise AssertionError("correct causal-state trajectory must retain positive total reward")
    if not correct and total_reward > 1e-9:
        raise AssertionError("failed causal-state trajectory cannot receive positive total reward")
    if correct and positive_mass > 0 and not math.isclose(
        sum(positive_allocations), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise AssertionError("correct causal-state positive mass must normalize to one")

    return CausalStateDeltaAllocation(
        rewards=tuple(rewards),
        positive_allocations=tuple(positive_allocations),
        negative_allocations=tuple(negative_allocations),
        raw_progress=tuple(raw_progress),
        eligible_progress=tuple(eligible_progress),
        positive_mass=positive_mass,
        raw_penalty_mass=raw_penalty_mass,
        capped_penalty=capped_penalty,
        total_reward=total_reward,
        process_update=process_update,
    )


def task_record(row: dict[str, Any], database_root: Path) -> dict[str, Any]:
    db_id = str(row["db_id"])
    return {
        "example_id": f"bird_dev_{int(row['example_index']):05d}",
        "dataset": "bird-sql",
        "split": "dev",
        "db_id": db_id,
        "db_path": str(database_root / db_id / f"{db_id}.sqlite"),
        "question": row["question"],
        "gold_sql": row["gold_sql"],
        "external_knowledge": row.get("evidence") or row.get("external_knowledge"),
        "denotation_comparison": "bird-set",
    }


def semantic_sample(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    samples = row.get("samples") or []
    if len(samples) != 1:
        raise ValueError(
            f"example_index={row.get('example_index')} must contain exactly one sample"
        )
    sample = samples[0]
    failure_type = sample.get("failure_type")
    if failure_type in INFRASTRUCTURE_FAILURE_TYPES:
        return None, f"infrastructure_failure:{failure_type}"
    trajectory_id = f"bird_dev_{int(row['example_index']):05d}_sample_0"
    return {**sample, "trajectory_id": trajectory_id}, None


def _feature_from_reward_step(step: dict[str, Any]) -> StepFeature:
    allowed = set(StepFeature.__dataclass_fields__)
    payload = {
        key: value
        for key, value in (step.get("features") or {}).items()
        if key in allowed
    }
    payload.setdefault("action_index", int(step["action_index"]))
    payload.setdefault("step_id", str(step["step_id"]))
    payload.setdefault("tool", step.get("tool"))
    payload.setdefault("legal_success", bool(payload.get("legal_success")))
    return StepFeature(**payload)


def _progress_bin(remaining: float) -> int:
    return min(4, max(0, int(math.floor(min(0.999999999, max(0.0, remaining)) * 5))))


def _depth_bin(step_index: int, step_count: int) -> int:
    return min(3, int(4 * step_index / max(1, step_count)))


def build_step_rows(
    *,
    trajectory_id: str,
    correct: bool,
    current_steps: list[dict[str, Any]],
    causal: CausalStateDeltaAllocation,
) -> list[dict[str, Any]]:
    if len(current_steps) != len(causal.rewards):
        raise ValueError("current and causal allocations do not align")
    progress_before = 0.0
    rows = []
    for index, (step, causal_reward) in enumerate(
        zip(current_steps, causal.rewards, strict=True)
    ):
        feature = step["features"]
        remaining = max(0.0, 1.0 - progress_before)
        bucket = (
            bool(correct),
            bool(feature.get("legal_success")),
            bool(feature.get("is_terminal")),
            _progress_bin(remaining),
            _depth_bin(index, len(current_steps)),
        )
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "correct": bool(correct),
                "step_index": index,
                "step_count": len(current_steps),
                "tool": step.get("tool") or "__unparsed__",
                "bucket": bucket,
                "current_multisignal": float(step["reward"]),
                "causal_state_delta": float(causal_reward),
                "target_potential_delta": float(
                    causal.raw_progress[index]
                ),
                "eligible_target_potential_delta": float(
                    causal.eligible_progress[index]
                ),
                "current_positive_allocation": float(step["c_positive"]),
                "causal_positive_allocation": float(
                    causal.positive_allocations[index]
                ),
                "local_penalty": float(step["p_local"]),
                "legal_success": bool(feature.get("legal_success")),
                "is_terminal": bool(feature.get("is_terminal")),
                "back_slice": float(feature.get("back_slice", 0.0)),
                "new_used_evidence": float(
                    feature.get("new_used_evidence", 0.0)
                ),
                "progress_before": progress_before,
                "remaining_potential": remaining,
            }
        )
        progress_before = min(
            1.0,
            progress_before + float(causal.raw_progress[index]),
        )
    return rows


def score_row(
    row: dict[str, Any],
    *,
    database_root: Path,
    config: ProcessRewardConfig,
    episode_timeout_seconds: float | None,
) -> dict[str, Any]:
    sample, excluded = semantic_sample(row)
    if sample is None:
        return {"excluded": str(excluded)}
    task = task_record(row, database_root)
    db_path = Path(task["db_path"])
    if not db_path.is_file():
        raise FileNotFoundError(
            f"missing database for example_index={row['example_index']}: {db_path}"
        )
    normalized, reason = normalize_failure_record(sample, task)
    if normalized is None:
        return {"excluded": str(reason)}

    try:
        with replay_time_limit(episode_timeout_seconds):
            features, diagnostics = replay_step_features(
                normalized,
                denotation_comparison="bird-set",
            )
            replay_correct = bool(diagnostics["replay_correct"])
            recorded_correct = bool(sample.get("correct"))
            current = allocate_process_rewards(
                str(normalized["trajectory_id"]),
                copy.deepcopy(features),
                correct=replay_correct,
                config=config,
                diagnostics=diagnostics,
            )
            causal = allocate_causal_state_delta(
                features,
                correct=replay_correct,
                config=config,
            )
            current_payload = current.to_dict()
            rows = build_step_rows(
                trajectory_id=str(normalized["trajectory_id"]),
                correct=replay_correct,
                current_steps=current_payload["steps"],
                causal=causal,
            )
    except EpisodeReplayTimeout:
        return {
            "excluded": "episode_replay_timeout",
            "timeout_example_index": int(row["example_index"]),
            "timeout_db_id": str(row["db_id"]),
        }

    disagreement = None
    if replay_correct != recorded_correct:
        disagreement = {
            "example_index": int(row["example_index"]),
            "trajectory_id": normalized["trajectory_id"],
            "recorded_correct": recorded_correct,
            "replay_correct": replay_correct,
            "recorded_failure_type": sample.get("failure_type"),
        }
    return {
        "step_rows": rows,
        "replay_correct": replay_correct,
        "recorded_correct": recorded_correct,
        "disagreement": disagreement,
        "current_total": float(current.total_reward),
        "causal_total": float(causal.total_reward),
        "causal_process_update": bool(causal.process_update),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def summarize_variant(
    step_rows: list[dict[str, Any]],
    reward_key: str,
    positive_key: str,
) -> dict[str, Any]:
    bucket_values: dict[tuple[Any, ...], list[float]] = collections.defaultdict(list)
    for row in step_rows:
        bucket_values[row["bucket"]].append(float(row[reward_key]))
    bucket_means = {key: _mean(values) for key, values in bucket_values.items()}

    by_tool: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    total_positive = sum(float(row[positive_key]) for row in step_rows)
    for row in step_rows:
        residual = float(row[reward_key]) - bucket_means[row["bucket"]]
        by_tool[str(row["tool"])].append({**row, "residual": residual})

    tools: dict[str, Any] = {}
    total_steps = len(step_rows)
    for tool in sorted(by_tool):
        rows = by_tool[tool]
        count = len(rows)
        positive_mass = sum(float(row[positive_key]) for row in rows)
        tools[tool] = {
            "steps": count,
            "step_share": count / total_steps if total_steps else 0.0,
            "episodes": len({row["trajectory_id"] for row in rows}),
            "correct_steps": sum(bool(row["correct"]) for row in rows),
            "mean_reward": _mean(float(row[reward_key]) for row in rows),
            "mean_reward_correct": _mean(
                float(row[reward_key]) for row in rows if row["correct"]
            ),
            "mean_reward_failure": _mean(
                float(row[reward_key]) for row in rows if not row["correct"]
            ),
            "mean_state_conditioned_residual": _mean(
                float(row["residual"]) for row in rows
            ),
            "positive_mass": positive_mass,
            "positive_mass_share": (
                positive_mass / total_positive if total_positive > 0 else 0.0
            ),
            "positive_mass_to_step_share": (
                (positive_mass / total_positive) / (count / total_steps)
                if total_positive > 0 and count > 0 and total_steps > 0
                else 0.0
            ),
            "positive_steps": sum(float(row[reward_key]) > 0 for row in rows),
            "negative_steps": sum(float(row[reward_key]) < 0 for row in rows),
            "zero_steps": sum(
                math.isclose(float(row[reward_key]), 0.0, abs_tol=1e-12)
                for row in rows
            ),
            "target_progress_steps": sum(
                float(row["target_potential_delta"]) > 0 for row in rows
            ),
            "eligible_target_progress_steps": sum(
                float(row["eligible_target_potential_delta"]) > 0 for row in rows
            ),
        }

    stable = [payload for payload in tools.values() if payload["steps"] >= 20]
    raw_means = [float(payload["mean_reward"]) for payload in stable]
    residual_means = [
        float(payload["mean_state_conditioned_residual"])
        for payload in stable
    ]
    return {
        "reward_key": reward_key,
        "steps": total_steps,
        "state_opportunity_buckets": len(bucket_values),
        "bucket_definition": [
            "trajectory_correct",
            "legal_success",
            "is_terminal",
            "remaining_target_potential_quintile",
            "relative_depth_quartile",
        ],
        "baseline_excludes_selected_tool": True,
        "tools": tools,
        "stable_tool_min_steps": 20,
        "raw_tool_mean_range": (
            max(raw_means) - min(raw_means) if raw_means else 0.0
        ),
        "state_conditioned_residual_mean_range": (
            max(residual_means) - min(residual_means)
            if residual_means
            else 0.0
        ),
        "max_positive_mass_to_step_share": max(
            (
                float(payload["positive_mass_to_step_share"])
                for payload in stable
            ),
            default=0.0,
        ),
    }


def audit(
    *,
    eval_jsonl: Path,
    database_root: Path,
    config: ProcessRewardConfig,
    start_index: int = 0,
    limit: int | None = None,
    workers: int = 1,
    episode_timeout_seconds: float | None = 30.0,
) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    step_rows: list[dict[str, Any]] = []
    exclusions: collections.Counter[str] = collections.Counter()
    replay_disagreements: list[dict[str, Any]] = []
    episode_counts: collections.Counter[str] = collections.Counter()
    causal_totals: list[float] = []
    causal_outcomes: list[bool] = []
    current_totals: list[float] = []
    causal_g0_correct = 0
    timeout_examples: list[dict[str, Any]] = []

    def consume(result: dict[str, Any]) -> None:
        nonlocal causal_g0_correct
        episode_counts["processed"] += 1
        excluded = result.get("excluded")
        if excluded:
            exclusions[str(excluded)] += 1
            if excluded == "episode_replay_timeout":
                timeout_examples.append(
                    {
                        "example_index": result["timeout_example_index"],
                        "db_id": result["timeout_db_id"],
                    }
                )
            return
        step_rows.extend(result["step_rows"])
        replay_correct = bool(result["replay_correct"])
        recorded_correct = bool(result["recorded_correct"])
        disagreement = result.get("disagreement")
        if disagreement is not None:
            replay_disagreements.append(disagreement)
        current_totals.append(float(result["current_total"]))
        causal_totals.append(float(result["causal_total"]))
        causal_outcomes.append(replay_correct)
        episode_counts["scored"] += 1
        episode_counts["replay_correct" if replay_correct else "replay_failure"] += 1
        episode_counts["recorded_correct" if recorded_correct else "recorded_failure"] += 1
        if replay_correct and not bool(result["causal_process_update"]):
            causal_g0_correct += 1

    def selected_rows() -> Iterable[dict[str, Any]]:
        with eval_jsonl.open("r", encoding="utf-8") as handle:
            for row_index, line in enumerate(handle):
                if row_index < start_index:
                    continue
                if limit is not None and episode_counts["input"] >= limit:
                    break
                if not line.strip():
                    continue
                episode_counts["input"] += 1
                yield json.loads(line)

    if workers == 1:
        for row in selected_rows():
            consume(
                score_row(
                    row,
                    database_root=database_root,
                    config=config,
                    episode_timeout_seconds=episode_timeout_seconds,
                )
            )
            if episode_counts["processed"] % 100 == 0:
                print(
                    json.dumps(
                        {
                            "processed": episode_counts["processed"],
                            "scored": episode_counts["scored"],
                            "excluded": sum(exclusions.values()),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
    else:
        max_pending = workers * 2
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            pending: set[concurrent.futures.Future] = set()
            for row in selected_rows():
                pending.add(
                    executor.submit(
                        score_row,
                        row,
                        database_root=database_root,
                        config=config,
                        episode_timeout_seconds=episode_timeout_seconds,
                    )
                )
                if len(pending) < max_pending:
                    continue
                done, pending = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    consume(future.result())
                if episode_counts["processed"] // 100 != (
                    episode_counts["processed"] - len(done)
                ) // 100:
                    print(
                        json.dumps(
                            {
                                "processed": episode_counts["processed"],
                                "scored": episode_counts["scored"],
                                "excluded": sum(exclusions.values()),
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
            for future in concurrent.futures.as_completed(pending):
                consume(future.result())

    current_summary = summarize_variant(
        step_rows,
        "current_multisignal",
        "current_positive_allocation",
    )
    causal_summary = summarize_variant(
        step_rows,
        "causal_state_delta",
        "causal_positive_allocation",
    )
    return {
        "schema_version": "atomic-v26-gold-state-reward-bias-audit-v1",
        "input": {
            "eval_jsonl": str(eval_jsonl.resolve()),
            "eval_jsonl_sha256": sha256_file(eval_jsonl),
            "database_root": str(database_root.resolve()),
            "start_index": start_index,
            "limit": limit,
            "workers": workers,
            "episode_timeout_seconds": episode_timeout_seconds,
        },
        "reward_contract": {
            "gold_visible_to_actor": False,
            "state_progress": "first normalized gold table/column/result-row support",
            "success_gate": "terminal backward slice or used evidence",
            "correct_positive_budget": "one normalized unit per eligible episode",
            "failure_progress_cap": config.eta_failure_progress,
            "failure_terminal_penalty": config.lambda_terminal_failure,
            "tool_identity_used_as_reward_feature": False,
            "per_tool_mean_centering_used_for_reward": False,
        },
        "config": asdict(config),
        "episodes": dict(sorted(episode_counts.items())),
        "excluded": dict(sorted(exclusions.items())),
        "episode_replay_timeout_examples": timeout_examples[:100],
        "replay_correctness": {
            "agreement": not replay_disagreements,
            "disagreement_count": len(replay_disagreements),
            "examples": replay_disagreements[:20],
        },
        "causal_state_delta": {
            "correct_g0_excluded": causal_g0_correct,
            "all_failure_totals_nonpositive": all(
                correct or total <= 1e-9
                for correct, total in zip(
                    causal_outcomes,
                    causal_totals,
                    strict=True,
                )
            ),
            "mean_episode_reward": _mean(causal_totals),
        },
        "current_multisignal": {
            "mean_episode_reward": _mean(current_totals),
        },
        "tool_bias_audit": {
            "current_multisignal": current_summary,
            "causal_state_delta": causal_summary,
            "interpretation": (
                "Raw per-tool means need not be equal. The state-conditioned residual is the "
                "relevant diagnostic; it uses a baseline independent of the selected tool."
            ),
        },
        "training_ready": False,
        "training_ready_reason": (
            "This is a reward-distribution diagnostic on greedy dev trajectories. A fresh "
            "BIRD-train K-way rollout pool, counterfactual completeness gate, and matched "
            "result-only control are still required before optimization."
        ),
    }
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-jsonl", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument(
        "--config-json",
        type=Path,
        default=ROOT / "src" / "rl" / "configs" / "atomic_process_reward.json",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--episode-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.output_json.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_json}")
    summary = audit(
        eval_jsonl=args.eval_jsonl,
        database_root=args.database_root,
        config=load_process_config(args.config_json),
        start_index=args.start_index,
        limit=args.limit,
        workers=args.workers,
        episode_timeout_seconds=args.episode_timeout_seconds,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
