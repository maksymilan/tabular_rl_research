#!/usr/bin/env python3
"""Audit every frozen BIRD-dev error against the existing BIRD-train error inventory.

This is a read-only, post-hoc evaluation diagnostic.  It never exports BIRD-dev records to SFT or
RL, never changes a reward, and never calls a model.  Each wrong dev episode receives an auditable
record built from:

* the recorded terminal/failure state and Harness error events;
* deterministic relation-derivation metadata from successful tool calls;
* Gold-SQL semantic obligations used only on the evaluator side; and
* the already frozen train-error inventory, used only to measure support for the same signature.

The script deliberately distinguishes three statements:

1. ``exact_signature_seen``: the same terminal/semantic signature exists in train errors;
2. ``broad_family_seen_only``: the broad mechanism exists, but not the exact signature;
3. ``contributing_event_seen_only``: the terminal cause is new, but a preceding tool-error event
   exists in train; and
4. ``not_observed_in_train``: neither terminal, broad-family, nor event support exists.

None of these labels says that a dev item may be used for training.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from rl.diagnostics.io import read_jsonl as _diagnostic_read_jsonl
from rl.diagnostics.io import sha256_file as _diagnostic_sha256_file


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "src" / "rl")]

from rl.runtime.external_failure_adapter import normalize_failure_record  # noqa: E402
from rl.scenarios.diagnostics.analyze_all_incorrect_trajectory_patterns import (  # noqa: E402
    TAIL_CATEGORIES,
    classify_incorrect,
    gold_sql_shape,
    question_shape,
    semantic_features,
    trajectory_features,
)
from rl.diagnostics.semantic.tool_state_obligation_quality import score_tool_state_trajectory  # noqa: E402
from rl.diagnostics.semantic.trajectory_semantic_ranker import load_sqlite_columns  # noqa: E402


EXPECTED_DEV_TOTAL = 1534
EXPECTED_DEV_CORRECT = 838
EXPECTED_DEV_INCORRECT = 696


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Compatibility generator backed by the shared diagnostics parser."""
    yield from (dict(row) for row in _diagnostic_read_jsonl(path))


def sha256_file(path: Path) -> str:
    """Compatibility export backed by :mod:`rl.diagnostics.io`."""
    return _diagnostic_sha256_file(path)


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _sample(row: Mapping[str, Any]) -> dict[str, Any]:
    samples = row.get("samples") or []
    if len(samples) != 1:
        raise ValueError(
            f"example_index={row.get('example_index')} must contain exactly one sample"
        )
    return dict(samples[0])


def _task(row: Mapping[str, Any], database_root: Path) -> dict[str, Any]:
    example_index = int(row["example_index"])
    db_id = str(row["db_id"])
    return {
        "task_id": f"bird_dev_{example_index:05d}",
        "example_id": f"bird_dev_{example_index:05d}",
        "example_index": example_index,
        "dataset": "bird-sql",
        "split": "dev",
        "db_id": db_id,
        "db_path": str(database_root / db_id / f"{db_id}.sqlite"),
        "question": row.get("question"),
        "gold_sql": row.get("gold_sql"),
        "external_knowledge": row.get("evidence") or row.get("external_knowledge"),
        "denotation_comparison": "bird-set",
    }


def _score_record(
    row: Mapping[str, Any],
    sample: dict[str, Any],
    task: dict[str, Any],
    schema_cache: dict[str, dict[str, tuple[str, ...]]],
) -> dict[str, Any] | None:
    record = {**sample, "trajectory_id": f"{task['task_id']}_sample_0"}
    trajectory, _ = normalize_failure_record(record, task)
    if trajectory is None:
        return None
    db_path = str(task["db_path"])
    if db_path not in schema_cache:
        if not Path(db_path).is_file():
            raise FileNotFoundError(db_path)
        schema_cache[db_path] = load_sqlite_columns(db_path)
    quality = score_tool_state_trajectory(
        gold_sql=str(task["gold_sql"] or ""),
        table_columns=schema_cache[db_path],
        steps=trajectory.get("steps") or [],
    )
    return {
        "task_id": task["task_id"],
        "score": quality.score,
        "semantic_eligible": quality.semantic_eligible,
        "c_terminal": quality.c_terminal,
        "c_max": quality.c_max,
        "terminal_evidence": quality.terminal_evidence,
        "max_handle": quality.max_handle,
        "scoring_mode": quality.scoring_mode,
        "reasons": list(quality.reasons),
        "terminal_overlap": quality.terminal_overlap,
        "max_overlap": quality.max_overlap,
    }


def _semantic_signature(cohort: str, tail_mismatches: Iterable[str]) -> str:
    tail = "+".join(tail_mismatches)
    if cohort in {
        "one_tail_near_miss",
        "two_tail_near_miss",
        "route_solved_multi_tail",
    }:
        return f"semantic:{cohort}:{tail}"
    return f"semantic:{cohort}"


def _guard_empty_terminal_overlap(
    semantic: dict[str, Any], score: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Prevent an unresolved terminal handle from looking semantically exact.

    The shared overlap helper treats an absent category as jointly empty.  That convention is
    correct when some other categories establish a compiled artifact, but not when the entire
    terminal overlap is empty and its score is zero (for example, answering from a base table or
    perception step that has no compiled relation artifact).
    """
    if score is None or semantic.get("scoring_mode") != "terminal":
        return semantic
    overlap = score.get("terminal_overlap") or {}
    if float(overlap.get("score") or 0.0) != 0.0 or (overlap.get("per_category") or {}):
        return semantic
    return {
        **semantic,
        "semantic_eligible": False,
        "scoring_mode": "unresolved_terminal_artifact",
        "source_score": None,
        "join_score": None,
        "tail_mismatches": [],
        "tail_mismatch_count": None,
    }


def _audit_signature(
    *,
    failure_type: str,
    legal: bool,
    cohort: str,
    tail_mismatches: Iterable[str],
) -> str:
    if legal and failure_type == "wrong_answer":
        return _semantic_signature(cohort, tail_mismatches)
    return f"terminal:{failure_type}"


def _broad_family(
    *,
    failure_type: str,
    legal: bool,
    cohort: str,
) -> str:
    if not legal or failure_type != "wrong_answer":
        if failure_type in {"protocol_error", "provider_carrier_error"}:
            return "protocol_or_carrier"
        if failure_type == "argument_validation_error":
            return "argument_schema"
        if failure_type in {"execution_error", "nonrecoverable_execution_error"}:
            return "execution"
        if failure_type == "max_steps":
            return "budget_exhaustion"
        if failure_type in {"context_overflow", "context_length_exceeded"}:
            return "context_capacity"
        if failure_type in {"api_error", "transport_error", "task_timeout"}:
            return "infrastructure"
        return "other_nonterminal"
    if cohort in {"wrong_source_or_join_route", "partial_join_route"}:
        return "source_or_join_route"
    if cohort in {
        "one_tail_near_miss",
        "two_tail_near_miss",
        "route_solved_multi_tail",
    }:
        return "tail_semantics"
    if cohort == "semantic_exact_but_wrong":
        return "semantic_exact_denotation_mismatch"
    if cohort == "semantic_unscorable":
        return "semantic_compiler_gap"
    if cohort == "no_grounded_terminal":
        return "missing_grounded_terminal"
    return cohort


def _distance_band(*, legal: bool, cohort: str) -> str:
    """A deterministic structural band, not a human judgment of reasoning quality."""
    if not legal:
        return "no_legal_answer"
    if cohort == "one_tail_near_miss":
        return "local_one_family"
    if cohort == "two_tail_near_miss":
        return "local_two_families"
    if cohort in {"route_solved_multi_tail", "partial_join_route"}:
        return "multi_decision"
    if cohort == "wrong_source_or_join_route":
        return "route_level"
    if cohort == "semantic_exact_but_wrong":
        return "scorer_blind_spot_or_denotation_detail"
    return "not_determined"


def _compact_overlap(overlap: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for category, payload in (overlap.get("per_category") or {}).items():
        result[str(category)] = {
            "score": payload.get("score"),
            "gold": payload.get("gold"),
            "candidate": payload.get("candidate"),
            "matched": payload.get("matched"),
            "missing_count": len(payload.get("missing") or []),
            "extra_count": len(payload.get("extra") or []),
            "missing": list(payload.get("missing") or []),
            "extra": list(payload.get("extra") or []),
        }
    return result


def _error_events(sample: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for event in sample.get("error_events") or []:
        message = str(event.get("message") or "")
        result.append({
            "action_index": event.get("action_index"),
            "error_type": event.get("error_type"),
            "attempted_tool": event.get("attempted_tool"),
            "attempted_arguments": event.get("attempted_arguments"),
            "message": message,
            "message_family": _error_message_family(message),
        })
    return result


def _error_message_family(message: str) -> str:
    text = str(message or "").casefold()
    if "expected exactly one <think>" in text:
        return "carrier_shape"
    if 'action must contain exactly "tool" and "arguments" keys' in text:
        return "action_extra_or_missing_keys"
    if "action is not valid json" in text:
        return "invalid_json"
    if "read_subtable" in text and "unexpected arguments ['offset']" in text:
        return "unsupported_read_pagination"
    if "read_subtable" in text and "limit must be an integer" in text:
        return "invalid_read_limit"
    if "join_tables" in text and "right must be a bare column" in text:
        return "join_right_column_schema"
    if "join_tables" in text and "relation namespaces must be unique" in text:
        return "repeated_relation_role_missing"
    if "join_tables" in text and ("on must be []" in text or "equality edge" in text):
        return "join_edge_schema"
    if "scalargroundingerror" in text or "value_ref must cite" in text:
        return "scalar_grounding"
    if "no such column" in text or "unknown_column" in text:
        return "unknown_column"
    if "unexpected arguments" in text:
        return "unexpected_tool_arguments"
    if "no_progress" in text or "identical" in text and "action" in text:
        return "repeated_no_progress_action"
    if "operationalerror" in text:
        return "other_sql_execution"
    if "protocolerror" in text:
        return "other_protocol_validation"
    return "other_error"


def _answer_preview(sample: Mapping[str, Any]) -> dict[str, Any]:
    gold = sample.get("gold_sample")
    pred = sample.get("pred_sample")
    return {
        "gold_available": isinstance(gold, list),
        "pred_available": isinstance(pred, list),
        "gold_preview_rows": len(gold) if isinstance(gold, list) else None,
        "pred_preview_rows": len(pred) if isinstance(pred, list) else None,
        "preview_exact_equal": (
            gold == pred if isinstance(gold, list) and isinstance(pred, list) else None
        ),
        "note": "stored samples are previews and may be truncated; equality is not correctness",
    }


def _case_note(
    *,
    failure_type: str,
    legal: bool,
    cohort: str,
    tail_mismatches: list[str],
    overlap: Mapping[str, Any],
    error_events: list[dict[str, Any]],
) -> str:
    if not legal:
        event_types = Counter(str(event.get("error_type") or "unknown") for event in error_events)
        return (
            f"No legal terminal answer: final failure={failure_type}; "
            f"recorded error events={dict(event_types)}."
        )
    categories = overlap.get("per_category") or {}
    mismatched = {
        name: {
            "score": payload.get("score"),
            "missing": len(payload.get("missing") or []),
            "extra": len(payload.get("extra") or []),
        }
        for name, payload in categories.items()
        if payload.get("score") is not None and float(payload["score"]) < 1.0
    }
    return (
        f"Legal wrong answer; cohort={cohort}; tail={tail_mismatches}; "
        f"mismatched obligation counts={mismatched}."
    )


def _train_support(rows: list[dict[str, Any]]) -> dict[str, Any]:
    signatures: Counter[str] = Counter()
    families: Counter[str] = Counter()
    event_episodes: Counter[str] = Counter()
    event_counts: Counter[str] = Counter()
    for row in rows:
        failure_type = str(row.get("failure_type") or "unknown")
        legal = bool(row.get("legal"))
        cohort = str(row.get("analysis_cohort") or "unknown")
        tail = list(row.get("tail_mismatches") or [])
        signatures[_audit_signature(
            failure_type=failure_type,
            legal=legal,
            cohort=cohort,
            tail_mismatches=tail,
        )] += 1
        families[_broad_family(
            failure_type=failure_type,
            legal=legal,
            cohort=cohort,
        )] += 1
        types = [str(value) for value in row.get("error_types") or []]
        event_counts.update(types)
        event_episodes.update(set(types))
    return {
        "signatures": signatures,
        "families": families,
        "event_episodes": event_episodes,
        "event_counts": event_counts,
    }


def _consistency(
    *,
    signature: str,
    family: str,
    failure_type: str,
    error_events: list[dict[str, Any]],
    train: Mapping[str, Counter[str]],
) -> dict[str, Any]:
    exact = int(train["signatures"][signature])
    broad = int(train["families"][family])
    event_types = {str(event.get("error_type") or "unknown") for event in error_events}
    if not event_types and failure_type:
        event_types = {failure_type}
    event_support = {
        error_type: int(train["event_episodes"][error_type])
        for error_type in sorted(event_types)
    }
    if exact:
        label = "exact_signature_seen"
    elif broad:
        label = "broad_family_seen_only"
    elif any(event_support.values()):
        label = "contributing_event_seen_only"
    else:
        label = "not_observed_in_train"
    return {
        "label": label,
        "exact_train_error_count": exact,
        "broad_train_error_count": broad,
        "train_error_episode_support_by_event": event_support,
    }


def _outcome_table(source: Mapping[str, Counter[str]], incorrect_total: int) -> dict[str, Any]:
    result = {}
    for name, counts in sorted(source.items()):
        total = counts["total"]
        result[name] = {
            "tasks": total,
            "correct": counts["correct"],
            "incorrect": counts["incorrect"],
            "accuracy": _rate(counts["correct"], total),
            "share_of_all_incorrect": _rate(counts["incorrect"], incorrect_total),
        }
    return result


def _comparison_table(
    dev_counts: Counter[str],
    train_counts: Mapping[str, int],
    *,
    dev_total: int,
    train_total: int,
) -> dict[str, Any]:
    result = {}
    for key in sorted(set(dev_counts) | set(train_counts)):
        dev_count = int(dev_counts[key])
        train_count = int(train_counts.get(key, 0))
        dev_rate = _rate(dev_count, dev_total)
        train_rate = _rate(train_count, train_total)
        result[key] = {
            "dev_count": dev_count,
            "dev_rate": dev_rate,
            "train_count": train_count,
            "train_rate": train_rate,
            "dev_minus_train_percentage_points": (
                100.0 * (dev_rate - train_rate)
                if dev_rate is not None and train_rate is not None else None
            ),
        }
    return result


def audit(
    *,
    dev_results: Path,
    dev_manifest: Path,
    database_root: Path,
    train_incorrect_rows: Path,
    train_summary_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = json.loads(dev_manifest.read_text(encoding="utf-8"))
    if manifest.get("dataset_purpose") != "evaluation":
        raise ValueError("dev manifest must declare dataset_purpose=evaluation")
    if bool(manifest.get("sft_export_eligible")):
        raise ValueError("dev manifest must be SFT-ineligible")
    if manifest.get("protocol_version") != "version26":
        raise ValueError("expected frozen version26 dev result")
    if manifest.get("assistant_carrier") != "think-json-v1":
        raise ValueError("expected frozen think-json-v1 carrier")
    if int(manifest.get("requested_size", -1)) != EXPECTED_DEV_TOTAL:
        raise ValueError("expected frozen 1534-task dev result")

    train_rows = list(read_jsonl(train_incorrect_rows))
    train_summary = json.loads(train_summary_path.read_text(encoding="utf-8"))
    if len(train_rows) != 564 or train_summary.get("dataset_split") != "train":
        raise ValueError("expected frozen 564-error BIRD-train inventory")
    train_support = _train_support(train_rows)

    schema_cache: dict[str, dict[str, tuple[str, ...]]] = {}
    cases: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    population: Counter[str] = Counter()
    failure_types: Counter[str] = Counter()
    dev_cohorts: Counter[str] = Counter()
    dev_signatures: Counter[str] = Counter()
    dev_families: Counter[str] = Counter()
    consistency_counts: Counter[str] = Counter()
    error_event_counts: Counter[str] = Counter()
    error_event_episodes: Counter[str] = Counter()
    error_message_counts: Counter[str] = Counter()
    error_message_episodes: Counter[str] = Counter()
    sql_shapes: dict[str, Counter[str]] = defaultdict(Counter)
    question_shapes: dict[str, Counter[str]] = defaultdict(Counter)
    database_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    seen_indices: set[int] = set()

    for row in read_jsonl(dev_results):
        example_index = int(row["example_index"])
        if example_index in seen_indices:
            raise ValueError(f"duplicate dev example_index={example_index}")
        seen_indices.add(example_index)
        sample = _sample(row)
        correct = bool(sample.get("correct"))
        db_id = str(row["db_id"])
        outcome = "correct" if correct else "incorrect"
        population["records"] += 1
        population[outcome] += 1
        database_outcomes[db_id]["total"] += 1
        database_outcomes[db_id][outcome] += 1
        if not correct:
            database_outcomes[db_id][
                "legal_wrong" if bool(sample.get("legal")) else "no_legal_answer"
            ] += 1
            database_outcomes[db_id][f"failure:{sample.get('failure_type') or 'unknown'}"] += 1
        sql_shape = gold_sql_shape(str(row.get("gold_sql") or ""))
        q_shape = question_shape(str(row.get("question") or ""))
        for name, present in sql_shape.items():
            if present:
                sql_shapes[name]["total"] += 1
                sql_shapes[name][outcome] += 1
        for name, present in q_shape.items():
            if present:
                question_shapes[name]["total"] += 1
                question_shapes[name][outcome] += 1
        if correct:
            continue

        task = _task(row, database_root)
        score = _score_record(row, sample, task, schema_cache)
        semantic = _guard_empty_terminal_overlap(semantic_features(score), score)
        failure_type = str(sample.get("failure_type") or "unknown")
        legal = bool(sample.get("legal"))
        cohort = classify_incorrect(
            failure_type=failure_type,
            legal=legal,
            semantic=semantic,
        )
        signature = _audit_signature(
            failure_type=failure_type,
            legal=legal,
            cohort=cohort,
            tail_mismatches=semantic["tail_mismatches"],
        )
        family = _broad_family(
            failure_type=failure_type,
            legal=legal,
            cohort=cohort,
        )
        events = _error_events(sample)
        event_types = [str(event.get("error_type") or "unknown") for event in events]
        message_families = [str(event["message_family"]) for event in events]
        consistency = _consistency(
            signature=signature,
            family=family,
            failure_type=failure_type,
            error_events=events,
            train=train_support,
        )
        traj = trajectory_features(sample)
        terminal_overlap = (score or {}).get("terminal_overlap") or {}
        case = {
            "task_id": task["task_id"],
            "example_index": example_index,
            "db_id": task["db_id"],
            "question": task["question"],
            "gold_sql": task["gold_sql"],
            "correct": False,
            "legal": legal,
            "failure_type": failure_type,
            "analysis_cohort": cohort,
            "audit_signature": signature,
            "broad_error_family": family,
            "structural_distance_band": _distance_band(legal=legal, cohort=cohort),
            "semantic_score": semantic["semantic_score"],
            "semantic_scoring_mode": semantic["scoring_mode"],
            "source_score": semantic["source_score"],
            "join_score": semantic["join_score"],
            "tail_mismatches": semantic["tail_mismatches"],
            "tail_mismatch_count": semantic["tail_mismatch_count"],
            "semantic_category_audit": _compact_overlap(terminal_overlap),
            "score_reasons": list((score or {}).get("reasons") or []),
            "terminal_evidence": (score or {}).get("terminal_evidence"),
            "tools": traj["tools"],
            "tool_counts": traj["tool_counts"],
            "turn_count": traj["turn_count"],
            "error_event_count": len(events),
            "error_events": events,
            "answer_preview": _answer_preview(sample),
            "gold_sql_shape": sql_shape,
            "question_shape": q_shape,
            "train_consistency": consistency,
            "audit_note": _case_note(
                failure_type=failure_type,
                legal=legal,
                cohort=cohort,
                tail_mismatches=list(semantic["tail_mismatches"]),
                overlap=terminal_overlap,
                error_events=events,
            ),
            "audit_basis": [
                "recorded_harness_terminal_state",
                "recorded_harness_error_events",
                "relation_derivation_v1",
                "gold_sql_obligations_evaluator_only",
                "frozen_train_error_inventory",
            ],
            "dev_training_use_prohibited": True,
        }
        cases.append(case)
        if score is not None:
            scores.append({
                "example_index": example_index,
                "db_id": task["db_id"],
                "correct": False,
                "legal": legal,
                "failure_type": failure_type,
                **score,
            })
        failure_types[failure_type] += 1
        dev_cohorts[cohort] += 1
        dev_signatures[signature] += 1
        dev_families[family] += 1
        consistency_counts[consistency["label"]] += 1
        error_event_counts.update(event_types)
        error_event_episodes.update(set(event_types))
        error_message_counts.update(message_families)
        error_message_episodes.update(set(message_families))

    expected_population = {
        "records": EXPECTED_DEV_TOTAL,
        "correct": EXPECTED_DEV_CORRECT,
        "incorrect": EXPECTED_DEV_INCORRECT,
    }
    for key, expected in expected_population.items():
        if population[key] != expected:
            raise ValueError(f"expected dev {key}={expected}, found {population[key]}")
    if len(cases) != EXPECTED_DEV_INCORRECT or len(seen_indices) != EXPECTED_DEV_TOTAL:
        raise AssertionError("dev one-row-per-task audit invariant failed")

    train_cohorts = {
        name: int(payload["count"])
        for name, payload in train_summary["cohorts"].items()
    }
    train_failure_types = {
        name: int(count)
        for name, count in train_summary["incorrect_failure_types"].items()
    }
    train_event_counts = {
        name: int(count)
        for name, count in train_summary["process_errors"]["types"].items()
    }
    legal_dev = sum(case["legal"] for case in cases)
    legal_train = int(train_summary["incorrect_legal"]["legal"])
    dev_legal_cohorts = Counter(
        case["analysis_cohort"] for case in cases if case["legal"]
    )
    train_legal_cohorts = Counter(
        str(row["analysis_cohort"]) for row in train_rows if bool(row["legal"])
    )
    dev_tail = Counter(
        category
        for case in cases
        if case["analysis_cohort"] in {"one_tail_near_miss", "two_tail_near_miss"}
        for category in case["tail_mismatches"]
    )
    train_tail = Counter(
        train_summary["near_miss"]["mismatch_category_occurrences"]
    )
    preview_equal_wrong = sum(
        case["answer_preview"]["preview_exact_equal"] is True for case in cases
    )

    summary = {
        "schema_version": "bird-dev-all-error-train-consistency-audit-v1",
        "dataset_boundary": {
            "dev_role": "read_only_post_hoc_evaluation_diagnostic",
            "dev_records_used_for_training": 0,
            "dev_records_used_for_reward_design_or_selection": 0,
            "sft_export_eligible": False,
            "rl_admitted": False,
        },
        "identity": {
            "model": manifest.get("model"),
            "protocol_version": manifest.get("protocol_version"),
            "protocol_hash": manifest.get("protocol_hash"),
            "assistant_carrier": manifest.get("assistant_carrier"),
            "denotation_comparison": manifest.get("denotation_comparison"),
            "temperature": manifest.get("temperature"),
            "top_p": manifest.get("top_p"),
            "max_steps": manifest.get("max_steps"),
        },
        "population": dict(population),
        "per_case_audit": {
            "incorrect_records": len(cases),
            "unique_example_indices": len({case["example_index"] for case in cases}),
            "records_with_semantic_score": sum(case["semantic_score"] is not None for case in cases),
            "records_with_error_events": sum(case["error_event_count"] > 0 for case in cases),
            "wrong_records_with_equal_stored_answer_preview": preview_equal_wrong,
        },
        "main_finding": {
            "same_error_vocabulary": (
                "Most dev failures map to semantic or tool-error mechanisms already observed in "
                "train, but their distribution is not consistent."
            ),
            "distribution_consistent": False,
            "reason": (
                f"No-legal-answer errors are {EXPECTED_DEV_INCORRECT - legal_dev}/"
                f"{EXPECTED_DEV_INCORRECT} on dev versus {564 - legal_train}/564 on train."
            ),
        },
        "train_consistency": {
            "counts": dict(consistency_counts),
            "rates": {
                key: _rate(count, len(cases))
                for key, count in sorted(consistency_counts.items())
            },
            "exact_or_mechanism_seen": sum(
                count for key, count in consistency_counts.items()
                if key != "not_observed_in_train"
            ),
            "not_observed_task_ids": [
                case["task_id"]
                for case in cases
                if case["train_consistency"]["label"] == "not_observed_in_train"
            ],
        },
        "legal_terminal_comparison": {
            "dev": {
                "legal_wrong": legal_dev,
                "no_legal_answer": len(cases) - legal_dev,
                "no_legal_answer_rate": _rate(len(cases) - legal_dev, len(cases)),
            },
            "train": {
                "legal_wrong": legal_train,
                "no_legal_answer": 564 - legal_train,
                "no_legal_answer_rate": _rate(564 - legal_train, 564),
            },
        },
        "failure_type_comparison_all_incorrect": _comparison_table(
            failure_types,
            train_failure_types,
            dev_total=len(cases),
            train_total=564,
        ),
        "semantic_cohort_comparison_all_incorrect": _comparison_table(
            dev_cohorts,
            train_cohorts,
            dev_total=len(cases),
            train_total=564,
        ),
        "semantic_cohort_comparison_conditioned_on_legal_wrong": _comparison_table(
            dev_legal_cohorts,
            train_legal_cohorts,
            dev_total=legal_dev,
            train_total=legal_train,
        ),
        "broad_error_families_dev": dict(dev_families.most_common()),
        "exact_signatures_dev": dict(dev_signatures.most_common()),
        "strict_near_miss_tail_occurrences": {
            "dev": dict(dev_tail.most_common()),
            "train": dict(train_tail.most_common()),
        },
        "process_error_events": {
            "dev_episode_counts": dict(error_event_episodes.most_common()),
            "dev_event_counts": dict(error_event_counts.most_common()),
            "train_event_counts": dict(Counter(train_event_counts).most_common()),
            "dev_message_family_episode_counts": dict(error_message_episodes.most_common()),
            "dev_message_family_event_counts": dict(error_message_counts.most_common()),
            "message_level_train_comparison": "unavailable in compact frozen train inventory",
        },
        "gold_sql_shape_outcomes_dev": _outcome_table(sql_shapes, len(cases)),
        "question_shape_outcomes_dev": _outcome_table(question_shapes, len(cases)),
        "database_outcomes_dev": {
            db_id: {
                **dict(counts),
                "accuracy": _rate(counts["correct"], counts["total"]),
                "no_legal_share_of_incorrect": _rate(
                    counts["no_legal_answer"], counts["incorrect"]
                ),
            }
            for db_id, counts in sorted(database_outcomes.items())
        },
        "limitations": [
            "The audit compares one frozen dev rollout per task with one existing train rollout per task; it does not measure same-task policy frontier probability.",
            "Semantic obligation equality is not denotation equality; semantic_exact_but_wrong remains an explicit blind-spot cohort.",
            "Stored gold/pred samples are previews and may be truncated.",
            "Unscorable means compiler coverage failure, not a model error severity judgment.",
            "No dev label, question, trajectory, signature, or task id may select or shape training data or rewards.",
        ],
        "model_calls": 0,
        "optimizer_updates": 0,
    }
    score_rows = sorted(scores, key=lambda item: int(item["example_index"]))
    cases.sort(key=lambda item: int(item["example_index"]))
    return summary, cases, score_rows


def _markdown_cell(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def write_casebook_index(path: Path, cases: list[dict[str, Any]]) -> None:
    lines = [
        "# BIRD-dev 696 条错误逐题审核索引",
        "",
        "> 只读评测诊断。完整 Gold/终态义务差异与 Harness 错误证据见同目录 JSONL；本索引不得进入训练。",
        "",
        "| 题号 | DB | 合法 | 终止错误 / 语义 cohort | 结构距离 | 尾部差异 | train 对照 | 问题 |",
        "|---:|---|:---:|---|---|---|---|---|",
    ]
    for case in cases:
        consistency = case["train_consistency"]
        comparison = (
            f"{consistency['label']} (exact={consistency['exact_train_error_count']}, "
            f"broad={consistency['broad_train_error_count']})"
        )
        outcome = (
            case["failure_type"] if not case["legal"] else case["analysis_cohort"]
        )
        lines.append(
            "| "
            + " | ".join(
                _markdown_cell(value)
                for value in (
                    case["example_index"],
                    case["db_id"],
                    "是" if case["legal"] else "否",
                    outcome,
                    case["structural_distance_band"],
                    "+".join(case["tail_mismatches"]),
                    comparison,
                    case["question"],
                )
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-results", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--database-root", type=Path, required=True)
    parser.add_argument("--train-incorrect-rows", type=Path, required=True)
    parser.add_argument("--train-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary, cases, scores = audit(
        dev_results=args.dev_results,
        dev_manifest=args.dev_manifest,
        database_root=args.database_root,
        train_incorrect_rows=args.train_incorrect_rows,
        train_summary_path=args.train_summary,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "dev_incorrect_casebook.jsonl").open("w", encoding="utf-8") as target:
        for row in cases:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "dev_incorrect_semantic_scores.jsonl").open("w", encoding="utf-8") as target:
        for row in scores:
            target.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    write_casebook_index(args.output_dir / "dev_incorrect_casebook_index.md", cases)
    generated = [
        "summary.json",
        "dev_incorrect_casebook.jsonl",
        "dev_incorrect_semantic_scores.jsonl",
        "dev_incorrect_casebook_index.md",
    ]
    result_manifest = {
        "schema_version": "bird-dev-error-train-consistency-manifest-v1",
        "analysis_only": True,
        "dataset_split": "dev",
        "dev_records_used_for_training": 0,
        "reward_admitted": False,
        "model_calls": 0,
        "optimizer_updates": 0,
        "input": {
            "dev_results": str(args.dev_results.resolve()),
            "dev_results_sha256": sha256_file(args.dev_results),
            "dev_manifest": str(args.dev_manifest.resolve()),
            "dev_manifest_sha256": sha256_file(args.dev_manifest),
            "train_incorrect_rows": str(args.train_incorrect_rows.resolve()),
            "train_incorrect_rows_sha256": sha256_file(args.train_incorrect_rows),
            "train_summary": str(args.train_summary.resolve()),
            "train_summary_sha256": sha256_file(args.train_summary),
        },
        "outputs": {
            name: sha256_file(args.output_dir / name) for name in generated
        },
        "code": {
            str(Path(__file__).resolve().relative_to(ROOT)): sha256_file(Path(__file__).resolve()),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(result_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
