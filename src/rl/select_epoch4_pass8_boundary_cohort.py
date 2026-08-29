#!/usr/bin/env python3
"""Select an Epoch-4 K=8 policy-boundary cohort from a completed pass@8 screen.

This is a diagnostic, identity-bound selector for the current Epoch-4 RL screen.  It
does *not* turn correct samples into an offline SFT set: result-only RL consumes a
fresh K=8 group for every selected task.  A task is boundary-eligible only when its
eight samples contain both rewards (1 <= correct_count <= 7); the core boundary is
2 <= correct_count <= 6.  Gold SQL is used only for a deterministic SQL-structure
proxy and is never copied into the manifest or model-visible metadata.

The frozen historical selector in ``select_policy_boundary_grpo_tasks.py`` remains
unchanged.  This module is intentionally separate so an Epoch-4/expanded-data
screen cannot silently become the formal version26 RL cohort.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import sqlglot
from sqlglot import exp

try:
    from harness.sql_task_coverage_profile import profile_identity, profile_sql
except ImportError:  # pragma: no cover - the standalone fallback remains usable
    profile_identity = None
    profile_sql = None


SCHEMA_VERSION = "epoch4-pass8-boundary-cohort-v1"
GROUP_SIZE = 8
EXPECTED_PROTOCOL_VERSION = "version26"
EXPECTED_TOOL_SCHEME = "atomic"
EXPECTED_CARRIER = "think-json-v1"
EXPECTED_REGISTRY_VERSION = "tool-scheme-registry-v2"
INFRA_FAILURE_TYPES = frozenset(
    {
        "api_error",
        "context_overflow",
        "context_length_exceeded",
        "generation_oom",
        "incomplete_api_response",
        "provider_carrier_error",
        "task_timeout",
        "transport_error",
        "runner_error",
    }
)
SOURCE_PREFIXES = (("bird_", "BIRD"), ("spider_", "Spider"), ("synsql_", "SynSQL"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def task_id(row: Mapping[str, Any]) -> str:
    value = row.get("example_id") or row.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("task is missing example_id/instance_id")
    return value


def source_name(identifier: str) -> str:
    for prefix, name in SOURCE_PREFIXES:
        if identifier.startswith(prefix):
            return name
    return "Other"


def stable_rank(seed: str, namespace: str, identifier: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{identifier}".encode()).hexdigest()


def _count(tree: exp.Expression, node_type: type[exp.Expression]) -> int:
    return sum(1 for _ in tree.find_all(node_type))


def _max_depth(tree: exp.Expression) -> int:
    def depth(node: exp.Expression) -> int:
        children = [child for child in node.iter_expressions()]
        return 1 + max((depth(child) for child in children), default=0)

    return max(0, depth(tree) - 1)


def sql_structure(sql: str) -> dict[str, Any]:
    """Compile SQL into a content-free structural feature record.

    The feature names deliberately mirror the existing BIRD SQL complexity proxy,
    with a few additional executable-shape indicators.  The SQL text itself is
    never returned by this function.
    """

    raw = str(sql or "").strip()
    if not raw:
        return {"parse_status": "missing", "bin": "unknown", "score": None}
    try:
        tree = sqlglot.parse_one(raw, read="sqlite")
    except Exception as exc:  # dialect-specific SQL stays eligible but is marked unknown
        return {
            "parse_status": "error",
            "bin": "unknown",
            "score": None,
            "error_type": type(exc).__name__,
        }

    features: dict[str, int] = {
        "joins": _count(tree, exp.Join),
        "subqueries": _count(tree, exp.Subquery),
        "groups": _count(tree, exp.Group),
        "havings": _count(tree, exp.Having),
        "aggregates": sum(1 for node in tree.walk() if isinstance(node, exp.AggFunc)),
        "windows": _count(tree, exp.Window),
        "set_operations": sum(
            1 for node in tree.walk() if isinstance(node, (exp.Union, exp.Intersect, exp.Except))
        ),
        "cases": _count(tree, exp.Case),
        "orders": _count(tree, exp.Order),
        "limits": _count(tree, exp.Limit),
        "distincts": _count(tree, exp.Distinct),
        "predicates": sum(
            1
            for node in tree.walk()
            if isinstance(node, (exp.Where, exp.Having, exp.Predicate))
        ),
        "tables": _count(tree, exp.Table),
        "max_ast_depth": _max_depth(tree),
    }
    score = (
        2 * features["joins"]
        + 3 * features["subqueries"]
        + 2 * features["groups"]
        + 2 * features["havings"]
        + features["aggregates"]
        + 2 * features["windows"]
        + 3 * features["set_operations"]
        + features["cases"]
        + features["orders"]
        + features["limits"]
        + features["distincts"]
        + features["predicates"]
        + max(0, features["tables"] - 1)
        + max(0, features["max_ast_depth"] - 4)
    )
    features["score"] = score
    features["parse_status"] = "ok"
    features["bin"] = "easy" if score <= 2 else "medium" if score <= 5 else "hard"
    # A coarse family is more useful than the raw SQL text (or a very sparse
    # exact AST signature) for preventing the selected cohort from collapsing
    # onto one relational pattern.  Counts remain in ``sql_features`` for audit.
    if features["set_operations"] or features["windows"]:
        family = "set_or_window"
    elif features["subqueries"]:
        family = "nested"
    elif features["joins"] and features["groups"]:
        family = "join_aggregate"
    elif features["joins"]:
        family = "join"
    elif features["groups"] or features["aggregates"]:
        family = "aggregate"
    elif features["predicates"] or features["orders"] or features["limits"]:
        family = "filter_order"
    else:
        family = "projection"
    features["family"] = family
    # Reuse the repository's compiler-side coverage profile when available.  It
    # is strictly a sampling annotation: only difficulty/support/signature
    # metadata is retained, never canonical SQL or a rendered action sequence.
    if profile_sql is not None:
        try:
            compiled = profile_sql(raw)
        except Exception:
            compiled = None
        if compiled is not None:
            compiled_json = compiled.to_json()
            features["compiler_difficulty"] = compiled.difficulty
            features["compiler_score"] = compiled.difficulty_score
            features["compiler_atomic_support"] = compiled.atomic_support
            features["compiler_scope_count"] = compiled.counts.get("select_scopes", 0)
            features["compiler_shape_hash"] = (
                profile_identity(compiled) if profile_identity is not None else None
            )
            # Keep the compiler's calibrated difficulty as the primary bin.  The
            # hand-written score above remains in the feature record for audit.
            features["bin"] = compiled.difficulty
            features["score"] = compiled.difficulty_score
    return features


def _question_bin(row: Mapping[str, Any], cutpoints: Sequence[int]) -> str:
    length = len(str(row.get("question") or "").strip())
    for index, boundary in enumerate(cutpoints, start=1):
        if length <= boundary:
            return f"q{index}"
    return "q5"


def _knowledge_bin(row: Mapping[str, Any]) -> str:
    value = row.get("external_knowledge")
    return "present" if isinstance(value, str) and value.strip() else "absent"


def _cutpoints(rows: Sequence[Mapping[str, Any]]) -> tuple[int, ...]:
    values = sorted(len(str(row.get("question") or "").strip()) for row in rows)
    if not values or values[0] < 1:
        raise ValueError("every task must have a non-empty question")
    return tuple(values[max(0, math.ceil(p * len(values)) - 1)] for p in (0.2, 0.4, 0.6, 0.8))


def _tv(reference: Sequence[str], sample: Sequence[str]) -> float:
    if not sample:
        return 1.0
    left, right = Counter(reference), Counter(sample)
    categories = set(left) | set(right)
    return 0.5 * sum(abs(left[k] / len(reference) - right[k] / len(sample)) for k in categories)


def _row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str, str]:
    identifier = task_id(row)
    return (
        source_name(identifier),
        str(row.get("db_id") or ""),
        str(row.get("sql_bin") or "unknown"),
        str(row.get("question_bin") or "q5"),
        identifier,
    )


def _sample_is_clean(sample: Mapping[str, Any]) -> bool:
    if str(sample.get("failure_type") or "") in INFRA_FAILURE_TYPES:
        return False
    return not any(
        int(sample.get(field, 0) or 0) != 0
        for field in ("api_transport_retries", "api_context_retries")
    )


def _record_is_clean(record: Mapping[str, Any]) -> bool:
    """Exclude provider/runtime contamination without discarding semantic failures."""
    failure_type = record.get("failure_type")
    # ``all_samples_failed`` is a valid homogeneous zero-reward group and is
    # excluded by the mixed-reward gate anyway.  Any other record-level failure
    # means the group did not receive the same clean K=8 treatment.
    if failure_type not in (None, "all_samples_failed"):
        return False
    return not any(
        int(record.get(field, 0) or 0) != 0
        for field in ("api_transport_retries", "api_context_retries")
    )


def build_group_index(
    tasks: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    *,
    cutpoints: Sequence[int],
) -> list[dict[str, Any]]:
    task_by_index: dict[int, Mapping[str, Any]] = {}
    for row in tasks:
        identifier = task_id(row)
        index = row.get("example_index")
        if type(index) is not int or index in task_by_index:
            raise ValueError(f"duplicate/invalid example_index for {identifier}")
        task_by_index[index] = row

    groups: list[dict[str, Any]] = []
    seen: set[int] = set()
    for record in records:
        index = record.get("example_index")
        if type(index) is not int or index in seen or index not in task_by_index:
            raise ValueError(f"record/task identity mismatch at {index!r}")
        seen.add(index)
        samples = record.get("samples")
        if not isinstance(samples, list) or len(samples) != GROUP_SIZE:
            raise ValueError(f"record {index} does not contain exactly K=8 samples")
        if record.get("n_samples") != GROUP_SIZE:
            raise ValueError(f"record {index} has wrong n_samples")
        if record.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
            raise ValueError(f"record {index} has a non-version26 protocol")
        if record.get("tool_scheme") not in (None, EXPECTED_TOOL_SCHEME):
            raise ValueError(f"record {index} has a non-atomic tool scheme")
        if record.get("assistant_carrier") not in (None, EXPECTED_CARRIER):
            raise ValueError(f"record {index} has an unexpected carrier")
        if record.get("tool_scheme_registry_version") not in (
            None,
            EXPECTED_REGISTRY_VERSION,
        ):
            raise ValueError(f"record {index} has an unexpected registry version")
        sample_indices = [sample.get("sample_index") for sample in samples]
        if sample_indices != list(range(GROUP_SIZE)):
            raise ValueError(f"record {index} samples are not ordered 0..7")
        correct_count = sum(bool(sample.get("correct")) for sample in samples)
        reported_correct = record.get("sample_correct_count")
        if reported_correct is not None and reported_correct != correct_count:
            raise ValueError(f"record {index} sample_correct_count disagrees with samples")
        clean = _record_is_clean(record) and all(_sample_is_clean(sample) for sample in samples)
        row = task_by_index[index]
        structure = sql_structure(str(row.get("gold_sql") or row.get("query") or ""))
        groups.append(
            {
                "example_index": index,
                "task_id": task_id(row),
                "source": source_name(task_id(row)),
                "db_id": str(row.get("db_id") or ""),
                "question_bin": _question_bin(row, cutpoints),
                "sql_bin": structure.get("bin", "unknown"),
                "sql_family": structure.get("family", "unknown"),
                "sql_atomic_support": structure.get("compiler_atomic_support", "unknown"),
                "knowledge_bin": _knowledge_bin(row),
                "sql_parse_status": structure.get("parse_status"),
                "sql_score": structure.get("score"),
                "sql_features": {
                    key: value
                    for key, value in structure.items()
                    if key not in {"error_type", "score", "bin", "family", "parse_status"}
                },
                "correct_count": correct_count,
                "uncertainty": correct_count * (GROUP_SIZE - correct_count),
                "usable": bool(clean and 1 <= correct_count <= GROUP_SIZE - 1),
                "core_boundary": bool(clean and 2 <= correct_count <= GROUP_SIZE - 2),
                "clean": clean,
            }
        )
    if len(seen) != len(tasks):
        missing = sorted(set(task_by_index) - seen)
        raise ValueError(f"screen is incomplete: {len(missing)} task records missing")
    return groups


def _target_distribution(rows: Sequence[Mapping[str, Any]]) -> dict[str, Counter[str]]:
    return {
        "source": Counter(source_name(task_id(row)) for row in rows),
        "db_id": Counter(str(row.get("db_id") or "") for row in rows),
        "sql_bin": Counter(str(row.get("sql_bin") or "unknown") for row in rows),
        "sql_family": Counter(str(row.get("sql_family") or "unknown") for row in rows),
        "sql_atomic_support": Counter(
            str(row.get("sql_atomic_support") or "unknown") for row in rows
        ),
        "question_bin": Counter(str(row.get("question_bin") or "q5") for row in rows),
        "knowledge_bin": Counter(str(row.get("knowledge_bin") or "absent") for row in rows),
    }


def select_groups(
    groups: Sequence[dict[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    *,
    count: int,
    seed: str = "epoch4-pass8-boundary-v1",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    clean_mixed = [group for group in groups if group["usable"]]
    core = [group for group in clean_mixed if group["core_boundary"]]
    if len(clean_mixed) < count:
        raise ValueError(f"only {len(clean_mixed)} clean mixed groups; need {count}")

    task_by_id = {task_id(row): row for row in tasks}
    # SQL features are selection-only annotations on the public task copy.  The
    # target population is the full source pool, not only the already-successful
    # groups, so a selected cohort cannot collapse to one dataset or one SQL bin.
    cutpoints = _cutpoints(tasks)
    reference_rows: list[dict[str, Any]] = []
    for row in tasks:
        copied = dict(row)
        copied["source"] = source_name(task_id(row))
        structure = sql_structure(str(row.get("gold_sql") or row.get("query") or ""))
        copied["sql_bin"] = structure.get("bin", "unknown")
        copied["sql_family"] = structure.get("family", "unknown")
        copied["sql_atomic_support"] = structure.get("compiler_atomic_support", "unknown")
        copied["question_bin"] = _question_bin(row, cutpoints)
        copied["knowledge_bin"] = _knowledge_bin(row)
        reference_rows.append(copied)
    reference = _target_distribution(reference_rows)

    # Prefer the core (2..6) boundary, then the edge mixed groups (1/7) only
    # when the requested cohort cannot be filled from core groups.
    pool = list(core)
    if len(pool) < count:
        pool.extend(group for group in clean_mixed if group not in pool)
    selected: list[dict[str, Any]] = []
    selected_counts = {key: Counter() for key in reference}
    while pool and len(selected) < count:
        next_size = len(selected) + 1

        def candidate_score(group: dict[str, Any]) -> tuple[float, int, str]:
            # Uncertainty is primary.  Distribution divergence is secondary;
            # SQL bin is an explicit axis so all-correct easy SQL cannot dominate.
            divergence = 0.0
            for axis in reference:
                value = str(group[axis])
                categories = set(reference[axis]) | set(selected_counts[axis]) | {value}
                divergence += 0.5 * sum(
                    abs(
                        (selected_counts[axis][category] + int(category == value)) / next_size
                        - reference[axis].get(category, 0) / len(tasks)
                    )
                    for category in categories
                )
            return (
                divergence,
                -int(group["uncertainty"]),
                stable_rank(seed, "task-order", str(group["task_id"])),
            )

        chosen = min(pool, key=candidate_score)
        pool.remove(chosen)
        selected.append(chosen)
        for axis in selected_counts:
            selected_counts[axis][str(chosen[axis])] += 1

    if len(selected) != count:
        raise RuntimeError("selector stopped before requested cohort size")
    selected_ids = {group["task_id"] for group in selected}
    if len(selected_ids) != count or not selected_ids <= set(task_by_id):
        raise RuntimeError("selected task identities are not unique/bound")
    details = {
        "selector": SCHEMA_VERSION,
        "selection_rule": (
            "clean K=8 mixed rewards first; prefer core 2<=correct_count<=6; "
            "maximize c*(8-c), then minimize source/db/sql-bin/question-bin TV"
        ),
        "screen_groups": len(groups),
        "clean_mixed_groups": len(clean_mixed),
        "clean_core_groups": len(core),
        "selected_records": len(selected),
        "selected_core_records": sum(group["core_boundary"] for group in selected),
        "selected_edge_records": sum(not group["core_boundary"] for group in selected),
        "source_distribution": dict(Counter(group["source"] for group in selected)),
        "sql_bin_distribution": dict(Counter(group["sql_bin"] for group in selected)),
        "uncertainty_distribution": dict(
            Counter(str(group["correct_count"]) for group in selected)
        ),
        "distribution_tv_vs_source": {
            axis: _tv(
                [str(row[axis]) for row in reference_rows],
                [str(group[axis]) for group in selected],
            )
            for axis in (
                "source",
                "db_id",
                "sql_bin",
                "sql_family",
                "sql_atomic_support",
                "question_bin",
                "knowledge_bin",
            )
        },
        "gold_sql_used_for": "selection-only AST structure proxy; never model-visible",
    }
    return selected, details


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> bytes:
    payload = "".join(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows).encode()
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
    return payload


def build_selection(
    *,
    tasks_path: Path,
    screen_paths: Sequence[Path],
    output_dir: Path,
    count: int,
    seed: str,
) -> dict[str, Any]:
    task_bytes = tasks_path.read_bytes()
    tasks = read_jsonl(tasks_path)
    if not tasks:
        raise ValueError("tasks file is empty")
    records: list[dict[str, Any]] = []
    screen_hashes: list[dict[str, Any]] = []
    for path in screen_paths:
        payload = path.read_bytes()
        loaded = read_jsonl(path)
        records.extend(loaded)
        screen_hashes.append({"path": str(path.resolve()), "sha256": sha256_bytes(payload), "records": len(loaded)})
    protocol_hashes = {str(row.get("protocol_hash")) for row in records if row.get("protocol_hash") is not None}
    if len(protocol_hashes) > 1:
        raise ValueError("screen files contain multiple protocol hashes")
    if not records:
        raise ValueError("screen files are empty")
    cutpoints = _cutpoints(tasks)
    groups = build_group_index(tasks, records, cutpoints=cutpoints)
    selected, details = select_groups(groups, tasks, count=count, seed=seed)
    selected_indices = {group["example_index"] for group in selected}
    selected_rows = [row for row in tasks if row.get("example_index") in selected_indices]
    selected_rows.sort(key=lambda row: next(i for i, group in enumerate(selected) if group["example_index"] == row["example_index"]))
    if len(selected_rows) != count:
        raise RuntimeError("selected task materialization is incomplete")

    output_dir.mkdir(parents=True, exist_ok=False)
    selected_path = output_dir / "selected_tasks.jsonl"
    selected_payload = _write_jsonl(selected_path, selected_rows)
    group_path = output_dir / "selected_groups.jsonl"
    group_payload = _write_jsonl(group_path, selected)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "diagnostic_boundary_selection_not_rl_admitted",
        "purpose": "Epoch-4 pass@8 task selection for fresh result-only RL screening",
        "inputs": {
            "tasks": {"path": str(tasks_path.resolve()), "sha256": sha256_bytes(task_bytes), "records": len(tasks)},
            "screen_files": screen_hashes,
        },
        "contract": {
            "group_size": GROUP_SIZE,
            "reward_mode": "result-only",
            "reward": "1=correct terminal denotation, 0=otherwise",
            "mixed_boundary": "1<=correct_count<=7",
            "core_boundary": "2<=correct_count<=6",
            "fresh_training_rollouts": True,
            "screen_trajectories_reused": False,
            "model_checkpoint": "Epoch4-checkpoint-6380",
            "tool_scheme": EXPECTED_TOOL_SCHEME,
            "assistant_carrier": EXPECTED_CARRIER,
            "protocol_version": EXPECTED_PROTOCOL_VERSION,
            "protocol_hash": next(iter(protocol_hashes), None),
        },
        "selection": {"count": count, "seed": seed, "question_length_cutpoints": list(cutpoints), **details},
        "outputs": {
            "selected_tasks": {"path": str(selected_path.resolve()), "sha256": sha256_bytes(selected_payload), "records": count},
            "selected_groups": {"path": str(group_path.resolve()), "sha256": sha256_bytes(group_payload), "records": count},
        },
        "task_ids": [group["task_id"] for group in selected],
        "gold_visibility": "gold SQL remains Harness-only and is absent from selection metadata",
    }
    _write_json(output_dir / "selection_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--screen", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", default="epoch4-pass8-boundary-v1")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_selection(
        tasks_path=args.tasks.resolve(),
        screen_paths=[path.resolve() for path in args.screen],
        output_dir=args.output_dir.resolve(),
        count=args.count,
        seed=args.seed,
    )
    print(json.dumps({"output_dir": manifest["outputs"]["selected_tasks"]["path"], "records": args.count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
