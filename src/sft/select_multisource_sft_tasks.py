#!/usr/bin/env python3
"""Select 9,000 coverage-rich BIRD/Spider/SynSQL question tasks for causal rollout.

This script creates *rollout candidates*, never SFT records.  Gold SQL is used only by the local
private profiler and read-only nonempty-result gate.  Candidate tasks become SFT only after a real
causal model<->Harness episode, strict terminal correctness, fresh replay, no-leak/quality gates,
and a checkpoint-relalg scheme-aware exporter and admission decision.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import math
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from harness.sql_task_coverage_profile import (  # noqa: E402
    DIFFICULTY_VERSION,
    PROFILE_VERSION,
    SqlTaskCoverageProfile,
    profile_identity,
    profile_sql,
)


SELECTOR_VERSION = "multisource-sft-question-selector-v2"
NONEMPTY_GATE_VERSION = "gold-denotation-nonempty-task-filter-v1"
DEFAULT_SEED = "bird-spider-synsql-sft9k-questions-v2-20260811"
SOURCE_DIFFICULTY_QUOTAS = {
    "bird": {"easy": 1100, "medium": 1850, "hard": 200},
    "spider": {"easy": 400, "medium": 1700, "hard": 600},
    "synsql": {"easy": 300, "medium": 1850, "hard": 1000},
}
DIFFICULTY_ORDER = ("easy", "medium", "hard")
SOURCE_ORDER = ("bird", "spider", "synsql")
DEFAULT_EXCLUDES = (
    "data/eval_inputs/bird_train_baseline300_v1.jsonl",
    "data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl",
    "data/eval_inputs/iterative_sql_v6_output_shape_target_gate20_v1.jsonl",
)


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_int(seed: str, *parts: Any) -> int:
    return int(sha_text("\x1f".join([seed, *(str(part) for part in parts)])), 16)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def iter_json_array(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Stream a top-level JSON array without materializing SynSQL's 8.7 GiB file."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8", buffering=chunk_size) as handle:
        buffer = ""
        position = 0
        eof = False
        opened = False
        while True:
            if not eof and (len(buffer) - position < chunk_size // 2):
                if position:
                    buffer = buffer[position:]
                    position = 0
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not opened:
                if position >= len(buffer):
                    if eof:
                        raise ValueError(f"{path}: empty JSON input")
                    continue
                if buffer[position] != "[":
                    raise ValueError(f"{path}: expected a top-level JSON array")
                opened = True
                position += 1
                continue
            while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            if position >= len(buffer):
                if eof:
                    raise ValueError(f"{path}: truncated JSON array")
                continue
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise
                if position:
                    buffer = buffer[position:]
                    position = 0
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            if not isinstance(value, dict):
                raise ValueError(f"{path}: expected object elements, got {type(value).__name__}")
            yield value
            position = end


def source_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    metadata_path = path.parent / ".cache" / "huggingface" / "download" / f"{path.name}.metadata"
    out: dict[str, Any] = {"path": str(path.relative_to(ROOT)), "bytes": stat.st_size}
    if metadata_path.exists():
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
        if len(lines) >= 2:
            out["huggingface_commit"] = lines[0]
            out["huggingface_blob_identity"] = lines[1]
    else:
        out["sha256"] = sha_file(path)
    return out


def build_synsql_shortlist(
    source: Path,
    cache: Path,
    cache_manifest: Path,
    *,
    seed: str,
    per_label: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fingerprint = source_fingerprint(source)
    expected = {
        "version": "synsql-stable-minhash-shortlist-v1",
        "source": fingerprint,
        "seed": seed,
        "per_source_complexity_label": per_label,
        "stored_fields": ["original_index", "db_id", "sql_complexity", "question_style", "question", "external_knowledge", "sql"],
        "cot_stored": False,
    }
    if cache.exists() and cache_manifest.exists():
        manifest = json.loads(cache_manifest.read_text(encoding="utf-8"))
        if all(manifest.get(key) == value for key, value in expected.items()):
            rows = read_jsonl(cache)
            if manifest.get("shortlist_count") == len(rows) and manifest.get("shortlist_sha256") == sha_file(cache):
                return rows, manifest

    reservoirs: dict[str, list[tuple[int, int, dict[str, Any]]]] = collections.defaultdict(list)
    label_counts: collections.Counter[str] = collections.Counter()
    total = 0
    for index, row in enumerate(iter_json_array(source)):
        total += 1
        label = str(row.get("sql_complexity") or "unknown").strip().lower()
        label_counts[label] += 1
        slim = {
            "original_index": index,
            "db_id": row.get("db_id"),
            "sql_complexity": row.get("sql_complexity"),
            "question_style": row.get("question_style"),
            "question": row.get("question"),
            "external_knowledge": row.get("external_knowledge"),
            "sql": row.get("sql"),
        }
        rank = stable_int(seed, "synsql", index, row.get("db_id"), row.get("question"), row.get("sql"))
        heap = reservoirs[label]
        item = (-rank, -index, slim)
        if len(heap) < per_label:
            heapq.heappush(heap, item)
        elif rank < -heap[0][0]:
            heapq.heapreplace(heap, item)

    rows = []
    for label in sorted(reservoirs):
        rows.extend(item[2] for item in reservoirs[label])
    rows.sort(key=lambda row: stable_int(seed, "shortlist-order", row["original_index"]))
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(compact(row) + "\n")
    os.replace(tmp, cache)
    manifest = {
        **expected,
        "source_count": total,
        "source_complexity_histogram": dict(sorted(label_counts.items())),
        "shortlist_count": len(rows),
        "shortlist_sha256": sha_file(cache),
    }
    cache_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows, manifest


@dataclass(frozen=True)
class Candidate:
    source: str
    task: dict[str, Any]
    profile: SqlTaskCoverageProfile

    @property
    def example_id(self) -> str:
        return str(self.task["example_id"])

    @property
    def db_id(self) -> str:
        return str(self.task["db_id"])


def normalize_question(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def make_task(
    *, source: str, index: int, db_id: Any, question: Any, sql: Any, db_path: Path,
    external_knowledge: Any = None, metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if source == "bird":
        example_id = f"bird_train_{index:05d}"
        dataset = "bird-sql"
    elif source == "spider":
        example_id = f"spider_train_{index:05d}"
        dataset = "spider"
    else:
        example_id = f"synsql_train_{index:07d}"
        dataset = "synsql-2.5m"
    return {
        "dataset": dataset,
        "split": "train",
        "example_index": index,
        "example_id": example_id,
        "db_id": str(db_id),
        "question": str(question or "").strip(),
        "backend": "sqlite",
        "db_path": str(db_path.resolve()),
        "gold_sql": str(sql or "").strip(),
        "gold_sql_path": None,
        "gold_exec_results": [],
        "external_knowledge": external_knowledge,
        "metadata": {"selection_source": source, **(metadata or {})},
        "query": str(sql or "").strip(),
        "instance_id": example_id,
        "index": index,
    }


def load_bird(path: Path, excluded_ids: set[str]) -> tuple[list[Candidate], collections.Counter[str]]:
    candidates: list[Candidate] = []
    audit: collections.Counter[str] = collections.Counter()
    for row in read_jsonl(path):
        example_id = str(row.get("example_id") or row.get("instance_id"))
        if example_id in excluded_ids:
            audit["excluded_evaluation_task"] += 1
            continue
        try:
            profile = profile_sql(str(row.get("gold_sql") or row.get("query") or ""))
        except Exception:
            audit["profile_error"] += 1
            continue
        candidates.append(Candidate("bird", row, profile))
    return candidates, audit


def load_spider(paths: list[Path], database_root: Path) -> tuple[list[Candidate], collections.Counter[str]]:
    candidates: list[Candidate] = []
    audit: collections.Counter[str] = collections.Counter()
    offset = 0
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for local_index, row in enumerate(rows):
            index = offset + local_index
            db_id = row.get("db_id")
            task = make_task(
                source="spider", index=index, db_id=db_id, question=row.get("question"),
                sql=row.get("query"), db_path=database_root / str(db_id) / f"{db_id}.sqlite",
                metadata={"source_file": path.name},
            )
            if not Path(task["db_path"]).is_file():
                audit["missing_database"] += 1
                continue
            try:
                profile = profile_sql(task["gold_sql"])
            except Exception:
                audit["profile_error"] += 1
                continue
            candidates.append(Candidate("spider", task, profile))
        offset += len(rows)
    return candidates, audit


def load_synsql(rows: list[dict[str, Any]], database_root: Path) -> tuple[list[Candidate], collections.Counter[str]]:
    candidates: list[Candidate] = []
    audit: collections.Counter[str] = collections.Counter()
    for row in rows:
        index = int(row["original_index"])
        db_id = row.get("db_id")
        task = make_task(
            source="synsql", index=index, db_id=db_id, question=row.get("question"), sql=row.get("sql"),
            db_path=database_root / str(db_id) / f"{db_id}.sqlite",
            external_knowledge=row.get("external_knowledge"),
            metadata={"source_sql_complexity": row.get("sql_complexity"), "question_style": row.get("question_style")},
        )
        if not Path(task["db_path"]).is_file():
            audit["missing_database"] += 1
            continue
        try:
            profile = profile_sql(task["gold_sql"])
        except Exception:
            audit["profile_error"] += 1
            continue
        candidates.append(Candidate("synsql", task, profile))
    return candidates, audit


def read_only_nonempty(task: dict[str, Any], *, timeout_seconds: float) -> tuple[bool, str]:
    path = Path(task["db_path"])
    if not path.is_file():
        return False, "missing_database"
    sql = str(task["gold_sql"]).strip().rstrip(";").strip()
    started = time.monotonic()
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro&immutable=1", uri=True, timeout=timeout_seconds)
        connection.execute("PRAGMA query_only=ON")
        connection.set_progress_handler(lambda: int(time.monotonic() - started > timeout_seconds), 10_000)
        cursor = connection.execute(f"SELECT 1 FROM ({sql}) AS __candidate_nonempty LIMIT 1")
        row = cursor.fetchone()
        return row is not None, "nonempty" if row is not None else "empty"
    except sqlite3.Error as exc:
        name = type(exc).__name__.lower()
        return False, "timeout" if "interrupt" in str(exc).lower() else f"sqlite_{name}"
    finally:
        if connection is not None:
            connection.close()


def _candidate_gain(
    candidate: Candidate,
    feature_weights: dict[str, float],
    selected_features: collections.Counter[str],
    selected_dbs: collections.Counter[str],
) -> float:
    features = candidate.profile.feature_keys
    coverage = sum(feature_weights[key] / (1.0 + selected_features[key]) for key in features)
    coverage /= math.sqrt(max(1, len(features)))
    return coverage + 2.0 / (1.0 + selected_dbs[candidate.db_id])


def select_cell(
    pool: list[Candidate], quota: int, *, seed: str, selected_question_sql: set[tuple[str, str]],
    execution_timeout: float, skip_execution_check: bool,
) -> tuple[list[Candidate], collections.Counter[str]]:
    audit: collections.Counter[str] = collections.Counter()
    if len(pool) < quota:
        raise ValueError(f"eligible cell has only {len(pool)} candidates for quota {quota}")
    frequencies: collections.Counter[str] = collections.Counter(
        key for candidate in pool for key in candidate.profile.feature_keys
    )
    weights = {key: 1.0 + math.log1p(len(pool) / count) for key, count in frequencies.items()}
    selected_features: collections.Counter[str] = collections.Counter()
    selected_dbs: collections.Counter[str] = collections.Counter()
    template_counts: collections.Counter[str] = collections.Counter()
    db_template_counts: collections.Counter[tuple[str, str]] = collections.Counter()
    heap: list[tuple[float, int, int, int, Candidate]] = []
    for ordinal, candidate in enumerate(pool):
        gain = _candidate_gain(candidate, weights, selected_features, selected_dbs)
        tie = stable_int(seed, candidate.source, candidate.example_id)
        heapq.heappush(heap, (-gain, tie, ordinal, 0, candidate))
    selected: list[Candidate] = []
    while heap and len(selected) < quota:
        _, tie, ordinal, scored_round, candidate = heapq.heappop(heap)
        if scored_round != len(selected):
            gain = _candidate_gain(candidate, weights, selected_features, selected_dbs)
            heapq.heappush(heap, (-gain, tie, ordinal, len(selected), candidate))
            continue
        template = candidate.profile.literal_masked_template_sha256
        qsql = (normalize_question(candidate.task["question"]), candidate.profile.canonical_sql_sha256)
        if qsql in selected_question_sql:
            audit["cross_source_question_sql_duplicate"] += 1
            continue
        if template_counts[template] >= 25:
            audit["template_cap"] += 1
            continue
        if db_template_counts[(candidate.db_id, template)] >= 3:
            audit["database_template_cap"] += 1
            continue
        if not skip_execution_check:
            ok, reason = read_only_nonempty(candidate.task, timeout_seconds=execution_timeout)
            if not ok:
                audit[f"nonempty_gate:{reason}"] += 1
                continue
        selected.append(candidate)
        selected_question_sql.add(qsql)
        template_counts[template] += 1
        db_template_counts[(candidate.db_id, template)] += 1
        selected_dbs[candidate.db_id] += 1
        selected_features.update(candidate.profile.feature_keys)
    if len(selected) != quota:
        raise ValueError(f"selected {len(selected)} of required {quota}; rejection audit={dict(audit)}")
    return selected, audit


def weighted_interleave(groups: dict[str, list[Candidate]], order: tuple[str, ...]) -> list[Candidate]:
    total = sum(len(values) for values in groups.values())
    targets = {key: len(groups.get(key, [])) for key in order}
    used = collections.Counter()
    positions = collections.Counter()
    out: list[Candidate] = []
    for step in range(total):
        available = [key for key in order if positions[key] < targets[key]]
        key = max(available, key=lambda item: (targets[item] * (step + 1) / total - used[item], -order.index(item)))
        out.append(groups[key][positions[key]])
        positions[key] += 1
        used[key] += 1
    return out


def final_order(selected: dict[tuple[str, str], list[Candidate]], seed: str) -> list[Candidate]:
    by_difficulty: dict[str, list[Candidate]] = {}
    for difficulty in DIFFICULTY_ORDER:
        groups: dict[str, list[Candidate]] = {}
        for source in SOURCE_ORDER:
            values = list(selected[(source, difficulty)])
            values.sort(key=lambda item: stable_int(seed, "cell-order", source, difficulty, item.example_id))
            groups[source] = values
        by_difficulty[difficulty] = weighted_interleave(groups, SOURCE_ORDER)
    easy = by_difficulty["easy"]
    medium = by_difficulty["medium"]
    hard = by_difficulty["hard"]
    if not (len(easy) * 3 == len(medium) and len(easy) == len(hard)):
        raise ValueError("difficulty totals must support the exact E,M,M,M,H prefix cycle")
    out: list[Candidate] = []
    for index in range(len(easy)):
        out.extend([easy[index], *medium[index * 3:index * 3 + 3], hard[index]])
    return out


def write_outputs(
    ordered: list[Candidate], output: Path, profiles_output: Path, visible_output: Path,
    manifest_output: Path, *, seed: str, input_fingerprints: dict[str, Any], source_audits: dict[str, Any],
    selection_audits: dict[str, Any], primary_tasks: int, target_questions: int,
) -> dict[str, Any]:
    for path in (output, profiles_output, visible_output, manifest_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    difficulty_hist = collections.Counter()
    source_hist = collections.Counter()
    cross_hist = collections.Counter()
    feature_hist = collections.Counter()
    skeleton_hist = collections.Counter()
    with output.open("w", encoding="utf-8") as tasks, profiles_output.open("w", encoding="utf-8") as profiles, visible_output.open("w", encoding="utf-8") as visible:
        for position, candidate in enumerate(ordered):
            difficulty = candidate.profile.difficulty
            wave = "primary" if position < primary_tasks else "reserve"
            task = dict(candidate.task)
            metadata = dict(task.get("metadata") or {})
            metadata.update({
                "selection_version": SELECTOR_VERSION,
                "selection_source": candidate.source,
                "selection_position": position,
                "selection_wave": wave,
                "private_difficulty": difficulty,
            })
            task["metadata"] = metadata
            tasks.write(compact(task) + "\n")
            profiles.write(compact({
                "example_id": candidate.example_id,
                "selection_position": position,
                "selection_wave": wave,
                "profile_identity_sha256": profile_identity(candidate.profile),
                "profile": candidate.profile.to_json(),
            }) + "\n")
            visible.write(compact({
                "dataset": task["dataset"], "split": task["split"], "example_index": task["example_index"],
                "example_id": task["example_id"], "db_id": task["db_id"], "question": task["question"],
                "external_knowledge": task.get("external_knowledge"),
            }) + "\n")
            difficulty_hist[difficulty] += 1
            source_hist[candidate.source] += 1
            cross_hist[f"{candidate.source}:{difficulty}"] += 1
            feature_hist.update(candidate.profile.feature_keys)
            skeleton_hist.update(candidate.profile.scope_skeletons)

    primary = ordered[:primary_tasks]
    reserve = ordered[primary_tasks:]
    manifest = {
        "version": SELECTOR_VERSION,
        "status": "rollout-candidates-only-not-sft-admitted",
        "seed": seed,
        "selection_contract": {
            "sampling_profile": PROFILE_VERSION,
            "difficulty_classifier": DIFFICULTY_VERSION,
            "private_gold_sql_use": ["local_ast_profile", "read_only_nonempty_gate"],
            "gold_visible_to_actor_or_teacher": False,
            "sql_compiled_to_trajectory": False,
            "nonempty_gate": NONEMPTY_GATE_VERSION,
            "causal_rollout_required": True,
            "fresh_replay_no_leak_quality_required": True,
            "checkpoint_relalg_scheme_aware_exporter_required": True,
            "explicit_sft_admission_required": True,
        },
        "target": {
            "unit": "question_tasks",
            "question_tasks": target_questions,
            "candidate_tasks": len(ordered),
            "primary_candidate_tasks": len(primary),
            "reserve_candidate_tasks": len(reserve),
            "action_count_is_not_the_size_target": True,
        },
        "quotas": {
            "source_difficulty": SOURCE_DIFFICULTY_QUOTAS,
            "difficulty_ratio": {"easy": 0.2, "medium": 0.6, "hard": 0.2},
            "source_ratio": {"bird": 0.35, "spider": 0.30, "synsql": 0.35},
            "human_authored_source_floor": 0.65,
        },
        "selected": {
            "total": len(ordered),
            "source_histogram": dict(sorted(source_hist.items())),
            "difficulty_histogram": dict(sorted(difficulty_hist.items())),
            "source_difficulty_histogram": dict(sorted(cross_hist.items())),
            "database_count": len({candidate.db_id for candidate in ordered}),
            "feature_count": len(feature_hist),
            "operator_skeleton_count": len(skeleton_hist),
            "coverage": {
                "primitive_presence": dict(sorted(
                    (key.removeprefix("primitive:"), value)
                    for key, value in feature_hist.items() if key.startswith("primitive:")
                )),
                "join_types": dict(sorted(
                    (key.removeprefix("join_type:"), value)
                    for key, value in feature_hist.items() if key.startswith("join_type:")
                )),
                "predicate_types": dict(sorted(
                    (key.removeprefix("predicate_op:"), value)
                    for key, value in feature_hist.items() if key.startswith("predicate_op:")
                )),
                "aggregate_types": dict(sorted(
                    (key.removeprefix("aggregate_op:"), value)
                    for key, value in feature_hist.items() if key.startswith("aggregate_op:")
                )),
                "set_types": dict(sorted(
                    (key.removeprefix("set_op:"), value)
                    for key, value in feature_hist.items() if key.startswith("set_op:")
                )),
                "pair_types": sum(key.startswith("pair:") for key in feature_hist),
                "triple_types": sum(key.startswith("triple:") for key in feature_hist),
            },
            "sql_identity": {
                "unique_canonical_sql": len({candidate.profile.canonical_sql_sha256 for candidate in ordered}),
                "unique_literal_masked_templates": len({candidate.profile.literal_masked_template_sha256 for candidate in ordered}),
                "maximum_literal_masked_template_multiplicity": max(collections.Counter(
                    candidate.profile.literal_masked_template_sha256 for candidate in ordered
                ).values()),
            },
            "atomic_support_histogram": dict(sorted(collections.Counter(
                candidate.profile.atomic_support for candidate in ordered
            ).items())),
            "primary_difficulty_histogram": dict(sorted(collections.Counter(c.profile.difficulty for c in primary).items())),
            "reserve_difficulty_histogram": dict(sorted(collections.Counter(c.profile.difficulty for c in reserve).items())),
            "prefix_cycle": ["easy", "medium", "medium", "medium", "hard"],
        },
        "inputs": input_fingerprints,
        "source_audits": source_audits,
        "selection_rejection_audits": selection_audits,
        "outputs": {
            "tasks": str(output.relative_to(ROOT)), "tasks_sha256": sha_file(output),
            "private_profiles": str(profiles_output.relative_to(ROOT)), "private_profiles_sha256": sha_file(profiles_output),
            "teacher_visible": str(visible_output.relative_to(ROOT)), "teacher_visible_sha256": sha_file(visible_output),
        },
    }
    manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bird", type=Path, default=ROOT / "data/eval_inputs/bird_train_tool_compatible_nonempty_v1.jsonl")
    parser.add_argument("--spider", type=Path, nargs="+", default=[ROOT / "data/spider_data/train_spider.json", ROOT / "data/spider_data/train_others.json"])
    parser.add_argument("--spider-databases", type=Path, default=ROOT / "data/spider_data/database")
    parser.add_argument("--synsql", type=Path, default=ROOT / "data/SynSQL-2.5M/data.json")
    parser.add_argument("--synsql-databases", type=Path, default=ROOT / "data/SynSQL-2.5M/databases")
    parser.add_argument("--exclude", type=Path, nargs="*", default=[ROOT / path for path in DEFAULT_EXCLUDES])
    parser.add_argument("--synsql-shortlist-per-label", type=int, default=15_000)
    parser.add_argument("--synsql-shortlist-cache", type=Path, default=ROOT / "data/sft_task_selection/cache/synsql_sft9k_questions_v2_shortlist.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "data/sft_task_selection/bird_spider_synsql_sft9k_questions_v2.tasks.jsonl")
    parser.add_argument("--profiles-output", type=Path, default=ROOT / "data/sft_task_selection/bird_spider_synsql_sft9k_questions_v2.private_profiles.jsonl")
    parser.add_argument("--visible-output", type=Path, default=ROOT / "data/sft_task_selection/bird_spider_synsql_sft9k_questions_v2.teacher_visible.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/sft_task_selection/bird_spider_synsql_sft9k_questions_v2.manifest.json")
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--primary-tasks", type=int, default=9_000)
    parser.add_argument("--target-questions", type=int, default=9_000)
    parser.add_argument("--execution-timeout", type=float, default=3.0)
    parser.add_argument("--skip-execution-check", action="store_true", help="tests only; never for a frozen rollout cohort")
    args = parser.parse_args()

    if sum(sum(values.values()) for values in SOURCE_DIFFICULTY_QUOTAS.values()) != 9_000:
        raise ValueError("default quota matrix must total 9,000 questions")
    if args.target_questions != 9_000:
        raise ValueError("the frozen v2 quota matrix is identity-bound to exactly 9,000 questions")
    if args.primary_tasks != args.target_questions:
        raise ValueError("v2 contains exactly 9,000 selected questions and no action-yield reserve wave")
    if args.primary_tasks % 5:
        raise ValueError("primary task count must be divisible by five to preserve exact 20/60/20")

    excluded_ids: set[str] = set()
    exclude_inputs: list[dict[str, Any]] = []
    for path in args.exclude:
        if not path.exists():
            raise FileNotFoundError(path)
        rows = read_jsonl(path)
        excluded_ids.update(str(row.get("example_id") or row.get("instance_id")) for row in rows)
        exclude_inputs.append({"path": str(path.relative_to(ROOT)), "sha256": sha_file(path), "records": len(rows)})

    shortlist_manifest_path = args.synsql_shortlist_cache.with_suffix(".manifest.json")
    synsql_rows, shortlist_manifest = build_synsql_shortlist(
        args.synsql, args.synsql_shortlist_cache, shortlist_manifest_path,
        seed=args.seed, per_label=args.synsql_shortlist_per_label,
    )
    bird, bird_audit = load_bird(args.bird, excluded_ids)
    spider, spider_audit = load_spider(args.spider, args.spider_databases)
    synsql, synsql_audit = load_synsql(synsql_rows, args.synsql_databases)
    pools = {"bird": bird, "spider": spider, "synsql": synsql}
    source_audits: dict[str, Any] = {
        "bird": dict(bird_audit), "spider": dict(spider_audit), "synsql": dict(synsql_audit),
        "eligible_difficulty": {
            source: dict(sorted(collections.Counter(c.profile.difficulty for c in values).items()))
            for source, values in pools.items()
        },
    }

    selected: dict[tuple[str, str], list[Candidate]] = {}
    selection_audits: dict[str, Any] = {}
    selected_question_sql: set[tuple[str, str]] = set()
    for source in SOURCE_ORDER:
        for difficulty in ("hard", "medium", "easy"):
            cell_pool = [candidate for candidate in pools[source] if candidate.profile.difficulty == difficulty]
            values, audit = select_cell(
                cell_pool, SOURCE_DIFFICULTY_QUOTAS[source][difficulty],
                seed=f"{args.seed}:{source}:{difficulty}", selected_question_sql=selected_question_sql,
                execution_timeout=args.execution_timeout,
                skip_execution_check=args.skip_execution_check or source == "bird",
            )
            selected[(source, difficulty)] = values
            selection_audits[f"{source}:{difficulty}"] = dict(sorted(audit.items()))

    ordered = final_order(selected, args.seed)
    fingerprints = {
        "bird": source_fingerprint(args.bird),
        "spider": [source_fingerprint(path) for path in args.spider],
        "synsql": source_fingerprint(args.synsql),
        "synsql_shortlist": shortlist_manifest,
        "evaluation_exclusions": exclude_inputs,
    }
    manifest = write_outputs(
        ordered, args.output, args.profiles_output, args.visible_output, args.manifest,
        seed=args.seed, input_fingerprints=fingerprints, source_audits=source_audits,
        selection_audits=selection_audits, primary_tasks=args.primary_tasks,
        target_questions=args.target_questions,
    )
    print(json.dumps({
        "status": manifest["status"], "selected": manifest["selected"],
        "target": manifest["target"], "outputs": manifest["outputs"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
