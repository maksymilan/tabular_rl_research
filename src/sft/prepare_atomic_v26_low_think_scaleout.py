#!/usr/bin/env python3
"""Prepare diverse, disjoint task cohorts for Atomic-v26 low-think rollout scale-out.

Gold SQL is used only by the local structural profiler and read-only nonempty gate.  The output
contains rollout tasks, not trajectories, and remains ineligible for SFT until causal generation,
terminal correctness, fresh replay, no-leak, reasoning-length, and exact version26 projection
gates pass.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from harness.sql_task_coverage_profile import profile_identity, profile_sql
from sft.select_multisource_sft_tasks import (
    Candidate,
    load_spider,
    load_synsql,
    normalize_question,
    read_jsonl,
    read_only_nonempty,
    select_cell,
    stable_int,
)


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "atomic-v26-low-think-diverse-scaleout-selection-v1"
SEED = "atomic-v26-low-think-diverse-scaleout-20260827"

SOURCE_V2 = ROOT / "data/sft_task_selection/bird_spider_synsql_sft9k_questions_v2.tasks.jsonl"
SOURCE_MINUS_FIXED = ROOT / "data/sft_task_selection/bird_spider_synsql_sft9k_minus_sft1_fixed1000_v1.tasks.jsonl"
PROFILES_V2 = ROOT / "data/sft_task_selection/bird_spider_synsql_sft9k_questions_v2.private_profiles.jsonl"
BATCH1 = ROOT / "data/sft_task_selection/bird_sft1_expansion_low_think_pure1000_v1.tasks.jsonl"
BATCH2 = ROOT / "data/sft_task_selection/bird_sft1_expansion_low_think_pure1000_batch2_v1.tasks.jsonl"
FIXED1000 = ROOT / "data/eval_inputs/bird_train_external_teacher_fixed1000.jsonl"
BIRD = ROOT / "data/eval_inputs/bird_train_tool_compatible_nonempty_v1.jsonl"
SPIDER = [ROOT / "data/spider_data/train_spider.json", ROOT / "data/spider_data/train_others.json"]
SPIDER_DATABASES = ROOT / "data/spider_data/database"
SYNSQL_SHORTLIST = ROOT / "data/sft_task_selection/cache/synsql_sft9k_questions_v2_shortlist.jsonl"
SYNSQL_DATABASES = ROOT / "data/SynSQL-2.5M/databases"
EVAL_EXCLUSIONS = [
    ROOT / "data/eval_inputs/bird_train_baseline300_v1.jsonl",
    ROOT / "data/eval_inputs/bird_train_tool_interface_validation200_version4.jsonl",
    ROOT / "data/eval_inputs/iterative_sql_v6_output_shape_target_gate20_v1.jsonl",
]

FIRST_PASS_EXTRA_BIRD = {"easy": 1891, "medium": 68, "hard": 40}
FIRST_PASS_RETAIN_FROM_FROZEN = {
    ("bird", "easy"): 224,
    ("bird", "medium"): 345,
    ("bird", "hard"): 33,
    ("spider", "easy"): 192,
    ("spider", "medium"): 1240,
    ("spider", "hard"): 593,
    ("synsql", "easy"): 97,
    ("synsql", "medium"): 1289,
    ("synsql", "hard"): 988,
}
EXPANSION_QUOTAS = {
    # The broader Spider pool has only about 72 hard and 158 medium previously unused,
    # read-only, nonempty queries after the frozen v2 cohort is removed.  Keep conservative
    # quotas from both and move unusable quota to the structurally diverse easy cell rather than
    # admitting empty-result examples.  Across first-pass + expansion, medium/hard still form a
    # majority because the frozen v2 remainder is already rich in those cells.
    ("spider", "easy"): 2037,
    ("spider", "medium"): 150,
    ("spider", "hard"): 70,
    ("synsql", "medium"): 1000,
    ("synsql", "hard"): 960,
}


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_source(task: dict[str, Any]) -> str:
    return {"bird-sql": "bird", "spider": "spider", "synsql-2.5m": "synsql"}[str(task["dataset"])]


def write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(compact(value) + "\n")


def portable_task(task: dict[str, Any]) -> dict[str, Any]:
    out = dict(task)
    db_id = str(out["db_id"])
    source = dataset_source(out)
    if source == "bird":
        path = Path("/home/dengyan/tabular_rl_project/data/bird/train/train_databases") / db_id / f"{db_id}.sqlite"
    elif source == "spider":
        path = Path("/home/dengyan/tabular_rl_project/data/spider_data/database") / db_id / f"{db_id}.sqlite"
    else:
        path = Path("/home/dengyan/tabular_rl_outputs/text2sql_reproduction/data/SynSQL-2.5M/databases") / db_id / f"{db_id}.sqlite"
    out["db_path"] = str(path)
    return out


def profile_identity_map(path: Path) -> dict[str, str]:
    return {str(row["example_id"]): str(row["profile_identity_sha256"]) for row in read_jsonl(path)}


def candidate_from_task(task: dict[str, Any]) -> Candidate:
    return Candidate(dataset_source(task), task, profile_sql(str(task.get("gold_sql") or task.get("query") or "")))


def diversity_pick(
    pool: list[Candidate], quota: int, *, seed: str, selected: list[Candidate], execute_nonempty: bool,
) -> list[Candidate]:
    if len(pool) < quota:
        raise ValueError(f"pool contains {len(pool)} tasks but quota is {quota}")
    feature_counts = collections.Counter(key for value in selected for key in value.profile.feature_keys)
    db_counts = collections.Counter(value.db_id for value in selected)
    template_counts = collections.Counter(value.profile.literal_masked_template_sha256 for value in selected)
    question_sql = {
        (normalize_question(value.task.get("question")), value.profile.canonical_sql_sha256)
        for value in selected
    }
    remaining = list(pool)
    chosen: list[Candidate] = []
    while len(chosen) < quota:
        best_index = None
        best_key = None
        for index, value in enumerate(remaining):
            template = value.profile.literal_masked_template_sha256
            qsql = (normalize_question(value.task.get("question")), value.profile.canonical_sql_sha256)
            if qsql in question_sql or template_counts[template] >= 25:
                continue
            novelty = sum(1.0 / (1.0 + feature_counts[key]) for key in value.profile.feature_keys)
            novelty /= math.sqrt(max(1, len(value.profile.feature_keys)))
            score = novelty + 3.0 / (1.0 + db_counts[value.db_id]) + 2.0 / (1.0 + template_counts[template])
            key = (score, -stable_int(seed, value.example_id))
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index is None:
            raise ValueError(f"diversity constraints exhausted after {len(chosen)} of {quota}")
        value = remaining.pop(best_index)
        if execute_nonempty:
            ok, _ = read_only_nonempty(value.task, timeout_seconds=3.0)
            if not ok:
                continue
        chosen.append(value)
        selected.append(value)
        question_sql.add((normalize_question(value.task.get("question")), value.profile.canonical_sql_sha256))
        feature_counts.update(value.profile.feature_keys)
        db_counts[value.db_id] += 1
        template_counts[value.profile.literal_masked_template_sha256] += 1
    return chosen


def balanced_order(values: list[Candidate], *, seed: str) -> list[Candidate]:
    groups: dict[tuple[str, str], list[Candidate]] = collections.defaultdict(list)
    for value in values:
        groups[(value.source, value.profile.difficulty)].append(value)
    for key, group in groups.items():
        group.sort(key=lambda item: stable_int(seed, *key, item.db_id, item.example_id))
    targets = {key: len(group) for key, group in groups.items()}
    used = collections.Counter()
    position = collections.Counter()
    keys = sorted(groups)
    ordered: list[Candidate] = []
    total = len(values)
    for step in range(total):
        available = [key for key in keys if position[key] < targets[key]]
        key = max(
            available,
            key=lambda item: (targets[item] * (step + 1) / total - used[item], -keys.index(item)),
        )
        ordered.append(groups[key][position[key]])
        position[key] += 1
        used[key] += 1
    return ordered


def summarize(values: list[Candidate]) -> dict[str, Any]:
    features = collections.Counter(key for value in values for key in value.profile.feature_keys)
    templates = collections.Counter(value.profile.literal_masked_template_sha256 for value in values)
    return {
        "records": len(values),
        "unique_example_ids": len({value.example_id for value in values}),
        "unique_database_ids": len({value.db_id for value in values}),
        "source_histogram": dict(sorted(collections.Counter(value.source for value in values).items())),
        "difficulty_histogram": dict(sorted(collections.Counter(value.profile.difficulty for value in values).items())),
        "source_difficulty_histogram": dict(sorted(collections.Counter(
            f"{value.source}:{value.profile.difficulty}" for value in values
        ).items())),
        "atomic_support_histogram": dict(sorted(collections.Counter(value.profile.atomic_support for value in values).items())),
        "feature_count": len(features),
        "operator_skeleton_count": len({item for value in values for item in value.profile.scope_skeletons}),
        "unique_literal_masked_templates": len(templates),
        "maximum_template_multiplicity": max(templates.values()),
    }


def emit_cohort(name: str, values: list[Candidate], output_dir: Path, provenance: dict[str, Any]) -> dict[str, Any]:
    ordered = balanced_order(values, seed=f"{SEED}:{name}:order")
    tasks = output_dir / f"{name}.tasks.jsonl"
    profiles = output_dir / f"{name}.private_profiles.jsonl"
    portable = output_dir / f"{name}.table_rl.tasks.jsonl"
    manifest_path = output_dir / f"{name}.manifest.json"
    write_jsonl(tasks, (value.task for value in ordered))
    write_jsonl(portable, (portable_task(value.task) for value in ordered))
    write_jsonl(profiles, ({
        "example_id": value.example_id,
        "profile_identity_sha256": profile_identity(value.profile),
        "profile": value.profile.to_json(),
    } for value in ordered))
    summary = summarize(ordered)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_generation_candidates_not_sft_admitted",
        "name": name,
        "seed": SEED,
        "selection_contract": {
            "causal_rollout_required": True,
            "gold_visible_to_teacher": False,
            "gold_compiled_to_trajectory": False,
            "private_gold_use": ["sql_structure_diversity", "read_only_nonempty_gate"],
            "terminal_correctness_required": True,
            "fresh_replay_required": True,
            "no_leak_required": True,
            "reason_tokens_per_turn_max": 1024,
            "reason_tokens_per_episode_max": 4096,
            "student_projection": "atomic-version26-think-json-v1",
        },
        "teacher_identity": {
            "endpoint": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "thinking": "enabled",
            "reasoning_effort": "low",
            "response_format": "json_object",
            "atomic_protocol_version": "version24",
            "provider_system_prompt_sha256": "d211b9d7e2c8f03a68fbf37efb846844cebc7028072d5eeed75302bdcc4f4cf0",
            "provider_protocol_hash": "25ac4c10ef96365c",
            "history": "rolling-legal-history-recent4-full",
        },
        "selection": summary,
        "provenance": provenance,
        "outputs": {
            "tasks": str(tasks.relative_to(ROOT)), "tasks_sha256": sha_file(tasks),
            "private_profiles": str(profiles.relative_to(ROOT)), "private_profiles_sha256": sha_file(profiles),
            "table_rl_tasks": str(portable.relative_to(ROOT)), "table_rl_tasks_sha256": sha_file(portable),
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "data/sft_task_selection/atomic_v26_low_think_scaleout_20260827",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    v2_tasks = read_jsonl(SOURCE_V2)
    minus = read_jsonl(SOURCE_MINUS_FIXED)
    batch1_ids = {str(row["example_id"]) for row in read_jsonl(BATCH1)}
    batch2_ids = {str(row["example_id"]) for row in read_jsonl(BATCH2)}
    pilot_ids = {str(row["example_id"]) for row in minus[:100]}
    fixed_ids = {str(row["example_id"]) for row in read_jsonl(FIXED1000)}
    prior_ids = batch1_ids | batch2_ids | pilot_ids | fixed_ids
    profile_identities_v2 = profile_identity_map(PROFILES_V2)
    remaining_tasks = [row for row in minus if str(row["example_id"]) not in prior_ids]
    if len(remaining_tasks) != 6384:
        raise ValueError(f"expected 6384 frozen remaining tasks, got {len(remaining_tasks)}")
    remaining = [Candidate(dataset_source(row), row, profile_sql(str(row.get("gold_sql") or row.get("query") or ""))) for row in remaining_tasks]
    if any(profile_identity(value.profile) != profile_identities_v2[value.example_id] for value in remaining):
        raise ValueError("recomputed v2 profile identity mismatch")

    retained_frozen: list[Candidate] = []
    selected_question_sql: set[tuple[str, str]] = set()
    retained_audits: dict[str, dict[str, int]] = {}
    for source, difficulty in (
        ("bird", "hard"), ("bird", "medium"), ("bird", "easy"),
        ("spider", "hard"), ("spider", "medium"), ("spider", "easy"),
        ("synsql", "hard"), ("synsql", "medium"), ("synsql", "easy"),
    ):
        pool = [value for value in remaining if value.source == source and value.profile.difficulty == difficulty]
        quota = FIRST_PASS_RETAIN_FROM_FROZEN[(source, difficulty)]
        chosen, cell_audit = select_cell(
            pool,
            quota,
            seed=f"{SEED}:first-pass-retain:{source}:{difficulty}",
            selected_question_sql=selected_question_sql,
            execution_timeout=3.0,
            skip_execution_check=True,
        )
        retained_frozen.extend(chosen)
        retained_audits[f"{source}:{difficulty}"] = dict(sorted(cell_audit.items()))
    retained_ids = {value.example_id for value in retained_frozen}
    deferred_frozen = [value for value in remaining if value.example_id not in retained_ids]
    if len(retained_frozen) != 5001 or len(deferred_frozen) != 1383:
        raise ValueError("frozen first-pass/deferred split mismatch")

    exclusion_ids = {str(row["example_id"]) for row in v2_tasks} | fixed_ids
    for path in EVAL_EXCLUSIONS:
        exclusion_ids.update(str(row["example_id"]) for row in read_jsonl(path))
    unused_bird: list[Candidate] = []
    for row in read_jsonl(BIRD):
        if str(row["example_id"]) in exclusion_ids:
            continue
        unused_bird.append(candidate_from_task(row))
    extra: list[Candidate] = []
    selected_context = list(retained_frozen)
    for difficulty in ("hard", "medium", "easy"):
        quota = FIRST_PASS_EXTRA_BIRD[difficulty]
        pool = [value for value in unused_bird if value.profile.difficulty == difficulty and value.example_id not in {x.example_id for x in extra}]
        extra.extend(diversity_pick(
            pool, quota, seed=f"{SEED}:first-pass-extra-bird:{difficulty}",
            selected=selected_context, execute_nonempty=False,
        ))
    first_pass = retained_frozen + extra
    if len(first_pass) != 7000 or len({value.example_id for value in first_pass}) != 7000:
        raise ValueError("first-pass cohort is not exactly 7000 unique tasks")

    first_manifest = emit_cohort(
        "atomic_v26_low_think_diverse_firstpass7000", first_pass, args.output_dir,
        provenance={
            "frozen_remaining_source": str(SOURCE_MINUS_FIXED.relative_to(ROOT)),
            "frozen_remaining_source_sha256": sha_file(SOURCE_MINUS_FIXED),
            "frozen_remaining_records": 6384,
            "frozen_remaining_retained_records": len(retained_frozen),
            "frozen_remaining_deferred_records": len(deferred_frozen),
            "frozen_retain_selection_audits": retained_audits,
            "new_unused_bird_records": len(extra),
            "bird_frontloaded_for_target_domain": True,
            "dedup": {"fixed1000": len(fixed_ids), "pilot100": len(pilot_ids), "batch1": len(batch1_ids), "batch2": len(batch2_ids)},
        },
    )

    used_ids = prior_ids | {value.example_id for value in first_pass} | {value.example_id for value in deferred_frozen}
    selected_for_expansion: list[Candidate] = list(deferred_frozen)
    bird_pool = [candidate_from_task(row) for row in read_jsonl(BIRD) if str(row["example_id"]) not in used_ids]
    spider_pool, spider_audit = load_spider(SPIDER, SPIDER_DATABASES)
    spider_pool = [value for value in spider_pool if value.example_id not in used_ids]
    synsql_pool, synsql_audit = load_synsql(read_jsonl(SYNSQL_SHORTLIST), SYNSQL_DATABASES)
    synsql_pool = [value for value in synsql_pool if value.example_id not in used_ids]
    pools = {"bird": bird_pool, "spider": spider_pool, "synsql": synsql_pool}
    selected_question_sql = {
        (normalize_question(value.task["question"]), value.profile.canonical_sql_sha256)
        for value in first_pass + deferred_frozen
    }
    expansion_selection_audits: dict[str, dict[str, int]] = {}
    for source, difficulty in (
        ("spider", "hard"), ("spider", "medium"), ("spider", "easy"),
        ("synsql", "hard"), ("synsql", "medium"),
    ):
        quota = EXPANSION_QUOTAS[(source, difficulty)]
        pool = [value for value in pools[source] if value.profile.difficulty == difficulty]
        chosen, cell_audit = select_cell(
            pool,
            quota,
            seed=f"{SEED}:expansion:{source}:{difficulty}",
            selected_question_sql=selected_question_sql,
            execution_timeout=3.0,
            skip_execution_check=source == "bird",
        )
        expansion_selection_audits[f"{source}:{difficulty}"] = dict(sorted(cell_audit.items()))
        selected_for_expansion.extend(chosen)
        used_ids.update(value.example_id for value in chosen)
    if len(selected_for_expansion) != 5600 or len({value.example_id for value in selected_for_expansion}) != 5600:
        raise ValueError("expansion cohort is not exactly 5600 unique tasks")
    if {value.example_id for value in first_pass} & {value.example_id for value in selected_for_expansion}:
        raise ValueError("first-pass and expansion cohorts overlap")
    expansion_manifest = emit_cohort(
        "atomic_v26_low_think_diverse_expansion5600", selected_for_expansion, args.output_dir,
        provenance={
            "purpose": "unique-task quota reserve to reach 10000 admitted episodes at the observed 65.4 percent admission rate",
            "deferred_frozen_records": len(deferred_frozen),
            "first_pass_manifest_sha256": sha_file(args.output_dir / "atomic_v26_low_think_diverse_firstpass7000.manifest.json"),
            "spider_source_audit": dict(spider_audit),
            "synsql_source_audit": dict(synsql_audit),
            "selection_rejection_audits": expansion_selection_audits,
            "all_tasks_disjoint_from_first_pass_and_prior_training": True,
        },
    )
    expected = 1986 + round((7000 + 5600) * 0.654)
    campaign = {
        "schema_version": "atomic-v26-low-think-scaleout-campaign-v1",
        "status": "prepared_not_launched",
        "existing_admitted_episodes": 1986,
        "observed_admission_rate": 0.654,
        "first_pass_tasks": 7000,
        "expansion_reserve_tasks": 5600,
        "expected_total_admitted_episodes": expected,
        "target_total_admitted_episodes": 10000,
        "dispatch_policy": "run all 7000 first-pass tasks, then dispatch the diversity-ordered expansion reserve only until the audited cumulative admitted count reaches 10000",
        "first_pass_manifest_sha256": sha_file(args.output_dir / "atomic_v26_low_think_diverse_firstpass7000.manifest.json"),
        "expansion_manifest_sha256": sha_file(args.output_dir / "atomic_v26_low_think_diverse_expansion5600.manifest.json"),
        "selection": {"first_pass": first_manifest["selection"], "expansion": expansion_manifest["selection"]},
    }
    campaign_path = args.output_dir / "CAMPAIGN.json"
    campaign_path.write_text(json.dumps(campaign, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(campaign, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
