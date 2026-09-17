"""Build an auditable union of completed K-sample RL screening artifacts.

The pool keeps every screening observation instead of collapsing a task to one
mutable ``correct_count``.  Eligibility can therefore be recomputed without
rescanning multi-gigabyte rollout files, while diagnostic rescreens cannot
silently add or remove a task from a frozen candidate scope.
"""
from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rl.shared.io import atomic_write_text, read_jsonl, sha256_file


SCHEMA_VERSION = "atomic-v26-screened-rl-candidate-pool-v1"
SOURCE_SPEC_VERSION = "atomic-v26-screened-rl-source-spec-v1"
VALID_SCOPES = {"strict", "historical", "diagnostic"}


def task_id(row: Mapping[str, Any]) -> str:
    value = row.get("task_id") or row.get("example_id") or row.get("instance_id")
    if not isinstance(value, str) or not value:
        raise ValueError("task row is missing task_id/example_id/instance_id")
    return value


def question_sha256(row: Mapping[str, Any]) -> str:
    question = row.get("question")
    if not isinstance(question, str) or not question:
        raise ValueError(f"{task_id(row)}: missing question")
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def _semantic_signature(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("dataset"),
        row.get("db_id"),
        row.get("question"),
        row.get("external_knowledge"),
        row.get("gold_sql") or row.get("query"),
    )


def _portable_db_path(value: Any) -> str:
    normalized = str(value or "").replace("\\", "/")
    if not normalized:
        raise ValueError("task is missing db_path")
    if normalized.startswith("data/"):
        return normalized
    marker = "/data/"
    if marker not in normalized:
        raise ValueError(f"cannot make db_path portable: {normalized!r}")
    return "data/" + normalized.split(marker, 1)[1]


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _scope_rank(scope: str) -> int:
    return {"diagnostic": 0, "historical": 1, "strict": 2}[scope]


def load_task_catalog(
    repo_root: Path,
    source_specs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    tasks: dict[str, dict[str, Any]] = {}
    owners: dict[str, str] = {}
    signatures: dict[str, tuple[Any, ...]] = {}
    artifacts: list[dict[str, Any]] = []
    ordered = sorted(source_specs, key=lambda source: int(source.get("priority", 100)))
    for source in ordered:
        label = str(source["label"])
        path = _resolve(repo_root, str(source["path"]))
        rows = read_jsonl(path)
        source_ids: set[str] = set()
        for row in rows:
            identifier = task_id(row)
            if identifier in source_ids:
                raise ValueError(f"{label}: duplicate task {identifier}")
            source_ids.add(identifier)
            signature = _semantic_signature(row)
            if identifier in signatures and signatures[identifier] != signature:
                raise ValueError(
                    f"conflicting semantic task payload for {identifier}: "
                    f"{owners[identifier]} vs {label}"
                )
            signatures.setdefault(identifier, signature)
            if identifier not in tasks:
                tasks[identifier] = copy.deepcopy(row)
                owners[identifier] = label
        artifacts.append(
            {
                "label": label,
                "path": str(path.relative_to(repo_root)),
                "sha256": sha256_file(path),
                "records": len(rows),
                "unique_task_ids": len(source_ids),
                "priority": int(source.get("priority", 100)),
            }
        )
    return tasks, owners, artifacts


def _base_observation(
    *,
    identifier: str,
    source: str,
    scope: str,
    correct_count: int,
    n: int,
    artifact: str,
    source_spec: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if scope not in VALID_SCOPES:
        raise ValueError(f"{source}: unsupported candidate scope {scope!r}")
    observation = {
        "task_id": identifier,
        "source": source,
        "scope": scope,
        "correct_count": int(correct_count),
        "n": int(n),
        "complete": int(n) == int(source_spec["expected_group_size"]),
        "eligible": False,
        "evidence_artifacts": [artifact],
        "model_checkpoint": source_spec.get("model_checkpoint"),
        "protocol_version": source_spec.get("protocol_version"),
        "protocol_hash": source_spec.get("protocol_hash"),
        "identity_status": source_spec.get("identity_status", "manifest_bound"),
        "source_group": source_spec.get("source_group", source),
    }
    if extra:
        observation.update({key: value for key, value in extra.items() if value is not None})
    minimum = int(source_spec["correct_count_min"])
    maximum = int(source_spec["correct_count_max"])
    observation["eligible"] = bool(
        observation["complete"] and minimum <= int(correct_count) <= maximum
    )
    return observation


def _task_metadata_observations(
    rows: Sequence[dict[str, Any]],
    *,
    source_spec: Mapping[str, Any],
    artifact: str,
) -> list[dict[str, Any]]:
    metadata_key = str(source_spec["metadata_key"])
    source_label = str(source_spec["source_label"])
    scope = str(source_spec["scope"])
    observations: list[dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata") or {}
        screen = metadata.get(metadata_key)
        if not isinstance(screen, dict):
            raise ValueError(f"{task_id(row)}: missing metadata.{metadata_key}")
        observations.append(
            _base_observation(
                identifier=task_id(row),
                source=source_label,
                scope=scope,
                correct_count=int(screen["screen_correct_count"]),
                n=int(screen["screen_group_size"]),
                artifact=artifact,
                source_spec=source_spec,
                extra={
                    "db_id": row.get("db_id"),
                    "question_sha256": question_sha256(row),
                    "source_example_index": screen.get("source_example_index"),
                    "outcome_class": screen.get("screen_outcome_class"),
                },
            )
        )
    return observations


def _compact_passk_observations(
    rows: Sequence[dict[str, Any]],
    *,
    source_spec: Mapping[str, Any],
    artifact: str,
) -> list[dict[str, Any]]:
    scope = str(source_spec["scope"])
    expected_n = int(source_spec["expected_group_size"])
    observations: list[dict[str, Any]] = []
    for row in rows:
        source = str(row.get("source") or source_spec.get("source_label") or "")
        if not source:
            raise ValueError(f"{artifact}: compact row is missing source")
        n = int(row.get("n_samples") or 0)
        attempted = int(row.get("attempted_samples") or 0)
        if n != expected_n or attempted != expected_n:
            raise ValueError(
                f"{source}/{task_id(row)}: incomplete K={expected_n} screen "
                f"(n={n}, attempted={attempted})"
            )
        for field in ("protocol_version", "protocol_hash"):
            expected = source_spec.get(field)
            actual = row.get(field)
            if expected is not None and actual != expected:
                raise ValueError(
                    f"{source}/{task_id(row)}: {field} {actual!r} != {expected!r}"
                )
        observations.append(
            _base_observation(
                identifier=task_id(row),
                source=source,
                scope=scope,
                correct_count=int(row.get("sample_correct_count") or 0),
                n=n,
                artifact=artifact,
                source_spec=source_spec,
                extra={
                    "attempted_samples": attempted,
                    "legal_count": int(row.get("sample_legal_count") or 0),
                    "failure_type": row.get("failure_type"),
                    "db_id": row.get("db_id"),
                    "question_sha256": row.get("question_sha256"),
                    "source_example_index": row.get("example_index"),
                    "assistant_carrier": row.get("assistant_carrier"),
                    "temperature": row.get("temperature"),
                    "top_p": row.get("top_p"),
                    "max_tokens": row.get("max_tokens"),
                    "max_steps": row.get("max_steps"),
                    "recorded_at_utc": row.get("recorded_at_utc"),
                },
            )
        )
    return observations


def _legacy_index_observations(
    rows: Sequence[dict[str, Any]],
    *,
    source_spec: Mapping[str, Any],
    artifact: str,
) -> list[dict[str, Any]]:
    scope = str(source_spec["scope"])
    observations: list[dict[str, Any]] = []
    for row in rows:
        identifier = task_id(row)
        primary_source = str(row["source"])
        base_extra = {
            "db_id": row.get("db_id"),
            "question_sha256": row.get("question_sha256"),
            "source_example_index": row.get("source_example_index"),
        }
        observations.append(
            _base_observation(
                identifier=identifier,
                source=primary_source,
                scope=scope,
                correct_count=int(row["correct_count"]),
                n=int(row["n"]),
                artifact=artifact,
                source_spec=source_spec,
                extra=base_extra,
            )
        )
        secondary_sources = [
            str(value)
            for value in row.get("sources") or []
            if str(value) != primary_source
        ]
        if secondary_sources:
            secondary_count = row.get("newgnn_correct_count")
            if secondary_count is None or len(secondary_sources) != 1:
                raise ValueError(
                    f"{identifier}: ambiguous secondary historical observations"
                )
            observations.append(
                _base_observation(
                    identifier=identifier,
                    source=secondary_sources[0],
                    scope=scope,
                    correct_count=int(secondary_count),
                    n=int(row["n"]),
                    artifact=artifact,
                    source_spec=source_spec,
                    extra=base_extra,
                )
            )
    return observations


def load_screening_observations(
    repo_root: Path,
    source_specs: Sequence[Mapping[str, Any]],
    *,
    target: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: dict[tuple[str, str], dict[str, Any]] = {}
    artifacts: list[dict[str, Any]] = []
    for raw_spec in source_specs:
        source_spec = dict(target) | dict(raw_spec)
        path = _resolve(repo_root, str(source_spec["path"]))
        relative = str(path.relative_to(repo_root))
        rows = read_jsonl(path)
        input_format = str(source_spec["format"])
        if input_format == "task_metadata":
            parsed = _task_metadata_observations(
                rows, source_spec=source_spec, artifact=relative
            )
        elif input_format == "compact_passk":
            parsed = _compact_passk_observations(
                rows, source_spec=source_spec, artifact=relative
            )
        elif input_format == "legacy_candidate_index":
            parsed = _legacy_index_observations(
                rows, source_spec=source_spec, artifact=relative
            )
        else:
            raise ValueError(f"unsupported screening source format: {input_format}")

        for observation in parsed:
            key = (str(observation["task_id"]), str(observation["source"]))
            existing = observations.get(key)
            if existing is None:
                observations[key] = observation
                continue
            comparable = (
                "correct_count",
                "n",
                "model_checkpoint",
                "protocol_version",
                "protocol_hash",
            )
            if any(existing.get(field) != observation.get(field) for field in comparable):
                raise ValueError(
                    f"conflicting screening observation for {key}: "
                    f"{[(field, existing.get(field), observation.get(field)) for field in comparable]}"
                )
            existing_scope = str(existing["scope"])
            observation_scope = str(observation["scope"])
            existing["scope"] = max(
                (existing_scope, observation_scope),
                key=_scope_rank,
            )
            existing["eligible"] = bool(existing["eligible"] or observation["eligible"])
            if _scope_rank(observation_scope) > _scope_rank(existing_scope):
                existing["source_group"] = observation["source_group"]
            existing["evidence_artifacts"] = sorted(
                set(existing["evidence_artifacts"]) | set(observation["evidence_artifacts"])
            )
        artifacts.append(
            {
                "label": str(source_spec["label"]),
                "format": input_format,
                "scope": str(source_spec["scope"]),
                "path": relative,
                "sha256": sha256_file(path),
                "records": len(rows),
                "parsed_observations": len(parsed),
                "upstream": source_spec.get("upstream"),
            }
        )
    ordered = sorted(observations.values(), key=lambda row: (row["task_id"], row["source"]))
    return ordered, artifacts


def load_id_sets(
    repo_root: Path,
    source_specs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    output: dict[str, set[str]] = {}
    artifacts: list[dict[str, Any]] = []
    for source in source_specs:
        label = str(source["label"])
        field = str(source["field"])
        path = _resolve(repo_root, str(source["path"]))
        rows = read_jsonl(path)
        identifiers = {
            str(row[field])
            for row in rows
            if isinstance(row.get(field), str) and row[field]
        }
        if len(identifiers) != len(rows):
            raise ValueError(f"{label}: missing or duplicate {field} values")
        output[label] = identifiers
        artifacts.append(
            {
                "label": label,
                "field": field,
                "path": str(path.relative_to(repo_root)),
                "sha256": sha256_file(path),
                "records": len(rows),
                "upstream": source.get("upstream"),
            }
        )
    return output, artifacts


def _selection_observation(
    rows: Sequence[dict[str, Any]],
    precedence: Mapping[str, int],
) -> dict[str, Any]:
    eligible = [row for row in rows if row["eligible"] and row["scope"] != "diagnostic"]
    if not eligible:
        raise ValueError("candidate has no eligible non-diagnostic observation")
    return min(
        eligible,
        key=lambda row: (
            int(precedence.get(str(row["source"]), 10_000)),
            -_scope_rank(str(row["scope"])),
            str(row["source"]),
        ),
    )


def build_candidate_rows(
    *,
    tasks: Mapping[str, dict[str, Any]],
    task_owners: Mapping[str, str],
    observations: Sequence[dict[str, Any]],
    source_precedence: Sequence[str],
    id_sets: Mapping[str, set[str]],
    prior_cohort_task_ids: set[str],
) -> list[dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        by_task[str(observation["task_id"])].append(observation)
    precedence = {source: index for index, source in enumerate(source_precedence)}
    candidates: list[dict[str, Any]] = []
    for identifier, task_observations in sorted(by_task.items()):
        strict = any(
            row["eligible"] and row["scope"] == "strict"
            for row in task_observations
        )
        expanded = strict or any(
            row["eligible"] and row["scope"] == "historical"
            for row in task_observations
        )
        if not expanded:
            continue
        if identifier not in tasks:
            raise ValueError(f"eligible task {identifier} has no task payload")
        selected = _selection_observation(task_observations, precedence)
        task = tasks[identifier]
        candidate = {
            "task_id": identifier,
            "dataset": task.get("dataset"),
            "db_id": task.get("db_id"),
            "question_sha256": question_sha256(task),
            "strict_candidate": strict,
            "expanded_candidate": True,
            "historical_only": not strict,
            "selection_observation": {
                key: selected.get(key)
                for key in (
                    "source",
                    "scope",
                    "correct_count",
                    "n",
                    "model_checkpoint",
                    "protocol_version",
                    "protocol_hash",
                )
            },
            "screening_observations": task_observations,
            "task_payload_source": task_owners[identifier],
            "prior_saam700_cohort_member": identifier in prior_cohort_task_ids,
            "known_set_membership": {
                label: identifier in identifiers
                for label, identifiers in sorted(id_sets.items())
            },
        }
        candidate["fresh_under_known_exclusions"] = bool(
            not candidate["prior_saam700_cohort_member"]
            and not any(candidate["known_set_membership"].values())
        )
        candidates.append(candidate)
    return candidates


def trainer_tasks(
    candidates: Sequence[dict[str, Any]],
    tasks: Mapping[str, dict[str, Any]],
    *,
    pool_kind: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for position, candidate in enumerate(candidates):
        task = copy.deepcopy(tasks[str(candidate["task_id"])])
        source_example_index = task.get("example_index", task.get("index"))
        task["db_path"] = _portable_db_path(task.get("db_path"))
        task["example_id"] = str(candidate["task_id"])
        task["instance_id"] = str(candidate["task_id"])
        task["example_index"] = position
        task["index"] = position
        task["split"] = "train"
        metadata = dict(task.get("metadata") or {})
        selected = candidate["selection_observation"]
        metadata["screened_rl_candidate_pool"] = {
            "schema_version": SCHEMA_VERSION,
            "pool_kind": pool_kind,
            "source_example_index": source_example_index,
            "selection_source": selected["source"],
            "screen_correct_count": selected["correct_count"],
            "screen_group_size": selected["n"],
            "strict_candidate": candidate["strict_candidate"],
            "historical_only": candidate["historical_only"],
            "prior_saam700_cohort_member": candidate[
                "prior_saam700_cohort_member"
            ],
            "known_set_membership": candidate["known_set_membership"],
            "fresh_under_known_exclusions": candidate[
                "fresh_under_known_exclusions"
            ],
        }
        task["metadata"] = metadata
        output.append(task)
    return output


def jsonl_text(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )


def retained_observation_rows(
    *,
    observations: Sequence[Mapping[str, Any]],
    tasks: Mapping[str, Mapping[str, Any]],
    correct_count_min: int,
    correct_count_max: int,
) -> list[dict[str, Any]]:
    """Record screening outcomes outside the training band instead of dropping them.

    Tasks the policy never solves, or solves almost always, are not training
    candidates.  Their screening identity is still kept so later work can reuse
    them as difficulty evidence without rescanning multi-gigabyte artifacts.
    """
    rows: list[dict[str, Any]] = []
    for observation in sorted(
        observations, key=lambda item: (str(item["task_id"]), str(item["source"]))
    ):
        correct = int(observation["correct_count"])
        complete = bool(observation.get("complete"))
        if complete and correct_count_min <= correct <= correct_count_max:
            continue
        task = tasks.get(str(observation["task_id"])) or {}
        question_digest = observation.get("question_sha256")
        if not question_digest and task.get("question"):
            question_digest = question_sha256(task)
        if not complete:
            reason = "incomplete_screen"
        elif correct == 0:
            reason = "never_solved"
        else:
            reason = "always_solved"
        rows.append(
            {
                "task_id": str(observation["task_id"]),
                "dataset": task.get("dataset"),
                "db_id": observation.get("db_id") or task.get("db_id"),
                "question_sha256": question_digest,
                "source": observation.get("source"),
                "source_group": observation.get("source_group"),
                "scope": observation.get("scope"),
                "correct_count": correct,
                "n": int(observation["n"]),
                "complete": complete,
                "legal_count": observation.get("legal_count"),
                "failure_type": observation.get("failure_type"),
                "retention_reason": reason,
                "training_band": [int(correct_count_min), int(correct_count_max)],
            }
        )
    return rows


def build_pool(
    *,
    repo_root: Path,
    source_spec_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_spec = json.loads(source_spec_path.read_text(encoding="utf-8"))
    if source_spec.get("schema_version") != SOURCE_SPEC_VERSION:
        raise ValueError("unsupported screened-pool source spec")
    target = source_spec["target"]
    tasks, task_owners, task_artifacts = load_task_catalog(
        repo_root, source_spec["task_sources"]
    )
    observations, screening_artifacts = load_screening_observations(
        repo_root, source_spec["screening_sources"], target=target
    )
    id_sets, id_set_artifacts = load_id_sets(
        repo_root, source_spec.get("known_sets") or []
    )
    prior_cohort_path = _resolve(
        repo_root, str(source_spec["prior_saam700_cohort_path"])
    )
    prior_cohort_task_ids = {task_id(row) for row in read_jsonl(prior_cohort_path)}
    candidates = build_candidate_rows(
        tasks=tasks,
        task_owners=task_owners,
        observations=observations,
        source_precedence=source_spec["source_precedence"],
        id_sets=id_sets,
        prior_cohort_task_ids=prior_cohort_task_ids,
    )
    strict_candidates = [row for row in candidates if row["strict_candidate"]]
    fresh_candidates = [row for row in candidates if row["fresh_under_known_exclusions"]]
    correct_count_min = int(target["correct_count_min"])
    correct_count_max = int(target["correct_count_max"])
    retained_observations = retained_observation_rows(
        observations=observations,
        tasks=tasks,
        correct_count_min=correct_count_min,
        correct_count_max=correct_count_max,
    )

    rendered = {
        "screening_observations.jsonl": jsonl_text(observations),
        "retained_non_candidate_observations.jsonl": jsonl_text(retained_observations),
        "candidate_index.jsonl": jsonl_text(candidates),
        "strict_candidate_tasks.jsonl": jsonl_text(
            trainer_tasks(strict_candidates, tasks, pool_kind="strict")
        ),
        "expanded_candidate_tasks.jsonl": jsonl_text(
            trainer_tasks(candidates, tasks, pool_kind="expanded")
        ),
        "fresh_candidate_tasks.jsonl": jsonl_text(
            trainer_tasks(fresh_candidates, tasks, pool_kind="fresh_known_exclusions")
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_artifacts: dict[str, Any] = {}
    for filename, text in rendered.items():
        path = output_dir / filename
        atomic_write_text(path, text)
        output_artifacts[filename] = {
            "path": str(path.relative_to(repo_root)),
            "records": text.count("\n"),
            "sha256": sha256_file(path),
        }

    selection_histogram = Counter(
        int(row["selection_observation"]["correct_count"]) for row in candidates
    )
    strict_histogram = Counter(
        int(row["selection_observation"]["correct_count"])
        for row in strict_candidates
    )
    completed_histogram = Counter(
        int(row["correct_count"]) for row in observations if row.get("complete")
    )
    dataset_histogram = Counter(str(row.get("dataset")) for row in candidates)
    known_overlap = {
        label: sum(bool(row["known_set_membership"][label]) for row in candidates)
        for label in id_sets
    }
    screening_groups: dict[str, dict[str, Any]] = {}
    for group in sorted({str(row["source_group"]) for row in observations}):
        group_rows = [row for row in observations if row["source_group"] == group]
        screening_groups[group] = {
            "observations": len(group_rows),
            "unique_tasks": len({str(row["task_id"]) for row in group_rows}),
            "eligible_observations": sum(bool(row["eligible"]) for row in group_rows),
            "eligible_tasks": len(
                {
                    str(row["task_id"])
                    for row in group_rows
                    if row["eligible"]
                }
            ),
        }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "candidate_definition": (
            f"complete K=8 observation with correct_count in "
            f"[{correct_count_min},{correct_count_max}]; strict uses "
            f"{target.get('strict_source_description') or 'the explicit strict screening sources declared in the source spec'}; "
            "expanded also includes historical-scope observations; diagnostic "
            "rescreens never alter membership"
        ),
        "eligibility": {
            "expected_group_size": int(target["expected_group_size"]),
            "correct_count_min": correct_count_min,
            "correct_count_max": correct_count_max,
        },
        "unique_screening_observations": len(observations),
        "completed_observation_correct_count_histogram": {
            str(key): completed_histogram[key] for key in sorted(completed_histogram)
        },
        "retained_non_candidate_observations": len(retained_observations),
        "strict_candidates": len(strict_candidates),
        "expanded_candidates": len(candidates),
        "historical_only_candidates": len(candidates) - len(strict_candidates),
        "fresh_under_known_exclusions": len(fresh_candidates),
        "prior_saam700_cohort_overlap": sum(
            bool(row["prior_saam700_cohort_member"]) for row in candidates
        ),
        "known_set_overlap": known_overlap,
        "screening_groups": screening_groups,
        "expanded_dataset_histogram": dict(sorted(dataset_histogram.items())),
        "strict_selection_correct_count_histogram": {
            str(key): strict_histogram[key] for key in sorted(strict_histogram)
        },
        "expanded_selection_correct_count_histogram": {
            str(key): selection_histogram[key] for key in sorted(selection_histogram)
        },
        "outputs": output_artifacts,
    }
    summary_path = output_dir / "summary.json"
    atomic_write_text(
        summary_path,
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    output_artifacts["summary.json"] = {
        "path": str(summary_path.relative_to(repo_root)),
        "records": 1,
        "sha256": sha256_file(summary_path),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_inventory_not_final_training_admission",
        "source_spec": {
            "path": str(source_spec_path.relative_to(repo_root)),
            "sha256": sha256_file(source_spec_path),
        },
        "target": target,
        "task_sources": task_artifacts,
        "screening_sources": screening_artifacts,
        "known_sets": id_set_artifacts,
        "prior_saam700_cohort": {
            "path": str(prior_cohort_path.relative_to(repo_root)),
            "sha256": sha256_file(prior_cohort_path),
            "records": len(prior_cohort_task_ids),
        },
        "summary": summary,
        "admission_note": (
            "Candidate files are trainer-shaped but are not a frozen formal cohort. "
            "A formal launch must choose a scope, apply the registered exclusion policy, "
            "run fresh replay/admission gates, and freeze a separate cohort manifest."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    atomic_write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


__all__ = [
    "SCHEMA_VERSION",
    "SOURCE_SPEC_VERSION",
    "build_candidate_rows",
    "build_pool",
    "jsonl_text",
    "load_screening_observations",
    "load_task_catalog",
    "retained_observation_rows",
    "task_id",
    "trainer_tasks",
]
