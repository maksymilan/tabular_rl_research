#!/usr/bin/env python3
"""Audit paired K8 SFT1/checkpoint-{4,6} results on frozen missing32 cohorts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.rl.evaluation.prepare_earlystop_missing32_gate import (
    EXPECTED_EARLYSTOP_MANIFEST_SHA256,
    EXPECTED_OUTPUT_RECORDS,
    EXPECTED_SOURCE_TASKS_SHA256,
    SCHEMA_VERSION as COHORT_SCHEMA,
    STATUS as COHORT_STATUS,
    task_id,
)


SCHEMA_VERSIONS = {
    "checkpoint-4": "qwen3-v26-earlystop-missing32-checkpoint4-paired-behavior-gate-v1",
    "checkpoint-6": "qwen3-v26-earlystop-missing32-checkpoint6-paired-behavior-gate-v1",
}
EXPECTED_PROTOCOL_VERSION = "version26"
EXPECTED_PROTOCOL_HASH = "4da19387399bd3a5"
EXPECTED_RUNTIME_TREE = "5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab"
EXPECTED_STUDENT_PROMPT = "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
EXPECTED_SFT1_ADAPTER = "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5"
GROUP_SIZE = 8
TRAJECTORIES_PER_ARM = EXPECTED_OUTPUT_RECORDS * GROUP_SIZE
ARM_CONTRACTS = {
    "checkpoint-4": {"step": 4, "seed": 20260816},
    "checkpoint-6": {"step": 6, "seed": 20260817},
}
BOOTSTRAP_REPLICATES = 20000
MINIMUM_GAIN_PP = 5.0
ONE_SIDED_CONFIDENCE = 0.80
MAX_LEGAL_REGRESSION = 5


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def parse_jsonl(payload: bytes, *, path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(payload.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: row must be an object")
        rows.append(value)
    return rows


def _timeout_evidence(value: Any) -> bool:
    if isinstance(value, dict):
        for key, nested in value.items():
            if (
                key in {"failure_type", "error_type", "execution_error_type", "recovered_from_error_type"}
                and nested == "timeout_error"
            ) or (key in {"error_code", "code"} and nested == "tool_execution_timeout"):
                return True
            if isinstance(nested, (dict, list)) and _timeout_evidence(nested):
                return True
    if isinstance(value, list):
        return any(_timeout_evidence(item) for item in value)
    return False


def _legal(sample: Mapping[str, Any], audit: Mapping[str, Any]) -> bool:
    result_reward = audit.get("result_reward") or {}
    values = [audit.get("legal"), sample.get("legal"), result_reward.get("executable_terminal")]
    present = [value for value in values if isinstance(value, bool)]
    if not present:
        raise ValueError("trajectory lacks an explicit legal marker")
    if len(set(present)) != 1:
        raise ValueError("trajectory legal markers disagree")
    return present[0]


def _clean(sample: Mapping[str, Any], audit: Mapping[str, Any]) -> bool:
    return (
        sample.get("process_update") is True
        and sample.get("failure_type") not in {"generation_oom", "generation_length", "context_overflow", "timeout_error"}
        and audit.get("generation_truncation") is None
        and audit.get("optimization_exclusion") is None
        and audit.get("rollout_tokenization_warning") is not True
        and not _timeout_evidence(audit.get("turns"))
        and not _timeout_evidence(audit.get("error_events"))
    )


def _policy_evidence_valid(turns: Any) -> bool:
    if not isinstance(turns, list) or not turns:
        return False
    for turn in turns:
        if not isinstance(turn, dict):
            return False
        prompt = turn.get("prompt_ids")
        response = turn.get("response_ids")
        logprobs = turn.get("sampling_logprobs")
        if (
            not isinstance(prompt, list)
            or not prompt
            or not isinstance(response, list)
            or not response
            or not isinstance(logprobs, list)
            or len(response) != len(logprobs)
            or not all(type(value) is int for value in prompt + response)
            or not all(type(value) in {int, float} and math.isfinite(float(value)) for value in logprobs)
        ):
            return False
    return True


def audit_arm(
    *,
    arm: str,
    manifest: Mapping[str, Any],
    trajectories: Sequence[dict[str, Any]],
    task_rows: Sequence[dict[str, Any]],
    tasks_sha256: str,
    trajectories_sha256: str,
    expected_adapter_sha256: str | None,
    evaluation_seed: int,
) -> dict[str, Any]:
    expected_manifest = {
        "schema_version": "table-agent-fixed-rollout-pool-pending-v1",
        "status": "generated_pending_counterfactual_validation",
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "protocol_hash": EXPECTED_PROTOCOL_HASH,
        "protocol_runtime_content_tree_sha256": EXPECTED_RUNTIME_TREE,
        "student_prompt_sha256": EXPECTED_STUDENT_PROMPT,
        "tasks_sha256": tasks_sha256,
        "tasks": EXPECTED_OUTPUT_RECORDS,
        "group_size": GROUP_SIZE,
        "trajectories": TRAJECTORIES_PER_ARM,
        "reward_mode": "result-only",
        "result_reward_profile": "binary",
        "temperature": 0.8,
        "top_p": 1.0,
        "max_steps": 30,
        "max_new_tokens": 2048,
        "max_context_tokens": 16384,
        "history_turns": 4,
        "enable_thinking": True,
        "seed": evaluation_seed,
        "generation_seed_scheme": "sha256-task-sample-turn-v1",
        "denotation_comparison": "bird-set",
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(f"{arm} manifest {field} mismatch")
    if manifest.get("trajectories_sha256") != trajectories_sha256:
        raise ValueError(f"{arm} trajectory digest mismatch")
    adapter_sha = manifest.get("adapter_sha256")
    if expected_adapter_sha256 is not None and adapter_sha != expected_adapter_sha256:
        raise ValueError(f"{arm} adapter SHA-256 mismatch")
    if not isinstance(adapter_sha, str) or len(adapter_sha) != 64:
        raise ValueError(f"{arm} adapter SHA-256 is invalid")

    tasks = {task_id(row): row for row in task_rows}
    groups: dict[str, list[tuple[int, bool, bool]]] = defaultdict(list)
    for position, row in enumerate(trajectories):
        if row.get("schema_version") != "table-agent-fixed-policy-episode-v1":
            raise ValueError(f"{arm} trajectory schema mismatch at {position}")
        if row.get("sequence") != position:
            raise ValueError(f"{arm} trajectory sequence mismatch at {position}")
        environment = row.get("environment") or {}
        sample = row.get("sample") or {}
        audit = sample.get("audit_record") or {}
        identity = environment.get("task_id")
        if identity not in tasks:
            raise ValueError(f"{arm} unknown task at {position}")
        task = tasks[identity]
        expected_environment = {
            "task_id": identity,
            "example_index": task["example_index"],
            "db_id": task["db_id"],
            "db_path": task["db_path"],
            "question": task["question"],
            "gold_sql": task["gold_sql"],
            "external_knowledge": task.get("external_knowledge"),
        }
        if any(environment.get(field) != value for field, value in expected_environment.items()):
            raise ValueError(f"{arm} environment/task mismatch at {position}")
        sample_index = audit.get("sample_index")
        if type(sample_index) is not int:
            raise ValueError(f"{arm} sample index missing at {position}")
        if audit.get("protocol_version") != EXPECTED_PROTOCOL_VERSION or audit.get("protocol_hash") != EXPECTED_PROTOCOL_HASH:
            raise ValueError(f"{arm} trajectory protocol mismatch at {position}")
        optimizer_evidence = {
            field: audit[field]
            for field in ("policy_global_step", "policy_micro_step", "optimizer_step", "global_step")
            if field in audit
        }
        if any(type(value) is not int or value != 0 for value in optimizer_evidence.values()):
            raise ValueError(f"{arm} fixed-policy trajectory has optimizer evidence at {position}")
        if not _policy_evidence_valid(row.get("policy_turns")):
            raise ValueError(f"{arm} policy token/logprob evidence invalid at {position}")
        correct = sample.get("correct")
        reward = sample.get("reward")
        result_reward = audit.get("result_reward") or {}
        if not (
            isinstance(correct, bool)
            and type(reward) in {int, float}
            and math.isfinite(float(reward))
            and float(reward) == float(correct)
            and result_reward.get("profile") == "binary"
            and result_reward.get("correct") is correct
            and float(result_reward.get("value", -1.0)) == float(reward)
            and sample.get("step_rewards") is None
            and not audit.get("process_reward")
        ):
            raise ValueError(f"{arm} binary result reward mismatch at {position}")
        clean = _clean(sample, audit)
        legal = _legal(sample, audit)
        if not clean:
            raise ValueError(f"{arm} runtime-contaminated trajectory at {position}")
        groups[identity].append((sample_index, correct, legal))
    if len(trajectories) != TRAJECTORIES_PER_ARM or set(groups) != set(tasks):
        raise ValueError(f"{arm} does not contain exact 32xK8 trajectories")

    summaries = {}
    total_correct = total_legal = 0
    for identity in [task_id(row) for row in task_rows]:
        entries = sorted(groups[identity])
        if [entry[0] for entry in entries] != list(range(GROUP_SIZE)):
            raise ValueError(f"{arm} task {identity} lacks exact sample indices 0..7")
        correct = sum(int(entry[1]) for entry in entries)
        legal = sum(int(entry[2]) for entry in entries)
        summaries[identity] = {"correct": correct, "legal": legal}
        total_correct += correct
        total_legal += legal
    return {
        "arm": arm,
        "adapter_sha256": adapter_sha,
        "tasks": EXPECTED_OUTPUT_RECORDS,
        "trajectories": TRAJECTORIES_PER_ARM,
        "runtime_clean_trajectories": TRAJECTORIES_PER_ARM,
        "correct": total_correct,
        "legal": total_legal,
        "groups": summaries,
    }


def cluster_bootstrap_lower(
    differences: Sequence[float], *, bootstrap_seed: int = 20260817
) -> float:
    if len(differences) != EXPECTED_OUTPUT_RECORDS:
        raise ValueError("cluster bootstrap requires exactly 32 task differences")
    rng = random.Random(bootstrap_seed)
    means = []
    for _ in range(BOOTSTRAP_REPLICATES):
        means.append(sum(rng.choice(differences) for _ in differences) / len(differences))
    means.sort()
    # One-sided 80% lower confidence bound is the 20th percentile.
    rank = max(0, math.ceil((1.0 - ONE_SIDED_CONFIDENCE) * len(means)) - 1)
    return means[rank]


def audit_pair(
    *,
    cohort_manifest: Mapping[str, Any],
    tasks: Sequence[dict[str, Any]],
    tasks_sha256: str,
    sft1_manifest: Mapping[str, Any],
    sft1_rows: Sequence[dict[str, Any]],
    sft1_rows_sha256: str,
    checkpoint_manifest: Mapping[str, Any],
    checkpoint_rows: Sequence[dict[str, Any]],
    checkpoint_rows_sha256: str,
    expected_checkpoint_adapter_sha256: str,
    checkpoint_arm: str = "checkpoint-6",
) -> dict[str, Any]:
    if checkpoint_arm not in ARM_CONTRACTS:
        raise ValueError("checkpoint arm must be checkpoint-4 or checkpoint-6")
    arm_contract = ARM_CONTRACTS[checkpoint_arm]
    evaluation_seed = int(arm_contract["seed"])
    if cohort_manifest.get("schema_version") != COHORT_SCHEMA or cohort_manifest.get("status") != COHORT_STATUS:
        raise ValueError("unsupported missing32 cohort manifest")
    if (cohort_manifest.get("inputs") or {}).get("source_tasks", {}).get("sha256") != EXPECTED_SOURCE_TASKS_SHA256:
        raise ValueError("missing32 cohort source600 binding mismatch")
    if (cohort_manifest.get("inputs") or {}).get("earlystop_manifest", {}).get("sha256") != EXPECTED_EARLYSTOP_MANIFEST_SHA256:
        raise ValueError("missing32 cohort early-stop binding mismatch")
    ordered_ids = [task_id(row) for row in tasks]
    if len(tasks) != EXPECTED_OUTPUT_RECORDS or len(set(ordered_ids)) != EXPECTED_OUTPUT_RECORDS:
        raise ValueError("missing32 task identity/count mismatch")
    if (cohort_manifest.get("selection") or {}).get("ordered_task_ids") != ordered_ids:
        raise ValueError("missing32 ordered identities differ from cohort manifest")
    if (cohort_manifest.get("output") or {}).get("sha256") != tasks_sha256:
        raise ValueError("missing32 task digest differs from cohort manifest")
    cohort_eval = cohort_manifest.get("evaluation_contract") or {}
    if (
        cohort_eval.get("arms") != ["sft1", checkpoint_arm]
        or cohort_eval.get("seed") != evaluation_seed
    ):
        raise ValueError("missing32 cohort arm/seed contract mismatch")
    sft1 = audit_arm(
        arm="sft1", manifest=sft1_manifest, trajectories=sft1_rows,
        task_rows=tasks, tasks_sha256=tasks_sha256,
        trajectories_sha256=sft1_rows_sha256,
        expected_adapter_sha256=EXPECTED_SFT1_ADAPTER,
        evaluation_seed=evaluation_seed,
    )
    checkpoint = audit_arm(
        arm=checkpoint_arm, manifest=checkpoint_manifest, trajectories=checkpoint_rows,
        task_rows=tasks, tasks_sha256=tasks_sha256,
        trajectories_sha256=checkpoint_rows_sha256,
        expected_adapter_sha256=expected_checkpoint_adapter_sha256,
        evaluation_seed=evaluation_seed,
    )
    if checkpoint["adapter_sha256"] == sft1["adapter_sha256"]:
        raise ValueError(f"{checkpoint_arm} adapter is not distinct from SFT1")

    differences = [
        (checkpoint["groups"][identity]["correct"] - sft1["groups"][identity]["correct"]) / GROUP_SIZE
        for identity in ordered_ids
    ]
    gain = sum(differences) / len(differences)
    lower = cluster_bootstrap_lower(differences, bootstrap_seed=evaluation_seed)
    legal_delta = checkpoint["legal"] - sft1["legal"]
    checks = {
        "strict_identity_and_runtime_clean": True,
        "exact_paired_32xk8": True,
        "mean_gain_at_least_5pp": gain * 100.0 >= MINIMUM_GAIN_PP,
        "one_sided_80pct_cluster_bootstrap_lower_above_zero": lower > 0.0,
        "legal_regression_at_most_5_of_256": legal_delta >= -MAX_LEGAL_REGRESSION,
    }
    green = all(checks.values())
    task_deltas = [
        {
            "task_id": identity,
            "db_id": next(row["db_id"] for row in tasks if task_id(row) == identity),
            "sft1_correct": sft1["groups"][identity]["correct"],
            f"checkpoint{arm_contract['step']}_correct": checkpoint["groups"][identity]["correct"],
            "correct_delta": checkpoint["groups"][identity]["correct"] - sft1["groups"][identity]["correct"],
            "sft1_legal": sft1["groups"][identity]["legal"],
            f"checkpoint{arm_contract['step']}_legal": checkpoint["groups"][identity]["legal"],
        }
        for identity in ordered_ids
    ]
    db_clusters = Counter(row["db_id"] for row in tasks)
    return {
        "schema_version": SCHEMA_VERSIONS[checkpoint_arm],
        "status": {
            "green_light": green,
            "interpretation": "early_behavior_signal_only",
            "formal_dev_consumed": False,
            "formal_checkpoint_selection_authority": False,
        },
        "contract": {
            "tasks": 32, "group_size": 8, "trajectories_per_arm": 256,
            "checkpoint_arm": checkpoint_arm,
            "checkpoint_global_step": arm_contract["step"],
            "seed": evaluation_seed, "temperature": 0.8, "top_p": 1.0,
            "max_new_tokens": 2048, "max_context_tokens": 16384,
            "minimum_mean_gain_percentage_points": MINIMUM_GAIN_PP,
            "cluster_unit": "task", "one_sided_confidence": ONE_SIDED_CONFIDENCE,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": evaluation_seed,
            "maximum_legal_regression_trajectories": MAX_LEGAL_REGRESSION,
        },
        "arms": {"sft1": {k: v for k, v in sft1.items() if k != "groups"}, checkpoint_arm: {k: v for k, v in checkpoint.items() if k != "groups"}},
        "observed": {
            "accuracy_gain_percentage_points": gain * 100.0,
            "one_sided_80pct_cluster_bootstrap_lower_percentage_points": lower * 100.0,
            "legal_delta_of_256": legal_delta,
            "improved_tasks": sum(value > 0 for value in differences),
            "regressed_tasks": sum(value < 0 for value in differences),
            "tied_tasks": sum(value == 0 for value in differences),
            "database_clusters": dict(sorted(db_clusters.items())),
        },
        "checks": checks,
        "task_deltas": task_deltas,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-manifest", type=Path, required=True)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--sft1-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-adapter-sha256", required=True)
    parser.add_argument(
        "--checkpoint-arm", choices=tuple(ARM_CONTRACTS), default="checkpoint-6"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        task_bytes = args.tasks.read_bytes()
        sft_rows_bytes = (args.sft1_dir / "trajectories.jsonl").read_bytes()
        cp_rows_bytes = (args.checkpoint_dir / "trajectories.jsonl").read_bytes()
        result = audit_pair(
            cohort_manifest=json.loads(args.cohort_manifest.read_bytes()),
            tasks=parse_jsonl(task_bytes, path=args.tasks),
            tasks_sha256=sha256_bytes(task_bytes),
            sft1_manifest=json.loads((args.sft1_dir / "manifest.pending.json").read_bytes()),
            sft1_rows=parse_jsonl(sft_rows_bytes, path=args.sft1_dir / "trajectories.jsonl"),
            sft1_rows_sha256=sha256_bytes(sft_rows_bytes),
            checkpoint_manifest=json.loads((args.checkpoint_dir / "manifest.pending.json").read_bytes()),
            checkpoint_rows=parse_jsonl(cp_rows_bytes, path=args.checkpoint_dir / "trajectories.jsonl"),
            checkpoint_rows_sha256=sha256_bytes(cp_rows_bytes),
            expected_checkpoint_adapter_sha256=args.expected_checkpoint_adapter_sha256,
            checkpoint_arm=args.checkpoint_arm,
        )
        if args.output.exists() or args.output.is_symlink():
            raise ValueError(f"refusing existing output: {args.output}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".next")
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(args.output)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        print(f"early behavior gate blocked: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps({"green_light": result["status"]["green_light"], **result["observed"]}, sort_keys=True))
    return 0 if result["status"]["green_light"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
