#!/usr/bin/env python3
"""Streaming SFT2+Exp15 teacher union: student rollout, prefill, then update.

This is intentionally not an offline cache-and-train job.  The student is
initialized from SFT2 once.  For each frozen small batch it rolls out under its
current parameters, frozen teachers score those exact samples and attempt
parallel verified repairs, and one optimizer update occurs before the next
batch is sampled.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src" / "rl"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from protocol import PROTOCOL_VERSION  # noqa: E402
from src.rl.distillation.hf_backend import (  # noqa: E402
    MultiAdapterPolicyBackend,
    adapter_weight_path,
    sha256_file,
)
from src.rl.distillation.repair_rollout import generate_parallel_repairs  # noqa: E402
from src.rl.distillation.repair_plan import normalized_group_branch_scales  # noqa: E402
from src.rl.distillation.routing import choose_dense_teacher, route_teachers  # noqa: E402
from src.rl.distillation.streaming_checkpoint import (  # noqa: E402
    SCHEMA_VERSION as CHECKPOINT_SCHEMA_VERSION,
    jsonl_line_count,
    latest_complete_checkpoint,
    load_checkpoint_state,
    prune_complete_checkpoints,
    truncate_jsonl,
)
from src.rl.distillation.streaming_update import (  # noqa: E402
    backward_branch_dpo_surrogate,
    backward_dense_opd,
    branch_policy_token_count,
    mean_teacher_logprob,
    score_turns_for_dense_opd,
)
from src.rl.frameworks.trl.fixed_rollout_pool import serialize_episode  # noqa: E402
from src.rl.frameworks.trl.rollout import (  # noqa: E402
    RolloutSettings,
    TableAgentRolloutCollector,
)
from task_loader import load_rl_task_records  # noqa: E402


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.next.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as target:
        target.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        target.flush()
        os.fsync(target.fileno())


def adapter_for_teacher(backend, teacher: str) -> str:
    return backend.SFT2 if teacher == "sft2" else backend.EXP15


def public_repair_summary(repairs) -> dict[str, Any]:
    selection = repairs.selection
    return {
        "anchors": list(repairs.anchors),
        "selection_strength": selection.strength,
        "selected_teacher": (
            selection.candidate.teacher if selection.candidate is not None else None
        ),
        "selected_anchor_turn": (
            selection.candidate.anchor_turn if selection.candidate is not None else None
        ),
        "branch_counts": {
            "total": len(repairs.branches),
            "correct": sum(branch.correct for branch in repairs.branches),
            "legal": sum(branch.legal for branch in repairs.branches),
        },
        "strong_group_count": len(repairs.strong_groups),
        "retained_correct_branch_count": sum(
            len(group.chosen_branches) for group in repairs.strong_groups
        ),
        "by_teacher_anchor": [
            {
                "teacher": teacher,
                "anchor_turn": anchor,
                "trials": len(values),
                "correct": sum(value.correct for value in values),
                "legal": sum(value.legal for value in values),
                "errors": sum(value.errors for value in values),
            }
            for (teacher, anchor), values in sorted(
                {
                    key: [
                        branch
                        for branch in repairs.branches
                        if (branch.teacher, branch.anchor_turn) == key
                    ]
                    for key in {
                        (branch.teacher, branch.anchor_turn) for branch in repairs.branches
                    }
                }.items()
            )
        ],
    }


def serialize_repair_audit(
    repairs,
    *,
    sequence: int,
    batch_index: int,
    example_index: int,
    dpo_pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Persist every generated continuation and exact tokenized prefix."""
    used = {
        (str(row["teacher"]), int(row["anchor_turn"]), int(row["trial_index"]))
        for row in dpo_pairs
    }
    return {
        "schema_version": "streaming-teacher-union-repair-audit-v2",
        "sequence": sequence,
        "batch_index": batch_index,
        "example_index": example_index,
        "anchors": list(repairs.anchors),
        "strong_groups": [
            {
                "teacher": group.candidate.teacher,
                "anchor_turn": group.candidate.anchor_turn,
                "trials": group.candidate.trials,
                "correct": group.candidate.correct,
                "legal": group.candidate.legal,
                "retained_trial_indices": [
                    branch.trial_index for branch in group.chosen_branches
                ],
            }
            for group in repairs.strong_groups
        ],
        "dpo_pairs": dpo_pairs,
        "branches": [
            {
                "teacher": branch.teacher,
                "anchor_turn": branch.anchor_turn,
                "trial_index": branch.trial_index,
                "correct": branch.correct,
                "legal": branch.legal,
                "failure_type": branch.failure_type,
                "steps": branch.steps,
                "errors": branch.errors,
                "tokenization_warning": branch.tokenization_warning,
                "used_for_dpo": (
                    branch.teacher,
                    branch.anchor_turn,
                    branch.trial_index,
                ) in used,
                "policy_turns": [
                    {
                        "prompt_ids": list(turn.prompt_ids),
                        "response_ids": list(turn.response_ids),
                        "sampling_logprobs": list(turn.sampling_logprobs),
                    }
                    for turn in branch.policy_turns
                ],
            }
            for branch in repairs.branches
        ],
    }


def validate_eligibility(
    eligibility_path: Path,
    manifest_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = load_json(manifest_path)
    if manifest.get("status") != "frozen" or manifest.get("dataset_split") != "train":
        raise ValueError("teacher eligibility manifest is not a frozen train artifact")
    if manifest.get("protocol_version") != "version26":
        raise ValueError("streaming teacher union requires version26 rollout eligibility")
    if sha256_file(eligibility_path) != manifest.get("eligibility_sha256"):
        raise ValueError("teacher eligibility SHA256 does not match its manifest")
    rows = load_jsonl(eligibility_path)
    if len(rows) != int(manifest["tasks"]):
        raise ValueError("teacher eligibility task count drifted")
    if any(row.get("dataset_split") != "train" for row in rows):
        raise ValueError("teacher eligibility contains a non-train row")
    if any(key in row for row in rows for key in ("gold_sql", "gold_query", "reference_sql")):
        raise ValueError("teacher eligibility leaked verifier-only SQL")
    order = manifest["training_order"]
    if sha256_json(order) != manifest["training_order_sha256"]:
        raise ValueError("frozen streaming training order drifted")
    return rows, manifest


def save_training_checkpoint(
    backend,
    *,
    checkpoint_root: Path,
    run_identity_sha256: str,
    completed_batches: int,
    completed_tasks: int,
    optimizer_steps: int,
    generator_call_index: int,
    update_log: Path,
    task_log: Path,
    repair_log: Path,
    retention: int,
) -> Path:
    """Atomically persist one fully committed streaming-batch boundary."""
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    final = checkpoint_root / f"checkpoint-batch-{completed_batches:05d}"
    if final.exists():
        raise ValueError(f"refusing to overwrite checkpoint: {final}")
    temporary = checkpoint_root / f".{final.name}.next.{os.getpid()}"
    temporary.mkdir()
    adapter = backend.save_student(temporary / "student_adapter")
    backend.save_optimizer(temporary / "optimizer.pt")
    state = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "run_identity_sha256": run_identity_sha256,
        "completed_batches": completed_batches,
        "completed_tasks": completed_tasks,
        "optimizer_steps": optimizer_steps,
        "generator_call_index": generator_call_index,
        "updates_lines": jsonl_line_count(update_log),
        "tasks_lines": jsonl_line_count(task_log),
        "repair_audits_lines": jsonl_line_count(repair_log),
        "updates_sha256": sha256_file(update_log),
        "tasks_sha256": sha256_file(task_log),
        "repair_audits_sha256": sha256_file(repair_log),
        "student_adapter_relative_path": str(adapter.relative_to(temporary)),
    }
    write_json_atomic(temporary / "checkpoint_state.json", state)
    (temporary / "COMPLETE").write_text("complete\n", encoding="utf-8")
    temporary.replace(final)
    prune_complete_checkpoints(checkpoint_root, keep=retention)
    return final


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--eligibility", required=True, type=Path)
    parser.add_argument("--eligibility-manifest", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--sft2-adapter", required=True, type=Path)
    parser.add_argument("--exp15-adapter", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=120)
    parser.add_argument("--question-batch-size", type=int, default=4)
    parser.add_argument("--repair-k", type=int, default=4)
    parser.add_argument("--repair-generation-batch-size", type=int, default=8)
    parser.add_argument("--max-positive-repairs-per-group", type=int, default=2)
    parser.add_argument("--checkpoint-every-batches", type=int, default=5)
    parser.add_argument("--checkpoint-total-limit", type=int, default=3)
    parser.add_argument(
        "--resume-latest",
        action="store_true",
        help="resume only this output directory's latest complete checkpoint",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--advantage-clip", type=float, default=5.0)
    parser.add_argument("--repair-beta", type=float, default=0.1)
    parser.add_argument("--repair-lambda", type=float, default=1.0)
    parser.add_argument("--correct-preservation-scale", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--experiment-name", default="streaming_sft2_exp15_opd_repair_dpo")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if PROTOCOL_VERSION != "version26":
        raise SystemExit(
            f"refusing streaming training from mutable protocol {PROTOCOL_VERSION}; "
            "run inside a frozen version26 training runtime"
        )
    resume_checkpoint = None
    if args.output_dir.exists():
        if not args.resume_latest:
            raise SystemExit(f"refusing existing streaming output directory: {args.output_dir}")
        resume_checkpoint = latest_complete_checkpoint(args.output_dir / "checkpoints")
        if resume_checkpoint is None:
            raise SystemExit(
                f"existing output has no complete resumable checkpoint: {args.output_dir}"
            )
    if (
        args.limit < 1
        or args.question_batch_size < 1
        or args.repair_k < 2
        or args.repair_generation_batch_size < 1
        or args.max_positive_repairs_per_group < 1
        or args.checkpoint_every_batches < 1
        or args.checkpoint_total_limit < 1
    ):
        raise SystemExit("limit/batch/checkpoint values must be positive and repair-k at least two")
    args.output_dir.mkdir(parents=True, exist_ok=resume_checkpoint is not None)
    status_path = args.output_dir / "status.json"
    try:
        eligibility_rows, eligibility_manifest = validate_eligibility(
            args.eligibility,
            args.eligibility_manifest,
        )
        eligibility_by_index = {
            int(row["example_index"]): row for row in eligibility_rows
        }
        order = [
            int(index)
            for index in eligibility_manifest["training_order"]
            if int(index) in eligibility_by_index
        ][: args.limit]
        if not order:
            raise ValueError("frozen eligibility contains no trainable teacher-union tasks")
        selection_path = args.output_dir / "training_selection.jsonl"
        if resume_checkpoint is None:
            with selection_path.open("w", encoding="utf-8") as target:
                for index in order:
                    target.write(json.dumps({"example_index": index}) + "\n")
        else:
            existing_order = [
                int(row["example_index"]) for row in load_jsonl(selection_path)
            ]
            if existing_order != order:
                raise ValueError("resume selection differs from frozen training order")
        tasks = load_rl_task_records(
            args.project_root,
            split="train",
            selection=selection_path,
            context_mode="rolling-legal-history",
        )
        task_by_index = {int(row["environment"]["example_index"]): row for row in tasks}
        if set(order) != set(task_by_index):
            raise ValueError("frozen training order does not match loaded BIRD-train tasks")
        tasks = [task_by_index[index] for index in order]

        manifest = {
            "schema_version": "streaming-teacher-union-run-v2",
            "status": "initializing",
            "experiment_name": args.experiment_name,
            "protocol_version": PROTOCOL_VERSION,
            "dataset_split": "train",
            "algorithm": "action-token-mopd-plus-task-balanced-multi-repair-dpo-v2",
            "student_initialization": "sft2-checkpoint-1682-once",
            "student_reset_between_batches": False,
            "teacher_adapters_frozen": ["sft2", "exp15"],
            "frozen_teacher_device_policy": "cpu-when-inactive",
            "teacher_eligibility_sha256": sha256_file(args.eligibility),
            "training_order": order,
            "training_order_sha256": sha256_json(order),
            "tasks": len(tasks),
            "question_batch_size": args.question_batch_size,
            "repair_anchors": ["3/4", "1/2", "1/4", "start"],
            "repair_anchors_generated_conditionally": False,
            "repair_k": args.repair_k,
            "repair_generation_batch_size": args.repair_generation_batch_size,
            "repair_strong_group_policy": "all-teacher-anchor-groups-with-at-least-2-of-k",
            "max_positive_repairs_per_group": args.max_positive_repairs_per_group,
            "repair_loss_normalization": "equal-per-task-then-equal-per-group-then-branch",
            "repair_audit": "all-generated-branches-with-exact-policy-token-state",
            "dense_token_scope": "raw-json-action-and-terminal-evidence",
            "think_weight": 0.0,
            "environment_prompt_padding_weight": 0.0,
            "advantage_clip": args.advantage_clip,
            "repair_beta": args.repair_beta,
            "repair_lambda": args.repair_lambda,
            "correct_preservation_scale": args.correct_preservation_scale,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "rollout": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_steps": args.max_steps,
                "max_new_tokens": args.max_new_tokens,
                "max_context_tokens": args.max_context_tokens,
                "history_turns": args.history_turns,
                "denotation_comparison": "bird-set",
            },
            "model_path": str(args.model_path.resolve()),
            "sft2_adapter": str(args.sft2_adapter.resolve()),
            "sft2_adapter_sha256": sha256_file(adapter_weight_path(args.sft2_adapter)),
            "exp15_adapter": str(args.exp15_adapter.resolve()),
            "exp15_adapter_sha256": sha256_file(adapter_weight_path(args.exp15_adapter)),
            "seed": args.seed,
            "checkpoint_every_batches": args.checkpoint_every_batches,
            "checkpoint_total_limit": args.checkpoint_total_limit,
            "checkpoint_resume_scope": "same-run-latest-complete-only",
        }
        run_identity_sha256 = sha256_json(
            {key: value for key, value in manifest.items() if key != "status"}
        )
        manifest["run_identity_sha256"] = run_identity_sha256
        resume_state = None
        if resume_checkpoint is None:
            manifest["resume_history"] = []
        else:
            existing_manifest = load_json(args.output_dir / "run_manifest.json")
            if existing_manifest.get("run_identity_sha256") != run_identity_sha256:
                raise ValueError("resume checkpoint belongs to a different run configuration")
            resume_state = load_checkpoint_state(resume_checkpoint)
            if resume_state["run_identity_sha256"] != run_identity_sha256:
                raise ValueError("resume checkpoint run identity mismatch")
            manifest = existing_manifest
            manifest.setdefault("resume_history", []).append(
                {
                    "checkpoint": str(resume_checkpoint.resolve()),
                    "completed_batches": int(resume_state["completed_batches"]),
                    "completed_tasks": int(resume_state["completed_tasks"]),
                    "optimizer_steps": int(resume_state["optimizer_steps"]),
                }
            )
            manifest["status"] = "resuming"
        write_json_atomic(args.output_dir / "run_manifest.json", manifest)
        write_json_atomic(
            status_path,
            {
                "state": "loading_model",
                "completed_batches": (
                    int(resume_state["completed_batches"]) if resume_state else 0
                ),
                "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
            },
        )

        backend = MultiAdapterPolicyBackend(
            model_path=args.model_path,
            sft2_adapter_path=args.sft2_adapter,
            exp15_adapter_path=args.exp15_adapter,
            student_adapter_path=(
                resume_checkpoint / resume_state["student_adapter_relative_path"]
                if resume_checkpoint is not None and resume_state is not None
                else None
            ),
            max_context_tokens=args.max_context_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        if resume_checkpoint is not None:
            backend.load_optimizer(resume_checkpoint / "optimizer.pt")
        initial_generator_call_index = (
            int(resume_state["generator_call_index"]) if resume_state else 0
        )
        student_generator = backend.generator_callback(
            backend.STUDENT,
            seed=args.seed,
            initial_call_index=initial_generator_call_index,
        )
        collector = TableAgentRolloutCollector(
            backend.tokenizer,
            RolloutSettings(
                reward_mode="result-only",
                tool_scheme="atomic",
                max_steps=args.max_steps,
                max_new_tokens=args.max_new_tokens,
                max_context_tokens=args.max_context_tokens,
                history_turns=args.history_turns,
                denotation_comparison="bird-set",
                temperature=args.temperature,
                top_p=args.top_p,
            ),
            generate_batch=student_generator,
        )
        update_log = args.output_dir / "updates.jsonl"
        task_log = args.output_dir / "tasks.jsonl"
        repair_log = args.output_dir / "repair_audits.jsonl"
        repair_log.touch(exist_ok=True)
        if resume_state is not None:
            truncate_jsonl(update_log, int(resume_state["updates_lines"]))
            truncate_jsonl(task_log, int(resume_state["tasks_lines"]))
            truncate_jsonl(
                repair_log,
                int(resume_state.get("repair_audits_lines", 0)),
            )
            if sha256_file(update_log) != resume_state["updates_sha256"]:
                raise ValueError("resumed update log prefix hash mismatch")
            if sha256_file(task_log) != resume_state["tasks_sha256"]:
                raise ValueError("resumed task log prefix hash mismatch")
            expected_repair_sha256 = resume_state.get("repair_audits_sha256")
            if (
                expected_repair_sha256 is not None
                and sha256_file(repair_log) != expected_repair_sha256
            ):
                raise ValueError("resumed repair audit log prefix hash mismatch")
        completed_batches = int(resume_state["completed_batches"]) if resume_state else 0
        completed_tasks = int(resume_state["completed_tasks"]) if resume_state else 0
        optimizer_steps = int(resume_state["optimizer_steps"]) if resume_state else 0
        offsets = list(range(0, len(tasks), args.question_batch_size))
        if completed_batches > len(offsets):
            raise ValueError("checkpoint completed batch count exceeds frozen task schedule")
        for batch_index in range(completed_batches, len(offsets)):
            offset = offsets[batch_index]
            batch = tasks[offset : offset + args.question_batch_size]
            write_json_atomic(
                status_path,
                {
                    "state": "student_rollout",
                    "batch_index": batch_index,
                    "completed_tasks": completed_tasks,
                    "optimizer_steps": optimizer_steps,
                },
            )
            episodes = collector.collect(batch, trainer=None)
            backend.zero_grad()
            batch_has_gradient = False
            batch_metrics = Counter()
            for local_index, (task, episode) in enumerate(zip(batch, episodes, strict=True)):
                example_index = int(task["environment"]["example_index"])
                eligibility = eligibility_by_index[example_index]
                route = route_teachers(
                    eligibility,
                    student_correct=bool(episode.sample.correct),
                    correct_preservation_scale=args.correct_preservation_scale,
                )
                source_row = serialize_episode(
                    episode,
                    sequence=completed_tasks + local_index,
                    environment=task["environment"],
                )
                if episode.sample.audit_record.get("protocol_version") != "version26":
                    raise ValueError(
                        f"student rollout {example_index} did not use frozen version26"
                    )
                repairs = None
                if not episode.sample.correct and route.branch_candidates:
                    repairs = generate_parallel_repairs(
                        backend,
                        source_row,
                        teachers=route.branch_candidates,
                        trials_per_anchor=args.repair_k,
                        base_seed=stable_batch_seed(args.seed, batch_index, example_index),
                        max_steps=args.max_steps,
                        history_turns=args.history_turns,
                        generation_batch_size=args.repair_generation_batch_size,
                        max_correct_branches_per_group=(
                            args.max_positive_repairs_per_group
                        ),
                    )
                selected_repair_teacher = (
                    repairs.selection.candidate.teacher
                    if repairs is not None and repairs.selection.supports_dpo
                    else None
                )
                teacher_means = {}
                for teacher in route.dense_candidates:
                    try:
                        teacher_means[teacher] = mean_teacher_logprob(
                            backend,
                            episode.policy_turns,
                            teacher_adapter=adapter_for_teacher(backend, teacher),
                        )
                    except ValueError as exc:
                        if "no active student policy tokens" not in str(exc):
                            raise
                dense_teacher = choose_dense_teacher(
                    tuple(
                        teacher
                        for teacher in route.dense_candidates
                        if teacher in teacher_means
                    ),
                    teacher_means,
                    selected_repair_teacher=selected_repair_teacher,
                )
                task_metrics: dict[str, Any] = {
                    "schema_version": "streaming-teacher-union-task-v1",
                    "sequence": completed_tasks + local_index,
                    "batch_index": batch_index,
                    "example_index": example_index,
                    "category": eligibility["category"],
                    "student_correct": bool(episode.sample.correct),
                    "student_legal": bool(episode.sample.audit_record.get("legal")),
                    "student_steps": int(episode.sample.audit_record.get("steps") or 0),
                    "student_errors": int(episode.sample.audit_record.get("errors") or 0),
                    "dense_teacher": dense_teacher,
                    "teacher_mean_student_token_logprob": teacher_means,
                    "student_checkpoint_before_batch": batch_index,
                }
                per_task_scale = 1.0 / len(batch)
                per_task_has_gradient = False
                dpo_pair_audits: list[dict[str, Any]] = []
                if dense_teacher is not None:
                    scored = score_turns_for_dense_opd(
                        backend,
                        episode.policy_turns,
                        teacher_adapter=adapter_for_teacher(backend, dense_teacher),
                    )
                    if scored:
                        task_metrics.update(
                            backward_dense_opd(
                                backend,
                                scored,
                                scale=route.preservation_scale * per_task_scale,
                                advantage_clip=args.advantage_clip,
                            )
                        )
                        batch_has_gradient = True
                        per_task_has_gradient = True
                        batch_metrics["dense_tasks"] += 1
                    else:
                        task_metrics["dense_skipped_reason"] = "no_valid_policy_token_mask"
                if repairs is not None:
                    task_metrics["repair"] = public_repair_summary(repairs)
                    active_groups = []
                    for group in repairs.strong_groups:
                        anchor = group.candidate.anchor_turn
                        rejected = tuple(episode.policy_turns[anchor:])
                        if not branch_policy_token_count(backend, rejected):
                            continue
                        active = tuple(
                            branch
                            for branch in group.chosen_branches
                            if branch_policy_token_count(backend, branch.policy_turns)
                        )
                        if active:
                            active_groups.append((group, active, rejected))
                    scales = normalized_group_branch_scales(
                        (len(active) for _, active, _ in active_groups),
                        total_scale=args.repair_lambda * per_task_scale,
                    )
                    for (group, active, rejected), group_scales in zip(
                        active_groups, scales, strict=True
                    ):
                        for branch, pair_scale in zip(active, group_scales, strict=True):
                            diagnostics = backward_branch_dpo_surrogate(
                                backend,
                                branch.policy_turns,
                                rejected,
                                reference_adapter=backend.SFT2,
                                beta=args.repair_beta,
                                scale=pair_scale,
                            )
                            dpo_pair_audits.append(
                                {
                                    "teacher": group.candidate.teacher,
                                    "anchor_turn": group.candidate.anchor_turn,
                                    "trial_index": branch.trial_index,
                                    "scale": pair_scale,
                                    **diagnostics,
                                }
                            )
                    if dpo_pair_audits:
                        task_metrics["repair_dpo"] = {
                            "active_groups": len(active_groups),
                            "pairs": len(dpo_pair_audits),
                            "total_scale": sum(
                                float(row["scale"]) for row in dpo_pair_audits
                            ),
                            "pair_diagnostics": dpo_pair_audits,
                        }
                        batch_has_gradient = True
                        per_task_has_gradient = True
                        batch_metrics["repair_dpo_tasks"] += 1
                        batch_metrics["repair_dpo_groups"] += len(active_groups)
                        batch_metrics["repair_dpo_pairs"] += len(dpo_pair_audits)
                    elif repairs.strong_groups:
                        task_metrics["repair_dpo_skipped_reason"] = (
                            "chosen_or_rejected_branch_has_no_valid_policy_token_mask"
                        )
                    elif repairs.selection.strength == "weak":
                        batch_metrics["weak_repair_tasks"] += 1
                    append_jsonl(
                        repair_log,
                        serialize_repair_audit(
                            repairs,
                            sequence=completed_tasks + local_index,
                            batch_index=batch_index,
                            example_index=example_index,
                            dpo_pairs=dpo_pair_audits,
                        ),
                    )
                if not per_task_has_gradient:
                    batch_metrics["no_update_tasks"] += 1
                append_jsonl(task_log, task_metrics)

            gradient_norm = None
            if batch_has_gradient:
                gradient_norm = backend.step(gradient_clip=args.gradient_clip)
                optimizer_steps += 1
            completed_tasks += len(batch)
            update_row = {
                "schema_version": "streaming-teacher-union-update-v1",
                "batch_index": batch_index,
                "example_indices": [
                    int(task["environment"]["example_index"]) for task in batch
                ],
                "student_checkpoint_before": batch_index,
                "student_checkpoint_after": batch_index + int(batch_has_gradient),
                "optimizer_step_applied": batch_has_gradient,
                "gradient_norm": gradient_norm,
                "counts": dict(batch_metrics),
                "completed_tasks": completed_tasks,
                "optimizer_steps": optimizer_steps,
            }
            append_jsonl(update_log, update_row)
            write_json_atomic(status_path, {"state": "trained_batch", **update_row})
            completed_batches = batch_index + 1
            if (
                completed_batches % args.checkpoint_every_batches == 0
                or completed_batches == len(offsets)
            ):
                checkpoint = save_training_checkpoint(
                    backend,
                    checkpoint_root=args.output_dir / "checkpoints",
                    run_identity_sha256=run_identity_sha256,
                    completed_batches=completed_batches,
                    completed_tasks=completed_tasks,
                    optimizer_steps=optimizer_steps,
                    generator_call_index=int(student_generator.call_index),
                    update_log=update_log,
                    task_log=task_log,
                    repair_log=repair_log,
                    retention=args.checkpoint_total_limit,
                )
                manifest["latest_checkpoint"] = str(checkpoint.resolve())
                manifest["latest_checkpoint_completed_batches"] = completed_batches
                write_json_atomic(args.output_dir / "run_manifest.json", manifest)
                write_json_atomic(
                    status_path,
                    {
                        "state": "checkpointed",
                        **update_row,
                        "checkpoint": str(checkpoint.resolve()),
                    },
                )

        final_dir = args.output_dir / "final"
        final_temporary = args.output_dir / f".final.next.{os.getpid()}"
        saved_adapter = backend.save_student(final_temporary)
        adapter_relative = saved_adapter.relative_to(final_temporary)
        final_temporary.replace(final_dir)
        final_adapter = final_dir / adapter_relative
        manifest["status"] = "complete"
        manifest["optimizer_steps"] = optimizer_steps
        manifest["completed_tasks"] = completed_tasks
        manifest["final_adapter"] = str(final_adapter.resolve())
        manifest["updates_sha256"] = sha256_file(update_log)
        manifest["tasks_log_sha256"] = sha256_file(task_log)
        manifest["repair_audits_sha256"] = sha256_file(repair_log)
        manifest["repair_audits"] = str(repair_log.resolve())
        write_json_atomic(args.output_dir / "run_manifest.json", manifest)
        write_json_atomic(
            status_path,
            {
                "state": "complete",
                "completed_tasks": completed_tasks,
                "optimizer_steps": optimizer_steps,
                "final_adapter": str(final_adapter.resolve()),
            },
        )
    except Exception as exc:
        write_json_atomic(
            status_path,
            {
                "state": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def stable_batch_seed(base_seed: int, batch_index: int, example_index: int) -> int:
    payload = f"{base_seed}\0{batch_index}\0{example_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF


if __name__ == "__main__":
    main()
