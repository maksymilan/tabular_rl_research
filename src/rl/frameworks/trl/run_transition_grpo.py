#!/usr/bin/env python3
"""Launch TRL-backed clipped policy optimization over exact table-agent turns."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import bitsandbytes as bnb
import torch
from datasets import Dataset
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import GRPOConfig


ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [
    str(ROOT / "src" / "rl"),
    str(ROOT / "src" / "eval"),
    str(ROOT / "src" / "harness"),
    str(ROOT / "src" / "sft"),
]

from counterfactual_suite import (  # noqa: E402
    SCHEMA_VERSION as COUNTERFACTUAL_SCHEMA_VERSION,
    CounterfactualSuiteManifest,
    load_counterfactual_suite_manifest,
)
from experiment_config import RLExperimentConfig  # noqa: E402
from process_credit import ProcessRewardConfig  # noqa: E402
from protocol import student_runtime_system_prompt, tool_schema_hash  # noqa: E402
from reference_result_filter import filter_training_records  # noqa: E402
from task_loader import load_rl_task_records  # noqa: E402
from frameworks.trl.rollout import (  # noqa: E402
    RolloutSettings,
    TableAgentRolloutCollector,
)
from frameworks.trl.fixed_rollout_pool import FixedPoolRolloutCollector  # noqa: E402
from frameworks.trl.transition_grpo import TransitionGRPOTrainer  # noqa: E402


def parse_args() -> tuple[argparse.Namespace, RLExperimentConfig | None]:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--experiment-config", type=Path)
    bootstrap_args, _ = bootstrap.parse_known_args()
    experiment_config = (
        RLExperimentConfig.load(bootstrap_args.experiment_config)
        if bootstrap_args.experiment_config is not None
        else None
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--examples-json", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument(
        "--example-index",
        type=int,
        help="retain one exact example_index for deterministic debugging",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--reward-mode", choices=("result-only", "process"), default="result-only")
    parser.add_argument("--process-reward-config", type=Path)
    parser.add_argument(
        "--process-admission-policy",
        choices=(
            "counterfactual-completeness",
            "counterfactual-screened",
            "rank-local-features",
            "dense-outcome",
            "denotation-nonempty",
        ),
        default="counterfactual-completeness",
    )
    parser.add_argument("--counterfactual-suite-manifest", type=Path)
    parser.add_argument("--exclude-empty-reference-results", action="store_true")
    parser.add_argument("--optimizer-steps", type=int, default=200)
    parser.add_argument("--ppo-iterations", type=int, default=2)
    parser.add_argument("--prompts-per-update", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--transition-micro-batch-size", type=int, default=2)
    parser.add_argument("--policy-loss-coefficient", type=float, default=1.0)
    parser.add_argument("--train-turns", choices=("all", "last"), default="all")
    parser.add_argument("--trainable-part", choices=("all", "tool_only"), default="all")
    parser.add_argument("--rank-loss-coefficient", type=float, default=0.0)
    parser.add_argument("--rank-beta", type=float, default=0.1)
    parser.add_argument(
        "--rank-score-tokens",
        choices=("all", "tool_only"),
        default="all",
    )
    parser.add_argument(
        "--rank-update-scope",
        choices=("full_trajectory", "conservative_legal", "dense_outcome"),
        default="full_trajectory",
    )
    parser.add_argument(
        "--rank-score-scope",
        choices=("full_trajectory", "conservative_legal", "dense_outcome"),
        default="full_trajectory",
    )
    parser.add_argument(
        "--rank-score-reduction",
        choices=("sum_tokens", "mean_action"),
        default="sum_tokens",
    )
    parser.add_argument(
        "--optimizer-name",
        choices=("adamw_torch", "paged_adamw_8bit"),
        default="paged_adamw_8bit",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-7)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("constant", "cosine"),
        default="constant",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--kl-beta", type=float, default=0.001)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--max-agent-steps", type=int, default=30)
    parser.add_argument("--max-batch-calls", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--history-turns", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--fixed-rollout-pool", type=Path)
    parser.add_argument("--fixed-pool-manifest", type=Path)
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-group-port", type=int, default=51216)
    if experiment_config is not None:
        parser.set_defaults(**experiment_config.argparse_defaults(ROOT))
    return parser.parse_args(), experiment_config


def load_process_config(path: Path | None) -> ProcessRewardConfig:
    if path is None:
        return ProcessRewardConfig()
    values = {
        key: value
        for key, value in json.loads(path.read_text(encoding="utf-8")).items()
        if not key.startswith("_")
    }
    config = ProcessRewardConfig(**values)
    config.validate()
    return config


def sha256_file(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adapter_weight_path(adapter_path: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = adapter_path / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"adapter has no weight file: {adapter_path}")


def validate_fixed_pool_manifest(args: argparse.Namespace) -> dict[str, Any] | None:
    """Fail closed unless offline training consumes the exact frozen SFT2 K4 pool."""
    if args.fixed_pool_manifest is None:
        return None
    manifest_path = args.fixed_pool_manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_status = {
        "counterfactual-completeness": "frozen_passed",
        "counterfactual-screened": "frozen_process_screened",
        "rank-local-features": "frozen_rank_ready",
        "dense-outcome": "frozen_dense_ready",
    }.get(args.process_admission_policy)
    if expected_status is None:
        raise ValueError(
            "fixed-pool training requires a fixed-pool admission policy, got "
            f"{args.process_admission_policy!r}"
        )
    manifest_tasks = int(payload.get("tasks") or 0)
    manifest_trajectories = int(payload.get("trajectories") or 0)
    if manifest_tasks < 1 or manifest_trajectories != manifest_tasks * args.group_size:
        raise ValueError("fixed-pool manifest task/trajectory counts are inconsistent")
    expected_scalars = {
        "schema_version": "table-agent-fixed-rollout-pool-v1",
        "status": expected_status,
        "group_size": 4,
        "protocol_version": "version26",
        "temperature": 0.7,
        "top_p": 0.95,
        "max_steps": 30,
        "history_turns": 4,
    }
    for key, expected in expected_scalars.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"fixed-pool manifest {key} mismatch: {payload.get(key)!r} != {expected!r}"
            )
    runtime_scalars = {
        "group_size": args.group_size,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_steps": args.max_agent_steps,
        "history_turns": args.history_turns,
    }
    for key, actual in runtime_scalars.items():
        if actual != expected_scalars[key]:
            raise ValueError(
                f"offline runtime {key} mismatch: {actual!r} != {expected_scalars[key]!r}"
            )
    difficulty_counts = payload.get("difficulty_counts") or {}
    if set(difficulty_counts) != {"challenging", "moderate", "simple"}:
        raise ValueError("fixed-pool manifest is missing difficulty counts")
    if sum(int(value) for value in difficulty_counts.values()) != manifest_tasks:
        raise ValueError("fixed-pool difficulty counts do not sum to tasks")
    if len({int(value) for value in difficulty_counts.values()}) != 1:
        raise ValueError("fixed-pool manifest is not difficulty-balanced")
    reuse = payload.get("reuse_contract") or {}
    same_trajectories = reuse.get("same_trajectories") is True or (
        manifest_trajectories == 240 and reuse.get("same_240_trajectories") is True
    )
    if not same_trajectories or not all(
        reuse.get(key) is True
        for key in ("same_sequence", "online_resampling_forbidden")
    ):
        raise ValueError("fixed-pool reuse contract is incomplete")
    files = payload.get("files") or {}
    pool_record = files.get("validated_trajectories") or {}
    pool_path = args.fixed_rollout_pool.resolve()
    recorded_pool_path = Path(str(pool_record.get("path", ""))).resolve()
    if recorded_pool_path != pool_path:
        raise ValueError(
            f"fixed rollout path mismatch: {pool_path} != {recorded_pool_path}"
        )
    actual_pool_sha = sha256_file(pool_path)
    if pool_record.get("sha256") != actual_pool_sha:
        raise ValueError("fixed rollout pool SHA-256 does not match frozen manifest")
    process_record = payload.get("process_reward_config") or {}
    actual_process_sha = sha256_file(args.process_reward_config)
    if process_record.get("sha256") != actual_process_sha:
        raise ValueError("process reward config SHA-256 does not match frozen manifest")
    actual_adapter_sha = sha256_file(_adapter_weight_path(args.adapter_path))
    if payload.get("initial_adapter_sha256") != actual_adapter_sha:
        raise ValueError("training adapter is not the frozen SFT2 checkpoint-1682 adapter")
    if args.optimizer_steps != manifest_tasks or args.prompts_per_update != 1:
        raise ValueError(
            "controlled fixed-pool training requires exactly one optimizer step per task"
        )
    return payload


def require_clean_output(path: Path, resume: Path | None) -> None:
    if resume is not None:
        path.mkdir(parents=True, exist_ok=True)
        return
    if path.exists() and any(path.iterdir()):
        raise SystemExit(
            f"output directory is not empty: {path}; use an isolated directory"
        )
    path.mkdir(parents=True, exist_ok=True)


def load_counterfactual_suites(
    manifest_path: Path | None,
    records: list[dict[str, Any]],
) -> tuple[CounterfactualSuiteManifest | None, dict[str, Any]]:
    if manifest_path is None:
        return None, {}
    manifest = load_counterfactual_suite_manifest(manifest_path)
    suites = {}
    for record in records:
        metadata = record["environment"]
        suites[str(metadata["task_id"])] = manifest.suite_for(metadata)
    return manifest, suites


def load_qlora_model(args: argparse.Namespace):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
    )
    adapter_path = args.resume_from_checkpoint or args.adapter_path
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        is_trainable=True,
    )
    model.gradient_checkpointing_enable()
    return model, tokenizer


def main() -> None:
    args, experiment_config = parse_args()
    require_clean_output(args.output_dir, args.resume_from_checkpoint)
    if args.prompts_per_update < 1 or args.group_size < 2:
        raise SystemExit("prompts-per-update must be positive and group-size must be at least 2")
    if args.ppo_iterations < 1:
        raise SystemExit("ppo-iterations must be positive")
    if args.rank_loss_coefficient < 0:
        raise SystemExit("rank-loss-coefficient must be non-negative")
    if args.policy_loss_coefficient < 0:
        raise SystemExit("policy-loss-coefficient must be non-negative")
    if args.policy_loss_coefficient == 0.0 and args.rank_loss_coefficient == 0.0:
        raise SystemExit("at least one policy or rank loss must be enabled")
    if (args.fixed_rollout_pool is None) != (args.fixed_pool_manifest is None):
        raise SystemExit(
            "--fixed-rollout-pool and --fixed-pool-manifest must be provided together"
        )
    fixed_pool_manifest = validate_fixed_pool_manifest(args)
    if args.rank_beta <= 0:
        raise SystemExit("rank-beta must be positive")
    if (
        args.rank_update_scope == "conservative_legal"
        and args.rank_score_tokens != "tool_only"
    ):
        raise SystemExit(
            "conservative_legal rank updates require --rank-score-tokens tool_only"
        )
    if (
        args.rank_score_scope == "conservative_legal"
        and args.rank_update_scope != "conservative_legal"
    ):
        raise SystemExit(
            "conservative_legal rank scores require "
            "--rank-update-scope conservative_legal"
        )
    if args.rank_update_scope == "dense_outcome" and args.rank_score_tokens != "all":
        raise SystemExit("dense_outcome rank updates require --rank-score-tokens all")
    if (
        args.rank_score_scope == "dense_outcome"
        and args.rank_update_scope != "dense_outcome"
    ):
        raise SystemExit(
            "dense_outcome rank scores require --rank-update-scope dense_outcome"
        )
    allowed_experiment_statuses = {
        "allowed_result_only_control",
        "allowed_process_after_gates",
        "allowed_process_screened",
        "allowed_process_unscreened",
        "allowed_rank_only_control",
    }
    if (
        experiment_config is not None
        and experiment_config.admission_status not in allowed_experiment_statuses
    ):
        raise SystemExit(
            "this experiment config is evaluation-only or blocked by the process "
            f"admission gates: {experiment_config.admission_status}"
        )
    if (
        experiment_config is not None
        and experiment_config.admission_status == "allowed_process_screened"
        and (
            args.reward_mode != "process"
            or args.process_admission_policy != "counterfactual-screened"
            or args.fixed_pool_manifest is None
        )
    ):
        raise SystemExit(
            "allowed_process_screened requires process reward and a frozen screened pool"
        )
    if (
        experiment_config is not None
        and experiment_config.admission_status == "allowed_rank_only_control"
        and (
            args.reward_mode != "process"
            or args.process_admission_policy != "rank-local-features"
            or args.policy_loss_coefficient != 0.0
            or args.rank_loss_coefficient <= 0.0
            or args.fixed_pool_manifest is None
        )
    ):
        raise SystemExit(
            "allowed_rank_only_control requires a frozen rank pool, disabled process "
            "loss, and enabled rank loss"
        )
    if (
        experiment_config is not None
        and experiment_config.admission_status == "allowed_process_unscreened"
        and (
            args.reward_mode != "process"
            or args.process_admission_policy != "dense-outcome"
            or args.fixed_pool_manifest is None
            or args.policy_loss_coefficient <= 0.0
            or args.rank_loss_coefficient <= 0.0
            or args.trainable_part != "all"
            or args.rank_score_tokens != "all"
            or args.rank_score_scope != "dense_outcome"
            or args.rank_update_scope != "dense_outcome"
        )
    ):
        raise SystemExit(
            "allowed_process_unscreened requires the frozen dense pool, enabled "
            "process/rank losses, and full-response dense_outcome scoring"
        )
    if (
        experiment_config is not None
        and experiment_config.admission_status == "allowed_process_after_gates"
        and (
            args.reward_mode != "process"
            or args.process_admission_policy != "counterfactual-completeness"
            or (
                args.counterfactual_suite_manifest is None
                and args.fixed_pool_manifest is None
            )
        )
    ):
        raise SystemExit(
            "allowed_process_after_gates requires process reward, "
            "counterfactual-completeness, and a passed counterfactual manifest"
        )
    if (
        args.reward_mode == "process"
        and args.process_admission_policy == "counterfactual-completeness"
        and args.counterfactual_suite_manifest is None
        and args.fixed_pool_manifest is None
    ):
        raise SystemExit(
            "counterfactual-completeness requires --counterfactual-suite-manifest"
        )
    if (
        args.reward_mode == "process"
        and args.process_admission_policy == "denotation-nonempty"
        and not args.exclude_empty_reference_results
    ):
        raise SystemExit(
            "denotation-nonempty requires --exclude-empty-reference-results"
        )

    records = load_rl_task_records(
        ROOT,
        split="train",
        selection=args.selection,
        examples_json=args.examples_json,
        limit=args.limit,
        seed=args.seed,
        context_mode="rolling-legal-history",
    )
    if args.example_index is not None:
        records = [
            record
            for record in records
            if int(record["environment"]["example_index"]) == args.example_index
        ]
    reference_filter_audit = []
    if args.exclude_empty_reference_results:
        records, reference_filter_audit = filter_training_records(records)
    if not records:
        raise SystemExit("no training records selected")

    counterfactual_manifest, counterfactual_suites = (
        (None, {})
        if args.fixed_rollout_pool is not None
        else load_counterfactual_suites(args.counterfactual_suite_manifest, records)
    )
    process_config = (
        load_process_config(args.process_reward_config)
        if args.reward_mode == "process"
        else None
    )
    rollout_settings = RolloutSettings(
        reward_mode=args.reward_mode,
        process_admission_policy=args.process_admission_policy,
        tool_scheme="atomic",
        max_steps=args.max_agent_steps,
        max_batch_calls=args.max_batch_calls,
        max_new_tokens=args.max_new_tokens,
        max_context_tokens=args.max_context_tokens,
        context_mode="rolling-legal-history",
        history_turns=args.history_turns,
        denotation_comparison="bird-set",
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )

    per_device_batch = args.prompts_per_update * args.group_size
    training_args = GRPOConfig(
        output_dir=str(args.output_dir),
        max_steps=args.optimizer_steps,
        per_device_train_batch_size=per_device_batch,
        gradient_accumulation_steps=1,
        steps_per_generation=1,
        num_generations=args.group_size,
        num_iterations=args.ppo_iterations,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        disable_dropout=True,
        max_completion_length=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        use_vllm=args.fixed_rollout_pool is None,
        vllm_mode="server",
        vllm_server_host=args.vllm_host,
        vllm_server_port=args.vllm_port,
        vllm_group_port=args.vllm_group_port,
        vllm_importance_sampling_correction=True,
        vllm_importance_sampling_mode="token_truncate",
        vllm_importance_sampling_cap=3.0,
        importance_sampling_level="token",
        beta=args.kl_beta,
        epsilon=args.clip_epsilon,
        loss_type="grpo",
        scale_rewards="none",
        logging_strategy="steps",
        logging_steps=1,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
        log_completions=False,
    )
    model, tokenizer = load_qlora_model(args)
    rollout_collector = (
        FixedPoolRolloutCollector(
            args.fixed_rollout_pool,
            expected_group_size=args.group_size,
        )
        if args.fixed_rollout_pool is not None
        else TableAgentRolloutCollector(
            tokenizer,
            rollout_settings,
            process_config=process_config,
            counterfactual_suites=counterfactual_suites,
            rollout_log_path=args.output_dir / "rollouts.jsonl",
        )
    )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = (
        torch.optim.AdamW(
            trainable,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        if args.optimizer_name == "adamw_torch"
        else bnb.optim.PagedAdamW8bit(
            trainable,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    )
    dataset = Dataset.from_list(records)
    trainer = TransitionGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, None),
        rollout_collector=rollout_collector,
        reward_mode=args.reward_mode,
        train_turns=args.train_turns,
        trainable_part=args.trainable_part,
        rank_loss_coefficient=args.rank_loss_coefficient,
        rank_beta=args.rank_beta,
        rank_score_tokens=args.rank_score_tokens,
        rank_score_scope=args.rank_score_scope,
        rank_score_reduction=args.rank_score_reduction,
        rank_update_scope=args.rank_update_scope,
        policy_loss_coefficient=args.policy_loss_coefficient,
        transition_micro_batch_size=args.transition_micro_batch_size,
    )

    manifest = {
        "schema_version": "table-agent-trl-transition-grpo-v2",
        "framework": "trl",
        "framework_version": importlib.metadata.version("trl"),
        "torch_version": importlib.metadata.version("torch"),
        "transformers_version": importlib.metadata.version("transformers"),
        "vllm_version": importlib.metadata.version("vllm"),
        "accelerate_version": importlib.metadata.version("accelerate"),
        "peft_version": importlib.metadata.version("peft"),
        "bitsandbytes_version": importlib.metadata.version("bitsandbytes"),
        "model_path": str(args.model_path),
        "adapter_path": str(args.adapter_path),
        "resume_from_checkpoint": (
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        ),
        "experiment_config": (
            experiment_config.to_dict()
            if experiment_config is not None
            else None
        ),
        "experiment_config_sha256": sha256_file(args.experiment_config),
        "reward_mode": args.reward_mode,
        "process_admission_policy": args.process_admission_policy,
        "records": len(records),
        "example_index_filter": args.example_index,
        "group_size": args.group_size,
        "prompts_per_update": args.prompts_per_update,
        "optimizer_steps": args.optimizer_steps,
        "ppo_iterations": args.ppo_iterations,
        "seed": args.seed,
        "transition_micro_batch_size": args.transition_micro_batch_size,
        "train_turns": args.train_turns,
        "trainable_part": args.trainable_part,
        "policy_loss_coefficient": args.policy_loss_coefficient,
        "rank_loss_coefficient": args.rank_loss_coefficient,
        "rank_beta": args.rank_beta,
        "rank_score_tokens": args.rank_score_tokens,
        "rank_score_scope": args.rank_score_scope,
        "rank_score_reduction": args.rank_score_reduction,
        "rank_update_scope": args.rank_update_scope,
        "optimizer_name": args.optimizer_name,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": 1.0,
        "kl_beta": args.kl_beta,
        "clip_epsilon": args.clip_epsilon,
        "vllm_mode": "server",
        "vllm_importance_sampling_mode": "token_truncate",
        "vllm_importance_sampling_cap": 3.0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "rollout_settings": rollout_settings.__dict__,
        "student_prompt_sha256": hashlib.sha256(
            student_runtime_system_prompt(
                context_mode="rolling-legal-history",
                compact=False,
            ).encode("utf-8")
        ).hexdigest(),
        "tool_schema_sha256": tool_schema_hash(),
        "examples_json_sha256": sha256_file(args.examples_json),
        "selection_sha256": sha256_file(args.selection),
        "process_reward_config_sha256": sha256_file(args.process_reward_config),
        "counterfactual_manifest_sha256": sha256_file(
            args.counterfactual_suite_manifest
        ),
        "fixed_rollout_pool": (
            str(args.fixed_rollout_pool) if args.fixed_rollout_pool else None
        ),
        "fixed_rollout_pool_sha256": sha256_file(args.fixed_rollout_pool),
        "fixed_pool_manifest": (
            str(args.fixed_pool_manifest) if args.fixed_pool_manifest else None
        ),
        "fixed_pool_manifest_sha256": sha256_file(args.fixed_pool_manifest),
        "fixed_pool_status": (
            fixed_pool_manifest.get("status") if fixed_pool_manifest is not None else None
        ),
        "fixed_pool_reuse_contract": (
            fixed_pool_manifest.get("reuse_contract")
            if fixed_pool_manifest is not None
            else None
        ),
        "counterfactual_manifest_schema": (
            COUNTERFACTUAL_SCHEMA_VERSION
            if counterfactual_manifest is not None
            else None
        ),
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if reference_filter_audit:
        (args.output_dir / "reference_result_filter.json").write_text(
            json.dumps(
                {
                    "schema_version": "rl-empty-reference-task-filter-v1",
                    "retained_count": len(records),
                    "excluded_count": sum(
                        audit.excluded for audit in reference_filter_audit
                    ),
                    "tasks": [audit.to_dict() for audit in reference_filter_audit],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        )
    )
    trainer.save_model(str(args.output_dir / "final"))
    tokenizer.save_pretrained(args.output_dir / "final")


if __name__ == "__main__":
    main()
