#!/usr/bin/env python3
"""Strict YAML configuration for comparable table-agent RL experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = "table-agent-rl-experiment-v1"
REWARD_TYPES = {
    "result",
    "process",
    "process_no_backslice",
    "process_no_normalize",
}
TRAINABLE_PARTS = {"all", "tool_only"}
RANK_SCORE_TOKENS = {"all", "tool_only"}
RANK_SCORE_SCOPES = {"full_trajectory", "conservative_legal", "dense_outcome"}
RANK_SCORE_REDUCTIONS = {"sum_tokens", "mean_action"}
RANK_UPDATE_SCOPES = {"full_trajectory", "conservative_legal", "dense_outcome"}
ADMISSION_STATUSES = {
    "allowed_result_only_control",
    "allowed_process_after_gates",
    "allowed_process_screened",
    "allowed_process_unscreened",
    "allowed_rank_only_control",
    "evaluation_only_existing_checkpoint",
    "blocked_pending_process_gates",
}


def _strict_keys(value: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {context} keys: {unknown}")


@dataclass(frozen=True)
class RLExperimentConfig:
    path: Path
    payload: dict[str, Any]

    @classmethod
    def load(cls, path: Path) -> "RLExperimentConfig":
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("experiment config must be a YAML mapping")
        _strict_keys(
            payload,
            {
                "schema_version",
                "experiment_name",
                "admission_status",
                "reward_type",
                "process_reward_config",
                "process_admission_policy",
                "exclude_empty_reference_results",
                "process_loss",
                "trainable_part",
                "rank_loss",
                "optimizer",
                "rollout",
            },
            "experiment",
        )
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must be {SCHEMA_VERSION!r}"
            )
        if not payload.get("experiment_name"):
            raise ValueError("experiment_name must be non-empty")
        if payload.get("reward_type") not in REWARD_TYPES:
            raise ValueError(f"unsupported reward_type: {payload.get('reward_type')}")
        if payload.get("trainable_part") not in TRAINABLE_PARTS:
            raise ValueError(
                f"unsupported trainable_part: {payload.get('trainable_part')}"
            )
        if payload.get("admission_status") not in ADMISSION_STATUSES:
            raise ValueError(
                f"unsupported admission_status: {payload.get('admission_status')}"
            )
        if (
            payload["reward_type"] == "result"
            and payload["admission_status"] != "allowed_result_only_control"
        ):
            raise ValueError(
                "result reward must use allowed_result_only_control admission"
            )
        if (
            payload["reward_type"] != "result"
            and payload["admission_status"] == "allowed_result_only_control"
        ):
            raise ValueError(
                "process rewards cannot use result-only admission"
            )
        if (
            payload["reward_type"] == "result"
            and payload["admission_status"] == "allowed_process_after_gates"
        ):
            raise ValueError(
                "result reward cannot use process-gated admission"
            )
        if payload["admission_status"] == "allowed_process_after_gates":
            if payload.get("process_admission_policy") != "counterfactual-completeness":
                raise ValueError(
                    "process-gated admission requires counterfactual-completeness"
                )
            if bool(payload.get("exclude_empty_reference_results")):
                raise ValueError(
                    "process-gated admission cannot use the weaker empty-reference filter"
                )

        rank = payload.get("rank_loss") or {}
        optimizer = payload.get("optimizer") or {}
        rollout = payload.get("rollout") or {}
        if not all(isinstance(value, dict) for value in (rank, optimizer, rollout)):
            raise ValueError("rank_loss, optimizer, and rollout must be mappings")
        _strict_keys(
            rank,
            {
                "enabled",
                "coefficient",
                "beta",
                "score_tokens",
                "score_scope",
                "score_reduction",
                "update_scope",
            },
            "rank_loss",
        )
        _strict_keys(
            optimizer,
            {
                "name",
                "learning_rate",
                "weight_decay",
                "steps",
                "ppo_iterations",
                "lr_scheduler_type",
                "warmup_ratio",
                "kl_beta",
                "clip_epsilon",
            },
            "optimizer",
        )
        _strict_keys(
            rollout,
            {
                "prompts_per_update",
                "group_size",
                "max_agent_steps",
                "max_new_tokens",
                "max_context_tokens",
                "history_turns",
                "temperature",
                "top_p",
                "top_k",
            },
            "rollout",
        )
        if bool(rank.get("enabled")) and float(rank.get("coefficient", 0.0)) <= 0:
            raise ValueError("enabled rank_loss requires coefficient > 0")
        if rank.get("score_tokens", "all") not in RANK_SCORE_TOKENS:
            raise ValueError(
                f"unsupported rank_loss.score_tokens: {rank.get('score_tokens')}"
            )
        if rank.get("score_scope", "full_trajectory") not in RANK_SCORE_SCOPES:
            raise ValueError(
                f"unsupported rank_loss.score_scope: {rank.get('score_scope')}"
            )
        if rank.get("score_reduction", "sum_tokens") not in RANK_SCORE_REDUCTIONS:
            raise ValueError(
                "unsupported rank_loss.score_reduction: "
                f"{rank.get('score_reduction')}"
            )
        if rank.get("update_scope", "full_trajectory") not in RANK_UPDATE_SCOPES:
            raise ValueError(
                f"unsupported rank_loss.update_scope: {rank.get('update_scope')}"
            )
        if (
            rank.get("update_scope", "full_trajectory") == "conservative_legal"
            and rank.get("score_tokens", "all") != "tool_only"
        ):
            raise ValueError(
                "conservative_legal rank updates require score_tokens=tool_only"
            )
        if (
            rank.get("score_scope", "full_trajectory") == "conservative_legal"
            and rank.get("update_scope", "full_trajectory")
            != "conservative_legal"
        ):
            raise ValueError(
                "conservative_legal rank scores require conservative_legal updates"
            )
        if rank.get("update_scope", "full_trajectory") == "dense_outcome" and (
            rank.get("score_tokens", "all") != "all"
        ):
            raise ValueError("dense_outcome rank updates require score_tokens=all")
        if rank.get("score_scope", "full_trajectory") == "dense_outcome" and (
            rank.get("update_scope", "full_trajectory") != "dense_outcome"
        ):
            raise ValueError("dense_outcome rank scores require dense_outcome updates")
        if int(optimizer.get("steps", 0)) <= 0:
            raise ValueError("optimizer.steps must be positive")
        if int(rollout.get("group_size", 0)) < 2:
            raise ValueError("rollout.group_size must be at least 2")
        if not bool(payload.get("process_loss", True)) and not bool(rank.get("enabled")):
            raise ValueError("at least one of process_loss or rank_loss must be enabled")
        if payload["admission_status"] == "allowed_rank_only_control":
            if bool(payload.get("process_loss", True)) or not bool(rank.get("enabled")):
                raise ValueError(
                    "rank-only admission requires disabled process loss and enabled rank loss"
                )
            if payload.get("process_admission_policy") != "rank-local-features":
                raise ValueError(
                    "rank-only admission requires process_admission_policy=rank-local-features"
                )
        if payload["admission_status"] == "allowed_process_screened":
            if not bool(payload.get("process_loss", True)):
                raise ValueError("process-screened admission requires enabled process loss")
            if payload.get("process_admission_policy") != "counterfactual-screened":
                raise ValueError(
                    "process-screened admission requires counterfactual-screened policy"
                )
        if payload["admission_status"] == "allowed_process_unscreened":
            if not bool(payload.get("process_loss", True)):
                raise ValueError("process-unscreened admission requires enabled process loss")
            if payload.get("process_admission_policy") != "dense-outcome":
                raise ValueError(
                    "process-unscreened admission requires dense-outcome policy"
                )
            if payload.get("trainable_part") != "all":
                raise ValueError(
                    "dense-outcome process admission requires trainable_part=all"
                )
            if not bool(rank.get("enabled")):
                raise ValueError(
                    "dense-outcome process admission requires enabled rank loss"
                )
            if (
                rank.get("score_scope") != "dense_outcome"
                or rank.get("update_scope") != "dense_outcome"
            ):
                raise ValueError(
                    "dense-outcome admission requires dense_outcome rank score/update scopes"
                )
        return cls(path=path, payload=payload)

    def argparse_defaults(self, project_root: Path) -> dict[str, Any]:
        reward_type = self.payload["reward_type"]
        rank = self.payload.get("rank_loss") or {}
        optimizer = self.payload.get("optimizer") or {}
        rollout = self.payload.get("rollout") or {}
        process_config = self.payload.get("process_reward_config")
        if process_config:
            process_config = Path(process_config)
            if not process_config.is_absolute():
                process_config = project_root / process_config
        return {
            "reward_mode": "result-only" if reward_type == "result" else "process",
            "process_reward_config": process_config,
            "process_admission_policy": self.payload.get(
                "process_admission_policy",
                "counterfactual-completeness",
            ),
            "exclude_empty_reference_results": bool(
                self.payload.get("exclude_empty_reference_results", False)
            ),
            "trainable_part": self.payload["trainable_part"],
            "policy_loss_coefficient": (
                1.0 if bool(self.payload.get("process_loss", True)) else 0.0
            ),
            "rank_loss_coefficient": (
                float(rank.get("coefficient", 0.0))
                if rank.get("enabled")
                else 0.0
            ),
            "rank_beta": float(rank.get("beta", 0.1)),
            "rank_score_tokens": rank.get("score_tokens", "all"),
            "rank_score_scope": rank.get(
                "score_scope",
                "full_trajectory",
            ),
            "rank_score_reduction": rank.get(
                "score_reduction",
                "sum_tokens",
            ),
            "rank_update_scope": rank.get(
                "update_scope",
                "full_trajectory",
            ),
            "optimizer_name": optimizer.get("name", "adamw_torch"),
            "learning_rate": float(optimizer.get("learning_rate", 1e-6)),
            "weight_decay": float(optimizer.get("weight_decay", 0.01)),
            "optimizer_steps": int(optimizer["steps"]),
            "ppo_iterations": int(optimizer.get("ppo_iterations", 2)),
            "lr_scheduler_type": optimizer.get("lr_scheduler_type", "cosine"),
            "warmup_ratio": float(optimizer.get("warmup_ratio", 0.03)),
            "kl_beta": float(optimizer.get("kl_beta", 0.0)),
            "clip_epsilon": float(optimizer.get("clip_epsilon", 0.2)),
            "prompts_per_update": int(rollout.get("prompts_per_update", 1)),
            "group_size": int(rollout["group_size"]),
            "max_agent_steps": int(rollout.get("max_agent_steps", 30)),
            "max_new_tokens": int(rollout.get("max_new_tokens", 1024)),
            "max_context_tokens": int(rollout.get("max_context_tokens", 8192)),
            "history_turns": int(rollout.get("history_turns", 4)),
            "temperature": float(rollout.get("temperature", 0.7)),
            "top_p": float(rollout.get("top_p", 0.95)),
            "top_k": int(rollout.get("top_k", 0)),
        }

    @property
    def admission_status(self) -> str:
        return str(self.payload["admission_status"])

    def to_dict(self) -> dict[str, Any]:
        return dict(self.payload)
