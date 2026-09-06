#!/usr/bin/env python3
"""Short exact-token gradient audit for the frozen dense RL objective.

This diagnostic never rolls out or updates the model.  It loads one immutable
K=4 question batch at the SFT2 initialization and measures the policy-gradient
vectors induced by positive rewards, negative rewards, and the trajectory-rank
branch.  A tool-token counterfactual is measured on the same responses.
"""
from __future__ import annotations

import argparse
import inspect
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from rl.diagnostics.records import load_jsonl as _load_jsonl
from rl.diagnostics.trajectory import dense_reward_category
from rl.frameworks.trl.tool_loss_mask import tool_token_loss_mask as _tool_mask


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return _load_jsonl(path)


def tool_token_loss_mask(tokenizer, response_ids: list[int]) -> tuple[int, ...]:
    return _tool_mask(tokenizer, response_ids)


def reward_category(
    transition: dict[str, Any], feature: dict[str, Any]
) -> str:
    return dense_reward_category(transition, feature)


def pairwise_metrics(
    vectors: dict[str, list[Any]], torch_module
) -> dict[str, Any]:
    def dot(left: str, right: str) -> float:
        return sum(
            float((a.float() * b.float()).sum().item())
            for a, b in zip(vectors[left], vectors[right])
        )

    dots = {
        left: {right: dot(left, right) for right in vectors}
        for left in vectors
    }
    norms = {name: math.sqrt(max(dots[name][name], 0.0)) for name in vectors}
    cosines = {}
    names = list(vectors)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            denominator = norms[left] * norms[right]
            cosines[f"{left}::{right}"] = (
                dots[left][right] / denominator if denominator else None
            )

    positive = "policy_positive_full"
    negative = "policy_negative_full"
    rank = "rank_full"
    policy_norm2 = (
        dots[positive][positive]
        + dots[negative][negative]
        + 2.0 * dots[positive][negative]
    )
    policy_norm = math.sqrt(max(policy_norm2, 0.0))
    total_norm2 = (
        policy_norm2
        + dots[rank][rank]
        + 2.0 * (dots[positive][rank] + dots[negative][rank])
    )
    total_norm = math.sqrt(max(total_norm2, 0.0))
    cancellation_denominator = norms[positive] + norms[negative]
    rank_denominator = policy_norm + norms[rank]
    tool = "policy_tool_only"
    policy_tool_dot = dots[positive][tool] + dots[negative][tool]
    return {
        "norms": norms,
        "cosines": cosines,
        "derived": {
            "policy_full_norm": policy_norm,
            "policy_positive_negative_net_over_sum_norm": (
                policy_norm / cancellation_denominator
                if cancellation_denominator
                else None
            ),
            "policy_rank_cosine": (
                (dots[positive][rank] + dots[negative][rank])
                / (policy_norm * norms[rank])
                if policy_norm and norms[rank]
                else None
            ),
            "policy_plus_rank_norm": total_norm,
            "policy_rank_net_over_sum_norm": (
                total_norm / rank_denominator if rank_denominator else None
            ),
            "policy_full_tool_only_cosine": (
                policy_tool_dot / (policy_norm * norms[tool])
                if policy_norm and norms[tool]
                else None
            ),
        },
    }


class GradientAudit:
    def __init__(self, args: argparse.Namespace) -> None:
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self.torch = torch
        self.device = args.device
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map={"": args.device},
            trust_remote_code=True,
        )
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(
            base, use_gradient_checkpointing=True
        )
        self.keep_argument = next(
            (
                name
                for name in ("logits_to_keep", "num_logits_to_keep")
                if name in inspect.signature(base.forward).parameters
            ),
            None,
        )
        self.model = PeftModel.from_pretrained(
            base, args.adapter, is_trainable=True
        )
        self.model.gradient_checkpointing_enable()
        # The frozen trainer used disable_dropout=True.
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.model.train()
        self.parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]

    def score(self, row: dict[str, Any], *, tool_only: bool) -> Any:
        torch = self.torch
        prompt = list(row["prompt_ids"])
        response = list(row["response_ids"])
        ids = torch.tensor([prompt + response], dtype=torch.long, device=self.device)
        kwargs: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "use_cache": False,
        }
        if self.keep_argument:
            kwargs[self.keep_argument] = len(response) + 1
        logits = self.model(**kwargs).logits
        returned = int(logits.shape[1])
        response_logits = logits[0, returned - len(response) - 1 : returned - 1]
        labels = torch.tensor(response, dtype=torch.long, device=self.device)
        logps = response_logits.float().log_softmax(dim=-1).gather(
            -1, labels[:, None]
        ).squeeze(-1)
        if tool_only:
            mask = torch.tensor(
                row["tool_mask"], dtype=torch.bool, device=self.device
            )
            logps = logps[mask]
        result = logps.mean()
        del logits, response_logits, labels, logps, ids
        return result

    def gradient(
        self,
        rows: list[dict[str, Any]],
        coefficients: dict[tuple[str, int], float],
        *,
        tool_only: bool,
    ) -> list[Any]:
        self.model.zero_grad(set_to_none=True)
        active = 0
        for row in rows:
            key = (str(row["trajectory_id"]), int(row["turn_index"]))
            coefficient = float(coefficients.get(key, 0.0))
            if coefficient == 0.0:
                continue
            score = self.score(row, tool_only=tool_only)
            (score * coefficient).backward()
            active += 1
        if not active:
            raise ValueError("gradient objective has no active transitions")
        result = [
            (
                parameter.grad.detach().float().cpu().clone()
                if parameter.grad is not None
                else self.torch.zeros_like(parameter, device="cpu", dtype=self.torch.float32)
            )
            for parameter in self.parameters
        ]
        self.model.zero_grad(set_to_none=True)
        self.torch.cuda.empty_cache()
        return result

    def detached_scores(
        self, rows: list[dict[str, Any]], *, tool_only: bool
    ) -> dict[tuple[str, int], float]:
        result = {}
        with self.torch.no_grad():
            for row in rows:
                key = (str(row["trajectory_id"]), int(row["turn_index"]))
                result[key] = float(self.score(row, tool_only=tool_only).item())
        return result


def rank_coefficients(
    rows: list[dict[str, Any]],
    scores: dict[tuple[str, int], float],
    *,
    beta: float,
    coefficient: float,
) -> tuple[dict[tuple[str, int], float], dict[str, Any]]:
    by_trajectory: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_trajectory[str(row["trajectory_id"])].append(row)
    selected: dict[str, list[dict[str, Any]]] = {}
    trajectory_correct = {}
    for trajectory_id, values in by_trajectory.items():
        trajectory_correct[trajectory_id] = bool(values[0]["trajectory_correct"])
        selected[trajectory_id] = [
            row
            for row in values
            if (
                not row["trajectory_correct"]
                or (row["legal_success"] and float(row["local_penalty"]) <= 0.0)
            )
        ]
    trajectory_scores = {
        trajectory_id: sum(
            scores[(str(row["trajectory_id"]), int(row["turn_index"]))]
            for row in values
        )
        / len(values)
        for trajectory_id, values in selected.items()
        if values
    }
    positives = [
        key for key, correct in trajectory_correct.items()
        if correct and key in trajectory_scores
    ]
    negatives = [
        key for key, correct in trajectory_correct.items()
        if not correct and key in trajectory_scores
    ]
    pairs = [(positive, negative) for positive in positives for negative in negatives]
    if not pairs:
        raise ValueError("question has no supported correct/incorrect rank pair")
    trajectory_derivatives = Counter()
    losses = []
    for positive, negative in pairs:
        delta = trajectory_scores[positive] - trajectory_scores[negative]
        losses.append(math.log1p(math.exp(-beta * delta)))
        derivative = -beta / (1.0 + math.exp(beta * delta)) / len(pairs)
        trajectory_derivatives[positive] += derivative * coefficient
        trajectory_derivatives[negative] -= derivative * coefficient
    transition_coefficients = {}
    for trajectory_id, values in selected.items():
        for row in values:
            transition_coefficients[
                (str(row["trajectory_id"]), int(row["turn_index"]))
            ] = trajectory_derivatives[trajectory_id] / len(values)
    return transition_coefficients, {
        "pairs": len(pairs),
        "mean_rank_loss_before_coefficient": sum(losses) / len(losses),
        "trajectory_scores": trajectory_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--example-index", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank-beta", type=float, default=0.1)
    parser.add_argument("--rank-coefficient", type=float, default=0.5)
    args = parser.parse_args()

    transitions = [
        row
        for row in load_jsonl(args.transitions)
        if int(row["example_index"]) == args.example_index
    ]
    if not transitions:
        raise SystemExit("example index is absent from the transition pool")
    features = {
        (str(row["trajectory_id"]), int(row["step_index"])): row
        for row in load_jsonl(args.features)
    }
    audit = GradientAudit(args)
    rows = []
    skipped_tool_masks = 0
    for row in transitions:
        key = (str(row["trajectory_id"]), int(row["turn_index"]))
        enriched = dict(row)
        enriched["reward_category"] = reward_category(row, features[key])
        try:
            enriched["tool_mask"] = tool_token_loss_mask(
                audit.tokenizer, list(row["response_ids"])
            )
        except ValueError:
            enriched["tool_mask"] = tuple(0 for _ in row["response_ids"])
            skipped_tool_masks += 1
        rows.append(enriched)
    if skipped_tool_masks:
        raise SystemExit(
            f"selected question has {skipped_tool_masks} unavailable tool masks"
        )

    normalizer = len(rows)
    positive_coefficients = {
        (str(row["trajectory_id"]), int(row["turn_index"])):
        -float(row["advantage"]) / normalizer
        for row in rows
        if float(row["advantage"]) > 0.0
    }
    negative_coefficients = {
        (str(row["trajectory_id"]), int(row["turn_index"])):
        -float(row["advantage"]) / normalizer
        for row in rows
        if float(row["advantage"]) < 0.0
    }
    policy_coefficients = {**positive_coefficients, **negative_coefficients}
    old_scores = audit.detached_scores(rows, tool_only=False)
    rank_values, rank_audit = rank_coefficients(
        rows,
        old_scores,
        beta=args.rank_beta,
        coefficient=args.rank_coefficient,
    )
    vectors = {
        "policy_positive_full": audit.gradient(
            rows, positive_coefficients, tool_only=False
        ),
        "policy_negative_full": audit.gradient(
            rows, negative_coefficients, tool_only=False
        ),
        "rank_full": audit.gradient(rows, rank_values, tool_only=False),
        "policy_tool_only": audit.gradient(
            rows, policy_coefficients, tool_only=True
        ),
    }
    result = {
        "schema_version": "dense-reward-gradient-alignment-audit-v1",
        "measurement": (
            "exact saved tokens; SFT2 QLoRA initialization; no rollout and no update"
        ),
        "example_index": args.example_index,
        "transitions": len(rows),
        "trajectories": len({row["trajectory_id"] for row in rows}),
        "positive_transitions": len(positive_coefficients),
        "negative_transitions": len(negative_coefficients),
        "reward_categories": dict(Counter(row["reward_category"] for row in rows)),
        "response_tokens": sum(len(row["response_ids"]) for row in rows),
        "tool_tokens": sum(sum(row["tool_mask"]) for row in rows),
        "rank": rank_audit,
        "gradients": pairwise_metrics(vectors, audit.torch),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
