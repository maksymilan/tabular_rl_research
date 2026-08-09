#!/usr/bin/env python3
"""Five-step rule-routed coupled reasoning/action policy-gradient smoke.

The script consumes exact frozen mixed-pool tokens, initializes once from the
frozen SFT2 adapter, and compares several learning rates from the identical
initial adapter state.  It performs no rollout and saves adapters only when
``--save-root`` is explicitly provided.
"""
from __future__ import annotations

import argparse
import inspect
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


POSITIVE_CATEGORIES = {
    "correct_key_evidence",
    "correct_key_backslice",
    "correct_terminal",
}
NEGATIVE_CATEGORIES = {"severe_local_error"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def tool_token_loss_mask(tokenizer, response_ids: list[int]) -> tuple[int, ...]:
    def decode(ids: list[int]) -> str:
        return tokenizer.decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    text = decode(response_ids)
    think_end = text.find("</think>")
    if think_end < 0:
        raise ValueError("response has no closing think marker")
    json_start = text.find("{", think_end + len("</think>"))
    if json_start < 0:
        raise ValueError("response has no JSON action suffix")
    if text[think_end + len("</think>") : json_start].strip():
        raise ValueError("non-whitespace content lies before JSON")
    result = []
    previous = 0
    for end in range(1, len(response_ids) + 1):
        current = len(decode(response_ids[:end]))
        if current < previous:
            raise ValueError("tokenizer prefix decoding is not monotonic")
        result.append(int(current > json_start))
        previous = current
    if not any(result) or all(result):
        raise ValueError("response does not contain both reasoning and action tokens")
    return tuple(result)


def reward_category(
    transition: dict[str, Any], feature: dict[str, Any]
) -> str:
    values = feature["features"]
    if values["dense_severe_local_bad_event"]:
        return "severe_local_error"
    if transition["trajectory_correct"]:
        if values["dense_operator_backslice_bonus"]:
            return "correct_key_backslice"
        if values["dense_observation_support_bonus"]:
            return "correct_key_evidence"
        if values["is_terminal"]:
            return "correct_terminal"
        return "correct_other_clean"
    return "incorrect_other_clean"


def routed_advantage(row: dict[str, Any]) -> float:
    category = row["reward_category"]
    advantage = float(row["advantage"])
    if category in POSITIVE_CATEGORIES:
        if advantage <= 0.0:
            raise ValueError(f"positive category has non-positive advantage: {category}")
        return advantage
    if category in NEGATIVE_CATEGORIES:
        if advantage >= 0.0:
            raise ValueError(f"negative category has non-negative advantage: {category}")
        return advantage
    return 0.0


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["trajectory_id"]), int(row["turn_index"])


def normalized_loss_weights(
    active_by_question: list[list[dict[str, Any]]],
    normalization: str,
    category_weights: dict[str, float] | None = None,
) -> dict[tuple[str, int], float]:
    """Return transition weights summing to one for one optimizer batch."""
    if not active_by_question or any(not rows for rows in active_by_question):
        raise ValueError("every optimizer-batch question must have active transitions")
    if normalization == "active_mean":
        question_weight = 1.0 / len(active_by_question)
        return {
            row_key(row): question_weight / len(rows)
            for rows in active_by_question
            for row in rows
        }
    if normalization == "category_mean":
        by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rows in active_by_question:
            for row in rows:
                by_category[str(row["reward_category"])].append(row)
        requested = category_weights or {}
        total_category_weight = sum(
            requested.get(category, 1.0) for category in by_category
        )
        return {
            row_key(row): (
                requested.get(category, 1.0)
                / total_category_weight
                / len(category_rows)
            )
            for category, category_rows in by_category.items()
            for row in category_rows
        }
    raise ValueError(f"unknown loss normalization: {normalization}")


def summarize_deltas(
    rows: list[dict[str, Any]],
    baseline: dict[tuple[str, int], dict[str, float]],
    current: dict[tuple[str, int], dict[str, float]],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        advantage = float(row["routed_advantage"])
        if advantage:
            groups["active"].append(row)
            groups["positive" if advantage > 0 else "negative"].append(row)
            groups[row["reward_category"]].append(row)
        if row.get("retention"):
            groups["retention_correct"].append(row)
    result = {}
    for name, values in groups.items():
        records = []
        for row in values:
            key = (str(row["trajectory_id"]), int(row["turn_index"]))
            deltas = {
                span: float(current[key][span] - baseline[key][span])
                for span in ("think", "action", "joint")
            }
            advantage = float(row["routed_advantage"])
            records.append((advantage, deltas))
        entry = {"transitions": len(records)}
        for span in ("think", "action", "joint"):
            span_deltas = [record[1][span] for record in records]
            entry[f"{span}_mean_delta"] = sum(span_deltas) / len(span_deltas)
            entry[f"{span}_mean_abs_delta"] = sum(
                abs(value) for value in span_deltas
            ) / len(span_deltas)
            aligned = [
                float(advantage * deltas[span] > 0.0)
                for advantage, deltas in records
                if advantage
            ]
            entry[f"{span}_reward_aligned_fraction"] = (
                sum(aligned) / len(aligned) if aligned else None
            )
            entry[f"{span}_decrease_fraction"] = sum(
                value < 0.0 for value in span_deltas
            ) / len(span_deltas)
        result[name] = entry
    return result


class CoupledSmoke:
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
        self.think_weight = args.think_weight
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
        self.model = PeftModel.from_pretrained(base, args.adapter, is_trainable=True)
        self.model.gradient_checkpointing_enable()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        self.initial = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }

    def reset(self) -> None:
        with self.torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(
                        self.initial[name].to(
                            device=parameter.device, dtype=parameter.dtype
                        )
                    )
        self.model.zero_grad(set_to_none=True)
        self.torch.cuda.empty_cache()

    def span_scores(self, row: dict[str, Any]) -> dict[str, Any]:
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
        action_mask = torch.tensor(
            row["action_mask"], dtype=torch.bool, device=self.device
        )
        think = logps[~action_mask].mean()
        action = logps[action_mask].mean()
        joint = self.think_weight * think + action
        del logits, response_logits, labels, logps, action_mask, ids
        return {"think": think, "action": action, "joint": joint}

    def score_rows(
        self, rows: list[dict[str, Any]]
    ) -> dict[tuple[str, int], dict[str, float]]:
        self.model.eval()
        result = {}
        with self.torch.no_grad():
            for row in rows:
                key = (str(row["trajectory_id"]), int(row["turn_index"]))
                scores = self.span_scores(row)
                result[key] = {
                    name: float(value.item()) for name, value in scores.items()
                }
        return result

    def train_variant(
        self,
        train_groups: list[list[dict[str, Any]]],
        *,
        learning_rate: float,
        negative_scale: float,
        negative_grad_ratio_cap: float | None,
        question_batch_size: int,
        loss_normalization: str,
        category_weights: dict[str, float],
        weight_decay: float,
    ) -> dict[str, Any]:
        torch = self.torch
        self.reset()
        self.model.train()
        optimizer = torch.optim.AdamW(
            self.parameters, lr=learning_rate, weight_decay=weight_decay
        )
        metrics = []
        optimizer_batches = [
            train_groups[start : start + question_batch_size]
            for start in range(0, len(train_groups), question_batch_size)
        ]
        for step, batch_groups in enumerate(optimizer_batches, start=1):
            def effective_advantage(row: dict[str, Any]) -> float:
                value = float(row["routed_advantage"])
                return value * negative_scale if value < 0.0 else value

            active_by_question = [
                [row for row in rows if effective_advantage(row)]
                for rows in batch_groups
            ]
            if any(not rows for rows in active_by_question):
                raise ValueError("training question contains no routed reward")
            active = [row for rows in active_by_question for row in rows]
            positive = [row for row in active if effective_advantage(row) > 0.0]
            negative = [row for row in active if effective_advantage(row) < 0.0]
            loss_weights = normalized_loss_weights(
                active_by_question, loss_normalization, category_weights
            )

            def backward_rows(selected: list[dict[str, Any]]) -> list[float]:
                values = []
                for row in selected:
                    scores = self.span_scores(row)
                    loss = (
                        -effective_advantage(row)
                        * scores["joint"]
                        * loss_weights[row_key(row)]
                    )
                    loss.backward()
                    values.append(float(loss.detach().item()))
                return values

            optimizer.zero_grad(set_to_none=True)
            positive_grad_norm = None
            negative_grad_norm = None
            negative_grad_scale_applied = 1.0
            if negative_grad_ratio_cap is None or not negative:
                losses = backward_rows(active)
            else:
                positive_losses = backward_rows(positive)
                positive_grads = [
                    parameter.grad.detach().clone()
                    if parameter.grad is not None
                    else None
                    for parameter in self.parameters
                ]
                positive_grad_norm = math.sqrt(
                    sum(
                        float((gradient.float() ** 2).sum().item())
                        for gradient in positive_grads
                        if gradient is not None
                    )
                )
                optimizer.zero_grad(set_to_none=True)
                negative_losses = backward_rows(negative)
                negative_grad_norm = math.sqrt(
                    sum(
                        float((parameter.grad.detach().float() ** 2).sum().item())
                        for parameter in self.parameters
                        if parameter.grad is not None
                    )
                )
                if negative_grad_norm > 0.0:
                    negative_grad_scale_applied = min(
                        1.0,
                        negative_grad_ratio_cap
                        * positive_grad_norm
                        / negative_grad_norm,
                    )
                with torch.no_grad():
                    for parameter, positive_gradient in zip(
                        self.parameters, positive_grads
                    ):
                        if parameter.grad is not None:
                            parameter.grad.mul_(negative_grad_scale_applied)
                        if positive_gradient is None:
                            continue
                        if parameter.grad is None:
                            parameter.grad = positive_gradient
                        else:
                            parameter.grad.add_(positive_gradient)
                losses = positive_losses + negative_losses
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, 1.0)
            optimizer.step()
            metrics.append(
                {
                    "step": step,
                    "example_index": (
                        int(batch_groups[0][0]["example_index"])
                        if len(batch_groups) == 1
                        else None
                    ),
                    "example_indices": [
                        int(rows[0]["example_index"]) for rows in batch_groups
                    ],
                    "question_batch_size": len(batch_groups),
                    "loss_normalization": loss_normalization,
                    "active_transitions": len(active),
                    "positive_transitions": sum(
                        effective_advantage(row) > 0 for row in active
                    ),
                    "negative_transitions": sum(
                        effective_advantage(row) < 0 for row in active
                    ),
                    "negative_grad_ratio_cap": negative_grad_ratio_cap,
                    "positive_grad_norm": positive_grad_norm,
                    "negative_grad_norm": negative_grad_norm,
                    "negative_grad_scale_applied": negative_grad_scale_applied,
                    "loss": sum(losses),
                    "grad_norm_before_clip": float(grad_norm),
                    "clip_triggered": bool(float(grad_norm) > 1.0),
                }
            )
            print(json.dumps(metrics[-1]), flush=True)
        update2 = 0.0
        reference2 = 0.0
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                initial = self.initial[name].to(
                    device=parameter.device, dtype=torch.float32
                )
                current = parameter.detach().float()
                update2 += float(((current - initial) ** 2).sum().item())
                reference2 += float((initial**2).sum().item())
        return {
            "learning_rate": learning_rate,
            "negative_scale": negative_scale,
            "negative_grad_ratio_cap": negative_grad_ratio_cap,
            "question_batch_size": question_batch_size,
            "loss_normalization": loss_normalization,
            "category_weights": category_weights,
            "weight_decay": weight_decay,
            "steps": metrics,
            "raw_adapter_update_norm": math.sqrt(update2),
            "raw_adapter_update_over_reference": (
                math.sqrt(update2 / reference2) if reference2 else None
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--train-example-index", type=int, action="append", required=True)
    parser.add_argument("--retention-example-index", type=int, action="append", default=[])
    parser.add_argument("--learning-rate", type=float, action="append", required=True)
    parser.add_argument(
        "--negative-scale",
        type=float,
        action="append",
        default=[],
        help="Scale severe-local-error negative advantages; may be repeated.",
    )
    parser.add_argument(
        "--negative-grad-ratio-cap",
        type=float,
        help=(
            "Cap each question's negative gradient norm to this multiple of "
            "its positive gradient norm before combining them."
        ),
    )
    parser.add_argument(
        "--question-batch-size",
        type=int,
        default=1,
        help="Number of complete questions accumulated into one optimizer step.",
    )
    parser.add_argument(
        "--loss-normalization",
        choices=("active_mean", "category_mean"),
        default="active_mean",
        help=(
            "active_mean preserves the historical per-question mean; "
            "category_mean gives each present routed reward category equal mass."
        ),
    )
    parser.add_argument(
        "--category-weight",
        action="append",
        default=[],
        metavar="CATEGORY=WEIGHT",
        help=(
            "Optional positive category mass for category_mean; unspecified "
            "active categories retain weight 1. May be repeated."
        ),
    )
    parser.add_argument("--think-weight", type=float, default=0.5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--save-root",
        type=Path,
        help="Optionally save each trained adapter below this directory.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate selection, reward routing, and token masks without loading the model.",
    )
    args = parser.parse_args()
    category_weights: dict[str, float] = {}
    allowed_categories = POSITIVE_CATEGORIES | NEGATIVE_CATEGORIES
    for item in args.category_weight:
        if "=" not in item:
            raise SystemExit(f"invalid category weight: {item}")
        category, value_text = item.split("=", 1)
        if category not in allowed_categories:
            raise SystemExit(f"unknown routed reward category: {category}")
        if category in category_weights:
            raise SystemExit(f"duplicate category weight: {category}")
        try:
            value = float(value_text)
        except ValueError as error:
            raise SystemExit(f"invalid category weight: {item}") from error
        if not math.isfinite(value) or value <= 0.0:
            raise SystemExit(f"category weight must be finite and positive: {item}")
        category_weights[category] = value
    if category_weights and args.loss_normalization != "category_mean":
        raise SystemExit("category weights require --loss-normalization category_mean")
    if not (0.0 < args.think_weight <= 1.0):
        raise SystemExit("think weight must lie in (0, 1]")
    negative_scales = args.negative_scale or [1.0]
    if any(value < 0.0 for value in negative_scales):
        raise SystemExit("negative scales must be non-negative")
    if (
        args.negative_grad_ratio_cap is not None
        and args.negative_grad_ratio_cap <= 0.0
    ):
        raise SystemExit("negative gradient ratio cap must be positive")
    if len(args.train_example_index) != len(set(args.train_example_index)):
        raise SystemExit("training example indices contain duplicates")
    if args.question_batch_size <= 0:
        raise SystemExit("question batch size must be positive")
    if set(args.train_example_index) & set(args.retention_example_index):
        raise SystemExit("training and retention questions overlap")

    wanted = set(args.train_example_index) | set(args.retention_example_index)
    transitions = [
        row
        for row in load_jsonl(args.transitions)
        if int(row["example_index"]) in wanted
    ]
    features = {
        (str(row["trajectory_id"]), int(row["step_index"])): row
        for row in load_jsonl(args.features)
    }
    if args.validate_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True
        )
    else:
        smoke = CoupledSmoke(args)
        tokenizer = smoke.tokenizer
    rows = []
    for row in transitions:
        key = (str(row["trajectory_id"]), int(row["turn_index"]))
        enriched = dict(row)
        enriched["reward_category"] = reward_category(row, features[key])
        enriched["routed_advantage"] = routed_advantage(enriched)
        enriched["action_mask"] = tool_token_loss_mask(
            tokenizer, list(row["response_ids"])
        )
        enriched["retention"] = (
            int(row["example_index"]) in set(args.retention_example_index)
            and bool(row["trajectory_correct"])
        )
        rows.append(enriched)
    by_example: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_example[int(row["example_index"])].append(row)
    missing = wanted - set(by_example)
    if missing:
        raise SystemExit(f"selected example indices are missing: {sorted(missing)}")
    if args.validate_only:
        validation = {
            "selected_examples": sorted(wanted),
            "transitions": len(rows),
            "categories": dict(Counter(row["reward_category"] for row in rows)),
            "active_train_transitions": sum(
                bool(float(row["routed_advantage"]))
                for row in rows
                if int(row["example_index"]) in set(args.train_example_index)
            ),
            "retention_correct_transitions": sum(row["retention"] for row in rows),
            "response_tokens": sum(len(row["response_ids"]) for row in rows),
            "action_tokens": sum(sum(row["action_mask"]) for row in rows),
        }
        print(json.dumps(validation, ensure_ascii=False, indent=2), flush=True)
        return
    train_groups = [by_example[index] for index in args.train_example_index]
    measurement_rows = [
        row
        for row in rows
        if float(row["routed_advantage"]) or row["retention"]
    ]
    baseline = smoke.score_rows(measurement_rows)
    variants = []
    for learning_rate in args.learning_rate:
        for negative_scale in negative_scales:
            training = smoke.train_variant(
                train_groups,
                learning_rate=learning_rate,
                negative_scale=negative_scale,
                negative_grad_ratio_cap=args.negative_grad_ratio_cap,
                question_batch_size=args.question_batch_size,
                loss_normalization=args.loss_normalization,
                category_weights=category_weights,
                weight_decay=args.weight_decay,
            )
            current = smoke.score_rows(measurement_rows)
            training["deltas"] = summarize_deltas(
                measurement_rows, baseline, current
            )
            if args.save_root is not None:
                learning_rate_name = f"lr_{learning_rate:.0e}".replace("+", "")
                negative_name = format(negative_scale, "g").replace(".", "p")
                adapter_dir = args.save_root / f"{learning_rate_name}_neg_{negative_name}"
                if args.negative_grad_ratio_cap is not None:
                    cap_name = format(args.negative_grad_ratio_cap, "g").replace(
                        ".", "p"
                    )
                    adapter_dir = Path(f"{adapter_dir}_ngcap_{cap_name}")
                if args.question_batch_size != 1:
                    adapter_dir = Path(
                        f"{adapter_dir}_qbatch_{args.question_batch_size}"
                    )
                if args.loss_normalization != "active_mean":
                    adapter_dir = Path(
                        f"{adapter_dir}_{args.loss_normalization}"
                    )
                if category_weights:
                    adapter_dir = Path(f"{adapter_dir}_weighted")
                adapter_dir.mkdir(parents=True, exist_ok=True)
                smoke.model.save_pretrained(adapter_dir, safe_serialization=True)
                training["adapter_dir"] = str(adapter_dir)
            variants.append(training)
    result = {
        "schema_version": "routed-coupled-reward-smoke-v1",
        "measurement": "exact frozen tokens; independent reset to SFT2 for each LR",
        "train_example_indices": args.train_example_index,
        "retention_example_indices": args.retention_example_index,
        "think_weight": args.think_weight,
        "action_weight": 1.0,
        "rank_coefficient": 0.0,
        "negative_scales": negative_scales,
        "negative_grad_ratio_cap": args.negative_grad_ratio_cap,
        "question_batch_size": args.question_batch_size,
        "loss_normalization": args.loss_normalization,
        "category_weights": category_weights,
        "routing": {
            "positive": sorted(POSITIVE_CATEGORIES),
            "negative": sorted(NEGATIVE_CATEGORIES),
            "neutral": ["correct_other_clean", "incorrect_other_clean"],
            "normalization": (
                "mean over routed-active transitions per question"
                if args.loss_normalization == "active_mean"
                else "equal mean over present routed reward categories"
            ),
        },
        "train_categories": dict(
            Counter(
                row["reward_category"]
                for row in rows
                if int(row["example_index"]) in set(args.train_example_index)
            )
        ),
        "active_train_transitions": sum(
            bool(float(row["routed_advantage"]))
            for row in rows
            if int(row["example_index"]) in set(args.train_example_index)
        ),
        "retention_correct_transitions": sum(row["retention"] for row in rows),
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
