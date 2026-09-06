#!/usr/bin/env python3
"""Measure positive/negative and think/action gradient geometry on Gate60 rollouts.

This is a read-only diagnostic.  It reconstructs the exact causal chat prompt and
response text from the saved rollout, computes ordinary binary group GRPO
advantages, and measures gradients on the adapter parameters only.  Gold SQL is
never read.  The diagnostic is intentionally separate from the trainer: it does
not perform optimizer steps or alter any rollout.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def group_advantages(values: list[float]) -> list[float]:
    if not values or all(value == values[0] for value in values[1:]):
        return [0.0] * len(values)
    tensor = torch.tensor(values, dtype=torch.float32)
    std = tensor.std(unbiased=False)
    if float(std) == 0.0:
        return [0.0] * len(values)
    return ((tensor - tensor.mean()) / (std + 1e-6)).tolist()


def tool_mask(tokenizer, response_text: str, response_ids: list[int]) -> list[int]:
    think_end = response_text.find("</think>")
    if think_end < 0:
        raise ValueError("response has no closing </think>")
    json_start = response_text.find("{", think_end + len("</think>"))
    if json_start < 0:
        raise ValueError("response has no JSON action")
    encoded = tokenizer(
        response_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if ids != response_ids:
        raise ValueError("response tokenization mismatch")
    mask = [int(end > json_start) for _, end in offsets]
    if not any(mask):
        raise ValueError("empty tool mask")
    return mask


def build_rows(
    records: list[dict[str, Any]], tokenizer, step: int, limit_groups: int,
    example_index_filter: int | None = None,
) -> dict[int, list[dict[str, Any]]]:
    selected = [record for record in records if int(record["policy_global_step"]) == step]
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in selected:
        groups[int(record["example_index"])].append(record)
    example_ids = sorted(groups)
    if example_index_filter is not None:
        example_ids = [example_index_filter] if example_index_filter in groups else []
    if limit_groups > 0:
        example_ids = example_ids[:limit_groups]
    result: dict[int, list[dict[str, Any]]] = {}
    for example_index in example_ids:
        trajectories = groups[example_index]
        eligible = [bool(record.get("process_update", False)) for record in trajectories]
        rewards = [
            float((record.get("result_reward") or {}).get("value", 1.0 if record.get("correct") else 0.0))
            for record in trajectories
        ]
        advantages = group_advantages(
            [reward for reward, keep in zip(rewards, eligible, strict=True) if keep]
        )
        advantage_iter = iter(advantages)
        rows: list[dict[str, Any]] = []
        for record, keep in zip(trajectories, eligible, strict=True):
            advantage = next(advantage_iter) if keep else 0.0
            if not keep or advantage == 0.0:
                continue
            for turn in record["turns"]:
                prompt_ids = list(
                    tokenizer.apply_chat_template(
                        turn["model_input"], tokenize=True, add_generation_prompt=True
                    )
                )
                response_text = str(turn["model_output"])
                response_ids = list(
                    tokenizer(response_text, add_special_tokens=False)["input_ids"]
                )
                mask = tool_mask(tokenizer, response_text, response_ids)
                rows.append(
                    {
                        "example_index": example_index,
                        "trajectory_id": str(record["trajectory_id"]),
                        "turn_index": int(turn["turn_index"]),
                        "prompt_ids": prompt_ids,
                        "response_ids": response_ids,
                        "tool_mask": mask,
                        "advantage": float(advantage),
                        "correct": bool(record["correct"]),
                    }
                )
        result[example_index] = rows
    return result


class GradientScorer:
    def __init__(self, base_model: str, adapter: str, device: str):
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quantization,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True,
        )
        base.config.use_cache = False
        base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
        self.model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
        self.model.gradient_checkpointing_enable()
        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
        self.model.train()
        self.parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.keep_argument = next(
            (name for name in ("logits_to_keep", "num_logits_to_keep") if name in self.model.base_model.forward.__code__.co_varnames),
            None,
        )

    def score_batch(
        self,
        rows: list[dict[str, Any]],
        only_tool: bool = False,
        mode: str | None = None,
    ) -> torch.Tensor:
        """Score a small left-padded batch, returning one mean logp per row.

        Left padding makes every response end at the same position.  Qwen3's
        ``logits_to_keep`` then avoids materializing logits for the long causal
        prefix while retaining the exact response-token positions.
        """
        if mode is None:
            mode = "tool" if only_tool else "full"
        if mode not in {"full", "think", "tool", "span"}:
            raise ValueError(f"unsupported carrier mode: {mode}")
        return self.score_batch_modes(rows)[mode]

    def score_batch_modes(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Return full/tool/span scores from one shared model forward."""
        sequences = [row["prompt_ids"] + row["response_ids"] for row in rows]
        max_length = max(len(sequence) for sequence in sequences)
        max_response = max(len(row["response_ids"]) for row in rows)
        pad_id = int(self.tokenizer.pad_token_id)
        ids = torch.tensor(
            [[pad_id] * (max_length - len(sequence)) + sequence for sequence in sequences],
            dtype=torch.long,
            device=self.device,
        )
        attention = torch.tensor(
            [[0] * (max_length - len(sequence)) + [1] * len(sequence) for sequence in sequences],
            dtype=torch.long,
            device=self.device,
        )
        kwargs: dict[str, Any] = {
            "input_ids": ids,
            "attention_mask": attention,
            "use_cache": False,
            "logits_to_keep": max_response + 1,
        }
        logits = self.model(**kwargs).logits
        scores: dict[str, list[torch.Tensor]] = {"full": [], "think": [], "tool": [], "span": []}
        for index, row in enumerate(rows):
            response = row["response_ids"]
            response_length = len(response)
            response_logits = logits[index, max_response - response_length : max_response]
            labels = torch.tensor(response, dtype=torch.long, device=self.device)
            logps = response_logits.float().log_softmax(dim=-1).gather(
                -1, labels[:, None]
            ).squeeze(-1)
            mask = torch.tensor(row["tool_mask"], dtype=torch.bool, device=self.device)
            think_logps = logps[~mask]
            tool_logps = logps[mask]
            if not len(think_logps) or not len(tool_logps):
                raise ValueError("span carrier requires both think and tool tokens")
            scores["full"].append(logps.mean())
            scores["think"].append(think_logps.mean())
            scores["tool"].append(tool_logps.mean())
            scores["span"].append(0.5 * think_logps.mean() + tool_logps.mean())
        return {name: torch.stack(values) for name, values in scores.items()}

    def aggregate(
        self,
        rows: list[dict[str, Any]],
        only_tool: bool = False,
        sign: str = "positive",
        mode: str | None = None,
    ) -> list[torch.Tensor]:
        selected = [row for row in rows if (row["advantage"] > 0) == (sign == "positive")]
        if not selected:
            raise ValueError(f"no {sign} rows")
        self.model.zero_grad(set_to_none=True)
        scale = 1.0 / len(rows)
        batch_size = 2
        for start in range(0, len(selected), batch_size):
            batch = selected[start : start + batch_size]
            coefficients = torch.tensor(
                [-float(row["advantage"]) * scale for row in batch],
                dtype=torch.float32,
                device=self.device,
            )
            (self.score_batch(batch, only_tool=only_tool, mode=mode) * coefficients).sum().backward()
        gradients = [
            parameter.grad.detach().float().cpu().clone()
            if parameter.grad is not None
            else torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
            for parameter in self.parameters
        ]
        self.model.zero_grad(set_to_none=True)
        return gradients


def dot(left: list[torch.Tensor], right: list[torch.Tensor]) -> float:
    return sum(float((a * b).sum()) for a, b in zip(left, right, strict=True))


def geometry(left: list[torch.Tensor], right: list[torch.Tensor]) -> dict[str, float]:
    left_norm = math.sqrt(max(dot(left, left), 0.0))
    right_norm = math.sqrt(max(dot(right, right), 0.0))
    cross = dot(left, right)
    cosine = cross / (left_norm * right_norm) if left_norm and right_norm else 0.0
    return {
        "left_norm": left_norm,
        "right_norm": right_norm,
        "cosine": cosine,
        "sum_norm": math.sqrt(max(left_norm * left_norm + right_norm * right_norm + 2.0 * cross, 0.0)),
        "sum_over_norm_sum": (
            math.sqrt(max(left_norm * left_norm + right_norm * right_norm + 2.0 * cross, 0.0))
            / (left_norm + right_norm)
            if left_norm + right_norm
            else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--limit-groups", type=int, default=0)
    parser.add_argument("--example-index", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_jsonl(args.rollouts)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    grouped = build_rows(
        records, tokenizer, args.step, args.limit_groups, args.example_index
    )
    scorer = GradientScorer(args.base_model, args.adapter, args.device)
    groups = []
    for example_index, rows in grouped.items():
        positives = sum(row["advantage"] > 0 for row in rows)
        negatives = sum(row["advantage"] < 0 for row in rows)
        if not positives or not negatives:
            continue
        pos_full = scorer.aggregate(rows, only_tool=False, sign="positive")
        neg_full = scorer.aggregate(rows, only_tool=False, sign="negative")
        pos_tool = scorer.aggregate(rows, only_tool=True, sign="positive")
        neg_tool = scorer.aggregate(rows, only_tool=True, sign="negative")
        groups.append(
            {
                "example_index": example_index,
                "rows": len(rows),
                "positive_rows": positives,
                "negative_rows": negatives,
                "response_tokens": sum(len(row["response_ids"]) for row in rows),
                "tool_tokens": sum(sum(row["tool_mask"]) for row in rows),
                "full_positive_negative": geometry(pos_full, neg_full),
                "tool_positive_negative": geometry(pos_tool, neg_tool),
                "positive_full_tool": geometry(pos_full, pos_tool),
                "negative_full_tool": geometry(neg_full, neg_tool),
            }
        )
        print(json.dumps(groups[-1], ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "gate60-gradient-conflict-audit-v1",
        "gold_sql_read": False,
        "measurement": "saved causal prompts/responses; adapter gradients only; no rollout or optimizer update",
        "step": args.step,
        "groups": groups,
        "group_count": len(groups),
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
