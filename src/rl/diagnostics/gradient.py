"""Reusable gradient probes for the frozen think + JSON action carrier.

The Gate60 probes historically copied the same rollout-to-token-row builder and
the same adapter scorer into several scenario files.  This module owns that
mechanism.  It is importable without Torch; the model-backed class raises a
clear error only when a caller actually constructs it in a dependency-light
environment.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

try:  # Keep pure record helpers usable on CPU-only audit hosts.
    import torch
except ImportError:  # pragma: no cover - exercised only without Torch installed.
    torch = None

from rl.frameworks.trl.tool_loss_mask import ToolMaskUnavailable, tool_token_loss_mask


def group_advantages(values: Sequence[float], *, epsilon: float = 1e-6) -> list[float]:
    """Compute the float32 population-standardized group advantages."""

    result = [0.0] * len(values)
    if not values or all(value == values[0] for value in values[1:]):
        return result
    if torch is None:
        mean = math.fsum(float(value) for value in values) / len(values)
        variance = math.fsum((float(value) - mean) ** 2 for value in values) / len(values)
        standard_deviation = math.sqrt(variance)
        if standard_deviation == 0.0:
            return result
        return [
            (float(value) - mean) / (standard_deviation + epsilon)
            for value in values
        ]
    tensor = torch.tensor(list(values), dtype=torch.float32)
    standard_deviation = tensor.std(unbiased=False)
    if float(standard_deviation) == 0.0:
        return result
    return ((tensor - tensor.mean()) / (standard_deviation + epsilon)).tolist()


def tool_mask(
    tokenizer: Any, response_text: str, response_ids: Sequence[int]
) -> tuple[int, ...]:
    """Validate a sampled response and return its canonical JSON suffix mask."""

    tokenized = tokenizer(response_text, add_special_tokens=False)["input_ids"]
    if list(tokenized) != list(response_ids):
        raise ValueError("response tokenization mismatch")
    return tool_token_loss_mask(tokenizer, response_ids)


def build_rows(
    records: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    step: int,
    limit_groups: int = 0,
    example_index_filter: int | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Build exact prompt/response rows for one binary group-gradient probe."""

    selected = [
        record for record in records if int(record["policy_global_step"]) == int(step)
    ]
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in selected:
        groups[int(record["example_index"])].append(record)
    example_ids = sorted(groups)
    if example_index_filter is not None:
        example_ids = [
            int(example_index_filter)
        ] if int(example_index_filter) in groups else []
    if limit_groups > 0:
        example_ids = example_ids[: int(limit_groups)]

    result: dict[int, list[dict[str, Any]]] = {}
    for example_index in example_ids:
        trajectories = groups[example_index]
        eligible = [bool(record.get("process_update", False)) for record in trajectories]
        rewards = [
            float(
                (record.get("result_reward") or {}).get(
                    "value", 1.0 if record.get("correct") else 0.0
                )
            )
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
            turns = record.get("turns") or []
            if not isinstance(turns, list):
                raise ValueError(
                    f"trajectory {record.get('trajectory_id')!r} has non-list turns"
                )
            for turn in turns:
                prompt_ids = list(
                    tokenizer.apply_chat_template(
                        turn["model_input"], tokenize=True, add_generation_prompt=True
                    )
                )
                response_text = str(turn["model_output"])
                response_ids = list(
                    tokenizer(response_text, add_special_tokens=False)["input_ids"]
                )
                try:
                    mask = tool_mask(tokenizer, response_text, response_ids)
                except ToolMaskUnavailable:
                    raise
                rows.append(
                    {
                        "example_index": example_index,
                        "trajectory_id": str(record["trajectory_id"]),
                        "turn_index": int(turn["turn_index"]),
                        "prompt_ids": prompt_ids,
                        "response_ids": response_ids,
                        "tool_mask": list(mask),
                        "advantage": float(advantage),
                        "correct": bool(record["correct"]),
                    }
                )
        result[example_index] = rows
    return result


def dot(left: Sequence[Any], right: Sequence[Any]) -> float:
    """Flattened dot product for aligned Torch tensors."""

    return sum(float((first * second).sum()) for first, second in zip(left, right, strict=True))


def vector_norm(vector: Sequence[Any]) -> float:
    """Euclidean norm for a list of gradient tensors."""

    return math.sqrt(max(dot(vector, vector), 0.0))


def geometry(left: Sequence[Any], right: Sequence[Any]) -> dict[str, float]:
    """Return stable norm/cosine geometry for two gradient vectors."""

    left_norm = vector_norm(left)
    right_norm = vector_norm(right)
    cross = dot(left, right)
    sum_norm = math.sqrt(max(left_norm * left_norm + right_norm * right_norm + 2.0 * cross, 0.0))
    return {
        "left_norm": left_norm,
        "right_norm": right_norm,
        "cosine": cross / (left_norm * right_norm) if left_norm and right_norm else 0.0,
        "sum_norm": sum_norm,
        "sum_over_norm_sum": sum_norm / (left_norm + right_norm)
        if left_norm + right_norm
        else 0.0,
    }


def alignment(
    rows: Sequence[Mapping[str, Any]],
    before: Sequence[float],
    after: Sequence[float],
) -> dict[str, float]:
    """Summarize whether sampled log-probabilities moved with their advantage."""

    deltas = [
        float(current) - float(previous)
        for current, previous in zip(after, before, strict=True)
    ]
    positive = [
        delta
        for row, delta in zip(rows, deltas, strict=True)
        if float(row["advantage"]) > 0
    ]
    negative = [
        delta
        for row, delta in zip(rows, deltas, strict=True)
        if float(row["advantage"]) < 0
    ]
    if not deltas:
        raise ValueError("alignment requires at least one scored row")
    return {
        "rows": float(len(deltas)),
        "mean_delta": sum(deltas) / len(deltas),
        "mean_abs_delta": sum(abs(delta) for delta in deltas) / len(deltas),
        "positive_rows": float(len(positive)),
        "positive_aligned": sum(delta > 0 for delta in positive) / len(positive)
        if positive
        else 0.0,
        "positive_mean_delta": sum(positive) / len(positive) if positive else 0.0,
        "negative_rows": float(len(negative)),
        "negative_aligned": sum(delta < 0 for delta in negative) / len(negative)
        if negative
        else 0.0,
        "negative_mean_delta": sum(negative) / len(negative) if negative else 0.0,
    }


class GradientScorer:
    """Score exact frozen responses under one trainable adapter."""

    def __init__(self, base_model: str, adapter: str, device: str):
        if torch is None:  # pragma: no cover - depends on optional runtime deps.
            raise RuntimeError("GradientScorer requires torch, peft, and transformers")
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

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
        self.parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

    def score_batch(
        self,
        rows: list[Mapping[str, Any]],
        only_tool: bool = False,
        mode: str | None = None,
    ):
        """Score one response mean log-probability per row."""

        selected_mode = mode or ("tool" if only_tool else "full")
        if selected_mode not in {"full", "think", "tool", "span"}:
            raise ValueError(f"unsupported carrier mode: {selected_mode}")
        return self.score_batch_modes(rows)[selected_mode]

    def score_batch_modes(self, rows: list[Mapping[str, Any]]):
        """Return full, reasoning, tool, and coupled span scores."""

        if torch is None:  # pragma: no cover
            raise RuntimeError("GradientScorer requires torch")
        if not rows:
            raise ValueError("cannot score an empty batch")
        sequences = [list(row["prompt_ids"]) + list(row["response_ids"]) for row in rows]
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
        scores: dict[str, list[Any]] = {"full": [], "think": [], "tool": [], "span": []}
        for index, row in enumerate(rows):
            response = list(row["response_ids"])
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
        rows: list[Mapping[str, Any]],
        only_tool: bool = False,
        sign: str = "positive",
        mode: str | None = None,
    ):
        """Aggregate one signed branch into CPU gradient tensors."""

        if torch is None:  # pragma: no cover
            raise RuntimeError("GradientScorer requires torch")
        selected = [
            row
            for row in rows
            if (float(row["advantage"]) > 0) == (sign == "positive")
        ]
        if not selected:
            raise ValueError(f"no {sign} rows")
        self.model.zero_grad(set_to_none=True)
        scale = 1.0 / len(rows)
        for start in range(0, len(selected), 2):
            batch = selected[start : start + 2]
            coefficients = torch.tensor(
                [-float(row["advantage"]) * scale for row in batch],
                dtype=torch.float32,
                device=self.device,
            )
            (
                self.score_batch(batch, only_tool=only_tool, mode=mode) * coefficients
            ).sum().backward()
        gradients = [
            parameter.grad.detach().float().cpu().clone()
            if parameter.grad is not None
            else torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
            for parameter in self.parameters
        ]
        self.model.zero_grad(set_to_none=True)
        return gradients


__all__ = [
    "GradientScorer",
    "ToolMaskUnavailable",
    "alignment",
    "build_rows",
    "dot",
    "geometry",
    "group_advantages",
    "tool_mask",
    "tool_token_loss_mask",
    "vector_norm",
]
