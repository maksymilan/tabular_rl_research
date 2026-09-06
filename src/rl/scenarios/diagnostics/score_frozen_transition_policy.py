#!/usr/bin/env python3
"""Rescore frozen rollout transitions under several LoRA checkpoints.

The input token IDs are used verbatim.  No prompt rendering, rollout, harness
execution, or gold data is involved.  This makes the chosen-token log-probability
shift an exact fixed-prefix measurement of policy movement on the training pool.
"""
from __future__ import annotations

import argparse
import inspect
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from rl.diagnostics.io import sha256_file
from rl.diagnostics.records import load_jsonl as _load_jsonl
from rl.diagnostics.trajectory import dense_reward_category
from rl.frameworks.trl.tool_loss_mask import tool_token_loss_mask as _tool_mask


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return _load_jsonl(path)


def file_sha256(path: Path) -> str:
    return sha256_file(path)


def transition_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["trajectory_id"]), int(row["turn_index"])


def feature_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["trajectory_id"]), int(row["step_index"])


def reward_category(
    transition: dict[str, Any], feature_row: dict[str, Any]
) -> str:
    return dense_reward_category(transition, feature_row)


def enrich_rows(
    transitions: list[dict[str, Any]], features: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    feature_map = {feature_key(row): row for row in features}
    if len(feature_map) != len(features):
        raise ValueError("process-feature keys are not unique")
    enriched = []
    for row in transitions:
        key = transition_key(row)
        if key not in feature_map:
            raise ValueError(f"missing process features for transition {key}")
        feature_row = feature_map[key]
        enriched.append(
            {
                **row,
                "reward_category": reward_category(row, feature_row),
                "tool": feature_row["tool"],
                "dense_raw_action_credit": feature_row["features"][
                    "dense_raw_action_credit"
                ],
            }
        )
    if len(enriched) != len(features):
        unused = set(feature_map) - {transition_key(row) for row in transitions}
        raise ValueError(f"unused process-feature rows: {len(unused)}")
    return enriched


def parse_adapter(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("adapter must be LABEL=PATH")
    return label, Path(path)


def tool_token_loss_mask(tokenizer, response_ids: list[int]) -> tuple[int, ...]:
    """Reproduce the frozen trainer's exact raw-JSON suffix mask."""
    return _tool_mask(tokenizer, response_ids)


class MultiAdapterScorer:
    def __init__(
        self,
        base_model: str,
        adapters: list[tuple[str, Path]],
        device: str,
        dtype: str,
        attention_implementation: str | None,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.labels = [label for label, _ in adapters]
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype_value = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype_value,
            "device_map": {"": device},
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if attention_implementation:
            load_kwargs["attn_implementation"] = attention_implementation
        base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        base_signature = inspect.signature(base.forward)
        first_label, first_path = adapters[0]
        self.model = PeftModel.from_pretrained(
            base,
            first_path,
            adapter_name=first_label,
            is_trainable=False,
        )
        for label, path in adapters[1:]:
            self.model.load_adapter(path, adapter_name=label, is_trainable=False)
        self.model.eval()
        self.keep_argument = next(
            (
                name
                for name in ("logits_to_keep", "num_logits_to_keep")
                if name in base_signature.parameters
            ),
            None,
        )

    def score_batch(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        torch = self.torch
        encoded = []
        max_response = 0
        max_total = 0
        for row in rows:
            prompt_ids = list(row["prompt_ids"])
            response_ids = list(row["response_ids"])
            if not prompt_ids or not response_ids:
                raise ValueError("transition has an empty prompt or response")
            try:
                tool_mask = tool_token_loss_mask(self.tokenizer, response_ids)
            except ValueError:
                # Dense Exp16-18 trained the complete sampled response, including
                # malformed historical failures.  Such rows remain valid for the
                # primary full-response score but have no JSON-only subscore.
                tool_mask = tuple(0 for _ in response_ids)
            max_response = max(max_response, len(response_ids))
            max_total = max(max_total, len(prompt_ids) + len(response_ids))
            encoded.append((prompt_ids, response_ids, tool_mask))
        input_rows = []
        attention_rows = []
        for prompt_ids, response_ids, _ in encoded:
            combined = prompt_ids + response_ids
            padding = max_total - len(combined)
            input_rows.append([self.tokenizer.pad_token_id] * padding + combined)
            attention_rows.append([0] * padding + [1] * len(combined))
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            attention_rows, dtype=torch.long, device=self.device
        )
        results = [
            {
                "prompt_tokens": len(prompt_ids),
                "response_tokens": len(response_ids),
                "tool_tokens": int(sum(tool_mask)),
                "scores": {},
            }
            for prompt_ids, response_ids, tool_mask in encoded
        ]
        for label in self.labels:
            self.model.set_adapter(label)
            kwargs: dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "use_cache": False,
            }
            if self.keep_argument:
                kwargs[self.keep_argument] = max_response + 1
            with torch.inference_mode():
                logits = self.model(**kwargs).logits
            returned = int(logits.shape[1])
            for index, (_, response_ids, tool_mask) in enumerate(encoded):
                response_length = len(response_ids)
                response_logits = logits[
                    index,
                    returned - response_length - 1 : returned - 1,
                ]
                if int(response_logits.shape[0]) != response_length:
                    raise RuntimeError("returned logits do not align to the response")
                labels = torch.tensor(
                    response_ids, dtype=torch.long, device=self.device
                )
                chosen = response_logits.gather(-1, labels[:, None]).squeeze(-1)
                token_logprobs = chosen.float() - torch.logsumexp(
                    response_logits.float(), dim=-1
                )
                mask = torch.tensor(tool_mask, dtype=torch.bool, device=self.device)
                tool_logprobs = token_logprobs[mask]
                results[index]["scores"][label] = {
                    "response_mean_logprob": float(token_logprobs.mean().item()),
                    "response_sum_logprob": float(token_logprobs.sum().item()),
                    "tool_mean_logprob": (
                        float(tool_logprobs.mean().item())
                        if int(tool_logprobs.numel())
                        else None
                    ),
                    "tool_sum_logprob": (
                        float(tool_logprobs.sum().item())
                        if int(tool_logprobs.numel())
                        else None
                    ),
                }
            del logits
        return results


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize(
    rows: list[dict[str, Any]], baseline: str, checkpoints: list[str]
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups["all"].append(row)
        groups[row["reward_category"]].append(row)
        groups["positive_reward" if row["advantage"] > 0 else "negative_reward"].append(row)
    summaries: dict[str, Any] = {}
    for group_name, group in sorted(groups.items()):
        group_result: dict[str, Any] = {
            "transitions": len(group),
            "response_tokens": sum(row["response_tokens"] for row in group),
            "advantage_mean": _mean(float(row["advantage"]) for row in group),
            "checkpoints": {},
        }
        for checkpoint in checkpoints:
            checkpoint_advantages = [
                float(
                    row.get("checkpoint_advantages", {}).get(
                        checkpoint, row["advantage"]
                    )
                )
                for row in group
            ]
            action_deltas = [
                row["scores"][checkpoint]["response_mean_logprob"]
                - row["scores"][baseline]["response_mean_logprob"]
                for row in group
            ]
            tool_deltas = [
                row["scores"][checkpoint]["tool_mean_logprob"]
                - row["scores"][baseline]["tool_mean_logprob"]
                for row in group
                if row["scores"][checkpoint]["tool_mean_logprob"] is not None
                and row["scores"][baseline]["tool_mean_logprob"] is not None
            ]
            token_delta_sum = sum(
                row["scores"][checkpoint]["response_sum_logprob"]
                - row["scores"][baseline]["response_sum_logprob"]
                for row in group
            )
            alignments = [
                advantage * delta
                for advantage, delta in zip(
                    checkpoint_advantages, action_deltas, strict=True
                )
            ]
            group_result["checkpoints"][checkpoint] = {
                "advantage_mean": _mean(checkpoint_advantages),
                "response_action_mean_logprob_delta": _mean(action_deltas),
                "response_token_weighted_logprob_delta": token_delta_sum
                / group_result["response_tokens"],
                "tool_action_mean_logprob_delta": _mean(tool_deltas),
                "tool_scored_transitions": len(tool_deltas),
                "mean_abs_response_action_delta": _mean(abs(value) for value in action_deltas),
                "reward_alignment_mean": _mean(alignments),
                "reward_aligned_fraction": _mean(value > 0 for value in alignments),
                "abs_response_delta_gt_0_01_fraction": _mean(
                    abs(value) > 0.01 for value in action_deltas
                ),
                "abs_response_delta_gt_0_05_fraction": _mean(
                    abs(value) > 0.05 for value in action_deltas
                ),
            }
        summaries[group_name] = group_result
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transitions", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True, action="append", type=parse_adapter)
    parser.add_argument(
        "--checkpoint-advantages", action="append", type=parse_adapter, default=[]
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--attention-implementation")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    labels = [label for label, _ in args.adapter]
    if len(labels) != len(set(labels)):
        raise SystemExit("adapter labels must be unique")
    if args.baseline not in labels:
        raise SystemExit("baseline must name one of the adapters")
    if args.batch_size < 1:
        raise SystemExit("batch-size must be positive")

    rows = enrich_rows(load_jsonl(args.transitions), load_jsonl(args.features))
    alternate_advantages: dict[str, dict[tuple[str, int], float]] = {}
    for label, path in args.checkpoint_advantages:
        if label not in labels or label == args.baseline:
            raise SystemExit(
                "checkpoint-advantages must name a non-baseline adapter"
            )
        values = {
            transition_key(row): float(row["advantage"])
            for row in load_jsonl(path)
        }
        expected_keys = {transition_key(row) for row in rows}
        if len(values) != len(rows) or set(values) != expected_keys:
            raise SystemExit(f"alternate advantages do not align for {label}")
        alternate_advantages[label] = values
    rows.sort(
        key=lambda row: (
            len(row["prompt_ids"]) + len(row["response_ids"]),
            row["trajectory_id"],
            row["turn_index"],
        )
    )
    if args.limit:
        rows = rows[: args.limit]
    completed: dict[tuple[str, int], dict[str, Any]] = {}
    if args.output_jsonl.exists():
        for row in load_jsonl(args.output_jsonl):
            completed[transition_key(row)] = row
        unknown = set(completed) - {transition_key(row) for row in rows}
        if unknown:
            raise SystemExit("resume output contains transitions outside the selected input")
        for row in completed.values():
            if set(row["scores"]) != set(labels):
                raise SystemExit("resume output checkpoint labels do not match")

    scorer = MultiAdapterScorer(
        args.base_model,
        args.adapter,
        args.device,
        args.dtype,
        args.attention_implementation,
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if completed else "w"
    pending = [row for row in rows if transition_key(row) not in completed]
    with args.output_jsonl.open(mode, encoding="utf-8") as target:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            scores = scorer.score_batch(batch)
            for row, score in zip(batch, scores, strict=True):
                sampling_mean = sum(row["sampling_logprobs"]) / len(
                    row["sampling_logprobs"]
                )
                result = {
                    "schema_version": "frozen-transition-policy-score-v1",
                    "trajectory_id": row["trajectory_id"],
                    "turn_index": row["turn_index"],
                    "example_index": row["example_index"],
                    "trajectory_correct": row["trajectory_correct"],
                    "legal_success": row["legal_success"],
                    "advantage": row["advantage"],
                    "checkpoint_advantages": {
                        label: values[transition_key(row)]
                        for label, values in alternate_advantages.items()
                    },
                    "local_penalty": row["local_penalty"],
                    "reward_category": row["reward_category"],
                    "tool": row["tool"],
                    "sampling_response_mean_logprob": sampling_mean,
                    **score,
                }
                completed[transition_key(row)] = result
                target.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            target.flush()
            done = len(completed)
            if done % 20 < args.batch_size or done == len(rows):
                print(json.dumps({"scored": done, "total": len(rows)}), flush=True)

    ordered = [completed[transition_key(row)] for row in rows]
    summary = {
        "schema_version": "frozen-transition-policy-summary-v1",
        "baseline": args.baseline,
        "checkpoint_labels": labels,
        "groups": summarize(
            ordered,
            args.baseline,
            [label for label in labels if label != args.baseline],
        ),
        "sampling_vs_baseline": {
            "response_action_mean_logprob_difference": _mean(
                row["scores"][args.baseline]["response_mean_logprob"]
                - row["sampling_response_mean_logprob"]
                for row in ordered
            ),
            "mean_abs_response_action_logprob_difference": _mean(
                abs(
                    row["scores"][args.baseline]["response_mean_logprob"]
                    - row["sampling_response_mean_logprob"]
                )
                for row in ordered
            ),
        },
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "frozen-transition-policy-manifest-v1",
        "transitions": str(args.transitions),
        "transitions_sha256": file_sha256(args.transitions),
        "features": str(args.features),
        "features_sha256": file_sha256(args.features),
        "base_model": args.base_model,
        "adapters": {
            label: {
                "path": str(path),
                "sha256": file_sha256(path / "adapter_model.safetensors"),
            }
            for label, path in args.adapter
        },
        "checkpoint_advantages": {
            label: {"path": str(path), "sha256": file_sha256(path)}
            for label, path in args.checkpoint_advantages
        },
        "rows": len(ordered),
        "dtype": args.dtype,
        "device": args.device,
        "attention_implementation": args.attention_implementation,
        "measurement": "chosen-token conditional log probability under exact saved token prefixes",
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
