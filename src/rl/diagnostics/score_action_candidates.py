#!/usr/bin/env python3
"""Score positive/negative tool actions under an identical visible prefix."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path
from typing import Any


def json_token_indices(response: str, offsets: list[tuple[int, int]]) -> list[int]:
    start = response.find("{")
    if start < 0:
        raise ValueError("scoring response has no JSON object")
    return [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start and left < len(response)
    ]


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for the mandated parquet artifact") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    # Tool arguments are intentionally heterogeneous (strings, numbers, lists,
    # and nested predicate objects).  Letting Arrow infer a nested struct from
    # the first row can therefore make later, perfectly valid actions fail the
    # export with a cross-row type error.  JSONL remains the structured source
    # of truth; Parquet stores composite cells as canonical JSON strings so its
    # schema is stable across every legal tool argument shape.
    parquet_rows = [
        {
            key: (
                json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                if isinstance(value, (dict, list))
                else value
            )
            for key, value in row.items()
        }
        for row in rows
    ]
    pq.write_table(pa.Table.from_pylist(parquet_rows), path)


class CandidateScorer:
    def __init__(self, base_model: str, adapter: str, device: str, dtype: str) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
        if not self.tokenizer.is_fast:
            raise RuntimeError("a fast tokenizer is required for exact JSON offset masking")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dtype_value = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype_value,
            device_map={"": device},
            low_cpu_mem_usage=True,
        )
        base_signature = inspect.signature(base.forward)
        self.model = PeftModel.from_pretrained(base, adapter, is_trainable=False)
        self.model.eval()
        self.device = device
        self.keep_argument = next(
            (
                name
                for name in ("logits_to_keep", "num_logits_to_keep")
                if name in base_signature.parameters
            ),
            None,
        )

    def _prompt_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        result = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(result, "tolist"):
            result = result.tolist()
        return list(result)

    def score_many(
        self,
        messages: list[dict[str, Any]],
        responses: list[str],
    ) -> list[dict[str, Any]]:
        """Score same-prefix candidates in one left-padded causal forward."""
        torch = self.torch
        prompt_ids = self._prompt_ids(messages)
        encoded_rows = []
        for response in responses:
            encoded = self.tokenizer(
                response,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            response_ids = list(encoded["input_ids"])
            offsets = [tuple(value) for value in encoded["offset_mapping"]]
            selected = json_token_indices(response, offsets)
            if not prompt_ids or not response_ids or not selected:
                raise ValueError("empty prompt, response, or JSON token mask")
            encoded_rows.append((response_ids, selected))
        max_total = max(len(prompt_ids) + len(ids) for ids, _ in encoded_rows)
        max_response = max(len(ids) for ids, _ in encoded_rows)
        input_rows = []
        attention_rows = []
        for response_ids, _ in encoded_rows:
            all_ids = prompt_ids + response_ids
            padding = max_total - len(all_ids)
            input_rows.append([self.tokenizer.pad_token_id] * padding + all_ids)
            attention_rows.append([0] * padding + [1] * len(all_ids))
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(
            attention_rows,
            dtype=torch.long,
            device=self.device,
        )
        forward_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if self.keep_argument:
            forward_kwargs[self.keep_argument] = max_response + 1
        results = []
        with torch.inference_mode():
            logits = self.model(**forward_kwargs).logits
            returned = int(logits.shape[1])
            for row_index, (response_ids, selected) in enumerate(encoded_rows):
                response_length = len(response_ids)
                response_logits = logits[
                    row_index,
                    returned - response_length - 1 : returned - 1,
                ]
                if response_logits.shape[0] != response_length:
                    raise RuntimeError("model did not return aligned candidate logits")
                labels = torch.tensor(
                    response_ids,
                    dtype=torch.long,
                    device=self.device,
                )
                chosen_logits = response_logits.gather(
                    -1,
                    labels.unsqueeze(-1),
                ).squeeze(-1)
                token_logprobs = chosen_logits.float() - torch.logsumexp(
                    response_logits.float(),
                    dim=-1,
                )
                selected_tensor = torch.tensor(
                    selected,
                    dtype=torch.long,
                    device=self.device,
                )
                json_logprobs = token_logprobs.index_select(
                    0,
                    selected_tensor,
                )
                mean_logprob = float(json_logprobs.mean().item())
                results.append(
                    {
                        "mean_tool_json_logprob": mean_logprob,
                        "sum_tool_json_logprob": float(json_logprobs.sum().item()),
                        "tool_json_tokens": len(selected),
                        "response_tokens": response_length,
                        "prompt_tokens": len(prompt_ids),
                        "finite": bool(math.isfinite(mean_logprob)),
                    }
                )
        return results

    def score(self, messages: list[dict[str, Any]], response: str) -> dict[str, Any]:
        return self.score_many(messages, [response])[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    rows = load_rows(args.dataset)
    if args.limit > 0:
        rows = rows[: args.limit]
    scorer = CandidateScorer(args.base_model, args.adapter, args.device, args.dtype)
    scores = []
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as target:
        for index, row in enumerate(rows, 1):
            positive, negative = scorer.score_many(
                row["state"],
                [
                    row["positive_scoring_response"],
                    row["negative_scoring_response"],
                ],
            )
            result = {
                "schema_version": "fixed-prefix-action-score-v1",
                "checkpoint": args.checkpoint_name,
                "pair_sha256": row["pair_sha256"],
                "question_id": row["question_id"],
                "example_index": row["example_index"],
                "category": row["category"],
                "positive_action": row["positive_action"],
                "negative_action": row["negative_action"],
                "positive_action_logprob": positive["mean_tool_json_logprob"],
                "negative_action_logprob": negative["mean_tool_json_logprob"],
                "margin": positive["mean_tool_json_logprob"]
                - negative["mean_tool_json_logprob"],
                "positive_tool_json_tokens": positive["tool_json_tokens"],
                "negative_tool_json_tokens": negative["tool_json_tokens"],
                "prompt_tokens": positive["prompt_tokens"],
            }
            scores.append(result)
            target.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            target.flush()
            if index % 25 == 0 or index == len(rows):
                print(json.dumps({"checkpoint": args.checkpoint_name, "scored": index, "total": len(rows)}))
    write_parquet(args.output_parquet, scores)
    manifest_path = args.manifest or args.output_jsonl.with_suffix(".manifest.json")
    adapter_weights = Path(args.adapter) / "adapter_model.safetensors"
    manifest = {
        "schema_version": "fixed-prefix-action-score-manifest-v1",
        "checkpoint": args.checkpoint_name,
        "base_model": args.base_model,
        "adapter": args.adapter,
        "adapter_sha256": hashlib.sha256(adapter_weights.read_bytes()).hexdigest(),
        "dataset": str(args.dataset),
        "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
        "pairs": len(scores),
        "scoring": "mean conditional log probability over tool JSON tokens only",
        "shared_reason": rows[0].get("shared_reason") if rows else None,
        "output_jsonl": str(args.output_jsonl),
        "output_jsonl_sha256": hashlib.sha256(args.output_jsonl.read_bytes()).hexdigest(),
        "output_parquet": str(args.output_parquet),
        "output_parquet_sha256": hashlib.sha256(args.output_parquet.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
