#!/usr/bin/env python3
"""Question-balanced Action-DPO over exact visible prefixes and tool JSON tokens."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

FLEX_ATTENTION_SMALL_BLOCK_OPTIONS = {
    "fwd_BLOCK_M": 32,
    "fwd_BLOCK_N": 32,
    "bwd_BLOCK_M1": 32,
    "bwd_BLOCK_N1": 32,
    "bwd_BLOCK_M2": 32,
    "bwd_BLOCK_N2": 32,
    "num_stages": 1,
    "num_warps": 4,
    "ROWS_GUARANTEED_SAFE": True,
    "BLOCKS_ARE_CONTIGUOUS": True,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def adapter_weight_path(path: Path) -> Path:
    for name in ("adapter_model.safetensors", "adapter_model.bin"):
        candidate = path / name
        if candidate.is_file():
            return candidate
    raise ValueError(f"adapter has no weight file: {path}")


def validate_dataset(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Action-DPO dataset is empty")
    seen = set()
    for row in rows:
        if row.get("schema_version") != "fixed-prefix-action-pair-v1":
            raise ValueError("unexpected Action-DPO pair schema")
        if row["pair_sha256"] in seen:
            raise ValueError(f"duplicate Action-DPO pair: {row['pair_sha256']}")
        seen.add(row["pair_sha256"])
        verification = row.get("positive_verified_by") or {}
        if not all(
            verification.get(key) is True
            for key in (
                "real_harness_execution",
                "successful_suffix",
                "source_trajectory_replayed",
            )
        ):
            raise ValueError("Action-DPO positive action lacks full replay verification")
        if not (verification.get("source_replay") or {}).get("correct"):
            raise ValueError("Action-DPO positive source replay is not correct")


class ActionScorer:
    def __init__(
        self,
        model,
        tokenizer,
        *,
        max_length: int,
        selected_tool_logits: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        base = getattr(model, "base_model", model)
        base = getattr(base, "model", base)
        signature = inspect.signature(base.forward)
        self.keep_argument = next(
            (
                name
                for name in ("logits_to_keep", "num_logits_to_keep")
                if name in signature.parameters
            ),
            None,
        )
        self.selected_tool_logits = selected_tool_logits
        if self.selected_tool_logits and self.keep_argument != "logits_to_keep":
            raise ValueError(
                "selected tool logits require a model forward with tensor logits_to_keep"
            )

    def encode(self, messages: list[dict[str, Any]], response: str):
        from frameworks.trl.tool_loss_mask import tool_token_loss_mask

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(prompt_ids, "tolist"):
            prompt_ids = prompt_ids.tolist()
        response_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        mask = tool_token_loss_mask(self.tokenizer, response_ids)
        if len(prompt_ids) + len(response_ids) > self.max_length:
            raise ValueError(
                f"exact Action-DPO sequence exceeds max length: "
                f"{len(prompt_ids) + len(response_ids)} > {self.max_length}"
            )
        return list(prompt_ids), list(response_ids), mask

    def mean_tool_logprob(self, messages: list[dict[str, Any]], response: str):
        return self.mean_tool_logprobs([(messages, response)])[0]

    def mean_tool_logprobs(
        self,
        items: list[tuple[list[dict[str, Any]], str]],
    ):
        """Score several tool responses in one left-padded causal forward."""
        import torch

        encoded = [self.encode(messages, response) for messages, response in items]
        device = next(self.model.parameters()).device
        max_total = max(len(prompt) + len(response) for prompt, response, _ in encoded)
        max_response = max(len(response) for _, response, _ in encoded)
        input_rows = []
        attention_rows = []
        selected_offsets_by_row = []
        prediction_positions_by_row = []
        for prompt_ids, response_ids, mask in encoded:
            ids = prompt_ids + response_ids
            padding = max_total - len(ids)
            input_rows.append([self.tokenizer.pad_token_id] * padding + ids)
            attention_rows.append([0] * padding + [1] * len(ids))
            selected_offsets = [index for index, selected in enumerate(mask) if selected]
            if not selected_offsets:
                raise RuntimeError("Action-DPO response has no selected tool tokens")
            selected_offsets_by_row.append(selected_offsets)
            prediction_positions_by_row.append(
                [
                    padding + len(prompt_ids) + response_offset - 1
                    for response_offset in selected_offsets
                ]
            )
        input_ids = torch.tensor(input_rows, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attention_rows, dtype=torch.long, device=device)
        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
        }
        if getattr(self.model.config, "_attn_implementation", None) == "flex_attention":
            kwargs["kernel_options"] = FLEX_ATTENTION_SMALL_BLOCK_OPTIONS
        kept_prediction_positions = None
        if self.selected_tool_logits:
            kept_prediction_positions = sorted(
                {
                    position
                    for row_positions in prediction_positions_by_row
                    for position in row_positions
                }
            )
            kwargs[self.keep_argument] = torch.tensor(
                kept_prediction_positions,
                dtype=torch.long,
                device=device,
            )
        elif self.keep_argument:
            kwargs[self.keep_argument] = max_response + 1
        logits = self.model(**kwargs).logits
        returned = int(logits.shape[1])
        results = []
        position_to_logit = (
            {
                position: index
                for index, position in enumerate(kept_prediction_positions)
            }
            if kept_prediction_positions is not None
            else None
        )
        for row_index, (_, response_ids, mask) in enumerate(encoded):
            if position_to_logit is not None:
                selected_offsets = selected_offsets_by_row[row_index]
                output_indices = torch.tensor(
                    [
                        position_to_logit[position]
                        for position in prediction_positions_by_row[row_index]
                    ],
                    dtype=torch.long,
                    device=device,
                )
                response_logits = logits[row_index].index_select(0, output_indices)
                labels = torch.tensor(
                    [response_ids[offset] for offset in selected_offsets],
                    dtype=torch.long,
                    device=device,
                )
                chosen = response_logits.gather(
                    -1, labels.unsqueeze(-1)
                ).squeeze(-1)
                token_logps = chosen.float() - torch.logsumexp(
                    response_logits.float(),
                    dim=-1,
                )
                results.append((token_logps.mean(), len(selected_offsets)))
                continue
            response_length = len(response_ids)
            response_logits = logits[
                row_index,
                returned - response_length - 1 : returned - 1,
            ]
            if response_logits.shape[0] != response_length:
                raise RuntimeError("model did not return aligned Action-DPO logits")
            labels = torch.tensor(response_ids, dtype=torch.long, device=device)
            chosen = response_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            token_logps = chosen.float() - torch.logsumexp(
                response_logits.float(),
                dim=-1,
            )
            selected = torch.tensor(mask, dtype=torch.bool, device=device)
            results.append((token_logps[selected].mean(), int(selected.sum().item())))
        return results


def load_policy(args):
    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs = {
        "quantization_config": quantization,
        "torch_dtype": torch.bfloat16,
        "device_map": {"": 0},
        "trust_remote_code": True,
    }
    if args.attention_implementation:
        model_kwargs["attn_implementation"] = args.attention_implementation
    base = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    base.config.use_cache = False
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = PeftModel.from_pretrained(base, args.adapter_path, is_trainable=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    return model, tokenizer


def relocate_optimizer_moments(optimizer, *, to_parameter_device: bool) -> None:
    """Move Adam moments without changing their dtype, values, or update rule."""
    import torch

    for parameter, state in optimizer.state.items():
        target = parameter.device if to_parameter_device else torch.device("cpu")
        for name, value in tuple(state.items()):
            # AdamW's scalar step intentionally remains on its original device.  Only
            # parameter-shaped moments account for meaningful persistent GPU memory.
            if name != "step" and torch.is_tensor(value) and value.device != target:
                state[name] = value.to(device=target, non_blocking=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--verification-audit", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--attention-implementation",
        choices=("eager", "sdpa", "flex_attention", "flash_attention_2"),
        default=None,
        help=(
            "explicit Transformers attention backend; sdpa avoids materializing the "
            "quadratic attention matrix while preserving causal-attention semantics"
        ),
    )
    parser.add_argument(
        "--sequential-pair-scoring",
        action="store_true",
        help=(
            "score positive and negative responses in separate forwards; this preserves "
            "the Action-DPO objective while avoiding batch-2 attention OOM on long prefixes"
        ),
    )
    parser.add_argument(
        "--memory-safe-dpo-backward",
        action="store_true",
        help=(
            "backpropagate positive and negative log-probabilities one graph at a time, "
            "recomputing the negative with the same RNG state after deriving the exact "
            "detached DPO scalar derivative"
        ),
    )
    parser.add_argument(
        "--selected-tool-logits",
        action="store_true",
        help=(
            "ask supported causal-LM forwards to materialize vocabulary logits only "
            "at the tool-token prediction positions used by the Action-DPO loss"
        ),
    )
    parser.add_argument(
        "--cpu-offload-optimizer-state",
        action="store_true",
        help=(
            "keep AdamW parameter-shaped moments on CPU during forward/backward, "
            "moving them to each parameter device only for the unchanged optimizer step"
        ),
    )
    parser.add_argument(
        "--experiment-name",
        default="exp15_fixed_prefix_action_dpo",
    )
    args = parser.parse_args()

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty Action-DPO output: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.beta <= 0 or args.learning_rate <= 0 or args.epochs != 1:
        raise SystemExit("initial controlled Action-DPO requires beta>0, lr>0, epochs=1")
    if args.memory_safe_dpo_backward and not args.sequential_pair_scoring:
        raise SystemExit("memory-safe DPO backward requires sequential pair scoring")
    audit = json.loads(args.verification_audit.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("verified_pairs", 0) < 1:
        raise SystemExit("Action-DPO verification audit is not a pass")
    rows = load_jsonl(args.dataset)
    validate_dataset(rows)
    if audit["output_sha256"] != sha256_file(args.dataset):
        raise SystemExit("Action-DPO dataset hash does not match verification audit")

    import torch
    import torch.nn.functional as F

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, tokenizer = load_policy(args)
    scorer = ActionScorer(
        model,
        tokenizer,
        max_length=args.max_length,
        selected_tool_logits=args.selected_tool_logits,
    )

    def score_pair(row: dict[str, Any]):
        items = (
            (row["state"], row["positive_scoring_response"]),
            (row["state"], row["negative_scoring_response"]),
        )
        if args.sequential_pair_scoring:
            return scorer.mean_tool_logprob(*items[0]), scorer.mean_tool_logprob(*items[1])
        return tuple(scorer.mean_tool_logprobs(list(items)))

    reference_rows = []
    reference_delta = {}
    model.eval()
    for index, row in enumerate(rows, 1):
        with torch.no_grad():
            (positive, positive_tokens), (negative, negative_tokens) = score_pair(row)
        delta = float((positive - negative).item())
        reference_delta[row["pair_sha256"]] = delta
        reference_rows.append(
            {
                "pair_sha256": row["pair_sha256"],
                "question_id": row["question_id"],
                "positive_mean_tool_logprob": float(positive.item()),
                "negative_mean_tool_logprob": float(negative.item()),
                "reference_delta": delta,
                "positive_tool_tokens": positive_tokens,
                "negative_tool_tokens": negative_tokens,
            }
        )
        print(json.dumps({"reference": index, "total": len(rows)}), flush=True)
    reference_path = args.output_dir / "reference_scores.jsonl"
    write_jsonl(reference_path, reference_rows)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["question_id"])].append(row)
    question_ids = sorted(grouped)
    rng = random.Random(args.seed)
    rng.shuffle(question_ids)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    metrics = []
    clipped = 0
    model.train()
    for step, question_id in enumerate(question_ids, 1):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        margins = []
        question_rows = grouped[question_id]
        for row in question_rows:
            if args.memory_safe_dpo_backward:
                positive, _ = scorer.mean_tool_logprob(
                    row["state"], row["positive_scoring_response"]
                )
                cpu_rng_before_negative = torch.random.get_rng_state()
                cuda_rng_before_negative = torch.cuda.get_rng_state(positive.device)
                with torch.no_grad():
                    negative_value, _ = scorer.mean_tool_logprob(
                        row["state"], row["negative_scoring_response"]
                    )
                policy_delta_value = positive.detach() - negative_value
                ref_delta = torch.tensor(
                    reference_delta[row["pair_sha256"]],
                    dtype=policy_delta_value.dtype,
                    device=policy_delta_value.device,
                )
                centered_delta = policy_delta_value - ref_delta
                loss_value = F.softplus(-args.beta * centered_delta)
                d_loss_d_delta = (
                    -args.beta * torch.sigmoid(-args.beta * centered_delta)
                ).detach() / len(question_rows)

                positive.backward(gradient=d_loss_d_delta)
                del positive
                torch.cuda.empty_cache()

                torch.random.set_rng_state(cpu_rng_before_negative)
                torch.cuda.set_rng_state(
                    cuda_rng_before_negative,
                    device=policy_delta_value.device,
                )
                negative, _ = scorer.mean_tool_logprob(
                    row["state"], row["negative_scoring_response"]
                )
                negative.backward(gradient=-d_loss_d_delta)
                del negative
                losses.append(float(loss_value.item()))
                margins.append(float(policy_delta_value.item()))
                del (
                    negative_value,
                    policy_delta_value,
                    ref_delta,
                    centered_delta,
                    loss_value,
                    d_loss_d_delta,
                )
                # The following positive may be a substantially longer prefix.  Release
                # cached blocks from the just-finished negative checkpoint backward so a
                # large activation/workspace allocation is not defeated by fragmentation.
                torch.cuda.empty_cache()
            else:
                (positive, _), (negative, _) = score_pair(row)
                policy_delta = positive - negative
                ref_delta = torch.tensor(
                    reference_delta[row["pair_sha256"]],
                    dtype=policy_delta.dtype,
                    device=policy_delta.device,
                )
                loss = F.softplus(-args.beta * (policy_delta - ref_delta))
                (loss / len(question_rows)).backward()
                losses.append(float(loss.detach().item()))
                margins.append(float(policy_delta.detach().item()))
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.gradient_clip)
        grad_norm_value = float(grad_norm.detach().item())
        clipped += int(grad_norm_value > args.gradient_clip)
        if not math.isfinite(grad_norm_value):
            raise RuntimeError("Action-DPO produced a non-finite gradient norm")
        if args.cpu_offload_optimizer_state:
            relocate_optimizer_moments(optimizer, to_parameter_device=True)
        optimizer.step()
        if args.cpu_offload_optimizer_state:
            relocate_optimizer_moments(optimizer, to_parameter_device=False)
            torch.cuda.empty_cache()
        metric = {
            "step": step,
            "question_id": question_id,
            "pairs": len(question_rows),
            "loss": fmean(losses),
            "policy_margin": fmean(margins),
            "grad_norm": grad_norm_value,
            "gradient_clipped": grad_norm_value > args.gradient_clip,
        }
        metrics.append(metric)
        print(json.dumps(metric), flush=True)

    final_dir = args.output_dir / "final"
    model.save_pretrained(final_dir, safe_serialization=True)
    tokenizer.save_pretrained(final_dir)
    write_jsonl(args.output_dir / "training_metrics.jsonl", metrics)
    clip_fraction = clipped / len(metrics)
    manifest = {
        "schema_version": "fixed-prefix-action-dpo-run-v1",
        "experiment_name": args.experiment_name,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": sha256_file(args.dataset),
        "verification_audit_sha256": sha256_file(args.verification_audit),
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()),
        "initial_adapter_sha256": sha256_file(adapter_weight_path(args.adapter_path)),
        "reference": "frozen SFT2 checkpoint-1682 precomputed before optimizer step 1",
        "reference_scores_sha256": sha256_file(reference_path),
        "beta": args.beta,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "gradient_clip": args.gradient_clip,
        "attention_implementation": (
            args.attention_implementation
            or getattr(model.config, "_attn_implementation", None)
        ),
        "flex_attention_kernel_options": (
            FLEX_ATTENTION_SMALL_BLOCK_OPTIONS
            if args.attention_implementation == "flex_attention"
            else None
        ),
        "tool_token_only": True,
        "pair_scoring_mode": (
            "sequential_memory_safe" if args.sequential_pair_scoring else "paired_batch"
        ),
        "backward_mode": (
            "single_graph_exact_dpo_derivative_recompute"
            if args.memory_safe_dpo_backward
            else "direct_autograd"
        ),
        "logits_mode": (
            "selected_tool_token_positions"
            if args.selected_tool_logits
            else "response_suffix"
        ),
        "optimizer_state_placement_during_backward": (
            "cpu" if args.cpu_offload_optimizer_state else "parameter_device"
        ),
        "question_balanced_sampling": True,
        "questions": len(question_ids),
        "pairs": len(rows),
        "optimizer_steps": len(metrics),
        "gradient_clip_fraction": clip_fraction,
        "learning_rate_followup": (
            "reduce_to_5e-7_without_changing_beta"
            if clip_fraction > 0.20
            else "retain_1e-6"
        ),
        "seed": args.seed,
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
