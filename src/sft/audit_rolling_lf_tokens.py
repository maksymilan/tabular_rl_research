#!/usr/bin/env python3
"""Exact LLaMA-Factory token audit for last-turn-only rolling SFT records.

Run this in the same LLaMA-Factory environment used for training. The audit distinguishes final
target preservation from full causal-prefix preservation so a hardware cutoff never silently
becomes an underspecified data-quality rule.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from llamafactory.data.processor.supervised import SupervisedDatasetProcessor, infer_seqlen
from llamafactory.data.template import get_template_and_fix_tokenizer
from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.hparams.data_args import DataArguments


ROLE_MAP = {"human": "user", "gpt": "assistant"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def encoded_pairs(template, tokenizer, messages: list[dict], system: str | None, mask_history: bool):
    processed = template.mm_plugin.process_messages(messages, [], [], [], None)
    pairs = template.encode_multiturn(
        tokenizer,
        processed,
        system,
        None,
        mask_history and not template.preserve_thinking,
    )
    if not pairs:
        raise ValueError("template produced no assistant target")
    return pairs


def truncation_detail(
    pairs: list[tuple[list[int], list[int]]],
    *,
    cutoff_len: int,
    efficient_eos: bool,
) -> dict:
    """Mirror LLaMA-Factory's reverse-pair truncation and expose each causal segment."""
    total_length = 1 if efficient_eos else 0
    original_total = sum(len(source) + len(target) for source, target in pairs)
    last_target_original = len(pairs[-1][1])
    last_source_original = len(pairs[-1][0])
    last_target_kept = 0
    current_source_kept = 0
    history_tokens_kept = 0
    history_pairs_kept = 0
    history_pairs_complete = 0
    first_pair_complete = len(pairs) == 1
    for reverse_index, (source_ids, target_ids) in enumerate(reversed(pairs)):
        if total_length >= cutoff_len:
            break
        source_len, target_len = infer_seqlen(
            len(source_ids),
            len(target_ids),
            cutoff_len - total_length,
        )
        total_length += source_len + target_len
        if reverse_index == 0:
            current_source_kept = source_len
            last_target_kept = target_len
        else:
            history_tokens_kept += source_len + target_len
            history_pairs_kept += int(bool(source_len or target_len))
            history_pairs_complete += int(
                source_len == len(source_ids) and target_len == len(target_ids)
            )
            if reverse_index == len(pairs) - 1:
                first_pair_complete = (
                    source_len == len(source_ids) and target_len == len(target_ids)
                )
    if efficient_eos:
        total_length += 1
        original_total += 1
    return {
        "target_status": (
            "complete"
            if last_target_kept == last_target_original
            else "partial"
            if last_target_kept
            else "missing"
        ),
        "target_tokens_original": last_target_original,
        "target_tokens_kept": last_target_kept,
        "current_source_tokens_original": last_source_original,
        "current_source_tokens_kept": current_source_kept,
        "history_pairs_original": len(pairs) - 1,
        "history_pairs_kept": history_pairs_kept,
        "history_pairs_complete": history_pairs_complete,
        "first_pair_complete": first_pair_complete,
        "history_tokens_kept": history_tokens_kept,
        "original_tokens": original_total,
        "encoded_tokens": total_length,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--cutoff-len", type=int, default=4096)
    parser.add_argument("--template", default="qwen")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Explicitly enable the reasoning model's thinking-mode chat template.",
    )
    parser.add_argument(
        "--preserve-thinking",
        action="store_true",
        help=(
            "Preserve reasoning blocks in masked assistant history. This must match the "
            "training config and inference chat template for reasoning models such as Qwen3."
        ),
    )
    parser.add_argument(
        "--filter-policy",
        choices=["complete-target", "full-prefix"],
        default="complete-target",
        help="full-prefix additionally requires the current source and oldest history pair intact",
    )
    args = parser.parse_args()
    if args.cutoff_len <= 0:
        parser.error("--cutoff-len must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    data_args = DataArguments(
        template=args.template,
        cutoff_len=args.cutoff_len,
        mask_history=True,
        preserve_thinking=args.preserve_thinking,
        enable_thinking=args.enable_thinking,
    )
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    processor = SupervisedDatasetProcessor(template, tokenizer, None, data_args)

    kept, dropped, lengths, target_lengths, details, all_token_details = [], [], [], [], [], []
    target_status = Counter()
    for row in read_jsonl(args.input):
        conversations = row["conversations"]
        messages = [{"role": ROLE_MAP[item["from"]], "content": item["value"]} for item in conversations]
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            raise ValueError(f"invalid final target: {row.get('metadata', {}).get('record_id')}")
        pairs = encoded_pairs(template, tokenizer, messages, row.get("system"), mask_history=True)
        expected = pairs[-1][1]
        input_ids, labels = processor._encode_data_example(
            prompt=messages[:-1],
            response=[messages[-1]],
            system=row.get("system"),
            tools=None,
            images=[],
            videos=[],
            audios=[],
        )
        observed = [token for token in labels if token != IGNORE_INDEX]
        record_id = row.get("metadata", {}).get("record_id")
        detail = {
            "record_id": record_id,
            "source_episode_id": row.get("metadata", {}).get("source_episode_id"),
            "source_step_id": row.get("metadata", {}).get("source_step_id"),
            **truncation_detail(
                pairs,
                cutoff_len=args.cutoff_len,
                efficient_eos=template.efficient_eos,
            ),
        }
        detail["processor_target_complete"] = observed == expected
        all_token_details.append(detail)
        target_status[detail["target_status"]] += 1
        target_complete = detail["processor_target_complete"] and detail["target_status"] == "complete"
        current_source_complete = (
            detail["current_source_tokens_kept"]
            == detail["current_source_tokens_original"]
        )
        oldest_pair_complete = (
            not detail["history_pairs_original"] or detail["first_pair_complete"]
        )
        full_prefix_complete = (
            target_complete and current_source_complete and oldest_pair_complete
        )
        selected = (
            target_complete
            if args.filter_policy == "complete-target"
            else full_prefix_complete
        )
        if selected:
            kept.append(row)
            lengths.append(len(input_ids))
            target_lengths.append(len(observed))
        else:
            dropped.append({
                "record_id": record_id,
                "encoded_length": len(input_ids),
                "expected_target_tokens": len(expected),
                "retained_target_tokens": len(observed),
                "target_complete": target_complete,
                "current_source_complete": current_source_complete,
                "oldest_history_pair_complete": oldest_pair_complete,
            })
        if not full_prefix_complete:
            details.append(detail)

    write_jsonl(args.out, kept)
    manifest = {
        "input": str(args.input),
        "output": str(args.out),
        "model": args.model,
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "mask_history": True,
        "preserve_thinking": args.preserve_thinking,
        "enable_thinking": args.enable_thinking,
        "filter_policy": args.filter_policy,
        "records": len(kept) + len(dropped),
        "total": len(kept) + len(dropped),
        "target_status": dict(target_status),
        "kept_by_filter_policy": len(kept),
        "kept_complete_final_target": sum(
            item["processor_target_complete"] and item["target_status"] == "complete"
            for item in all_token_details
        ),
        "dropped_truncated_final_target": sum(
            not item["processor_target_complete"] or item["target_status"] != "complete"
            for item in all_token_details
        ),
        "dropped_by_filter_policy": len(dropped),
        "episodes_with_partial_or_missing_targets": sorted({
            item["source_episode_id"]
            for item in details
            if item["target_status"] != "complete"
        }),
        "max_target_tokens": max(
            (item["target_tokens_original"] for item in all_token_details),
            default=0,
        ),
        "original_token_lengths": {
            "min": min((item["original_tokens"] for item in all_token_details), default=0),
            "p50": sorted(item["original_tokens"] for item in all_token_details)[
                len(all_token_details) // 2
            ] if all_token_details else 0,
            "p90": sorted(item["original_tokens"] for item in all_token_details)[
                min(len(all_token_details) - 1, int(.9 * len(all_token_details)))
            ] if all_token_details else 0,
            "max": max((item["original_tokens"] for item in all_token_details), default=0),
        },
        "longest_records": sorted(
            (
                {
                    "record_id": item["record_id"],
                    "source_episode_id": item["source_episode_id"],
                    "original_tokens": item["original_tokens"],
                }
                for item in all_token_details
            ),
            key=lambda item: (-item["original_tokens"], str(item["record_id"])),
        )[:64],
        "records_with_truncated_current_source": sum(
            item["current_source_tokens_kept"] < item["current_source_tokens_original"]
            for item in details
        ),
        "episodes_with_truncated_current_source": sorted({
            item["source_episode_id"]
            for item in details
            if item["current_source_tokens_kept"] < item["current_source_tokens_original"]
        }),
        "records_with_incomplete_initial_pair": sum(
            item["history_pairs_original"] and not item["first_pair_complete"]
            for item in details
        ),
        "episodes_with_incomplete_initial_pair": sorted({
            item["source_episode_id"]
            for item in details
            if item["history_pairs_original"] and not item["first_pair_complete"]
        }),
        "encoded_lengths": {
            "min": min(lengths) if lengths else 0,
            "p50": sorted(lengths)[len(lengths) // 2] if lengths else 0,
            "p90": sorted(lengths)[min(len(lengths) - 1, int(.9 * len(lengths)))] if lengths else 0,
            "max": max(lengths) if lengths else 0,
        },
        "target_lengths": {
            "min": min(target_lengths) if target_lengths else 0,
            "max": max(target_lengths) if target_lengths else 0,
        },
        "dropped": dropped,
        "details": details,
        "tool_hist": dict(Counter(row.get("metadata", {}).get("source_step_id", "unknown") for row in kept)),
    }
    args.out.with_suffix(".token_audit.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
