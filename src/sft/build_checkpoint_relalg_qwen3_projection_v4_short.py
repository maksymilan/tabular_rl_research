#!/usr/bin/env python3
"""Build an extractive short-reasoning view from projection-v3.

The expensive causal trajectory is immutable: system text, causal history, tool
name, arguments, and action JSON remain byte-identical.  Only the reasoning in
the current supervised target may be shortened, and the shortened text must be
one contiguous substring of the original provider reasoning.  No model call,
gold field, future tool result, or generated summary is used.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

from build_checkpoint_relalg_qwen3_projection_v3 import (
    TARGET_RE,
    read_jsonl,
    repeated_ngram_ratio,
    sha256,
)
from sft_dataset_registry import write_sharegpt_dataset_info


PROJECTION_VERSION = "checkpoint-relalg-qwen3-projection-v4-extractive-short-v1"
POLICY_VERSION = "checkpoint-relalg-qwen3-short-reasoning-policy-v1"
SOURCE_SCHEMA = "checkpoint-relalg-qwen3-projection-v3-dataset-v1"
OUTPUT_SCHEMA = "checkpoint-relalg-qwen3-projection-v4-short-dataset-v1"
DEFAULT_SOFT_MAX_TOKENS = 256
DEFAULT_HARD_MAX_TOKENS = 512
DEFAULT_MAX_REPEAT_RATIO = 0.12
MIN_REASONING_TOKENS = 4

SENTENCE_END = re.compile(r"[.!?](?:[\"'”’\)\]\}]*)?(?=\s|$)")
WORD = re.compile(r"[A-Za-z0-9_]+")

TOOL_ANCHORS: dict[str, tuple[str, ...]] = {
    "describe_table": ("describe", "schema"),
    "inspect_column": ("inspect", "column", "distinct", "values"),
    "read_rows": ("read rows", "read_rows", "sample rows", "rows"),
    "filter_rows": ("filter", "filter_rows", "population"),
    "shape_rows": ("shape", "shape_rows", "project", "output columns"),
    "join": ("join",),
    "group_aggregate": ("aggregate", "group_aggregate", "group by", "count", "sum", "average"),
    "scalar_compute": ("scalar", "scalar_compute", "compute", "calculate", "divide", "multiply", "subtract"),
    "rank_select": ("rank", "rank_select", "top", "order by"),
    "set_operation": ("set operation", "set_operation", "union", "intersect", "except"),
    "answer": ("answer", "submit"),
}


class Tokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def token_count(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def token_counts(tokenizer: Tokenizer, texts: list[str]) -> list[int]:
    if not texts:
        return []
    if callable(tokenizer):
        encoded = tokenizer(  # type: ignore[operator]
            texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        return [len(item) for item in encoded]
    return [token_count(tokenizer, text) for text in texts]


def percentile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "mean": sum(values) / len(values) if values else math.nan,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0),
    }


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return non-empty, exact contiguous sentence/paragraph spans."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in SENTENCE_END.finditer(text):
        end = match.end()
        if text[start:end].strip():
            left = start
            while left < end and text[left].isspace():
                left += 1
            spans.append((left, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    if start < len(text) and text[start:].strip():
        end = len(text.rstrip())
        spans.append((start, end))
    if not spans and text.strip():
        left = len(text) - len(text.lstrip())
        spans.append((left, len(text.rstrip())))
    return spans


def has_repeated_sentence(text: str) -> bool:
    normalized = [
        " ".join(text[start:end].lower().split())
        for start, end in sentence_spans(text)
        if len(" ".join(text[start:end].split())) >= 24
    ]
    return len(normalized) != len(set(normalized))


def _argument_strings(value: Any) -> list[str]:
    values: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            values.extend(_argument_strings(item))
    elif isinstance(value, list):
        for item in value:
            values.extend(_argument_strings(item))
    elif isinstance(value, str):
        normalized = " ".join(value.lower().split())
        if len(normalized) >= 2:
            values.append(normalized)
    return values


def action_anchors(action: dict[str, Any]) -> tuple[str, ...]:
    tool = str(action.get("tool", ""))
    anchors = {tool.lower(), tool.lower().replace("_", " "), *TOOL_ANCHORS.get(tool, ())}
    anchors.update(_argument_strings(action.get("arguments")))
    return tuple(sorted((item for item in anchors if len(item) >= 2), key=len, reverse=True))


def contains_anchor(text: str, anchors: tuple[str, ...]) -> bool:
    lowered = " ".join(text.lower().split())
    return any(anchor in lowered for anchor in anchors)


def parse_target(row: dict[str, Any], *, item_id: str) -> tuple[str, str, dict[str, Any]]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"{item_id}: conversations are missing")
    target = conversations[-1]
    value = target.get("value") if isinstance(target, dict) else None
    match = TARGET_RE.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"{item_id}: target carrier is invalid")
    reasoning = match.group("reasoning").strip()
    action_text = match.group("action")
    action = json.loads(action_text)
    if set(action) != {"tool", "arguments"}:
        raise ValueError(f"{item_id}: target action envelope is invalid")
    return reasoning, action_text, action


def project_reasoning(
    reasoning: str,
    action: dict[str, Any],
    tokenizer: Tokenizer,
    *,
    soft_max_tokens: int = DEFAULT_SOFT_MAX_TOKENS,
    hard_max_tokens: int = DEFAULT_HARD_MAX_TOKENS,
    max_repeat_ratio: float = DEFAULT_MAX_REPEAT_RATIO,
) -> tuple[str | None, dict[str, Any]]:
    if soft_max_tokens < MIN_REASONING_TOKENS or hard_max_tokens < soft_max_tokens:
        raise ValueError("invalid reasoning token limits")
    anchors = action_anchors(action)
    spans = sentence_spans(reasoning)
    if not spans:
        return None, {"reason": "empty_reasoning", "source_reasoning_tokens": 0}

    sentence_texts = [reasoning[start:end] for start, end in spans]
    base_counts = token_counts(tokenizer, [reasoning, *sentence_texts])
    raw_tokens = base_counts[0]
    sentence_counts = base_counts[1:]
    # Only suffixes near the hard budget can possibly be selected.  Counting
    # each sentence once avoids tokenizing every O(n^2) suffix of a long trace.
    # The margin covers small boundary-tokenization differences; final limits
    # always use an exact encoding of the contiguous candidate.
    raw_candidates: list[tuple[int, int, str]] = []
    end = spans[-1][1]
    approximate_tokens = 0
    for (start, _), sentence_count in zip(reversed(spans), reversed(sentence_counts), strict=True):
        approximate_tokens += sentence_count
        if approximate_tokens > hard_max_tokens + 128:
            break
        text = reasoning[start:end].strip()
        raw_candidates.append((start, end, text))
    counts = token_counts(tokenizer, [item[2] for item in raw_candidates])
    candidates: list[tuple[int, int, str, int, float, bool, bool]] = []
    for (start, end, text), count in zip(raw_candidates, counts, strict=True):
        if count > hard_max_tokens:
            continue
        candidates.append(
            (
                start,
                end,
                text,
                count,
                repeated_ngram_ratio(text),
                contains_anchor(text, anchors),
                has_repeated_sentence(text),
            )
        )

    raw_repeat_ratio = repeated_ngram_ratio(reasoning)
    raw_repeats_sentence = has_repeated_sentence(reasoning)
    if (
        raw_tokens <= soft_max_tokens
        and raw_repeat_ratio < max_repeat_ratio
        and not raw_repeats_sentence
    ):
        chosen = (
            0,
            len(reasoning),
            reasoning,
            raw_tokens,
            raw_repeat_ratio,
            True,
            False,
        )
    else:
        valid = [
            item
            for item in candidates
            if item[3] >= MIN_REASONING_TOKENS
            and item[4] < max_repeat_ratio
            and item[5]
            and not item[6]
        ]
        soft = [item for item in valid if item[3] <= soft_max_tokens]
        chosen = max(soft, key=lambda item: item[3], default=None)
        if chosen is None:
            chosen = min(valid, key=lambda item: item[3], default=None)
        if chosen is None:
            return None, {
                "reason": "no_grounded_nonrepetitive_suffix_within_hard_limit",
                "source_reasoning_tokens": raw_tokens,
                "source_reasoning_repeat_ratio": repeated_ngram_ratio(reasoning),
            }

    start, end, projected, projected_tokens, repeat_ratio, grounded, repeated_sentence = chosen
    if projected_tokens < MIN_REASONING_TOKENS:
        return None, {
            "reason": "projected_reasoning_too_short",
            "source_reasoning_tokens": raw_tokens,
        }
    if repeat_ratio >= max_repeat_ratio:
        return None, {
            "reason": "projected_reasoning_repetitive",
            "source_reasoning_tokens": raw_tokens,
            "projected_reasoning_tokens": projected_tokens,
            "projected_reasoning_repeat_ratio": repeat_ratio,
        }
    if repeated_sentence:
        return None, {
            "reason": "projected_reasoning_repeats_sentence",
            "source_reasoning_tokens": raw_tokens,
            "projected_reasoning_tokens": projected_tokens,
        }
    if raw_tokens > soft_max_tokens and not grounded:
        raise AssertionError("modified reasoning lacks an action anchor")
    if projected not in reasoning or reasoning[start:end].strip() != projected:
        raise AssertionError("reasoning projection is not an exact contiguous substring")
    return projected, {
        "reasoning_projection_version": PROJECTION_VERSION,
        "source_reasoning_sha256": digest_text(reasoning),
        "source_reasoning_characters": len(reasoning),
        "source_reasoning_tokens": raw_tokens,
        "projected_reasoning_sha256": digest_text(projected),
        "projected_reasoning_characters": len(projected),
        "projected_reasoning_tokens": projected_tokens,
        "source_character_span": [start, end],
        "byte_preserved_action": True,
        "contiguous_source_substring": True,
        "action_anchor_present": contains_anchor(projected, anchors),
        "projected_reasoning_repeat_ratio": repeat_ratio,
        "projected_exact_sentence_repeat": False,
        "was_shortened": projected != reasoning,
    }


def _validate_source_manifest(
    source_manifest: Path,
    source_canonical: Path,
    source_index: Path,
) -> dict[str, Any]:
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SOURCE_SCHEMA:
        raise ValueError("source is not a projection-v3 dataset")
    combined = manifest.get("outputs", {}).get("combined", {})
    expected = {
        "canonical_sha256": sha256(source_canonical),
        "index_sha256": sha256(source_index),
    }
    for key, value in expected.items():
        if combined.get(key) != value:
            raise ValueError(f"source manifest {key} differs")
    return manifest


def build(
    source_canonical: Path,
    source_index: Path,
    source_manifest: Path,
    output_dir: Path,
    dataset_name: str,
    tokenizer: Tokenizer,
    *,
    tokenizer_identity: str,
    soft_max_tokens: int = DEFAULT_SOFT_MAX_TOKENS,
    hard_max_tokens: int = DEFAULT_HARD_MAX_TOKENS,
    max_repeat_ratio: float = DEFAULT_MAX_REPEAT_RATIO,
) -> dict[str, Any]:
    source_canonical = source_canonical.resolve()
    source_index = source_index.resolve()
    source_manifest = source_manifest.resolve()
    _validate_source_manifest(source_manifest, source_canonical, source_index)
    rows = read_jsonl(source_canonical)
    indexes = read_jsonl(source_index)
    if len(rows) != len(indexes):
        raise ValueError("source canonical/index counts differ")

    output_dir.mkdir(parents=True, exist_ok=True)
    canonical_out = output_dir / f"{dataset_name}_canonical.jsonl"
    index_out = output_dir / f"{dataset_name}_index.jsonl"
    training_out = output_dir / f"{dataset_name}.jsonl"
    excluded_out = output_dir / f"{dataset_name}_excluded.jsonl"
    manifest_out = output_dir / f"{dataset_name}.manifest.json"
    for path in (canonical_out, index_out, training_out, excluded_out, manifest_out):
        if path.exists():
            raise FileExistsError(f"refusing existing projection-v4 output: {path}")

    selected_rows: list[dict[str, Any]] = []
    selected_indexes: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    tool_hist: Counter[str] = Counter()
    source_token_hist: Counter[str] = Counter()
    projected_token_hist: Counter[str] = Counter()
    episodes: set[str] = set()
    answer_episodes: set[str] = set()
    shortened = 0
    selected_source_tokens: list[int] = []
    selected_projected_tokens: list[int] = []

    def bucket(value: int) -> str:
        for limit in (128, 256, 384, 512, 1024, 2048, 4096, 8192):
            if value <= limit:
                return f"le_{limit}"
        return "gt_8192"

    for row, item in zip(rows, indexes, strict=True):
        metadata = row.get("metadata") or {}
        item_id = metadata.get("record_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("source row is missing metadata.record_id")
        if item.get("record_id") != item_id:
            raise ValueError(f"{item_id}: source index order differs")
        reasoning, action_text, action = parse_target(row, item_id=item_id)
        if item.get("tool_name") != action.get("tool"):
            raise ValueError(f"{item_id}: source action/index tool differs")
        projected, projection = project_reasoning(
            reasoning,
            action,
            tokenizer,
            soft_max_tokens=soft_max_tokens,
            hard_max_tokens=hard_max_tokens,
            max_repeat_ratio=max_repeat_ratio,
        )
        if projected is None:
            excluded.append(
                {
                    "record_id": item_id,
                    "source_episode_id": item.get("source_episode_id"),
                    "source_turn_index": item.get("source_turn_index"),
                    "tool_name": item.get("tool_name"),
                    **projection,
                }
            )
            continue

        derived_row = copy.deepcopy(row)
        target = derived_row["conversations"][-1]
        target["value"] = f"<think>\n{projected}\n</think>\n\n{action_text}"
        derived_metadata = dict(derived_row.get("metadata") or {})
        derived_metadata["reasoning_projection"] = projection
        derived_row["metadata"] = derived_metadata
        derived_index = dict(item)
        derived_index["reasoning_projection_policy_version"] = POLICY_VERSION
        derived_index["reasoning_projection"] = projection

        selected_rows.append(derived_row)
        selected_indexes.append(derived_index)
        shortened += int(projection["was_shortened"])
        source_token_hist[bucket(int(projection["source_reasoning_tokens"]))] += 1
        projected_token_hist[bucket(int(projection["projected_reasoning_tokens"]))] += 1
        selected_source_tokens.append(int(projection["source_reasoning_tokens"]))
        selected_projected_tokens.append(int(projection["projected_reasoning_tokens"]))
        tool = str(action["tool"])
        tool_hist[tool] += 1
        episode_id = str(item.get("source_episode_id"))
        episodes.add(episode_id)
        if tool == "answer":
            answer_episodes.add(episode_id)

    training_rows = [
        {"system": row.get("system"), "conversations": row.get("conversations")}
        for row in selected_rows
    ]
    write_jsonl_atomic(canonical_out, selected_rows)
    write_jsonl_atomic(index_out, selected_indexes)
    write_jsonl_atomic(training_out, training_rows)
    write_jsonl_atomic(excluded_out, excluded)
    write_sharegpt_dataset_info(training_out, dataset_name)

    manifest = {
        "schema_version": OUTPUT_SCHEMA,
        "reasoning_projection_version": PROJECTION_VERSION,
        "reasoning_projection_policy_version": POLICY_VERSION,
        "mutation": (
            "current target reasoning is one exact contiguous source substring; "
            "system, causal history, and action JSON are byte-preserved"
        ),
        "implementation": {
            "builder": str(Path(__file__).resolve()),
            "builder_sha256": sha256(Path(__file__).resolve()),
        },
        "source": {
            "manifest": str(source_manifest),
            "manifest_sha256": sha256(source_manifest),
            "canonical": str(source_canonical),
            "canonical_sha256": sha256(source_canonical),
            "index": str(source_index),
            "index_sha256": sha256(source_index),
            "records": len(rows),
        },
        "policy": {
            "tokenizer": tokenizer_identity,
            "soft_max_reasoning_tokens": soft_max_tokens,
            "hard_max_reasoning_tokens": hard_max_tokens,
            "min_reasoning_tokens": MIN_REASONING_TOKENS,
            "max_repeated_8gram_ratio": max_repeat_ratio,
            "requires_action_anchor_when_shortened": True,
            "abstractive_rewrite": False,
            "external_model_calls": 0,
            "gold_or_future_feedback_access": False,
        },
        "selection": {
            "source_records": len(rows),
            "selected_records": len(selected_rows),
            "excluded_records": len(excluded),
            "shortened_records": shortened,
            "unchanged_records": len(selected_rows) - shortened,
            "selected_episodes": len(episodes),
            "episodes_with_answer_target": len(answer_episodes),
            "tool_hist": dict(tool_hist.most_common()),
            "source_reasoning_token_hist": dict(source_token_hist),
            "projected_reasoning_token_hist": dict(projected_token_hist),
            "source_reasoning_token_distribution": distribution(selected_source_tokens),
            "projected_reasoning_token_distribution": distribution(selected_projected_tokens),
            "reasoning_tokens_before": sum(selected_source_tokens),
            "reasoning_tokens_after": sum(selected_projected_tokens),
            "reasoning_token_reduction": (
                1 - sum(selected_projected_tokens) / sum(selected_source_tokens)
                if selected_source_tokens
                else 0.0
            ),
            "exclusion_reason_hist": dict(
                Counter(str(item["reason"]) for item in excluded).most_common()
            ),
        },
        "outputs": {
            "canonical": str(canonical_out),
            "canonical_sha256": sha256(canonical_out),
            "index": str(index_out),
            "index_sha256": sha256(index_out),
            "training_view": str(training_out),
            "training_view_sha256": sha256(training_out),
            "excluded": str(excluded_out),
            "excluded_sha256": sha256(excluded_out),
        },
    }
    manifest_out.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-canonical", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--soft-max-reasoning-tokens", type=int, default=DEFAULT_SOFT_MAX_TOKENS)
    parser.add_argument("--hard-max-reasoning-tokens", type=int, default=DEFAULT_HARD_MAX_TOKENS)
    parser.add_argument("--max-repeated-8gram-ratio", type=float, default=DEFAULT_MAX_REPEAT_RATIO)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer_path = args.tokenizer.resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    manifest = build(
        args.source_canonical,
        args.source_index,
        args.source_manifest,
        args.output_dir,
        args.dataset_name,
        tokenizer,
        tokenizer_identity=str(tokenizer_path),
        soft_max_tokens=args.soft_max_reasoning_tokens,
        hard_max_tokens=args.hard_max_reasoning_tokens,
        max_repeat_ratio=args.max_repeated_8gram_ratio,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
