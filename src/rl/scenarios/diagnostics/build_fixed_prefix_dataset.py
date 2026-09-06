#!/usr/bin/env python3
"""Build verifier-backed same-prefix action pairs from full pass@k artifacts.

The builder is deliberately conservative: a positive action must be an actually
executed turn on a trajectory that reached the correct denotation, and the
negative action must have been sampled from the exact same model-visible input
on an incorrect trajectory.  It never invents an action or aligns merely
similar resident states.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

DEFAULT_QUOTAS = {
    "normal_first_step": 50,
    "semantic_divergence": 80,
    "error_recovery": 50,
    "population_grain_join": 50,
    "ranking_tie_order": 35,
    "final_output_slot": 35,
}
SHARED_REASON = "Compare the next legal tool action for the current database state."
ASSISTANT_REASONING_NORMALIZATION = "assistant-think-placeholder-v1"
ASSISTANT_REASONING_PLACEHOLDER = "[historical reasoning omitted]"
_THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalize_assistant_reasoning(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove authored reasoning semantics while preserving the action carrier.

    Only historical assistant message contents are changed.  The strict action
    JSON following ``</think>`` and every harness-authored/system/user message
    remain byte-for-byte identical.  A non-empty, constant placeholder keeps
    the model-visible carrier shape stable without selecting either sampled
    trajectory's reasoning as privileged context.
    """

    normalized = copy.deepcopy(messages)
    replacement = f"<think>{ASSISTANT_REASONING_PLACEHOLDER}</think>"
    for message in normalized:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        message["content"] = _THINK_BLOCK.sub(replacement, content, count=1)
    return normalized


def load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line)


def action_of(turn: dict[str, Any]) -> dict[str, Any] | None:
    parsed = turn.get("parsed") or {}
    tool = parsed.get("tool")
    arguments = parsed.get("arguments")
    if not isinstance(tool, str) or not isinstance(arguments, dict):
        return None
    return {"tool": tool, "arguments": arguments}


def action_text(action: dict[str, Any]) -> str:
    payload = json.dumps(action, ensure_ascii=False, separators=(",", ":"))
    return f"<think>{SHARED_REASON}</think>\n{payload}"


def turn_error(turn: dict[str, Any]) -> str | None:
    for key in ("execution_error_type", "error_type", "recovered_from_error_type"):
        value = turn.get(key)
        if value:
            return str(value)
    output = turn.get("tool_output")
    if isinstance(output, dict):
        error = output.get("error")
        if isinstance(error, dict) and error.get("type"):
            return str(error["type"])
        if output.get("error_type"):
            return str(output["error_type"])
    return None


def classify(positive: "Occurrence", negative: "Occurrence") -> str:
    tools = {positive.action["tool"], negative.action["tool"]}
    if positive.feedback_recovery or negative.feedback_recovery:
        return "error_recovery"
    if "answer_from_context" in tools:
        return "final_output_slot"
    if tools & {"extreme_value_select"} or any(
        any(token in canonical_json(occ.action["arguments"]).lower() for token in ("order_by", "top_k", "tie"))
        for occ in (positive, negative)
    ):
        return "ranking_tie_order"
    if tools & {"join_tables", "group_aggregate", "condition_filter", "set_op"}:
        return "population_grain_join"
    if positive.turn_index == 0 and negative.turn_index == 0:
        return "normal_first_step"
    return "semantic_divergence"


@dataclass(frozen=True)
class Occurrence:
    artifact: str
    example_index: int
    sample_index: int
    turn_index: int
    db_id: str
    question: str
    model_input: list[dict[str, Any]]
    action: dict[str, Any]
    trajectory_correct: bool
    trajectory_legal: bool
    feedback_recovery: bool
    explicit_error: str | None

    @property
    def action_hash(self) -> str:
        return sha256_json(self.action)

    @property
    def state_hash(self) -> str:
        return sha256_json(self.model_input)


def collect_occurrences(paths: list[Path]) -> list[Occurrence]:
    result: list[Occurrence] = []
    for path in paths:
        for record in load_jsonl(path):
            for sample in record.get("samples") or []:
                for turn in sample.get("turns") or []:
                    action = action_of(turn)
                    state = turn.get("model_input")
                    if action is None or not isinstance(state, list) or not state:
                        continue
                    result.append(
                        Occurrence(
                            artifact=str(path),
                            example_index=int(record["example_index"]),
                            sample_index=int(sample.get("sample_index") or 0),
                            turn_index=int(turn.get("turn_index") or 0),
                            db_id=str(record.get("db_id") or ""),
                            question=str(record.get("question") or ""),
                            model_input=state,
                            action=action,
                            trajectory_correct=bool(sample.get("correct")),
                            trajectory_legal=bool(sample.get("legal")),
                            feedback_recovery=bool(turn.get("feedback_recovery")),
                            explicit_error=turn_error(turn),
                        )
                    )
    return result


def provenance(occurrence: Occurrence) -> dict[str, Any]:
    return {
        "artifact": occurrence.artifact,
        "example_index": occurrence.example_index,
        "sample_index": occurrence.sample_index,
        "turn_index": occurrence.turn_index,
        "trajectory_correct": occurrence.trajectory_correct,
        "trajectory_legal": occurrence.trajectory_legal,
        "feedback_recovery": occurrence.feedback_recovery,
        "explicit_error": occurrence.explicit_error,
    }


def build_candidates(occurrences: list[Occurrence]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str], list[Occurrence]] = defaultdict(list)
    for occurrence in occurrences:
        groups[(occurrence.example_index, occurrence.state_hash)].append(occurrence)

    candidates = []
    for (example_index, state_hash), group in groups.items():
        positives = [item for item in group if item.trajectory_correct and item.trajectory_legal]
        negatives = [item for item in group if not item.trajectory_correct]
        if not positives or not negatives:
            continue
        positives.sort(key=lambda item: (item.turn_index, item.sample_index, item.artifact))
        negatives.sort(
            key=lambda item: (
                item.explicit_error is None,
                item.turn_index,
                item.sample_index,
                item.artifact,
            )
        )
        chosen: tuple[Occurrence, Occurrence] | None = None
        for positive in positives:
            for negative in negatives:
                if positive.action_hash != negative.action_hash:
                    chosen = positive, negative
                    break
            if chosen:
                break
        if chosen is None:
            continue
        positive, negative = chosen
        category = classify(positive, negative)
        candidates.append(
            {
                "schema_version": "fixed-prefix-action-pair-v1",
                "question_id": str(example_index),
                "example_index": example_index,
                "db_id": positive.db_id,
                "question": positive.question,
                "state": positive.model_input,
                "state_sha256": state_hash,
                "positive_action": positive.action,
                "negative_action": negative.action,
                "positive_scoring_response": action_text(positive.action),
                "negative_scoring_response": action_text(negative.action),
                "shared_reason": SHARED_REASON,
                "category": category,
                "positive_verified_by": {
                    "real_harness_execution": True,
                    "successful_suffix": True,
                    "source": provenance(positive),
                },
                "negative_observed_from": provenance(negative),
                "pair_sha256": sha256_json(
                    {
                        "state": state_hash,
                        "positive": positive.action,
                        "negative": negative.action,
                    }
                ),
            }
        )
    return candidates


def select_candidates(
    candidates: list[dict[str, Any]],
    quotas: dict[str, int],
    *,
    max_per_question: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    def order_key(row: dict[str, Any]) -> str:
        return hashlib.sha256(f"{seed}:{row['pair_sha256']}".encode()).hexdigest()

    ordered = sorted(candidates, key=order_key)
    selected: list[dict[str, Any]] = []
    per_question: dict[str, int] = defaultdict(int)
    selected_hashes: set[str] = set()
    for category, quota in quotas.items():
        for row in ordered:
            if sum(item["category"] == category for item in selected) >= quota:
                break
            if row["category"] != category or row["pair_sha256"] in selected_hashes:
                continue
            if per_question[row["question_id"]] >= max_per_question:
                continue
            selected.append(row)
            selected_hashes.add(row["pair_sha256"])
            per_question[row["question_id"]] += 1
    requested_total = sum(quotas.values())
    for row in ordered:
        if len(selected) >= requested_total:
            break
        if row["pair_sha256"] in selected_hashes:
            continue
        if per_question[row["question_id"]] >= max_per_question:
            continue
        selected.append(row)
        selected_hashes.add(row["pair_sha256"])
        per_question[row["question_id"]] += 1
    shortfalls = {
        category: max(0, quota - sum(row["category"] == category for row in selected))
        for category, quota in quotas.items()
    }
    return selected, shortfalls


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--quota-json", type=Path)
    parser.add_argument("--max-per-question", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--require-target", action="store_true")
    args = parser.parse_args()

    quotas = DEFAULT_QUOTAS
    if args.quota_json:
        quotas = {key: int(value) for key, value in json.loads(args.quota_json.read_text()).items()}
    occurrences = collect_occurrences(args.input)
    candidates = build_candidates(occurrences)
    selected, shortfalls = select_candidates(
        candidates,
        quotas,
        max_per_question=args.max_per_question,
        seed=args.seed,
    )
    write_jsonl(args.output, selected)
    manifest_path = args.manifest or args.output.with_suffix(".manifest.json")
    manifest = {
        "schema_version": "fixed-prefix-dataset-manifest-v1",
        "inputs": [str(path) for path in args.input],
        "input_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in args.input
        },
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "occurrences": len(occurrences),
        "candidate_pairs": len(candidates),
        "selected_pairs": len(selected),
        "requested_pairs": sum(quotas.values()),
        "quotas": quotas,
        "selected_by_category": {
            category: sum(row["category"] == category for row in selected)
            for category in quotas
        },
        "quota_shortfalls": shortfalls,
        "max_per_question": args.max_per_question,
        "seed": args.seed,
        "leakage_boundary": "dev artifacts are diagnostic-only and must never be used for training",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if args.require_target and (len(selected) < sum(quotas.values()) or any(shortfalls.values())):
        raise SystemExit("verified same-prefix coverage is below the requested target")


if __name__ == "__main__":
    main()
