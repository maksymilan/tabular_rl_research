#!/usr/bin/env python3
"""Summarize a small paired greedy behavior cohort against frozen SFT2."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def sample(row: dict[str, Any]) -> dict[str, Any]:
    samples = row.get("samples") or []
    if len(samples) != 1:
        raise ValueError(f"expected one sample for example {row.get('example_index')}")
    return samples[0]


def action_sequence(row: dict[str, Any]) -> tuple[str, ...]:
    result = []
    for turn in sample(row).get("turns") or []:
        parsed = turn.get("parsed") or {}
        tool = parsed.get("tool")
        arguments = parsed.get("arguments")
        if tool is None or arguments is None:
            continue
        result.append(
            json.dumps(
                {"tool": tool, "arguments": arguments},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    return tuple(result)


def adjacent_repeats(actions: tuple[str, ...]) -> int:
    return sum(left == right for left, right in zip(actions, actions[1:]))


def exact_p(gains: int, regressions: int) -> float:
    discordant = gains + regressions
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(gains, regressions) + 1)
    ) / (2**discordant)
    return min(1.0, 2.0 * tail)


def arm_summary(
    rows: dict[int, dict[str, Any]], difficulties: dict[int, str]
) -> dict[str, Any]:
    values = list(rows.values())
    correct = sum(bool(sample(row)["correct"]) for row in values)
    legal = sum(bool(sample(row)["legal"]) for row in values)
    steps = [int(sample(row)["steps"]) for row in values]
    actions = {index: action_sequence(row) for index, row in rows.items()}
    return {
        "correct": correct,
        "total": len(values),
        "accuracy": correct / len(values),
        "legal": legal,
        "legal_rate": legal / len(values),
        "mean_steps": sum(steps) / len(steps),
        "difficulty": {
            difficulty: {
                "correct": sum(
                    bool(sample(rows[index])["correct"])
                    for index in rows
                    if difficulties[index] == difficulty
                ),
                "total": sum(
                    difficulties[index] == difficulty for index in rows
                ),
            }
            for difficulty in ("simple", "moderate", "challenging")
        },
        "adjacent_exact_repeat_calls": sum(
            adjacent_repeats(sequence) for sequence in actions.values()
        ),
        "questions_with_adjacent_exact_repeat": sum(
            adjacent_repeats(sequence) > 0 for sequence in actions.values()
        ),
        "action_count": sum(len(sequence) for sequence in actions.values()),
    }


def paired_summary(
    baseline: dict[int, dict[str, Any]], candidate: dict[int, dict[str, Any]]
) -> dict[str, Any]:
    gains = []
    regressions = []
    legal_gains = []
    legal_regressions = []
    action_changed = []
    first_action_changed = []
    for index in sorted(baseline):
        base_sample = sample(baseline[index])
        candidate_sample = sample(candidate[index])
        base_correct = bool(base_sample["correct"])
        candidate_correct = bool(candidate_sample["correct"])
        if candidate_correct and not base_correct:
            gains.append(index)
        elif base_correct and not candidate_correct:
            regressions.append(index)
        base_legal = bool(base_sample["legal"])
        candidate_legal = bool(candidate_sample["legal"])
        if candidate_legal and not base_legal:
            legal_gains.append(index)
        elif base_legal and not candidate_legal:
            legal_regressions.append(index)
        base_actions = action_sequence(baseline[index])
        candidate_actions = action_sequence(candidate[index])
        if base_actions != candidate_actions:
            action_changed.append(index)
        if (base_actions[:1] or candidate_actions[:1]) and base_actions[:1] != candidate_actions[:1]:
            first_action_changed.append(index)
    return {
        "gains": gains,
        "regressions": regressions,
        "net": len(gains) - len(regressions),
        "exact_p": exact_p(len(gains), len(regressions)),
        "legal_gains": legal_gains,
        "legal_regressions": legal_regressions,
        "legal_net": len(legal_gains) - len(legal_regressions),
        "legal_exact_p": exact_p(len(legal_gains), len(legal_regressions)),
        "action_sequence_changed": action_changed,
        "action_sequence_changed_count": len(action_changed),
        "first_action_changed": first_action_changed,
        "first_action_changed_count": len(first_action_changed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--indices", type=Path, required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    indices_payload = json.loads(args.indices.read_text())
    indices_values = (
        indices_payload["indices"]
        if isinstance(indices_payload, dict)
        else indices_payload
    )
    indices = [int(value) for value in indices_values]
    if len(indices) != len(set(indices)):
        raise SystemExit("indices contain duplicates")
    wanted = set(indices)
    examples = {
        int(row["example_index"]): row
        for row in load_jsonl(args.examples)
        if int(row["example_index"]) in wanted
    }
    difficulties = {
        index: str(examples[index]["metadata"]["difficulty"])
        for index in wanted
    }

    def selected(path: Path) -> dict[int, dict[str, Any]]:
        rows = {
            int(row["example_index"]): row
            for row in load_jsonl(path)
            if int(row["example_index"]) in wanted
        }
        if set(rows) != wanted:
            raise ValueError(f"{path} does not contain exact selected cohort")
        for row in rows.values():
            if row.get("protocol_version") != "version36":
                raise ValueError("protocol mismatch")
            if float(row.get("temperature")) != 0.0 or float(row.get("top_p")) != 1.0:
                raise ValueError("sampling mismatch")
            if row.get("denotation_comparison") != "bird-set":
                raise ValueError("denotation mismatch")
        return rows

    baseline = selected(args.baseline)
    candidates = {name: selected(Path(path)) for name, path in args.candidate}
    result = {
        "schema_version": "routed-coupled-behavior-smoke-summary-v1",
        "indices": indices,
        "difficulty_counts": dict(Counter(difficulties.values())),
        "baseline": arm_summary(baseline, difficulties),
        "candidates": {
            name: {
                **arm_summary(rows, difficulties),
                "vs_sft2": paired_summary(baseline, rows),
            }
            for name, rows in candidates.items()
        },
    }
    result["candidate_pairwise"] = {
        f"{right}_vs_{left}": paired_summary(candidates[left], candidates[right])
        for left, right in combinations(candidates, 2)
    }
    if set(candidates) == {"lr1e-6", "lr4e-6"}:
        result["lr4e-6_vs_lr1e-6"] = paired_summary(
            candidates["lr1e-6"], candidates["lr4e-6"]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
