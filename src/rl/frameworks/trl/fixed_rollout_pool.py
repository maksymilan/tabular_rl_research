#!/usr/bin/env python3
"""Lossless serialization and replay of immutable policy rollout episodes."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from rl.runtime.rollout_scoring import RolloutSample

from rl.frameworks.trl.transition_batch import PolicyEpisode, PolicyTurn


SCHEMA_VERSION = "table-agent-fixed-policy-episode-v1"


def serialize_episode(
    episode: PolicyEpisode,
    *,
    sequence: int,
    environment: dict[str, Any],
) -> dict[str, Any]:
    episode.validate()
    return {
        "schema_version": SCHEMA_VERSION,
        "sequence": sequence,
        "environment": environment,
        "sample": {
            "reward": episode.sample.reward,
            "correct": episode.sample.correct,
            "failure_type": episode.sample.failure_type,
            "audit_record": episode.sample.audit_record,
            "step_rewards": episode.sample.step_rewards,
            "process_update": episode.sample.process_update,
        },
        "policy_turns": [
            {
                "prompt_ids": list(turn.prompt_ids),
                "response_ids": list(turn.response_ids),
                "sampling_logprobs": list(turn.sampling_logprobs),
            }
            for turn in episode.policy_turns
        ],
    }


def deserialize_episode(row: dict[str, Any]) -> PolicyEpisode:
    if row.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported fixed-pool schema: {row.get('schema_version')}")
    turns = [
        PolicyTurn(
            prompt_ids=tuple(int(value) for value in raw["prompt_ids"]),
            response_ids=tuple(int(value) for value in raw["response_ids"]),
            sampling_logprobs=tuple(float(value) for value in raw["sampling_logprobs"]),
        )
        for raw in row["policy_turns"]
    ]
    raw_sample = row["sample"]
    scored_turns = [
        (list(turn.prompt_ids), list(turn.response_ids)) for turn in turns
    ]
    sample = RolloutSample(
        reward=float(raw_sample["reward"]),
        correct=bool(raw_sample["correct"]),
        failure_type=raw_sample.get("failure_type"),
        turns=scored_turns,
        audit_record=raw_sample["audit_record"],
        step_rewards=(
            [float(value) for value in raw_sample["step_rewards"]]
            if raw_sample.get("step_rewards") is not None
            else None
        ),
        process_update=bool(raw_sample.get("process_update", True)),
    )
    episode = PolicyEpisode(sample=sample, policy_turns=turns)
    episode.validate()
    return episode


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    sequences = [int(row["sequence"]) for row in rows]
    if sequences != list(range(len(rows))):
        raise ValueError("fixed-pool sequences must be contiguous and ordered from zero")
    return rows


def write_rows_atomic(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".next")
    with temporary.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        target.flush()
    temporary.replace(path)


class FixedPoolRolloutCollector:
    """Return frozen episodes keyed by question instead of sampling a live policy."""

    requires_policy_sync = False
    is_offline = True

    def __init__(self, path: Path, *, expected_group_size: int):
        if expected_group_size < 2:
            raise ValueError("expected_group_size must be at least two")
        self.path = path
        self.expected_group_size = expected_group_size
        grouped: dict[int, list[tuple[int, PolicyEpisode]]] = defaultdict(list)
        for row in load_rows(path):
            episode = deserialize_episode(row)
            example_index = int(episode.sample.audit_record["example_index"])
            sample_index = int(episode.sample.audit_record["sample_index"])
            grouped[example_index].append((sample_index, episode))
        self.episodes = {}
        for example_index, values in grouped.items():
            values.sort(key=lambda value: value[0])
            if [index for index, _ in values] != list(range(expected_group_size)):
                raise ValueError(
                    f"example {example_index} does not contain exact sample indices "
                    f"0..{expected_group_size - 1}"
                )
            self.episodes[example_index] = [episode for _, episode in values]
        self.consumed: set[int] = set()

    def collect(self, inputs: list[dict[str, Any]], trainer=None) -> list[PolicyEpisode]:
        counts = Counter(int(item["environment"]["example_index"]) for item in inputs)
        if any(count != self.expected_group_size for count in counts.values()):
            raise ValueError(
                "each offline optimizer batch must contain exactly one frozen K group per question"
            )
        result = []
        offsets: Counter[int] = Counter()
        for item in inputs:
            example_index = int(item["environment"]["example_index"])
            if example_index in self.consumed:
                raise RuntimeError(f"fixed-pool question was requested twice: {example_index}")
            if example_index not in self.episodes:
                raise KeyError(f"fixed pool has no example_index={example_index}")
            result.append(self.episodes[example_index][offsets[example_index]])
            offsets[example_index] += 1
        self.consumed.update(counts)
        return result
