from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from rl.frameworks.trl.fixed_rollout_pool import (
    FixedPoolRolloutCollector,
    deserialize_episode,
    serialize_episode,
    write_rows_atomic,
)
from rl.frameworks.trl.transition_batch import PolicyEpisode, PolicyTurn
from rl.runtime.rollout_scoring import RolloutSample


def episode(example_index: int, sample_index: int) -> PolicyEpisode:
    turn = PolicyTurn((1, 2), (3,), (-0.2,))
    sample = RolloutSample(
        reward=1.0,
        correct=True,
        failure_type=None,
        turns=[([1, 2], [3])],
        audit_record={
            "trajectory_id": f"t-{sample_index}",
            "example_index": example_index,
            "sample_index": sample_index,
        },
    )
    return PolicyEpisode(sample=sample, policy_turns=[turn])


def test_round_trip_and_offline_group(tmp_path) -> None:
    rows = [
        serialize_episode(
            episode(9, sample_index),
            sequence=sample_index,
            environment={"example_index": 9},
        )
        for sample_index in range(2)
    ]
    path = tmp_path / "pool.jsonl"
    write_rows_atomic(path, rows)

    restored = deserialize_episode(json.loads(path.read_text().splitlines()[0]))
    assert restored.policy_turns[0].sampling_logprobs == (-0.2,)

    collector = FixedPoolRolloutCollector(path, expected_group_size=2)
    inputs = [{"environment": {"example_index": 9}}] * 2
    result = collector.collect(inputs)
    assert [item.sample.audit_record["sample_index"] for item in result] == [0, 1]
