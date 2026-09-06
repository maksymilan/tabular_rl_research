from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.diagnostics.replay import (
    group_rollouts,
    local_error_statuses,
    read_rows,
    stable_action_digest,
)


def test_replay_reader_and_grouping_are_strict_and_stable(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.jsonl"
    rows = [
        {"policy_global_step": 2, "example_index": 4, "trajectory_id": "b", "turns": []},
        {"policy_global_step": 2, "example_index": 4, "trajectory_id": "a", "turns": []},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    loaded = read_rows(path)
    grouped = group_rollouts(loaded)
    assert [row["trajectory_id"] for row in grouped[(2, 4)]] == ["a", "b"]
    assert stable_action_digest({"b": 2, "a": 1}) == stable_action_digest({"a": 1, "b": 2})


def test_local_error_statuses_detect_timeout_and_repeated_no_progress() -> None:
    row = {
        "trajectory_id": "t",
        "turns": [
            {"error_event": {"error_type": "timeout_error"}},
            {
                "parsed": {"tool": "project", "arguments": {"table": "t"}},
                "error_event": {
                    "error_type": "execution_error",
                    "state_before_hash": "a",
                    "state_after_hash": "a",
                },
            },
            {
                "parsed": {"tool": "project", "arguments": {"table": "t"}},
                "error_event": {
                    "error_type": "execution_error",
                    "state_before_hash": "a",
                    "state_after_hash": "a",
                },
            },
        ],
    }
    statuses = local_error_statuses(row)
    assert statuses[0]["kind"] == "infrastructure_timeout"
    assert statuses[1]["kind"] == "execution_error"
    assert statuses[2]["kind"] == "no_progress_repeat"


def test_local_error_statuses_reject_non_list_turns() -> None:
    with pytest.raises(ValueError, match="non-list turns"):
        local_error_statuses({"trajectory_id": "t", "turns": {}})
