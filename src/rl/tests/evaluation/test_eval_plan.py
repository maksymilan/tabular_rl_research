from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.evaluation.runners.eval_plan import make_plan, partition_rows, validate_gpu_ids, write_plan
from rl.evaluation.runners.make_eval_shards import main as make_eval_shards_main


def test_partition_is_deterministic_and_balanced() -> None:
    rows = [{"example_index": i, "question": str(i)} for i in range(7)]
    shards = partition_rows(rows, 3)
    assert [[row["example_index"] for row in shard] for shard in shards] == [[0, 3, 6], [1, 4], [2, 5]]
    contiguous = partition_rows(rows, 3, policy="contiguous")
    assert [[row["example_index"] for row in shard] for shard in contiguous] == [[0, 1, 2], [3, 4], [5, 6]]


def test_plan_parameterizes_split_checkpoint_and_gpu_count(tmp_path: Path) -> None:
    dataset = tmp_path / "dev.jsonl"
    dataset.write_text("".join(json.dumps({"example_index": i}) + "\n" for i in range(5)))
    plan, shards = make_plan(
        dataset,
        split="bird_dev",
        checkpoint=tmp_path / "checkpoint-6380",
        gpu_ids=[0, 2],
        allowed_gpu_ids=[0, 1, 2, 3],
    )
    assert plan.split == "bird_dev"
    assert plan.records == 5
    assert plan.gpu_ids == (0, 2)
    assert plan.shard_sizes == (3, 2)
    write_plan(plan, shards, tmp_path / "plan")
    assert (tmp_path / "plan/evaluation_plan.json").is_file()
    assert (tmp_path / "plan/shard_00.jsonl").read_text().count("\n") == 3


@pytest.mark.parametrize("ids", ([0, 0], [-1], []))
def test_gpu_validation_rejects_invalid_lists(ids: list[int]) -> None:
    with pytest.raises(ValueError):
        validate_gpu_ids(ids)


def test_gpu_allowlist_is_enforced() -> None:
    with pytest.raises(ValueError, match="allowlist"):
        validate_gpu_ids([4], allowed=[0, 1, 2, 3])


def test_generic_sharder_supports_arbitrary_gpu_count(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text("".join(json.dumps({"example_index": i}) + "\n" for i in range(5)))
    output = tmp_path / "shards"
    monkeypatch.setattr(
        "sys.argv",
        ["make_eval_shards", "--examples", str(dataset), "--output-dir", str(output), "--shard-count", "3"],
    )
    assert make_eval_shards_main() == 0
    assert [p.read_text().count("\n") for p in sorted(output.glob("shard_*.jsonl"))] == [2, 2, 1]
