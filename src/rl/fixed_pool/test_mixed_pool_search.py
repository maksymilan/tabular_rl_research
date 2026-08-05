import json
from pathlib import Path

from prepare_mixed_pool_search import main as _prepare_main  # noqa: F401
from summarize_mixed_outcome_pool import task_id


def test_task_id_prefers_example_id() -> None:
    assert task_id({"example_id": "bird_train_1", "trajectory_id": "other"}) == "bird_train_1"


def test_mixed_outcome_shape(tmp_path: Path) -> None:
    group = [
        {"sample": {"correct": value, "audit_record": {"sample_index": index}}}
        for index, value in enumerate((True, False, False, True))
    ]
    path = tmp_path / "bird_train_1.json"
    path.write_text(json.dumps(group))
    loaded = json.loads(path.read_text())
    assert len(loaded) == 4
    assert sum(bool(row["sample"]["correct"]) for row in loaded) == 2
