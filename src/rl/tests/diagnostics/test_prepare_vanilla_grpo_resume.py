from __future__ import annotations

import json
from pathlib import Path

import pytest

from rl.scenarios.diagnostics.prepare_vanilla_grpo_resume import (
    REQUIRED_CHECKPOINT_FILES,
    prepare_resume,
)


def _tasks(path: Path, count: int = 6) -> list[int]:
    indices = list(range(100, 100 + count))
    path.write_text(
        "".join(json.dumps({"example_index": index}) + "\n" for index in indices)
    )
    return indices


def _checkpoint(root: Path, step: int, *, complete: bool = True, max_steps: int = 6) -> Path:
    path = root / f"checkpoint-{step}"
    path.mkdir()
    names = REQUIRED_CHECKPOINT_FILES if complete else REQUIRED_CHECKPOINT_FILES[:3]
    for name in names:
        if name == "trainer_state.json":
            (path / name).write_text(json.dumps({"global_step": step, "max_steps": max_steps}))
        else:
            (path / name).write_bytes(b"x")
    return path


def _rollouts(path: Path, indices: list[int], *, steps: int, prompts: int = 2, k: int = 2) -> None:
    rows = []
    for step in range(steps):
        chosen = indices[(step * prompts) % len(indices) : (step * prompts) % len(indices) + prompts]
        if len(chosen) < prompts:
            chosen += indices[: prompts - len(chosen)]
        for index in chosen:
            for _ in range(k):
                rows.append({"policy_global_step": step, "example_index": index})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_resume_keeps_latest_complete_checkpoint_and_trims_uncommitted_rollouts(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks.jsonl"
    indices = _tasks(tasks)
    _checkpoint(tmp_path, 2)
    retained = _checkpoint(tmp_path, 4)
    incomplete = _checkpoint(tmp_path, 6, complete=False)
    _rollouts(tmp_path / "rollouts.jsonl", indices, steps=5)

    result = prepare_resume(
        train_output=tmp_path,
        tasks=tasks,
        expected_records=6,
        optimizer_steps=6,
        prompts_per_update=2,
        group_size=2,
        save_steps=2,
    )

    assert result["status"] == "resume_ready"
    assert result["checkpoint"]["global_step"] == 4
    assert Path(result["checkpoint"]["path"]) == retained.resolve()
    assert result["rollouts"]["original_rows"] == 20
    assert result["rollouts"]["retained_rows"] == 16
    assert result["rollouts"]["removed_uncommitted_rows"] == 4
    assert len(list(tmp_path.glob("checkpoint-*"))) == 1
    assert not incomplete.exists()
    assert len(list((tmp_path / "resume_quarantine").iterdir())) == 2


def test_resume_rejects_final_step_checkpoint_without_final_output(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    indices = _tasks(tasks)
    _checkpoint(tmp_path, 6)
    _rollouts(tmp_path / "rollouts.jsonl", indices, steps=6)
    with pytest.raises(ValueError, match="final optimizer step"):
        prepare_resume(
            train_output=tmp_path,
            tasks=tasks,
            expected_records=6,
            optimizer_steps=6,
            prompts_per_update=2,
            group_size=2,
            save_steps=2,
        )


def test_resume_retains_step5_for_stable_gate_reverification(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    indices = _tasks(tasks, count=8)
    gate = _checkpoint(tmp_path, 5, max_steps=8)
    latest = _checkpoint(tmp_path, 6, max_steps=8)
    _rollouts(tmp_path / "rollouts.jsonl", indices, steps=7)

    result = prepare_resume(
        train_output=tmp_path,
        tasks=tasks,
        expected_records=8,
        optimizer_steps=8,
        prompts_per_update=2,
        group_size=2,
        save_steps=1,
        retain_checkpoint_steps=[5],
    )

    assert Path(result["checkpoint"]["path"]) == latest.resolve()
    assert result["retained_audit_checkpoints"] == [str(gate.resolve())]
    assert sorted(path.name for path in tmp_path.glob("checkpoint-*")) == [
        "checkpoint-5",
        "checkpoint-6",
    ]


def test_resume_rejects_rollout_prefix_behind_or_wrong_step(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    indices = _tasks(tasks)
    _checkpoint(tmp_path, 4)
    _rollouts(tmp_path / "rollouts.jsonl", indices, steps=3)
    with pytest.raises(ValueError, match="behind checkpoint"):
        prepare_resume(
            train_output=tmp_path,
            tasks=tasks,
            expected_records=6,
            optimizer_steps=6,
            prompts_per_update=2,
            group_size=2,
            save_steps=2,
        )

    # The failed attempt did not quarantine the only complete checkpoint.
    _rollouts(tmp_path / "rollouts.jsonl", indices, steps=4)
    rows = [json.loads(line) for line in (tmp_path / "rollouts.jsonl").read_text().splitlines()]
    rows[4]["policy_global_step"] = 0
    (tmp_path / "rollouts.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    with pytest.raises(ValueError, match="wrong policy_global_step"):
        prepare_resume(
            train_output=tmp_path,
            tasks=tasks,
            expected_records=6,
            optimizer_steps=6,
            prompts_per_update=2,
            group_size=2,
            save_steps=2,
        )
