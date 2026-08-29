from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_checkpoint_relalg_qwen3_projection_v3 import build  # noqa: E402


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(record_id: str, reasoning: str, tool: str = "answer") -> dict:
    return {
        "system": "system",
        "conversations": [
            {"from": "human", "value": "current causal context"},
            {
                "from": "gpt",
                "value": (
                    f"<think>\n{reasoning}\n</think>\n\n"
                    f'{{"tool":"{tool}","arguments":{{}}}}'
                ),
            },
        ],
        "metadata": {"record_id": record_id},
    }


def _audit(path: Path, cutoff: int, records: int, rejected: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "records": records,
                "cutoff_len": cutoff,
                "filter_policy": "full-prefix",
                "model": "model",
                "template": "qwen3",
                "details": [
                    {
                        "record_id": record_id,
                        "target_status": "complete",
                        "processor_target_complete": True,
                        "current_source_tokens_original": 100,
                        "current_source_tokens_kept": 100,
                        "history_pairs_original": 1,
                        "first_pair_complete": False,
                    }
                    for record_id in rejected
                ],
            }
        )
    )


def test_builds_core_and_long_supplement_without_dropping_schema_mismatch(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.jsonl"
    index = tmp_path / "index.jsonl"
    audit8 = tmp_path / "8k.json"
    audit16 = tmp_path / "16k.json"
    output = tmp_path / "out"
    rows = [
        _row("r1", "The grounded table is ready."),
        _row("r2", "Use the verified long causal prefix."),
        _row("r3", "I remember this exact question and its gold SQL."),
        _row("r4", "This trajectory exceeds the 16K limit."),
    ]
    indexes = [
        {
            "record_id": item,
            "source_episode_id": f"e{number}",
            "source_turn_index": 0,
            "tool_name": "answer",
            "strict_artifact_accuracy": number == 1,
            "schema_match": number in {1, 2},
        }
        for number, item in enumerate(("r1", "r2", "r3", "r4"), 1)
    ]
    _write_jsonl(canonical, rows)
    _write_jsonl(index, indexes)
    _audit(audit8, 8192, 4, ["r2", "r3", "r4"])
    _audit(audit16, 16384, 4, ["r4"])

    manifest = build(canonical, index, audit8, audit16, output, "dataset_v3")

    combined = [json.loads(line) for line in (output / "dataset_v3.jsonl").read_text().splitlines()]
    derived = [json.loads(line) for line in (output / "dataset_v3_index.jsonl").read_text().splitlines()]
    assert len(combined) == 2
    assert [item["record_id"] for item in derived] == ["r1", "r2"]
    assert [item["context_bucket"] for item in derived] == [
        "core_8192",
        "supplement_8193_16384",
    ]
    assert derived[1]["artifact_tier"] == "value_correct_schema_exact"
    assert derived[1]["recommended_sample_weight"] == 0.75
    assert manifest["quality_filter"]["excluded_records"] == 1
    assert manifest["selection"]["episodes_with_answer_target"] == 2
    assert manifest["outputs"]["combined"]["records"] == 2


def test_rejects_adjacent_exact_successful_action_repeat(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical.jsonl"
    index = tmp_path / "index.jsonl"
    audit8 = tmp_path / "8k.json"
    audit16 = tmp_path / "16k.json"
    output = tmp_path / "out"
    row = _row("repeat", "Try the same action again.", tool="scalar_compute")
    row["conversations"] = [
        {"from": "human", "value": "context"},
        {"from": "gpt", "value": '{"tool":"scalar_compute","arguments":{}}'},
        {"from": "human", "value": "success feedback"},
        row["conversations"][-1],
    ]
    _write_jsonl(canonical, [row])
    _write_jsonl(
        index,
        [
            {
                "record_id": "repeat",
                "source_episode_id": "episode",
                "source_turn_index": 1,
                "tool_name": "scalar_compute",
                "strict_artifact_accuracy": True,
                "schema_match": True,
            }
        ],
    )
    _audit(audit8, 8192, 1, [])
    _audit(audit16, 16384, 1, [])

    manifest = build(canonical, index, audit8, audit16, output, "dataset_v3")

    assert manifest["outputs"]["combined"]["records"] == 0
    assert manifest["quality_filter"]["reason_hist"] == {
        "adjacent_exact_successful_action_repeat": 1
    }
