from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audit_checkpoint_relalg_qwen3_projection_v4_short import audit  # noqa: E402
from build_checkpoint_relalg_qwen3_projection_v3 import sha256  # noqa: E402
from build_checkpoint_relalg_qwen3_projection_v4_short import (  # noqa: E402
    POLICY_VERSION,
    build,
    parse_target,
    project_reasoning,
)


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(re.findall(r"\w+|[^\w\s]", text))))


class CallableWordTokenizer(WordTokenizer):
    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
    ) -> dict[str, list[list[int]]]:
        assert add_special_tokens is padding is truncation is False
        return {"input_ids": [self.encode(text) for text in texts]}


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(record_id: str, reasoning: str, action: str) -> dict:
    return {
        "system": "system",
        "conversations": [
            {"from": "human", "value": "current causal context"},
            {"from": "gpt", "value": f"<think>\n{reasoning}\n</think>\n\n{action}"},
        ],
        "metadata": {"record_id": record_id},
    }


def _source(tmp_path: Path, rows: list[dict], indexes: list[dict]) -> tuple[Path, Path, Path]:
    canonical = tmp_path / "source_canonical.jsonl"
    index = tmp_path / "source_index.jsonl"
    manifest = tmp_path / "source.manifest.json"
    _write_jsonl(canonical, rows)
    _write_jsonl(index, indexes)
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "checkpoint-relalg-qwen3-projection-v3-dataset-v1",
                "outputs": {
                    "combined": {
                        "canonical_sha256": sha256(canonical),
                        "index_sha256": sha256(index),
                    }
                },
            }
        )
    )
    return canonical, index, manifest


def test_projects_only_reasoning_and_audit_proves_source_substring(tmp_path: Path) -> None:
    action1 = '{"tool":"filter_rows","arguments":{"table":"people","conditions":{"op":"=","left":{"column":"city"},"right":{"value":"Paris"}}}}'
    action2 = '{"tool":"answer","arguments":{"table":"shape_004"}}'
    long_reasoning = (
        "We need to decide which population is relevant. "
        "Several alternatives are possible, but they do not match the observed relation. "
        "The active people relation already contains the exact city column. "
        "Filter people on city Paris now with filter_rows."
    )
    rows = [
        _row("r1", long_reasoning, action1),
        _row("r2", "The exact output is shape_004, so submit it as the answer.", action2),
    ]
    indexes = [
        {
            "record_id": "r1",
            "source_episode_id": "e1",
            "source_turn_index": 0,
            "tool_name": "filter_rows",
            "model_input_sha256": "a" * 64,
        },
        {
            "record_id": "r2",
            "source_episode_id": "e1",
            "source_turn_index": 1,
            "tool_name": "answer",
            "model_input_sha256": "b" * 64,
        },
    ]
    canonical, index, source_manifest = _source(tmp_path, rows, indexes)
    output = tmp_path / "out"
    tokenizer = WordTokenizer()

    manifest = build(
        canonical,
        index,
        source_manifest,
        output,
        "short_v4",
        tokenizer,
        tokenizer_identity="word-tokenizer",
        soft_max_tokens=18,
        hard_max_tokens=32,
        max_repeat_ratio=0.20,
    )

    selected = [json.loads(line) for line in (output / "short_v4_canonical.jsonl").read_text().splitlines()]
    selected_index = [json.loads(line) for line in (output / "short_v4_index.jsonl").read_text().splitlines()]
    projected, projected_action, _ = parse_target(selected[0], item_id="r1")
    source_reasoning, source_action, _ = parse_target(rows[0], item_id="r1")
    assert projected in source_reasoning
    assert projected_action == source_action == action1
    assert selected[0]["system"] == rows[0]["system"]
    assert selected[0]["conversations"][:-1] == rows[0]["conversations"][:-1]
    assert selected_index[0]["model_input_sha256"] == "a" * 64
    assert selected_index[0]["reasoning_projection_policy_version"] == POLICY_VERSION
    assert selected_index[0]["reasoning_projection"]["was_shortened"] is True
    assert manifest["selection"]["selected_records"] == 2
    assert manifest["selection"]["episodes_with_answer_target"] == 1

    result = audit(output, "short_v4", tokenizer)
    assert result["passed"] is True
    assert result["records"] == 2
    assert result["shortened_records"] == 1


def test_long_reasoning_without_grounded_suffix_is_excluded(tmp_path: Path) -> None:
    action = '{"tool":"join","arguments":{"left":"left_001","right":"right_001","type":"inner","on":[]}}'
    rows = [
        _row(
            "ungrounded",
            (
                "Join left_001 and right_001 after verifying their keys. "
                "This sentence consumes the remaining soft budget without naming the decision. "
                "Continue considering unrelated possibilities and postpone the choice."
            ),
            action,
        )
    ]
    indexes = [
        {
            "record_id": "ungrounded",
            "source_episode_id": "e1",
            "source_turn_index": 0,
            "tool_name": "join",
        }
    ]
    canonical, index, source_manifest = _source(tmp_path, rows, indexes)
    output = tmp_path / "out"

    manifest = build(
        canonical,
        index,
        source_manifest,
        output,
        "short_v4",
        WordTokenizer(),
        tokenizer_identity="word-tokenizer",
        soft_max_tokens=8,
        hard_max_tokens=12,
        max_repeat_ratio=0.20,
    )

    assert manifest["selection"]["selected_records"] == 0
    assert manifest["selection"]["exclusion_reason_hist"] == {
        "no_grounded_nonrepetitive_suffix_within_hard_limit": 1
    }


def test_projection_collapses_repetitive_reasoning_to_one_grounded_source_sentence() -> None:
    repeated = "Submit shape_004 as the answer. " * 20
    projected, reason = project_reasoning(
        repeated.strip(),
        {"tool": "answer", "arguments": {"table": "shape_004"}},
        WordTokenizer(),
        soft_max_tokens=16,
        hard_max_tokens=32,
        max_repeat_ratio=0.05,
    )
    assert projected == "Submit shape_004 as the answer."
    assert reason["was_shortened"] is True
    assert reason["contiguous_source_substring"] is True


def test_callable_tokenizer_handles_no_suffix_within_hard_limit() -> None:
    projected, reason = project_reasoning(
        " ".join(["unbounded"] * 700),
        {"tool": "answer", "arguments": {"table": "shape_004"}},
        CallableWordTokenizer(),
        soft_max_tokens=16,
        hard_max_tokens=32,
    )
    assert projected is None
    assert reason["reason"] == "no_grounded_nonrepetitive_suffix_within_hard_limit"
