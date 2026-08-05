from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from frameworks.trl.tool_loss_mask import (
    ToolMaskUnavailable,
    tool_token_loss_mask,
)


class PieceTokenizer:
    def __init__(self, pieces: dict[int, str]):
        self.pieces = pieces

    def decode(self, token_ids, **kwargs):
        return "".join(self.pieces[token_id] for token_id in token_ids)


def test_tool_mask_keeps_only_raw_json_suffix() -> None:
    tokenizer = PieceTokenizer(
        {
            1: "<think>",
            2: "reason",
            3: "</think>",
            4: "\n",
            5: '{"tool":',
            6: '"plan"}',
        }
    )
    assert tool_token_loss_mask(tokenizer, [1, 2, 3, 4, 5, 6]) == (
        0,
        0,
        0,
        0,
        1,
        1,
    )


def test_tool_mask_includes_token_that_crosses_json_boundary() -> None:
    tokenizer = PieceTokenizer(
        {
            1: "<think>x</think>",
            2: '\n{"tool"',
            3: ":1}",
        }
    )
    assert tool_token_loss_mask(tokenizer, [1, 2, 3]) == (0, 1, 1)


def test_tool_mask_rejects_missing_carrier_boundary() -> None:
    tokenizer = PieceTokenizer({1: '{"tool":"plan"}'})
    with pytest.raises(ToolMaskUnavailable, match="closing"):
        tool_token_loss_mask(tokenizer, [1])


def test_tool_mask_marks_truncated_think_as_unavailable() -> None:
    tokenizer = PieceTokenizer({1: "<think>", 2: "truncated reasoning"})
    with pytest.raises(ToolMaskUnavailable, match="closing"):
        tool_token_loss_mask(tokenizer, [1, 2])
