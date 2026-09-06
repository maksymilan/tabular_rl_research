from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from rl.frameworks.trl.transition_grpo import TransitionGRPOTrainer  # noqa: E402


class PieceTokenizer:
    def decode(self, token_ids, **kwargs):
        pieces = {
            1: "<think>",
            2: "reason",
            3: "</think>",
            4: "\n",
            5: '{"tool":',
            6: '"plan"}',
        }
        return "".join(pieces[int(token_id)] for token_id in token_ids)


def test_span_balance_weights_are_fractional_and_nonzero() -> None:
    trainer = object.__new__(TransitionGRPOTrainer)
    trainer.span_balance_alpha = 0.5
    trainer.processing_class = PieceTokenizer()

    weights = trainer._span_balanced_mask([1, 2, 3, 4, 5, 6])
    tensor = torch.tensor(weights, dtype=torch.float32)

    assert tensor.sum().item() == 1.0
    assert torch.all(tensor > 0)
    assert tensor.dtype.is_floating_point
