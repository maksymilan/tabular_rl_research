from __future__ import annotations

import pytest

from src.rl.distillation.masks import (
    PolicyTokenMaskUnavailable,
    policy_token_weights,
)


class PieceTokenizer:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, token_ids, **_kwargs):
        return "".join(self.pieces[token_id] for token_id in token_ids)


def test_first_version_masks_think_and_weights_entire_json():
    tokenizer = PieceTokenizer(
        {
            1: "<think>",
            2: "reason",
            3: "</think>\n",
            4: '{"tool":"project",',
            5: '"arguments":{"input":"step_1"}}',
        }
    )
    weights = policy_token_weights(tokenizer, [1, 2, 3, 4, 5])
    assert weights == (0.0, 0.0, 0.0, 1.0, 1.0)


def test_optional_weak_think_weight_is_explicit():
    tokenizer = PieceTokenizer(
        {1: "<think>", 2: "reason", 3: "</think>", 4: '{"tool":"x","arguments":{}}'}
    )
    weights = policy_token_weights(
        tokenizer,
        [1, 2, 3, 4],
        think_weight=0.25,
    )
    assert weights == (0.25, 0.25, 0.25, 1.0)


def test_decode_empty_special_token_is_not_mislabeled_as_json():
    tokenizer = PieceTokenizer(
        {
            1: "<think>x</think>",
            2: '{"tool":"x","arguments":{}}',
            3: "",
        }
    )
    assert policy_token_weights(tokenizer, [1, 2, 3]) == (0.0, 1.0, 0.0)


@pytest.mark.parametrize(
    "pieces,ids",
    [
        ({1: "{}"}, [1]),
        ({1: "<think>x</think> prose ", 2: "{}"}, [1, 2]),
        ({1: "<think>x</think>"}, [1]),
    ],
)
def test_invalid_carriers_fail_closed(pieces, ids):
    with pytest.raises(PolicyTokenMaskUnavailable):
        policy_token_weights(PieceTokenizer(pieces), ids)
