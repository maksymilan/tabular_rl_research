#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import torch
except ImportError:  # The lightweight local audit environment does not install GPU dependencies.
    torch = None

RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_DIR))

if torch is not None:
    from frameworks.accelerate.turn_logprobs import (  # noqa: E402
        build_turn_batch,
        response_logprobs_batched,
        response_token_logprobs_batched,
    )
    from process_objective import sampled_forward_kl, sampled_turn_forward_kl  # noqa: E402


class TinyTokenizer:
    pad_token_id = 0


class TinyCausalLM(torch.nn.Module if torch is not None else object):
    """A next-token table sufficient to test exact response alignment and gradients."""

    def __init__(self, vocab_size: int = 8):
        super().__init__()
        values = torch.arange(vocab_size * vocab_size, dtype=torch.float32)
        self.logit_table = torch.nn.Parameter(values.reshape(vocab_size, vocab_size) / 10.0)

    def forward(self, input_ids, attention_mask, use_cache=False, logits_to_keep=None):
        del attention_mask, use_cache
        logits = self.logit_table[input_ids]
        if logits_to_keep is not None:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)


@unittest.skipIf(torch is None, "torch is not installed in the local audit environment")
class AccelerateTurnLogProbTests(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.tokenizer = TinyTokenizer()
        self.model = TinyCausalLM()

    def test_batch_targets_align_different_response_lengths(self):
        turns = [([1, 2], [3, 4]), ([5], [6])]
        input_ids, attention_mask, targets = build_turn_batch(
            self.tokenizer,
            turns,
            self.device,
        )
        self.assertEqual(input_ids.tolist(), [[1, 2, 3], [0, 0, 5]])
        self.assertEqual(attention_mask.tolist(), [[1, 1, 1], [0, 0, 1]])
        self.assertEqual(targets.tolist(), [[3, 4], [-100, 6]])

    def test_turn_logprob_is_sum_of_exact_response_token_logprobs(self):
        turns = [([1, 2], [3, 4]), ([5], [6])]
        token_logps = response_token_logprobs_batched(
            self.model,
            self.tokenizer,
            turns,
            self.device,
        )
        turn_logps = response_logprobs_batched(
            self.model,
            self.tokenizer,
            turns,
            self.device,
        )
        expected_first = (
            self.model.logit_table[2].log_softmax(dim=-1)[3]
            + self.model.logit_table[3].log_softmax(dim=-1)[4]
        )
        expected_second = self.model.logit_table[5].log_softmax(dim=-1)[6]
        self.assertTrue(torch.allclose(turn_logps[0], expected_first))
        self.assertTrue(torch.allclose(turn_logps[1], expected_second))
        self.assertTrue(torch.allclose(turn_logps[0], token_logps[0].sum()))
        self.assertEqual([len(values) for values in token_logps], [2, 1])

    def test_complete_turn_logprob_retains_gradient(self):
        loss = -response_logprobs_batched(
            self.model,
            self.tokenizer,
            [([1, 2], [3, 4])],
            self.device,
        )[0]
        loss.backward()
        self.assertIsNotNone(self.model.logit_table.grad)
        self.assertGreater(float(self.model.logit_table.grad.abs().sum()), 0.0)

    def test_kl_is_computed_per_token_then_summed(self):
        current = torch.tensor([-1.0, -3.0])
        reference = torch.tensor([-2.0, -2.0])
        expected = sampled_forward_kl(current, reference).sum()
        actual = sampled_turn_forward_kl(current, reference)
        self.assertTrue(torch.allclose(actual, expected))
        self.assertGreater(float(actual), 0.0)
        self.assertEqual(
            float(sampled_turn_forward_kl(reference, reference)),
            0.0,
        )

    def test_k3_estimator_is_finite_for_extreme_log_ratios(self):
        estimate = sampled_forward_kl(
            torch.tensor([-1000.0, 1000.0]),
            torch.zeros(2),
        )
        self.assertTrue(bool(torch.isfinite(estimate).all()))
        self.assertTrue(bool((estimate >= 0).all()))
        self.assertTrue(bool((estimate <= 10).all()))

    def test_turn_kl_rejects_misaligned_tokens(self):
        with self.assertRaisesRegex(ValueError, "must align"):
            sampled_turn_forward_kl(torch.zeros(2), torch.zeros(3))


if __name__ == "__main__":
    unittest.main()
