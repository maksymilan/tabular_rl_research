from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from frameworks.trl.training_precision import (  # noqa: E402
    optimizer_moment_precision_audit,
    promote_trainable_parameters_to_fp32,
    require_adam_moments_fp32,
    require_trainable_parameters_fp32,
)


class TinyAdapter(torch.nn.Module):
    def __init__(self, *, dtype: torch.dtype):
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.ones(3, dtype=dtype), requires_grad=False)
        self.adapter = torch.nn.Parameter(torch.ones(3, dtype=dtype))

    def forward(self) -> torch.Tensor:
        return self.adapter.square().sum()


class TrainingPrecisionTest(unittest.TestCase):
    def test_promotes_only_trainable_parameters(self):
        model = TinyAdapter(dtype=torch.bfloat16)
        audit = promote_trainable_parameters_to_fp32(model)

        self.assertEqual(model.adapter.dtype, torch.float32)
        self.assertEqual(model.frozen.dtype, torch.bfloat16)
        self.assertEqual(audit["promoted_tensors"], 1)
        self.assertEqual(audit["promoted_parameters"], 3)
        self.assertEqual(audit["after"]["non_fp32_tensors"], [])

    def test_fp32_trainable_parameters_create_fp32_adam_moments(self):
        model = TinyAdapter(dtype=torch.bfloat16)
        promote_trainable_parameters_to_fp32(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-7)
        before = model.adapter.detach().clone()

        model().backward()
        optimizer.step()

        audit = require_adam_moments_fp32(optimizer)
        self.assertEqual(audit["moment_tensors"], 2)
        self.assertEqual(audit["non_fp32_moments"], [])
        self.assertFalse(torch.equal(before, model.adapter.detach()))

    def test_post_trainer_promotion_preserves_optimizer_parameter_references(self):
        model = TinyAdapter(dtype=torch.bfloat16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-7)

        promote_trainable_parameters_to_fp32(model)
        before = model.adapter.detach().clone()
        model().backward()
        optimizer.step()

        self.assertIs(optimizer.param_groups[0]["params"][1], model.adapter)
        self.assertFalse(torch.equal(before, model.adapter.detach()))
        require_adam_moments_fp32(optimizer)

    def test_bf16_adam_state_is_rejected(self):
        model = TinyAdapter(dtype=torch.bfloat16)
        optimizer = torch.optim.AdamW(model.parameters(), lr=8e-7)

        model().backward()
        optimizer.step()

        audit = optimizer_moment_precision_audit(optimizer)
        self.assertEqual(audit["moment_tensors"], 2)
        self.assertTrue(audit["non_fp32_moments"])
        with self.assertRaisesRegex(RuntimeError, "Adam moments must remain FP32"):
            require_adam_moments_fp32(optimizer)

    def test_non_fp32_trainable_parameters_are_rejected(self):
        model = TinyAdapter(dtype=torch.bfloat16)
        with self.assertRaisesRegex(RuntimeError, "must remain FP32"):
            require_trainable_parameters_fp32(model)


if __name__ == "__main__":
    unittest.main()
