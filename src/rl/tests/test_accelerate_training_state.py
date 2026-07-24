#!/usr/bin/env python3
from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # The lightweight local audit environment does not install GPU dependencies.
    torch = None

RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_DIR))

if torch is not None:
    from frameworks.accelerate.training_state import (  # noqa: E402
        load_training_state,
        save_training_state,
    )


@unittest.skipIf(torch is None, "torch is not installed in the local audit environment")
class AccelerateTrainingStateTests(unittest.TestCase):
    def test_optimizer_scheduler_and_rng_resume(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: 1.0 / (step + 1),
        )

        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()
        random.seed(17)
        torch.manual_seed(23)

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            save_training_state(
                checkpoint,
                step=7,
                optimizer=optimizer,
                scheduler=scheduler,
                metadata={"reward_mode": "process"},
            )
            expected_python = random.random()
            expected_torch = torch.rand(1)

            restored_model = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
            restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
                restored_optimizer,
                lr_lambda=lambda step: 1.0 / (step + 1),
            )
            completed_step = load_training_state(
                checkpoint,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                expected_metadata={"reward_mode": "process"},
            )

        self.assertEqual(completed_step, 7)
        self.assertAlmostEqual(random.random(), expected_python)
        self.assertTrue(torch.equal(torch.rand(1), expected_torch))
        self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())

    def test_resume_rejects_changed_control_condition(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory)
            save_training_state(
                checkpoint,
                step=1,
                optimizer=optimizer,
                scheduler=scheduler,
                metadata={"reward_mode": "process"},
            )
            with self.assertRaisesRegex(ValueError, "resume metadata mismatch"):
                load_training_state(
                    checkpoint,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    expected_metadata={"reward_mode": "result-only"},
                )


if __name__ == "__main__":
    unittest.main()
