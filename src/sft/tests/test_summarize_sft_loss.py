from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from summarize_sft_loss import summarize, write_summary  # noqa: E402


class SummarizeSftLossTest(unittest.TestCase):
    def test_groups_logged_losses_by_epoch_and_records_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            state = {
                "global_step": 8,
                "epoch": 2.0,
                "log_history": [
                    {"loss": 0.8, "step": 2, "epoch": 0.5, "learning_rate": 4e-5},
                    {"loss": 0.6, "step": 4, "epoch": 1.0, "learning_rate": 3e-5},
                    {"loss": 0.4, "step": 6, "epoch": 1.5, "learning_rate": 2e-5},
                    {"loss": 0.2, "step": 8, "epoch": 2.0, "learning_rate": 1e-5},
                    {"train_loss": 0.5, "train_runtime": 100, "step": 8, "epoch": 2.0},
                ],
            }
            for step, epoch in ((4, 1.0), (8, 2.0)):
                checkpoint = output / f"checkpoint-{step}"
                checkpoint.mkdir()
                checkpoint_state = {**state, "global_step": step, "epoch": epoch}
                (checkpoint / "trainer_state.json").write_text(
                    json.dumps(checkpoint_state), encoding="utf-8"
                )

            summary = summarize(output)
            self.assertEqual(summary["global_step"], 8)
            self.assertEqual(
                [item["checkpoint"] for item in summary["checkpoints"]],
                ["checkpoint-4", "checkpoint-8"],
            )
            self.assertEqual(len(summary["epochs"]), 2)
            self.assertAlmostEqual(summary["epochs"][0]["mean_logged_loss"], 0.7)
            self.assertAlmostEqual(summary["epochs"][1]["mean_logged_loss"], 0.3)
            self.assertEqual(summary["final_metrics"]["train_loss"], 0.5)

            write_summary(summary, output)
            self.assertTrue((output / "loss_by_epoch.json").is_file())
            self.assertTrue((output / "loss_by_epoch.csv").is_file())


if __name__ == "__main__":
    unittest.main()
