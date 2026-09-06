from __future__ import annotations

import unittest

from rl.scenarios.data.repair_passk_rollout_artifact import repaired_rows


class RepairPasskRolloutArtifactTests(unittest.TestCase):
    def test_replaces_only_named_contaminated_row_and_preserves_order(self) -> None:
        clean = {
            "example_index": 1,
            "samples": [{"sample_index": 0, "correct": True, "failure_type": None}],
        }
        contaminated = {
            "example_index": 2,
            "samples": [{
                "sample_index": 0,
                "correct": False,
                "failure_type": "context_overflow",
            }],
        }
        replacement = {
            "example_index": 2,
            "samples": [{"sample_index": 0, "correct": True, "failure_type": None}],
        }
        rows, replaced = repaired_rows([clean, contaminated], [replacement])
        self.assertEqual([row["example_index"] for row in rows], [1, 2])
        self.assertEqual(rows[1], replacement)
        self.assertEqual(replaced, [2])

    def test_rejects_replacement_that_is_still_contaminated(self) -> None:
        base = [{"example_index": 2, "samples": []}]
        replacement = [{
            "example_index": 2,
            "samples": [{"failure_type": "transport_error"}],
        }]
        with self.assertRaisesRegex(ValueError, "still contain"):
            repaired_rows(base, replacement)


if __name__ == "__main__":
    unittest.main()
