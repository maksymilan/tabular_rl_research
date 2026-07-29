from __future__ import annotations

import unittest

from select_passk_rl_examples import (
    has_infrastructure_failure,
    has_mixed_attempt_outcomes,
    sample_correct_count,
)


class SelectPasskRlExamplesTest(unittest.TestCase):
    def test_mixed_outcomes_from_summary_fields(self) -> None:
        record = {
            "n_samples": 8,
            "sample_correct_count": 3,
        }
        self.assertEqual(sample_correct_count(record), 3)
        self.assertTrue(has_mixed_attempt_outcomes(record))

    def test_all_equal_outcomes_are_not_mixed(self) -> None:
        self.assertFalse(has_mixed_attempt_outcomes({
            "n_samples": 8,
            "sample_correct_count": 0,
        }))
        self.assertFalse(has_mixed_attempt_outcomes({
            "n_samples": 8,
            "sample_correct_count": 8,
        }))

    def test_mixed_outcomes_fall_back_to_sample_records(self) -> None:
        record = {
            "samples": [
                {"correct": True},
                {"correct": False},
                {"is_correct": True},
            ],
        }
        self.assertEqual(sample_correct_count(record), 2)
        self.assertTrue(has_mixed_attempt_outcomes(record))

    def test_sample_level_infrastructure_failure_is_not_mixed(self) -> None:
        record = {
            "n_samples": 8,
            "sample_correct_count": 3,
            "samples": [
                {"correct": True, "failure_type": None, "turns": []},
                {"correct": False, "failure_type": "api_error", "turns": []},
            ],
        }
        self.assertTrue(has_infrastructure_failure(record))
        self.assertFalse(has_mixed_attempt_outcomes(record))


if __name__ == "__main__":
    unittest.main()
