#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_DIR))

from review_grounding_edges_external import _evidence_rows, _risk_flags, _validate_review  # noqa: E402


class ExternalGroundingReviewTests(unittest.TestCase):
    def setUp(self):
        self.package = {
            "trajectory_id": "trajectory_1",
            "grounding_edges": [{"edge_id": "step_1->step_2:schema_observation"}],
        }

    def test_strict_review_accepts_complete_exact_edge_set(self):
        review = {
            "trajectory_id": "trajectory_1",
            "final_dependency": {"label": "valid", "reason": "matches final table"},
            "edges": [{
                "edge_id": "step_1->step_2:schema_observation",
                "label": "valid",
                "reason": "column is present",
            }],
            "missing_edges": [],
            "overall": "pass",
        }
        self.assertEqual(_validate_review(self.package, json.dumps(review)), review)

    def test_strict_review_rejects_missing_edge(self):
        review = {
            "trajectory_id": "trajectory_1",
            "final_dependency": {"label": "valid", "reason": "matches final table"},
            "edges": [],
            "missing_edges": [],
            "overall": "pass",
        }
        with self.assertRaisesRegex(ValueError, "edge ids mismatch"):
            _validate_review(self.package, json.dumps(review))

    def test_common_literal_risk_ignores_structured_values(self):
        flags = _risk_flags("row_observation", {"values": [[1], {"value": 1}, "1"]}, None)
        self.assertEqual(flags, ["common_literal"])

    def test_evidence_rows_retain_target_match_beyond_preview(self):
        rows = [[index, f"value-{index}"] for index in range(20)]
        selected = _evidence_rows(rows, {"values": [17]})
        self.assertEqual(selected[:3], rows[:3])
        self.assertIn(rows[17], selected)


if __name__ == "__main__":
    unittest.main()
