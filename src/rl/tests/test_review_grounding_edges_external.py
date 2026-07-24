#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_DIR))

from review_grounding_edges_external import (  # noqa: E402
    _compact_output,
    _edge_id,
    _evidence_rows,
    _risk_flags,
    _validate_package_output_path,
    _validate_review,
)


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

    def test_relation_preview_is_compacted_like_read_observation(self):
        compact = _compact_output(
            "condition_filter",
            {
                "table": "filter_001",
                "columns": ["id"],
                "row_count": 20,
                "rows": [[index] for index in range(20)],
            },
            {"values": [17]},
        )
        self.assertEqual(compact["rows"][:3], [[0], [1], [2]])
        self.assertIn([17], compact["rows"])

    def test_parallel_grounding_edges_have_target_unique_ids(self):
        left = _edge_id(
            "step_1",
            "step_2",
            "grounding",
            "schema_observation",
            {"table": "Paper", "columns": ["Id"]},
        )
        right = _edge_id(
            "step_1",
            "step_2",
            "grounding",
            "schema_observation",
            {"table": "PaperAuthor", "columns": ["PaperId"]},
        )
        self.assertNotEqual(left, right)

    def test_subset_review_cannot_overwrite_complete_package_input(self):
        path = Path("/tmp/grounding-packages.jsonl")
        with self.assertRaisesRegex(ValueError, "must differ"):
            _validate_package_output_path(
                path,
                path,
                selected_ids={"trajectory_1"},
                limit=0,
            )

    def test_complete_package_reuse_may_keep_same_output_path(self):
        path = Path("/tmp/grounding-packages.jsonl")
        _validate_package_output_path(
            path,
            path,
            selected_ids=None,
            limit=0,
        )


if __name__ == "__main__":
    unittest.main()
