#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path

RL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RL_DIR))

from summarize_grounding_external_audit import (  # noqa: E402
    component_counts,
    index_unique,
    project_review_edges,
)


class GroundingAuditSummaryTests(unittest.TestCase):
    def test_index_unique_rejects_duplicate_trajectory(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            index_unique(
                [{"trajectory_id": "t1"}, {"trajectory_id": "t1"}],
                "reviews",
            )

    def test_projected_package_removes_retired_edge_and_recomputes_overall(self):
        packages = {
            "t1": {
                "trajectory_id": "t1",
                "grounding_edges": [{"edge_id": "kept"}],
            }
        }
        reviews = {
            "t1": {
                "trajectory_id": "t1",
                "review": {
                    "final_dependency": {"label": "valid"},
                    "edges": [
                        {"edge_id": "kept", "label": "valid"},
                        {"edge_id": "retired", "label": "invalid"},
                    ],
                    "missing_edges": [],
                    "overall": "fail",
                },
            }
        }
        projected, removed = project_review_edges(
            packages,
            reviews,
            label="reviews",
        )
        self.assertEqual(removed, 1)
        self.assertEqual(projected["t1"]["review"]["overall"], "pass")
        self.assertEqual(
            [edge["edge_id"] for edge in projected["t1"]["review"]["edges"]],
            ["kept"],
        )
        self.assertEqual(
            component_counts(projected, {"t1"})["edge_decided_precision"],
            1.0,
        )

    def test_projection_rejects_review_missing_current_edge(self):
        packages = {
            "t1": {
                "trajectory_id": "t1",
                "grounding_edges": [{"edge_id": "new"}],
            }
        }
        reviews = {
            "t1": {
                "trajectory_id": "t1",
                "review": {
                    "final_dependency": {"label": "valid"},
                    "edges": [],
                    "missing_edges": [],
                    "overall": "pass",
                },
            }
        }
        with self.assertRaisesRegex(ValueError, "missing current package edges"):
            project_review_edges(packages, reviews, label="reviews")


if __name__ == "__main__":
    unittest.main()
