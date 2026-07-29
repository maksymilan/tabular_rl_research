#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from select_schema_context_hard_cohort import (  # noqa: E402
    select_hard_stratum,
    summarize_reference,
)


def example(index: int, difficulty: str, score: int) -> dict:
    return {
        "example_id": f"task_{index}",
        "db_id": f"db_{index % 2}",
        "metadata": {
            "difficulty_proxy": difficulty,
            "difficulty_proxy_features": {"score": score},
        },
    }


class SelectSchemaContextHardCohortTest(unittest.TestCase):
    def test_selection_uses_only_position_and_difficulty_metadata(self):
        examples = [
            example(0, "hard", 9),
            example(1, "easy", 1),
            example(2, "hard", 5),
            example(3, "hard", 6),
            example(4, "hard", 10),
        ]
        examples[3]["gold_sql"] = "SELECT hidden"
        examples[3]["correct"] = False
        selected = select_hard_stratum(
            examples,
            exclude_prefix=1,
            difficulty="hard",
            minimum_complexity_score=6,
        )
        self.assertEqual([3, 4], [index for index, _ in selected])

    def test_reference_is_joined_after_selection(self):
        selected = [
            (3, example(3, "hard", 6)),
            (4, example(4, "hard", 10)),
        ]
        records = {
            "task_3": {
                "trajectory_id": "task_3",
                "correct": False,
                "legal": True,
                "errors": 1,
                "failure_type": "wrong_answer",
                "denotation_comparison": "bird-set",
            },
            "task_4": {
                "trajectory_id": "task_4",
                "correct": True,
                "legal": True,
                "errors": 0,
                "denotation_comparison": "bird-set",
            },
        }

        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reference.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records.values()) + "\n",
                encoding="utf-8",
            )
            summary = summarize_reference(selected, [path])

        self.assertEqual(2, summary["count"])
        self.assertEqual(1, summary["correct"])
        self.assertEqual(["task_3"], summary["failure_target_ids"])
        self.assertEqual(["task_4"], summary["correct_control_ids"])
        self.assertEqual(["bird-set"], summary["denotation_metrics"])


if __name__ == "__main__":
    unittest.main()
