import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_recovery_anchors import (  # noqa: E402
    evaluator_messages,
    parse_decision,
    selected_candidate,
)


def package() -> dict:
    return {
        "task": {
            "example_id": "task_1",
            "example_index": 1,
            "db_id": "db",
            "question": "Question?",
            "external_knowledge": None,
            "difficulty": "easy",
        },
        "student_rollout": {
            "source": "all.jsonl",
            "sample_index": 0,
            "failure_type": "execution_error",
            "legal": False,
            "steps": 2,
            "errors": 1,
        },
        "candidates": [
            {
                "candidate_id": "task_1:sample_0:action_2",
                "anchor_action_index": 2,
                "legal_prefix": [],
                "last_tool_error": {
                    "step_id": "step_2",
                    "status": "error",
                    "error": {"type": "execution_error", "message": "bad column"},
                },
            }
        ],
    }


class EvaluateRecoveryAnchorsTest(unittest.TestCase):
    def test_accepts_offered_candidate(self):
        decision = parse_decision(
            json.dumps(
                {
                    "decision": "select_candidate",
                    "candidate_id": "task_1:sample_0:action_2",
                    "rationale": "The prefix has a valid schema and actionable feedback.",
                }
            ),
            package(),
        )
        selected = selected_candidate(
            package(),
            decision,
            evaluator_model="evaluator",
            usage={"total_tokens": 10},
        )
        self.assertEqual(2, selected["selected_candidate"]["anchor_action_index"])
        self.assertFalse(selected["selection_audit"]["rationale_is_teacher_visible"])

    def test_rejects_hallucinated_candidate(self):
        with self.assertRaisesRegex(ValueError, "not offered"):
            parse_decision(
                json.dumps(
                    {
                        "decision": "select_candidate",
                        "candidate_id": "missing",
                        "rationale": "Looks useful.",
                    }
                ),
                package(),
            )

    def test_scratch_requires_null_candidate(self):
        with self.assertRaisesRegex(ValueError, "candidate_id=null"):
            parse_decision(
                json.dumps(
                    {
                        "decision": "route_to_teacher_from_scratch",
                        "candidate_id": "task_1:sample_0:action_2",
                        "rationale": "The prefix is confused.",
                    }
                ),
                package(),
            )

    def test_prompt_contains_no_harness_answer_field(self):
        rendered = json.dumps(evaluator_messages(package()), ensure_ascii=False)
        self.assertNotIn("gold_sql", rendered)
        self.assertNotIn("SELECT ", rendered)


if __name__ == "__main__":
    unittest.main()
