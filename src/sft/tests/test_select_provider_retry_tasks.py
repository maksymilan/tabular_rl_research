import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from select_provider_retry_tasks import build  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class SelectProviderRetryTasksTest(unittest.TestCase):
    def test_selects_only_empty_visible_content_protocol_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            attempts = root / "attempts.jsonl"
            output = root / "retry.jsonl"
            write_jsonl(source, [
                {"example_id": "task_1"},
                {"example_id": "task_2"},
                {"example_id": "task_3"},
            ])
            write_jsonl(attempts, [
                {
                    "trajectory_id": "task_1",
                    "correct": False,
                    "failure_type": "protocol_error",
                    "error_events": [{"message": "visible content was empty; retry JSON"}],
                },
                {
                    "trajectory_id": "task_2",
                    "correct": False,
                    "failure_type": "wrong_answer",
                    "error_events": [],
                },
                {
                    "trajectory_id": "task_3",
                    "correct": False,
                    "failure_type": "protocol_error",
                    "error_events": [{"message": "invalid JSON object"}],
                },
            ])

            manifest = build(source, attempts, output)

            selected = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([{"example_id": "task_1"}], selected)
            self.assertEqual(1, manifest["retry_tasks"])
            self.assertIn("provider-carrier incomplete", manifest["semantic_interpretation"])


if __name__ == "__main__":
    unittest.main()
