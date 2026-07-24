import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filter_sft_by_token_audit import build  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class FilterSftByTokenAuditTest(unittest.TestCase):
    def test_filters_only_incomplete_prefix_records_and_keeps_index_aligned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records_path = root / "source.jsonl"
            index_path = root / "source.index.jsonl"
            audit_path = root / "source.token_audit.json"
            output_path = root / "ready.jsonl"
            output_index_path = root / "ready.index.jsonl"
            records = [
                {
                    "system": "system",
                    "conversations": [
                        {"from": "human", "value": "question"},
                        {"from": "gpt", "value": "action"},
                    ],
                    "metadata": {
                        "record_id": "episode_1_step_1",
                        "source_episode_id": "episode_1",
                        "feedback_recovery": False,
                    },
                },
                {
                    "system": "system",
                    "conversations": [
                        {"from": "human", "value": "question and history"},
                        {"from": "gpt", "value": "later action"},
                    ],
                    "metadata": {
                        "record_id": "episode_1_step_2",
                        "source_episode_id": "episode_1",
                        "feedback_recovery": True,
                    },
                },
                {
                    "system": "system",
                    "conversations": [
                        {"from": "human", "value": "another question"},
                        {"from": "gpt", "value": "another action"},
                    ],
                    "metadata": {
                        "record_id": "episode_2_step_1",
                        "source_episode_id": "episode_2",
                        "feedback_recovery": False,
                    },
                },
                {
                    "system": "system",
                    "conversations": [
                        {"from": "human", "value": "fully rejected episode"},
                        {"from": "gpt", "value": "overlong action"},
                    ],
                    "metadata": {
                        "record_id": "episode_3_step_1",
                        "source_episode_id": "episode_3",
                        "feedback_recovery": False,
                    },
                },
            ]
            indexes = [
                {
                    "record_id": row["metadata"]["record_id"],
                    "source_episode_id": row["metadata"]["source_episode_id"],
                    "tool_name": "describe_table",
                }
                for row in records
            ]
            audit = {
                "records": 4,
                "model_path": "model",
                "template": "qwen",
                "cutoff_len": 6400,
                "mask_history": True,
                "details": [
                    {
                        "record_id": "episode_1_step_2",
                        "target_status": "complete",
                        "current_source_tokens_original": 10,
                        "current_source_tokens_kept": 10,
                        "history_pairs_original": 1,
                        "first_pair_complete": False,
                    },
                    {
                        "record_id": "episode_3_step_1",
                        "target_status": "complete",
                        "current_source_tokens_original": 10,
                        "current_source_tokens_kept": 10,
                        "history_pairs_original": 1,
                        "first_pair_complete": False,
                    },
                ],
            }
            write_jsonl(records_path, records)
            write_jsonl(index_path, indexes)
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

            manifest = build(
                [records_path],
                [index_path],
                [audit_path],
                output_path,
                output_index_path,
                "pilot",
            )

            kept_records = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]
            kept_indexes = [
                json.loads(line)
                for line in output_index_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                ["episode_1_step_1", "episode_2_step_1"],
                [row["metadata"]["record_id"] for row in kept_records],
            )
            self.assertEqual(
                ["episode_1_step_1", "episode_2_step_1"],
                [row["record_id"] for row in kept_indexes],
            )
            self.assertEqual(2, manifest["kept_records"])
            self.assertEqual(2, manifest["dropped_records"])
            self.assertEqual(2, manifest["episodes_with_dropped_records"])
            self.assertEqual(3, manifest["source_episodes"])
            self.assertEqual(2, manifest["contributing_episodes"])
            self.assertEqual(1, manifest["complete_episodes"])
            self.assertEqual(1, manifest["records_from_complete_episodes"])
            self.assertIsNone(manifest["selection_policy"]["reasoning_word_limit"])

    def test_rejects_misaligned_record_and_index_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records_path = root / "source.jsonl"
            index_path = root / "source.index.jsonl"
            audit_path = root / "source.token_audit.json"
            records = [
                {
                    "metadata": {
                        "record_id": "episode_step_1",
                        "source_episode_id": "episode",
                    }
                }
            ]
            write_jsonl(records_path, records)
            write_jsonl(
                index_path,
                [{
                    "record_id": "other_step_1",
                    "source_episode_id": "other",
                    "tool_name": "describe_table",
                }],
            )
            audit_path.write_text(
                json.dumps({"records": 1, "details": []}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "identical record-id order"):
                build(
                    [records_path],
                    [index_path],
                    [audit_path],
                    root / "ready.jsonl",
                    root / "ready.index.jsonl",
                    "pilot",
                )


if __name__ == "__main__":
    unittest.main()
