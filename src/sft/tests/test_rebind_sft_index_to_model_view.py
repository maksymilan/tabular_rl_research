from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rebind_sft_index_to_model_view import digest, rebind  # noqa: E402


class RebindSftIndexToModelViewTest(unittest.TestCase):
    def test_recomputes_input_hash_and_preserves_target_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            canonical = root / "canonical.jsonl"
            view = root / "view.jsonl"
            source_index = root / "source.index.jsonl"
            output = root / "view.index.jsonl"
            target = '<think>now</think>\n{"tool":"done","arguments":{}}'
            canonical_row = {
                "system": "s",
                "conversations": [
                    {"from": "human", "value": "q"},
                    {
                        "from": "gpt",
                        "value": '<think>old</think>\n{"tool":"inspect","arguments":{}}',
                    },
                    {"from": "human", "value": "obs"},
                    {"from": "gpt", "value": target},
                ],
                "metadata": {"record_id": "r"},
            }
            row = json.loads(json.dumps(canonical_row))
            row["conversations"][1]["value"] = '{"tool":"inspect","arguments":{}}'
            old_input_hash = "0" * 64
            old_target_hash = "1" * 64
            canonical.write_text(json.dumps(canonical_row) + "\n", encoding="utf-8")
            view.write_text(json.dumps(row) + "\n", encoding="utf-8")
            source_index.write_text(
                json.dumps(
                    {
                        "record_id": "r",
                        "model_input_sha256": old_input_hash,
                        "target_sha256": old_target_hash,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = rebind(view, canonical, source_index, output, "qwen3-test")
            actual = json.loads(output.read_text(encoding="utf-8"))
            expected_messages = [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": '{"tool":"inspect","arguments":{}}'},
                {"role": "user", "content": "obs"},
            ]
            self.assertEqual(actual["model_input_sha256"], digest(expected_messages))
            canonical_messages = [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "q"},
                {
                    "role": "assistant",
                    "content": '<think>old</think>\n{"tool":"inspect","arguments":{}}',
                },
                {"role": "user", "content": "obs"},
            ]
            self.assertEqual(actual["canonical_model_input_sha256"], digest(canonical_messages))
            self.assertEqual(actual["source_index_model_input_sha256"], old_input_hash)
            self.assertEqual(actual["source_index_target_sha256"], old_target_hash)
            self.assertEqual(actual["target_sha256"], digest(target))
            self.assertEqual(actual["canonical_target_sha256"], digest(target))
            self.assertEqual(actual["model_input_projection"], "qwen3-test")
            self.assertEqual(manifest["canonical_to_view_model_input_hashes_changed"], 1)
            self.assertEqual(manifest["target_hashes_changed_by_projection"], 0)
            self.assertEqual(
                manifest["source_index_model_input_hashes_matching_canonical"], 0
            )
            self.assertEqual(manifest["source_index_target_hashes_matching_canonical"], 0)


if __name__ == "__main__":
    unittest.main()
