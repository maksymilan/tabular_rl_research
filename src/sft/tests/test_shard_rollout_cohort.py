import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shard_rollout_cohort import shard  # noqa: E402


class ShardRolloutCohortTest(unittest.TestCase):
    def test_round_robin_is_balanced_disjoint_and_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.jsonl"
            rows = [
                {
                    "example_id": f"task_{index}",
                    "db_id": f"db_{index % 3}",
                    "metadata": {"difficulty_proxy": "easy" if index < 4 else "hard"},
                }
                for index in range(7)
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            manifest = shard(source, root / "out", shards=2, prefix="cohort")
            outputs = [
                [
                    json.loads(line)
                    for line in Path(item["path"]).read_text(encoding="utf-8").splitlines()
                ]
                for item in manifest["shards"]
            ]
            self.assertEqual([4, 3], [len(items) for items in outputs])
            first = {row["example_id"] for row in outputs[0]}
            second = {row["example_id"] for row in outputs[1]}
            self.assertFalse(first & second)
            self.assertEqual({row["example_id"] for row in rows}, first | second)
            self.assertEqual(0, manifest["coverage"]["overlap"])


if __name__ == "__main__":
    unittest.main()
