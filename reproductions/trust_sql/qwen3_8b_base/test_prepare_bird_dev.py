#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_bird_dev


class PrepareBirdDevTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="trustsql_bird_dev_")
        self.root = Path(self.temporary.name)
        self.dev_path = self.root / "dev.json"
        self.prompt_path = self.root / "prompt_template.txt"
        self.database_root = self.root / "dev_databases"
        self.database_root.mkdir()
        self.output_path = self.root / "prepared.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.system_prompt = "EXACT TRUST-SQL SYSTEM PROMPT"
        self.prompt_path.write_text(self.system_prompt, encoding="utf-8")
        self.source = [
            {
                "question_id": 10,
                "db_id": "alpha_db",
                "question": "Which alpha is largest?",
                "evidence": "largest means ORDER BY value DESC LIMIT 1",
                "SQL": "SELECT name FROM alpha ORDER BY value DESC LIMIT 1",
                "difficulty": "simple",
            },
            {
                "question_id": 11,
                "db_id": "beta_db",
                "question": "List beta names",
                "evidence": "",
                "SQL": "SELECT name FROM beta",
                "difficulty": "simple",
            },
            {
                "question_id": 12,
                "db_id": "gamma_db",
                "question": "Count gamma rows",
                "evidence": "count rows means COUNT(*)",
                "SQL": "SELECT COUNT(*) FROM gamma",
                "difficulty": "simple",
            },
        ]
        self.dev_path.write_text(
            json.dumps(self.source, ensure_ascii=False), encoding="utf-8"
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def read_rows(self) -> list[dict]:
        return [
            json.loads(line)
            for line in self.output_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    def test_prepares_runner_records_and_auditable_manifest(self) -> None:
        manifest = prepare_bird_dev.prepare_dataset(
            bird_dev_path=self.dev_path,
            prompt_template_path=self.prompt_path,
            database_root=self.database_root,
            output_path=self.output_path,
            manifest_path=self.manifest_path,
            limit=2,
        )
        rows = self.read_rows()
        self.assertEqual([row["id"] for row in rows], ["dev_10", "dev_11"])
        self.assertEqual(rows[0]["prompt"][0], {
            "role": "system",
            "content": self.system_prompt,
        })
        self.assertEqual(
            rows[0]["prompt"][1]["content"],
            "\n**Task Configuration**\n"
            "**Database Engine:** SQLite\n"
            "**Database:** alpha_db\n"
            "**External Knowledge:** largest means ORDER BY value DESC LIMIT 1\n"
            "**User Question:** Which alpha is largest??\n",
        )
        self.assertEqual(
            rows[1]["prompt"][1]["content"],
            "\n**Task Configuration**\n"
            "**Database Engine:** SQLite\n"
            "**Database:** beta_db\n"
            "**External Knowledge:** \n"
            "**User Question:** List beta names?\n",
        )
        for source, row in zip(self.source, rows):
            visible_prompt = json.dumps(row["prompt"], ensure_ascii=False)
            self.assertNotIn(source["SQL"], visible_prompt)
            self.assertNotIn("SQL", row)
            self.assertEqual(
                row["reward_model"]["ground_truth"]["target"], [source["SQL"]]
            )
            self.assertEqual(
                row["reward_model"]["database"], str(self.database_root.resolve())
            )

        expected_source_hash = hashlib.sha256(self.dev_path.read_bytes()).hexdigest()
        expected_prompt_hash = hashlib.sha256(self.prompt_path.read_bytes()).hexdigest()
        self.assertEqual(manifest["source"]["bird_dev_sha256"], expected_source_hash)
        self.assertEqual(manifest["prompt"]["template_sha256"], expected_prompt_hash)
        self.assertEqual(manifest["selection"]["source_indices"], [0, 1])
        self.assertEqual(manifest["output"]["record_count"], 2)
        self.assertEqual(
            json.loads(self.manifest_path.read_text(encoding="utf-8")), manifest
        )
        self.assertEqual(
            manifest["output"]["sha256"],
            hashlib.sha256(self.output_path.read_bytes()).hexdigest(),
        )

        first_output = self.output_path.read_bytes()
        first_manifest = self.manifest_path.read_bytes()
        prepare_bird_dev.prepare_dataset(
            bird_dev_path=self.dev_path,
            prompt_template_path=self.prompt_path,
            database_root=self.database_root,
            output_path=self.output_path,
            manifest_path=self.manifest_path,
            limit=2,
        )
        self.assertEqual(self.output_path.read_bytes(), first_output)
        self.assertEqual(self.manifest_path.read_bytes(), first_manifest)

    def test_explicit_indices_are_unique_in_range_and_source_ordered(self) -> None:
        manifest = prepare_bird_dev.prepare_dataset(
            bird_dev_path=self.dev_path,
            prompt_template_path=self.prompt_path,
            database_root=self.database_root,
            output_path=self.output_path,
            manifest_path=self.manifest_path,
            indices=[2, 0],
        )
        self.assertEqual([row["id"] for row in self.read_rows()], ["dev_10", "dev_12"])
        self.assertEqual(manifest["selection"]["source_indices"], [0, 2])
        with self.assertRaisesRegex(ValueError, "duplicates"):
            prepare_bird_dev.prepare_dataset(
                bird_dev_path=self.dev_path,
                prompt_template_path=self.prompt_path,
                database_root=self.database_root,
                output_path=self.output_path,
                indices=[1, 1],
            )
        with self.assertRaisesRegex(ValueError, "outside BIRD dev range"):
            prepare_bird_dev.prepare_dataset(
                bird_dev_path=self.dev_path,
                prompt_template_path=self.prompt_path,
                database_root=self.database_root,
                output_path=self.output_path,
                indices=[3],
            )

    def test_cli_rejects_limit_with_indices(self) -> None:
        argv = [
            "--bird-dev", str(self.dev_path),
            "--prompt-template", str(self.prompt_path),
            "--dev-databases", str(self.database_root),
            "--output", str(self.output_path),
            "--limit", "1",
            "--indices", "0",
        ]
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                prepare_bird_dev.parse_args(argv)
        self.assertEqual(raised.exception.code, 2)

        aliases = prepare_bird_dev.parse_args([
            "--bird-json", str(self.dev_path),
            "--system-prompt", str(self.prompt_path),
            "--database-root", str(self.database_root),
            "--output", str(self.output_path),
        ])
        self.assertEqual(aliases.bird_dev, self.dev_path)
        self.assertEqual(aliases.prompt_template, self.prompt_path)
        self.assertEqual(aliases.dev_databases, self.database_root)

    def test_rejects_any_gold_sql_that_would_become_model_visible(self) -> None:
        self.source[0]["question"] = self.source[0]["SQL"]
        self.dev_path.write_text(
            json.dumps(self.source, ensure_ascii=False), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "expose its gold SQL"):
            prepare_bird_dev.prepare_dataset(
                bird_dev_path=self.dev_path,
                prompt_template_path=self.prompt_path,
                database_root=self.database_root,
                output_path=self.output_path,
                limit=1,
            )


if __name__ == "__main__":
    unittest.main()
