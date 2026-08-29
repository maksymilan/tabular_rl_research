#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reproductions.trust_sql.qwen3_8b_sql_controls.remote_supervisor import (
    build_parser,
    validate_result,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class ResultValidationTests(unittest.TestCase):
    def test_concurrency_arguments_are_explicit(self):
        args = build_parser().parse_args([
            "--mode", "direct",
            "--run-dir", "/home/dengyan/tabular_rl_outputs/evaluations/qwen3_sql_controls/test_run",
            "--gpu", "5",
            "--port", "8042",
            "--runtime-sha256", "0" * 64,
            "--model-size", "8b",
            "--model-root", "/home/dengyan/models/Qwen3-8B-TrustSQL-baseline",
            "--max-num-batched-tokens", "16384",
            "--max-num-seqs", "8",
            "--workers", "8",
        ])
        self.assertEqual(args.max_num_batched_tokens, 16384)
        self.assertEqual(args.max_num_seqs, 8)
        self.assertEqual(args.workers, 8)

    def test_direct_result_contract_and_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                json.dumps({
                    "model": "qwen3-8b-direct-sql-base",
                    "denotation_comparison": "bird-set",
                    "enable_thinking": "1",
                    "runner": "direct_sql_passk",
                    "dev_size": 2,
                    "n_samples": 1,
                    "pass_k": [1],
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 2048,
                    "execution_feedback": False,
                    "prompt_profile": "canonical-json-v1",
                }),
                encoding="utf-8",
            )
            write_jsonl(root / "all.jsonl", [
                {
                    "example_index": 0,
                    "correct": True,
                    "samples": [{"correct": True, "predicted_row_count": 1}],
                },
                {
                    "example_index": 1,
                    "correct": False,
                    "failure_type": "all_samples_failed",
                    "samples": [{"correct": False, "failure_type": "no_sql"}],
                },
            ])
            report = validate_result(
                "direct", root, "qwen3-8b-direct-sql-base", 2
            )
            self.assertEqual(report["correct"], 1)
            self.assertEqual(report["legal"], 1)
            self.assertEqual(report["api_error_count"], 0)

    def test_iterative_api_error_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(
                json.dumps({
                    "model": "qwen3-8b-iterative-sql-base",
                    "denotation_comparison": "bird-set",
                    "enable_thinking": True,
                    "runner": "iterative_sql_feedback",
                    "tool_scheme": "iterative-sql",
                    "interface": "execute-sql-submit-sql-v6",
                    "context_profile": "lazy-catalog-v1",
                    "max_steps": 30,
                    "max_tokens": 2048,
                    "history_turns": 4,
                }),
                encoding="utf-8",
            )
            write_jsonl(root / "all.jsonl", [
                {
                    "example_index": 0,
                    "correct": False,
                    "legal": False,
                    "steps": 1,
                    "errors": 0,
                    "failure_type": "api_error",
                    "error": "ChatAPIError: disconnected",
                }
            ])
            with self.assertRaisesRegex(ValueError, "API error gate failed"):
                validate_result(
                    "iterative", root, "qwen3-8b-iterative-sql-base", 1
                )


if __name__ == "__main__":
    unittest.main()
