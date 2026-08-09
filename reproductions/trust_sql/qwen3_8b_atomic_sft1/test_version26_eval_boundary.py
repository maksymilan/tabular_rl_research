#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
PREPARE = HERE / "prepare_version26_runtime.sh"
VERIFY = HERE / "verify_version26_runtime.py"
LAUNCHER = HERE / "run_greedy_bird_dev1534_table_rl.sh"


class Version26EvaluationBoundaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="atomic-v26-eval-test-")
        cls.runtime = Path(cls.temporary.name) / "runtime"
        environment = {
            **os.environ,
            "PROJECT_DIR": str(PROJECT),
            "RUNTIME_ROOT": str(cls.runtime),
            "VERIFY_PYTHON": sys.executable,
        }
        subprocess.run(["bash", str(PREPARE)], check=True, env=environment, capture_output=True)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def runtime_report(self, runtime: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(VERIFY),
                "--runtime-root",
                str(runtime or self.runtime),
                "--json",
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def test_exact_runtime_prompt_and_version_gate(self) -> None:
        completed = self.runtime_report()
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(
            report["source_commit"],
            "4cd47c957fc6ae791e76a10594c8cd22f4d3b6de",
        )
        self.assertEqual(report["protocol"]["protocol_version"], "version26")
        self.assertEqual(report["protocol"]["tool_scheme"], "atomic")
        self.assertEqual(report["protocol"]["assistant_carrier"], "think-json-v1")
        self.assertEqual(
            report["protocol"]["rolling_system_prompt_sha256"],
            "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316",
        )
        self.assertIsNone(report["reasoning_parser"])
        self.assertFalse(report["current_checkout_protocol_modules_imported"])
        for module_path in report["protocol"]["module_paths"].values():
            self.assertTrue(Path(module_path).is_relative_to(self.runtime.resolve()))

    def test_runtime_gate_detects_one_byte_drift(self) -> None:
        changed = Path(self.temporary.name) / "changed-runtime"
        shutil.copytree(self.runtime, changed)
        protocol = changed / "src/sft/protocol.py"
        protocol.write_bytes(protocol.read_bytes() + b"\n")
        completed = self.runtime_report(changed)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("content tree mismatch", completed.stderr)

    def test_launcher_dry_run_performs_no_remote_work(self) -> None:
        environment = {
            **os.environ,
            "PROJECT_DIR": str(PROJECT),
            "RUNTIME_ROOT": str(self.runtime),
            "LOCAL_PYTHON": sys.executable,
            "MODEL_MODE": "base",
        }
        completed = subprocess.run(
            ["bash", str(LAUNCHER), "--dry-run"],
            check=False,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("dry-run gate: OK", completed.stdout)
        self.assertIn("qwen3_enable_thinking=true reasoning_parser=none", completed.stdout)
        self.assertIn("history=4 max_steps=30 bird-set", completed.stdout)

    def test_adapter_dry_run_uses_distinct_backbone_and_lora_aliases(self) -> None:
        environment = {
            **os.environ,
            "PROJECT_DIR": str(PROJECT),
            "RUNTIME_ROOT": str(self.runtime),
            "LOCAL_PYTHON": sys.executable,
            "MODEL_MODE": "adapter",
            "ADAPTER": "/remote/read-only/example-checkpoint",
        }
        completed = subprocess.run(
            ["bash", str(LAUNCHER), "--dry-run"],
            check=False,
            env=environment,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("served_model=qwen3-8b-atomic-v26-sft1-qlora", completed.stdout)
        self.assertIn("backbone_served_model=qwen3-8b-atomic-v26-backbone", completed.stdout)

    def test_launcher_contains_no_template_override_or_reasoning_parser(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertNotIn("--chat-template", source)
        self.assertNotIn("--reasoning-parser", source)
        self.assertNotIn("--enable-reasoning", source)
        self.assertIn("EVAL_ENABLE_THINKING=1", source)
        self.assertIn("PYTHONDONTWRITEBYTECODE=1", source)
        self.assertIn('--rolling-prompt-variant full', source)
        self.assertIn('--rolling-observation-style resident', source)
        self.assertIn('--denotation-comparison bird-set', source)
        self.assertIn('"$RUNTIME_ROOT/src/eval/rollout_passk.py"', source)
        self.assertNotIn("src/eval/run_bird_lora_tool_passk_table_rl.sh", source)

    def test_shell_and_python_files_compile(self) -> None:
        for shell_file in (PREPARE, LAUNCHER):
            subprocess.run(["bash", "-n", str(shell_file)], check=True)
        for python_file in (VERIFY, HERE / "verify_qwen3_model.py"):
            subprocess.run(
                [sys.executable, "-m", "py_compile", str(python_file)],
                check=True,
            )


if __name__ == "__main__":
    unittest.main()
