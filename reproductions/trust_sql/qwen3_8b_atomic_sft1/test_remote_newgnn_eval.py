#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from reproductions.trust_sql.qwen3_8b_atomic_sft1 import prepare_remote_eval_inputs
from reproductions.trust_sql.qwen3_8b_atomic_sft1 import remote_eval_supervisor


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
LOCK_PATH = HERE / "remote_eval_lock.json"
SOURCE = PROJECT / "data/eval_inputs/bird_dev_20240627.jsonl"
DB_ROOT = PROJECT / "data/bird/dev_20240627/dev_databases"
LAUNCHER = HERE / "launch_detached_remote_newgnn.sh"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RemoteInputPreparationTest(unittest.TestCase):
    def test_remap_is_pinned_deterministic_and_changes_only_db_path(self) -> None:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(prefix="qwen3-remote-input-") as temporary:
            root = Path(temporary)
            output = root / "bird.jsonl"
            manifest = root / "manifest.json"
            first = prepare_remote_eval_inputs.remap(
                SOURCE,
                output,
                manifest,
                db_root=DB_ROOT,
                lock_path=LOCK_PATH,
            )
            first_bytes = output.read_bytes()
            first_manifest = manifest.read_bytes()
            second = prepare_remote_eval_inputs.remap(
                SOURCE,
                output,
                manifest,
                db_root=DB_ROOT,
                lock_path=LOCK_PATH,
            )
            self.assertEqual(first, second)
            self.assertEqual(output.read_bytes(), first_bytes)
            self.assertEqual(manifest.read_bytes(), first_manifest)
            self.assertEqual(sha256(output), lock["evaluation_input"]["derived_sha256"])
            self.assertEqual(sha256(manifest), lock["evaluation_input"]["manifest_sha256"])

            source_rows = [json.loads(line) for line in SOURCE.read_text().splitlines()]
            derived_rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(source_rows), 1534)
            self.assertEqual(len(derived_rows), 1534)
            for source_row, derived_row in zip(source_rows, derived_rows, strict=True):
                expected_path = (
                    Path(lock["evaluation_input"]["remote_db_root"])
                    / source_row["db_id"]
                    / f"{source_row['db_id']}.sqlite"
                )
                self.assertEqual(derived_row["db_path"], str(expected_path))
                source_without = {k: v for k, v in source_row.items() if k != "db_path"}
                derived_without = {k: v for k, v in derived_row.items() if k != "db_path"}
                self.assertEqual(source_without, derived_without)

    def test_lock_pins_all_eleven_databases_and_adapter_560(self) -> None:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
        databases = lock["evaluation_input"]["databases"]
        self.assertEqual(len(databases), 11)
        self.assertEqual(sum(item["records"] for item in databases.values()), 1534)
        for db_id, item in databases.items():
            self.assertEqual(sha256(DB_ROOT / db_id / f"{db_id}.sqlite"), item["sha256"])
        adapter = lock["adapter"]
        self.assertEqual(adapter["checkpoint_name"], "checkpoint-560")
        self.assertEqual(adapter["global_step"], 560)
        self.assertEqual(
            adapter["files_sha256"]["adapter_model.safetensors"],
            "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5",
        )


class DetachedLauncherTest(unittest.TestCase):
    def launcher_dry_run(self, mode: str) -> subprocess.CompletedProcess[str]:
        environment = {
            **os.environ,
            "MODE": mode,
            "RUN_ID": f"{mode}_unit_test",
            "LOCAL_PYTHON": sys.executable,
        }
        return subprocess.run(
            ["bash", str(LAUNCHER), "--dry-run"],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )

    def test_dry_run_never_connects_and_uses_planned_gpu_port_pairs(self) -> None:
        base = self.launcher_dry_run("base")
        adapter = self.launcher_dry_run("adapter")
        self.assertEqual(base.returncode, 0, base.stderr)
        self.assertEqual(adapter.returncode, 0, adapter.stderr)
        self.assertIn("gpu_ids=5 port=8020", base.stdout)
        self.assertIn("gpu_ids=6 port=8021", adapter.stdout)
        self.assertIn("no SSH tunnel", base.stdout)
        self.assertIn("max_num_seqs4 workers4 max_inflight4", adapter.stdout)
        self.assertNotIn("submitted detached", base.stdout)

    def test_launcher_uses_vllm_environment_and_detaches(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("/home/dengyan/miniconda3/envs/vllm-qwen35/bin/python", source)
        self.assertIn("nohup setsid", source)
        self.assertIn("sha256sum -c SHA256SUMS", source)
        self.assertNotIn("ssh -N", source)
        self.assertNotIn("-L \"${LOCAL_PORT}", source)

    def test_supervisor_locks_historical_evaluation_semantics(self) -> None:
        source = (HERE / "remote_eval_supervisor.py").read_text(encoding="utf-8")
        for fragment in (
            '"--n",\n            "1534"',
            '"--workers",\n            "4"',
            '"--max-inflight-requests",\n            "4"',
            '"--max-steps",\n            "30"',
            '"--max-tokens",\n            "2048"',
            '"--history-turns",\n            "4"',
            '"--denotation-comparison",\n            "bird-set"',
            '"EVAL_ENABLE_THINKING": "1"',
            '"--max-num-seqs",\n            "4"',
            'f"http://127.0.0.1:{args.port}/v1"',
        ):
            self.assertIn(fragment, source)
        self.assertNotIn("--reasoning-parser", source)
        self.assertNotIn("ssh -N", source)

    def test_failure_after_status_initialization_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwen3-remote-supervisor-") as temporary:
            allowed_root = Path(temporary).resolve()
            run_dir = allowed_root / "base_failure_fixture"
            run_dir.mkdir()
            original_root = remote_eval_supervisor.ALLOWED_RUN_ROOT
            remote_eval_supervisor.ALLOWED_RUN_ROOT = allowed_root
            try:
                args = Namespace(
                    mode="base",
                    run_dir=run_dir,
                    gpu_ids="5",
                    port=8020,
                    source=run_dir / "missing-source.jsonl",
                    runtime_root=run_dir / "missing-runtime",
                    model_root=run_dir / "missing-model",
                    adapter=None,
                    max_gpu_memory_mib=512,
                    model_ready_timeout=1,
                )
                exit_status = remote_eval_supervisor.run(args)
            finally:
                remote_eval_supervisor.ALLOWED_RUN_ROOT = original_root
            self.assertEqual(exit_status, 1)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "failed")
            self.assertFalse(status["success"])
            self.assertEqual(status["exit_status"], 1)
            self.assertEqual(status["stage"], "preparing_input")
            self.assertIn("staged required file is missing", status["failure"]["message"])

    def test_result_gate_requires_runtime_and_model_identity_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="qwen3-remote-result-") as temporary:
            result = Path(temporary)
            manifest = {
                "protocol_version": "version26",
                "tool_scheme": "atomic",
                "assistant_carrier": "think-json-v1",
                "model": "qwen3-8b-atomic-v26-base",
                "base_url": "http://127.0.0.1:8020/v1",
                "dataset": "/tmp/derived.jsonl",
                "requested_size": 1534,
                "n_samples": 1,
                "pass_k": [1],
                "sample_workers": 1,
                "max_inflight_requests": 4,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 2048,
                "max_steps": 30,
                "enable_thinking": "1",
                "context_mode": "rolling-legal-history",
                "history_turns": 4,
                "rolling_prompt_variant": "full",
                "rolling_observation_style": "resident",
                "denotation_comparison": "bird-set",
            }
            (result / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (result / "summary.json").write_text(
                json.dumps({"total": 1534, "correct": 1, "accuracy": 1 / 1534}),
                encoding="utf-8",
            )
            with (result / "all.jsonl").open("w", encoding="utf-8") as handle:
                for index in range(1534):
                    handle.write(json.dumps({"example_index": index}) + "\n")
            (result / "version26_runtime_gate.json").write_text("{}\n", encoding="utf-8")
            (result / "qwen3_model_gate.json").write_text("{}\n", encoding="utf-8")
            report = remote_eval_supervisor.validate_final_result(
                result,
                served_model="qwen3-8b-atomic-v26-base",
                derived=Path("/tmp/derived.jsonl"),
                port=8020,
            )
            self.assertEqual(report["records"], 1534)
            self.assertEqual(report["api_error_count"], 0)

            records = [
                json.loads(line)
                for line in (result / "all.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            records[0]["samples"] = [
                {"sample_index": 0, "failure_type": "api_error", "error": "transport failed"}
            ]
            (result / "all.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "api_error_count=1"):
                remote_eval_supervisor.validate_final_result(
                    result,
                    served_model="qwen3-8b-atomic-v26-base",
                    derived=Path("/tmp/derived.jsonl"),
                    port=8020,
                )

            records[0]["samples"] = [
                {
                    "sample_index": 0,
                    "failure_type": "wrong_answer",
                    "error": "ChatAPIError: connection reset",
                }
            ]
            (result / "all.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "error_contains_ChatAPIError"):
                remote_eval_supervisor.validate_final_result(
                    result,
                    served_model="qwen3-8b-atomic-v26-base",
                    derived=Path("/tmp/derived.jsonl"),
                    port=8020,
                )

            records[0].pop("samples")
            (result / "all.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            (result / "qwen3_model_gate.json").unlink()
            with self.assertRaisesRegex(ValueError, "artifact is missing"):
                remote_eval_supervisor.validate_final_result(
                    result,
                    served_model="qwen3-8b-atomic-v26-base",
                    derived=Path("/tmp/derived.jsonl"),
                    port=8020,
                )

    def test_files_compile(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        for name in (
            "prepare_remote_eval_inputs.py",
            "verify_remote_eval_assets.py",
            "remote_eval_supervisor.py",
        ):
            subprocess.run(
                [sys.executable, "-m", "py_compile", str(HERE / name)], check=True
            )


if __name__ == "__main__":
    unittest.main()
