from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import tempfile
import unittest
from pathlib import Path

from reproductions.trust_sql.qwen3_8b_atomic_sft1 import archive_experiment_artifacts as archive


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class SyntheticExperiment:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.model = root / "model"
        self.model.mkdir()
        shard_hashes = {}
        for index in range(1, 6):
            name = f"model-{index:05d}-of-00005.safetensors"
            path = self.model / name
            path.write_bytes((f"fixed shard {index}\n" * index).encode())
            shard_hashes[name] = sha256(path)

        self.dataset = root / "data" / "train.jsonl"
        self.dataset.parent.mkdir()
        self.dataset.write_text('{"target": 1}\n{"target": 2}\n', encoding="utf-8")
        self.dataset_manifest = self.dataset.with_suffix(".manifest.json")
        write_json(
            self.dataset_manifest,
            {
                "output": str(self.dataset),
                "output_sha256": sha256(self.dataset),
                "records": 2,
            },
        )
        self.dataset_info = self.dataset.parent / "dataset_info.json"
        write_json(self.dataset_info, {"train": {"file_name": self.dataset.name}})
        self.preparation = self.dataset.parent / "preparation_manifest.json"
        write_json(
            self.preparation,
            {"files": {str(self.dataset): sha256(self.dataset)}},
        )
        self.config = root / "train.yaml"
        self.config.write_text("model: synthetic\n", encoding="utf-8")

        self.output_dir = root / "output"
        self.checkpoint = self.output_dir / "checkpoint-2"
        self.checkpoint.mkdir(parents=True)
        (self.checkpoint / "adapter_model.safetensors").write_bytes(b"adapter weights")
        write_json(
            self.checkpoint / "adapter_config.json",
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "base_model_name_or_path": str(self.model),
            },
        )
        write_json(self.checkpoint / "trainer_state.json", {"global_step": 2})
        (self.checkpoint / "training_args.bin").write_bytes(b"training args")

        self.launch = root / "logs" / "run.launch_manifest.json"
        launch_value = {
            "run_kind": "full",
            "experiment": "qwen3-8b-historical-atomic-version26-sft1-qlora",
            "model_repo": "Qwen/Qwen3-8B",
            "model_revision": "synthetic-revision",
            "model": str(self.model),
            "model_shards_sha256": shard_hashes,
            "config": str(self.config),
            "config_sha256": sha256(self.config),
            "dataset": str(self.dataset),
            "dataset_sha256": sha256(self.dataset),
            "dataset_manifest": str(self.dataset_manifest),
            "dataset_manifest_sha256": sha256(self.dataset_manifest),
            "dataset_info": str(self.dataset_info),
            "dataset_info_sha256": sha256(self.dataset_info),
            "preparation_manifest": str(self.preparation),
            "preparation_manifest_sha256": sha256(self.preparation),
            "records": 2,
            "world_size": 2,
            "effective_global_batch": 16,
            "expected_optimizer_steps": 2,
            "output_dir": str(self.output_dir),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "torch": "2.6.0+cu124",
            "transformers": "5.6.0",
            "peft": "0.18.1",
            "bitsandbytes": "0.46.1",
            "datasets": "4.0.0",
            "llamafactory": "0.9.5",
        }
        write_json(self.launch, launch_value)
        self.status = root / "logs" / "run.status.json"
        write_json(
            self.status,
            {
                "success": True,
                "exit_status": 0,
                "run_kind": "full",
                "expected_global_step": 2,
                "output_dir": str(self.output_dir),
                "log": str(root / "logs" / "run.log"),
                "adapter_model": str(self.output_dir / "adapter_model.safetensors"),
            },
        )
        self.log = root / "logs" / "run.log"
        self.log.write_text("final global_step gate passed: 2\n", encoding="utf-8")

        self.prompt = "synthetic frozen version26 prompt"
        self.evaluation_source_sha256 = "a" * 64
        self.evaluation_derived_sha256 = "b" * 64
        self.base_result = root / "base-result"
        self.adapter_result = root / "adapter-result"
        self._write_eval(self.base_result, adapter=False)
        self._write_eval(self.adapter_result, adapter=True)

        self.contract = archive.ExperimentContract(
            model_revision="synthetic-revision",
            shard_sha256=shard_hashes,
            launch_manifest_sha256=sha256(self.launch),
            dataset_sha256=sha256(self.dataset),
            dataset_manifest_sha256=sha256(self.dataset_manifest),
            preparation_manifest_sha256=sha256(self.preparation),
            expected_training_records=2,
            expected_global_step=2,
            expected_eval_tasks=2,
            evaluation_source_sha256=self.evaluation_source_sha256,
            evaluation_derived_sha256=self.evaluation_derived_sha256,
            version26_source_commit="synthetic-version26-commit",
            version26_content_tree_sha256="synthetic-runtime-tree",
            student_prompt_sha256=hashlib.sha256(self.prompt.encode()).hexdigest(),
            tokenizer_config_sha256="synthetic-tokenizer",
            chat_template_sha256="synthetic-template",
            training_python_version=platform.python_version(),
            evaluation_python_version=platform.python_version(),
            training_packages={},
            evaluation_packages={},
        )

    def write_remote_asset_gate(
        self,
        root: Path,
        *,
        mode: str,
        derived_sha256: str | None = None,
    ) -> Path:
        gate = {
            "status": "ok",
            "schema_version": "qwen3-atomic-v26-all-newgnn-assets-v1",
            "mode": mode,
            "runtime": {},
            "evaluation_input": {
                "source_sha256": self.evaluation_source_sha256,
                "derived_sha256": derived_sha256 or self.evaluation_derived_sha256,
                "records": 2,
                "only_db_path_changed": True,
                "remote_db_root": "/remote/bird",
                "databases": {
                    "db_one": {"records": 1, "sha256": "1" * 64},
                    "db_two": {"records": 1, "sha256": "2" * 64},
                },
            },
            "model": {},
            "evaluation_contract": {},
        }
        path = root / "remote_eval_asset_gate.json"
        write_json(path, gate)
        return path

    def _write_eval(self, root: Path, *, adapter: bool) -> None:
        root.mkdir()
        manifest = dict(archive.EVAL_MANIFEST_LOCK)
        manifest.update(
            {
                "tool_scheme_registry_version": "tool-scheme-registry-v2",
                "protocol_hash": "synthetic-protocol-hash",
                "dataset": "synthetic-bird-dev.jsonl",
                "requested_size": 2,
                "max_tokens": 2048,
                "enable_thinking": "1",
                "system_prompt_variant": "default",
                "system_prompt": self.prompt,
                "model": "adapter" if adapter else "base",
            }
        )
        write_json(root / "manifest.json", manifest)
        runtime_gate = {
            "status": "ok",
            "source_commit": "synthetic-version26-commit",
            "content_tree_sha256": "synthetic-runtime-tree",
            "reasoning_parser": None,
            "protocol": {
                "protocol_version": "version26",
                "tool_scheme": "atomic",
                "assistant_carrier": "think-json-v1",
                "rolling_system_prompt_sha256": hashlib.sha256(self.prompt.encode()).hexdigest(),
            },
        }
        write_json(root / "version26_runtime_gate.json", runtime_gate)
        model_gate = {
            "status": "ok",
            "assumed_huggingface_revision": "synthetic-revision",
            "model_type": "qwen3",
            "architecture": "Qwen3ForCausalLM",
            "tokenizer_config_sha256": "synthetic-tokenizer",
            "chat_template_sha256": "synthetic-template",
            "official_chat_template": True,
            "enable_thinking": True,
            "reasoning_parser": None,
            "adapter": (
                {
                    "path": str(self.checkpoint),
                    "peft_type": "LORA",
                    "rank": 16,
                    "base_model_name_or_path": str(self.model),
                }
                if adapter
                else None
            ),
        }
        write_json(root / "qwen3_model_gate.json", model_gate)
        with (root / "all.jsonl").open("w", encoding="utf-8") as handle:
            for index in range(2):
                row = {
                    "example_index": index,
                    "tool_scheme": "atomic",
                    "protocol_version": "version26",
                    "samples": [{"correct": index == 0, "legal": True}],
                }
                handle.write(json.dumps(row) + "\n")

    def args(self, mode: str = "final") -> argparse.Namespace:
        return argparse.Namespace(
            mode=mode,
            training_host=None,
            evaluation_host=None,
            remote_helper_python=sys.executable,
            rehash_base_shards_in_draft=False,
            training_launch_manifest=str(self.launch),
            training_status_manifest=str(self.status),
            training_log=str(self.log),
            base_model_root=None,
            checkpoint_dir=None,
            training_dataset=None,
            training_dataset_manifest=None,
            preparation_manifest=None,
            dataset_info=None,
            training_python=sys.executable,
            evaluation_python=sys.executable,
            base_result_dir=str(self.base_result),
            adapter_result_dir=str(self.adapter_result),
        )


class ArchiveExperimentArtifactsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.fixture = SyntheticExperiment(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_complete_final_archive_passes_and_hashes_required_files(self) -> None:
        manifest, status = archive.collect_archive(
            self.fixture.args("final"), self.fixture.contract
        )
        self.assertEqual(status, 0)
        self.assertEqual(manifest["status"], "ok")
        self.assertTrue(manifest["complete"])
        checkpoint = manifest["artifacts"]["training"]["checkpoint_560"]
        self.assertEqual(checkpoint["trainer_state_json"]["global_step"], 2)
        self.assertEqual(
            checkpoint["files"]["training_args.bin"]["sha256"],
            sha256(self.fixture.checkpoint / "training_args.bin"),
        )
        self.assertEqual(
            manifest["artifacts"]["evaluations"]["base"]["all_jsonl"]["records"], 2
        )
        self.assertTrue(manifest["cross_run_checks"]["runtime_gate_sha256_equal"])

    def test_draft_explicitly_reports_missing_adapter_evaluation(self) -> None:
        (self.fixture.adapter_result / "all.jsonl").unlink()
        manifest, status = archive.collect_archive(
            self.fixture.args("draft"), self.fixture.contract
        )
        self.assertEqual(status, 0)
        self.assertEqual(manifest["status"], "incomplete")
        self.assertFalse(manifest["complete"])
        codes = {item["code"] for item in manifest["issues"]["incomplete"]}
        self.assertIn("missing_artifact", codes)

    def test_final_mode_fails_closed_on_missing_artifact(self) -> None:
        (self.fixture.checkpoint / "training_args.bin").unlink()
        manifest, status = archive.collect_archive(
            self.fixture.args("final"), self.fixture.contract
        )
        self.assertEqual(status, 2)
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse(manifest["final_ready"])

    def test_shard_hash_drift_is_invalid_even_in_draft_mode(self) -> None:
        shard = self.fixture.model / "model-00003-of-00005.safetensors"
        shard.write_bytes(b"mutated shard")
        args = self.fixture.args("draft")
        args.rehash_base_shards_in_draft = True
        manifest, status = archive.collect_archive(
            args, self.fixture.contract
        )
        self.assertEqual(status, 2)
        self.assertEqual(manifest["status"], "invalid")
        codes = {item["code"] for item in manifest["issues"]["errors"]}
        self.assertIn("sha256_mismatch", codes)

    def test_partial_evaluation_is_incomplete_not_complete(self) -> None:
        all_jsonl = self.fixture.base_result / "all.jsonl"
        all_jsonl.write_text(all_jsonl.read_text(encoding="utf-8").splitlines()[0] + "\n")
        manifest, status = archive.collect_archive(
            self.fixture.args("draft"), self.fixture.contract
        )
        self.assertEqual(status, 0)
        self.assertEqual(manifest["status"], "incomplete")
        codes = {item["code"] for item in manifest["issues"]["incomplete"]}
        self.assertIn("evaluation_incomplete", codes)

    def test_manifest_writer_refuses_implicit_overwrite(self) -> None:
        path = Path(self.temporary.name) / "archive.json"
        archive._write_manifest(path, {"status": "one"}, overwrite=False)
        with self.assertRaises(FileExistsError):
            archive._write_manifest(path, {"status": "two"}, overwrite=False)
        self.assertEqual(json.loads(path.read_text())["status"], "one")

    def test_remote_asset_hash_replaces_absolute_dataset_path_equality(self) -> None:
        self.fixture.write_remote_asset_gate(self.fixture.base_result, mode="base")
        self.fixture.write_remote_asset_gate(self.fixture.adapter_result, mode="adapter")
        base_manifest_path = self.fixture.base_result / "manifest.json"
        adapter_manifest_path = self.fixture.adapter_result / "manifest.json"
        base_manifest = json.loads(base_manifest_path.read_text())
        adapter_manifest = json.loads(adapter_manifest_path.read_text())
        base_manifest["dataset"] = "/remote/run-base/input.newgnn.jsonl"
        adapter_manifest["dataset"] = "/remote/run-adapter/input.newgnn.jsonl"
        write_json(base_manifest_path, base_manifest)
        write_json(adapter_manifest_path, adapter_manifest)

        manifest, status = archive.collect_archive(
            self.fixture.args("final"), self.fixture.contract
        )
        self.assertEqual(status, 0)
        comparison = manifest["cross_run_checks"]
        self.assertEqual(
            comparison["dataset_identity_strategy"],
            "remote_eval_asset_gate_derived_sha256",
        )
        self.assertTrue(comparison["derived_sha256_equal"])
        self.assertTrue(comparison["equal"])
        archived_gate = manifest["artifacts"]["evaluations"]["base"][
            "remote_eval_asset_gate"
        ]
        self.assertEqual(
            archived_gate["sha256"],
            sha256(self.fixture.base_result / "remote_eval_asset_gate.json"),
        )

    def test_remote_asset_gate_rejects_wrong_mode_and_derived_hash(self) -> None:
        self.fixture.write_remote_asset_gate(
            self.fixture.base_result,
            mode="adapter",
            derived_sha256="c" * 64,
        )
        self.fixture.write_remote_asset_gate(self.fixture.adapter_result, mode="adapter")
        manifest, status = archive.collect_archive(
            self.fixture.args("final"), self.fixture.contract
        )
        self.assertEqual(status, 2)
        self.assertEqual(manifest["status"], "failed")
        codes = {item["code"] for item in manifest["issues"]["errors"]}
        self.assertIn("remote_eval_asset_gate_mismatch", codes)
        self.assertIn("evaluation_dataset_hashes_differ", codes)

    def test_one_remote_asset_gate_is_not_accepted_as_a_complete_pair(self) -> None:
        self.fixture.write_remote_asset_gate(self.fixture.base_result, mode="base")
        manifest, status = archive.collect_archive(
            self.fixture.args("final"), self.fixture.contract
        )
        self.assertEqual(status, 2)
        codes = {item["code"] for item in manifest["issues"]["incomplete"]}
        self.assertIn("remote_eval_asset_gate_pair_incomplete", codes)


if __name__ == "__main__":
    unittest.main()
