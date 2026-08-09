from __future__ import annotations

import io
import hashlib
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

from reproductions.trust_sql.qwen3_8b_atomic_sft1.analyze_paired_bird_dev import (
    EXPECTED_CHAT_TEMPLATE_SHA256,
    EXPECTED_QWEN3_REVISION,
    EXPECTED_REMOTE_ASSET_GATE_SCHEMA,
    EXPECTED_REMOTE_DATASET_SHA256,
    EXPECTED_RUNTIME_COMMIT,
    EXPECTED_SOURCE_DATASET_SHA256,
    EXPECTED_TOKENIZER_CONFIG_SHA256,
    build_analysis,
    exact_mcnemar_p,
    load_run,
    main,
    wilson_interval,
)


FIXTURE_SYSTEM_PROMPT = "frozen fixture system prompt"
FIXTURE_PROMPT_SHA256 = hashlib.sha256(FIXTURE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
PROMPT_HASH_PATCH_TARGET = (
    "reproductions.trust_sql.qwen3_8b_atomic_sft1.analyze_paired_bird_dev."
    "EXPECTED_STUDENT_PROMPT_SHA256"
)


def manifest(
    model: str,
    *,
    requested_size: int = 1534,
    dataset: str = "/frozen/bird_dev_20240627.jsonl",
) -> dict:
    return {
        "runner": "tool_rollout_passk",
        "tool_scheme": "atomic",
        "tool_scheme_registry_version": "tool-scheme-registry-v2",
        "assistant_carrier": "think-json-v1",
        "protocol_version": "version26",
        "protocol_hash": "4da19387399bd3a5",
        "model": model,
        "dataset": dataset,
        "requested_size": requested_size,
        "n_samples": 1,
        "stop_on_success": False,
        "pass_k": [1],
        "sample_detail": "full",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "max_steps": 30,
        "few_shot": 0,
        "enable_thinking": "1",
        "system_prompt_variant": "default",
        "system_prompt": FIXTURE_SYSTEM_PROMPT,
        "context_mode": "rolling-legal-history",
        "history_turns": 4,
        "rolling_prompt_variant": "full",
        "rolling_observation_style": "resident",
        "denotation_comparison": "bird-set",
        "dataset_purpose": "evaluation",
    }


def row(
    example_index: int,
    *,
    correct: bool,
    legal: bool,
    steps: int,
    errors: int,
) -> dict:
    error_events = [{"error_type": "protocol_error"} for _ in range(errors)]
    sample = {
        "tool_scheme": "atomic",
        "tool_scheme_registry_version": "tool-scheme-registry-v2",
        "assistant_carrier": "think-json-v1",
        "protocol_version": "version26",
        "protocol_hash": "4da19387399bd3a5",
        "sample_index": 0,
        "denotation_comparison": "bird-set",
        "correct": correct,
        "legal": legal,
        "steps": steps,
        "errors": errors,
        "failure_type": None if correct else ("wrong_answer" if legal else "protocol_error"),
        "error_events": error_events,
    }
    return {
        "tool_scheme": "atomic",
        "tool_scheme_registry_version": "tool-scheme-registry-v2",
        "assistant_carrier": "think-json-v1",
        "protocol_version": "version26",
        "protocol_hash": "4da19387399bd3a5",
        "example_index": example_index,
        "db_id": f"db_{example_index % 7}",
        "question": f"question {example_index}",
        "gold_sql": f"SELECT {example_index}",
        "n_samples": 1,
        "pass_k": [1],
        "temperature": 0.0,
        "top_p": 1.0,
        "denotation_comparison": "bird-set",
        "samples": [sample],
        "correct": correct,
        "sample_correct_count": int(correct),
        "sample_legal_count": int(legal),
        "pass_at": {"1": correct},
        "attempted_samples": 1,
    }


def runtime_gate(*, questions: int) -> dict:
    return {
        "status": "ok",
        "source_commit": EXPECTED_RUNTIME_COMMIT,
        "content_tree_sha256": "fixture-runtime-tree",
        "key_file_sha256": {"src/eval/rollout.py": "fixture-rollout-hash"},
        "isolation": "git-archive-exact-commit",
        "current_checkout_protocol_modules_imported": False,
        "reasoning_parser": None,
        "protocol": {
            "tool_scheme": "atomic",
            "tool_scheme_registry_version": "tool-scheme-registry-v2",
            "assistant_carrier": "think-json-v1",
            "protocol_version": "version26",
            "rolling_protocol_hash": "4da19387399bd3a5",
            "rolling_system_prompt_sha256": FIXTURE_PROMPT_SHA256,
        },
        "evaluation_contract": {
            "questions": questions,
            "rollouts_per_question": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_steps": 30,
            "minimum_max_tokens": 2048,
            "context_mode": "rolling-legal-history",
            "history_turns": 4,
            "rolling_observation_style": "resident",
            "denotation_comparison": "bird-set",
            "enable_thinking": True,
            "reasoning_parser": None,
        },
    }


def model_gate(*, adapter: bool) -> dict:
    adapter_report = None
    if adapter:
        adapter_report = {
            "path": "/frozen/qwen3-sft1/checkpoint-560",
            "peft_type": "LORA",
            "rank": 16,
            "base_model_name_or_path": "/frozen/Qwen3-8B",
        }
    return {
        "status": "ok",
        "model_root": "/frozen/Qwen3-8B",
        "assumed_huggingface_revision": EXPECTED_QWEN3_REVISION,
        "model_type": "qwen3",
        "architecture": "Qwen3ForCausalLM",
        "tokenizer_config_sha256": EXPECTED_TOKENIZER_CONFIG_SHA256,
        "chat_template_sha256": EXPECTED_CHAT_TEMPLATE_SHA256,
        "official_chat_template": True,
        "enable_thinking": True,
        "empty_think_suppression_injected": False,
        "reasoning_parser": None,
        "adapter": adapter_report,
    }


def remote_asset_gate(*, mode: str, runtime: dict, model: dict) -> dict:
    databases = {
        f"db_{index}": {
            "records": 139 if index < 10 else 144,
            "sha256": f"{index:064x}",
        }
        for index in range(11)
    }
    return {
        "status": "ok",
        "schema_version": EXPECTED_REMOTE_ASSET_GATE_SCHEMA,
        "mode": mode,
        "runtime": runtime,
        "evaluation_input": {
            "status": "ok",
            "source_sha256": EXPECTED_SOURCE_DATASET_SHA256,
            "derived_sha256": EXPECTED_REMOTE_DATASET_SHA256,
            "manifest_sha256": "fixture-derived-manifest",
            "records": 1534,
            "only_db_path_changed": True,
            "databases": databases,
        },
        "model": model,
        "evaluation_contract": {
            "questions": 1534,
            "n_samples": 1,
            "pass_k": "1",
            "temperature": 0,
            "top_p": 1,
            "max_steps": 30,
            "max_tokens": 2048,
            "context_mode": "rolling-legal-history",
            "history_turns": 4,
            "rolling_prompt_variant": "full",
            "rolling_observation_style": "resident",
            "denotation_comparison": "bird-set",
            "enable_thinking": True,
            "reasoning_parser": None,
            "max_num_seqs": 4,
            "workers": 4,
            "max_inflight_requests": 4,
        },
    }


def write_run(
    path: Path,
    *,
    model: str,
    rows: list[dict],
    adapter: bool = False,
    dataset: str = "/frozen/bird_dev_20240627.jsonl",
    remote_mode: str | None = None,
) -> None:
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps(
            manifest(model, requested_size=len(rows), dataset=dataset), sort_keys=True
        )
        + "\n",
        encoding="utf-8",
    )
    (path / "all.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in rows),
        encoding="utf-8",
    )
    runtime_report = runtime_gate(questions=len(rows))
    model_report = model_gate(adapter=adapter)
    (path / "version26_runtime_gate.json").write_text(
        json.dumps(runtime_report, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (path / "qwen3_model_gate.json").write_text(
        json.dumps(model_report, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if remote_mode is not None:
        (path / "remote_eval_asset_gate.json").write_text(
            json.dumps(
                remote_asset_gate(
                    mode=remote_mode,
                    runtime=runtime_report,
                    model=model_report,
                ),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


class StatisticsTest(unittest.TestCase):
    def test_wilson_known_values(self) -> None:
        low, high = wilson_interval(5, 10)
        self.assertAlmostEqual(low, 0.2365930905, places=9)
        self.assertAlmostEqual(high, 0.7634069095, places=9)
        zero_low, zero_high = wilson_interval(0, 10)
        self.assertEqual(zero_low, 0.0)
        self.assertAlmostEqual(zero_high, 0.2775327999, places=9)

    def test_exact_mcnemar(self) -> None:
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_p(3, 3), 1.0)
        self.assertEqual(exact_mcnemar_p(4, 0), 0.125)
        self.assertEqual(exact_mcnemar_p(10, 0), 0.001953125)


class PairedAnalysisTest(unittest.TestCase):
    def make_full_fixture(
        self, root: Path, *, remote: bool = False
    ) -> tuple[Path, Path]:
        base_rows = []
        adapter_rows = []
        adapter_correct = set(range(700)) | set(range(709, 744))
        for example_index in range(1534):
            base_rows.append(
                row(
                    example_index,
                    correct=example_index < 709,
                    legal=example_index < 1500,
                    steps=3,
                    errors=int(example_index < 100),
                )
            )
            adapter_rows.append(
                row(
                    example_index,
                    correct=example_index in adapter_correct,
                    legal=example_index < 1510,
                    steps=4,
                    errors=int(example_index < 80),
                )
            )
        base_path = root / "base"
        adapter_path = root / "adapter"
        base_dataset = "/frozen/bird_dev_20240627.jsonl"
        adapter_dataset = base_dataset
        if remote:
            base_dataset = "/remote/runs/base/input/bird_dev_20240627.newgnn.jsonl"
            adapter_dataset = "/remote/runs/adapter/input/bird_dev_20240627.newgnn.jsonl"
        write_run(
            base_path,
            model="qwen3-8b-atomic-v26-base",
            rows=base_rows,
            dataset=base_dataset,
            remote_mode="base" if remote else None,
        )
        write_run(
            adapter_path,
            model="qwen3-8b-atomic-v26-sft1-qlora",
            rows=adapter_rows,
            adapter=True,
            dataset=adapter_dataset,
            remote_mode="adapter" if remote else None,
        )
        return base_path, adapter_path

    @mock.patch(PROMPT_HASH_PATCH_TARGET, FIXTURE_PROMPT_SHA256)
    def test_full_analysis_thresholds_pairing_and_read_only_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base_path, adapter_path = self.make_full_fixture(root)
            base = load_run(base_path, label="base")
            adapter = load_run(adapter_path, label="adapter")
            analysis = build_analysis(base, adapter)

            self.assertEqual(analysis["base"]["ex"]["count"], 709)
            self.assertEqual(analysis["adapter"]["ex"]["count"], 735)
            self.assertEqual(analysis["base"]["process_errors"]["total"], 100)
            self.assertEqual(analysis["adapter"]["process_errors"]["total"], 80)
            self.assertEqual(analysis["base"]["steps"]["mean"], 3.0)
            self.assertEqual(analysis["adapter"]["steps"]["mean"], 4.0)

            paired = analysis["paired_accuracy"]
            self.assertEqual(paired["both_true"], 700)
            self.assertEqual(paired["both_false"], 790)
            self.assertEqual(paired["gains"], 35)
            self.assertEqual(paired["regressions"], 9)
            self.assertEqual(paired["net"], 26)
            self.assertEqual(paired["gain_example_indices"], list(range(709, 744)))
            self.assertEqual(paired["regression_example_indices"], list(range(700, 709)))

            local = analysis["reference_thresholds"][
                "author_protocol_raw_qwen3_8b_local_reproduction"
            ]
            paper = analysis["reference_thresholds"]["paper_raw_qwen3_8b_47_9_percent"]
            self.assertEqual(local["base"]["count_margin"], 0)
            self.assertTrue(local["base"]["meets_or_exceeds"])
            self.assertEqual(paper["base"]["count_margin"], -26)
            self.assertEqual(paper["adapter"]["count_margin"], 0)
            self.assertTrue(paper["adapter"]["meets_or_exceeds"])
            self.assertLess(
                analysis["adapter"]["ex"]["wilson_95_ci"]["low"],
                analysis["adapter"]["ex"]["rate"],
            )
            self.assertGreater(
                analysis["adapter"]["ex"]["wilson_95_ci"]["high"],
                analysis["adapter"]["ex"]["rate"],
            )

            before = {
                path: (path.read_bytes(), path.stat().st_mtime_ns)
                for path in (
                    base_path / "all.jsonl",
                    base_path / "manifest.json",
                    base_path / "version26_runtime_gate.json",
                    base_path / "qwen3_model_gate.json",
                    adapter_path / "all.jsonl",
                    adapter_path / "manifest.json",
                    adapter_path / "version26_runtime_gate.json",
                    adapter_path / "qwen3_model_gate.json",
                )
            }
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    main(
                        [
                            "--base",
                            str(base_path),
                            "--adapter",
                            str(adapter_path),
                            "--format",
                            "json",
                        ]
                    ),
                    0,
                )
            parsed = json.loads(stdout.getvalue())
            self.assertEqual(parsed["paired_accuracy"]["net"], 26)
            self.assertEqual(
                before,
                {
                    path: (path.read_bytes(), path.stat().st_mtime_ns)
                    for path in before
                },
            )
            self.assertEqual(
                set(base_path.iterdir()),
                {
                    base_path / "all.jsonl",
                    base_path / "manifest.json",
                    base_path / "version26_runtime_gate.json",
                    base_path / "qwen3_model_gate.json",
                },
            )
            self.assertEqual(
                set(adapter_path.iterdir()),
                {
                    adapter_path / "all.jsonl",
                    adapter_path / "manifest.json",
                    adapter_path / "version26_runtime_gate.json",
                    adapter_path / "qwen3_model_gate.json",
                },
            )

    @mock.patch(PROMPT_HASH_PATCH_TARGET, FIXTURE_PROMPT_SHA256)
    def test_incomplete_result_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_path = root / "incomplete"
            write_run(
                run_path,
                model="qwen3-8b-atomic-v26-base",
                rows=[row(0, correct=False, legal=False, steps=1, errors=0)],
            )
            (run_path / "manifest.json").write_text(
                json.dumps(
                    manifest("qwen3-8b-atomic-v26-base", requested_size=2),
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_path / "version26_runtime_gate.json").write_text(
                json.dumps(runtime_gate(questions=2), sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "must contain exactly indices"):
                load_run(run_path, label="base", expected_total=2)

    @mock.patch(PROMPT_HASH_PATCH_TARGET, FIXTURE_PROMPT_SHA256)
    def test_paired_manifest_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_path, adapter_path = self.make_full_fixture(Path(tmp))
            base = load_run(base_path, label="base")
            adapter = load_run(adapter_path, label="adapter")
            drifted_manifest = {**adapter.manifest, "max_tokens": 4096}
            adapter = replace(adapter, manifest=drifted_manifest)
            with self.assertRaisesRegex(ValueError, "evaluation manifests differ"):
                build_analysis(base, adapter)

    @mock.patch(PROMPT_HASH_PATCH_TARGET, FIXTURE_PROMPT_SHA256)
    def test_remote_run_dirs_may_differ_when_both_asset_gates_are_pinned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_path, adapter_path = self.make_full_fixture(Path(tmp), remote=True)
            base = load_run(base_path, label="base")
            adapter = load_run(adapter_path, label="adapter")
            analysis = build_analysis(base, adapter)
            identity = analysis["scope"]["dataset_identity"]
            self.assertFalse(identity["paths_equal"])
            self.assertEqual(identity["basename"], "bird_dev_20240627.newgnn.jsonl")
            self.assertTrue(identity["remote_asset_gated"])
            self.assertEqual(identity["source_sha256"], EXPECTED_SOURCE_DATASET_SHA256)
            self.assertEqual(identity["derived_sha256"], EXPECTED_REMOTE_DATASET_SHA256)
            self.assertTrue(identity["paired_gold_sql_hashes_equal"])
            self.assertIsNotNone(
                analysis["base"]["input"]["remote_eval_asset_gate_sha256"]
            )

    @mock.patch(PROMPT_HASH_PATCH_TARGET, FIXTURE_PROMPT_SHA256)
    def test_remote_dataset_requires_asset_gate_and_pinned_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_path, adapter_path = self.make_full_fixture(Path(tmp), remote=True)
            (base_path / "remote_eval_asset_gate.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing remote_eval_asset_gate"):
                load_run(base_path, label="base")

            adapter_gate_path = adapter_path / "remote_eval_asset_gate.json"
            adapter_gate = json.loads(adapter_gate_path.read_text(encoding="utf-8"))
            adapter_gate["evaluation_input"]["derived_sha256"] = "0" * 64
            adapter_gate_path.write_text(json.dumps(adapter_gate) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "derived_sha256 drifted"):
                load_run(adapter_path, label="adapter")

    @mock.patch(PROMPT_HASH_PATCH_TARGET, FIXTURE_PROMPT_SHA256)
    def test_legacy_paths_must_match_and_gold_sql_is_paired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_path, adapter_path = self.make_full_fixture(Path(tmp))
            base = load_run(base_path, label="base")
            adapter = load_run(adapter_path, label="adapter")

            different_path = replace(
                adapter,
                manifest={
                    **adapter.manifest,
                    "dataset": "/another-host/bird_dev_20240627.jsonl",
                },
            )
            with self.assertRaisesRegex(ValueError, "require both remote asset gates"):
                build_analysis(base, different_path)

            changed_tasks = dict(adapter.tasks)
            changed_tasks[0] = replace(
                changed_tasks[0], gold_sql_sha256="0" * 64
            )
            changed_gold = replace(adapter, tasks=changed_tasks)
            with self.assertRaisesRegex(ValueError, "gold_sql hash differs"):
                build_analysis(base, changed_gold)

    @mock.patch(PROMPT_HASH_PATCH_TARGET, FIXTURE_PROMPT_SHA256)
    def test_error_count_must_match_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad_rows = [row(0, correct=False, legal=False, steps=1, errors=1)]
            bad_rows[0]["samples"][0]["error_events"] = []
            run_path = root / "bad"
            write_run(
                run_path,
                model="qwen3-8b-atomic-v26-base",
                rows=bad_rows,
            )
            with self.assertRaisesRegex(ValueError, "has 0 error events"):
                load_run(run_path, label="base", expected_total=1)


if __name__ == "__main__":
    unittest.main()
