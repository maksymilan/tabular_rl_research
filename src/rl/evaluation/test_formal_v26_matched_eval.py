from __future__ import annotations

import copy
import hashlib
import inspect
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.rl.evaluation import formal_v26_matched_eval as formal


CONTRACT_PATH = (
    Path(__file__).resolve().parent
    / "qwen3_8b_v26_vanilla_formal_matched_contract.json"
)


def load_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_contract_and_implementation_are_pinned() -> None:
    contract = load_contract()
    formal.validate_contract_shape(contract)
    assert formal.verify_implementation(contract) == contract["implementation_sha256"]


def test_contract_rejects_historical_concurrency_four() -> None:
    contract = load_contract()
    contract["evaluation"]["workers"] = 4
    with pytest.raises(ValueError, match="evaluation.workers"):
        formal.validate_contract_shape(contract)


def test_task_identity_and_derived_input_match_frozen_hashes(tmp_path: Path) -> None:
    contract = load_contract()
    source = formal.PROJECT_ROOT / "data/eval_inputs/bird_dev_20240627.jsonl"
    rows = formal.load_jsonl(source)
    expected = contract["input"]["task_identity_sha256_without_db_path"]
    assert formal.task_identity_without_db_path(rows) == expected

    only_path_changed = copy.deepcopy(rows)
    only_path_changed[0]["db_path"] = "/different/root.sqlite"
    assert formal.task_identity_without_db_path(only_path_changed) == expected

    field_changed = copy.deepcopy(rows)
    field_changed[0]["question"] += " drift"
    assert formal.task_identity_without_db_path(field_changed) != expected
    assert formal.task_identity_without_db_path(list(reversed(rows))) != expected

    output = tmp_path / "bird_dev_20240627.newgnn.jsonl"
    manifest = tmp_path / "bird_dev_20240627.newgnn.manifest.json"
    formal.prepare_derived_input(
        contract,
        rows,
        Path(contract["host_paths"]["database_root"]),
        output,
        manifest,
    )
    assert formal.sha256_file(output) == contract["input"]["derived_sha256"]
    assert (
        formal.sha256_file(manifest)
        == contract["input"]["derived_manifest_sha256"]
    )


def test_commands_fix_matched_concurrency_and_forbid_resume() -> None:
    evaluator = formal.build_evaluator_command(
        Path("/python"),
        Path("/wrapper.py"),
        Path("/input.jsonl"),
        Path("/result"),
        port=8087,
        served_model="formal-final",
    )
    assert evaluator[evaluator.index("--workers") + 1] == "24"
    assert evaluator[evaluator.index("--max-inflight-requests") + 1] == "24"
    assert evaluator[evaluator.index("--n") + 1] == "1534"
    assert evaluator[evaluator.index("--n-samples") + 1] == "1"
    assert "--resume" not in evaluator
    assert "--stop-on-success" not in evaluator

    vllm = formal.build_vllm_command(
        Path("/python"),
        Path("/model"),
        Path("/adapter"),
        port=8087,
        served_model="formal-final",
        backbone_name="formal-backbone",
    )
    assert vllm[vllm.index("--max-num-seqs") + 1] == "24"
    assert vllm[vllm.index("--max-model-len") + 1] == "16384"
    assert "--reasoning-parser" not in vllm
    assert "--chat-template" not in vllm


def make_result_fixture(tmp_path: Path) -> tuple[dict, Path, list[dict]]:
    contract = load_contract()
    prompt = "frozen prompt"
    contract["runtime"]["student_prompt_sha256"] = hashlib.sha256(
        prompt.encode()
    ).hexdigest()
    derived = tmp_path / "input.jsonl"
    input_rows = [
        {
            "example_index": index,
            "db_id": "db",
            "question": f"question {index}",
            "gold_sql": f"SELECT {index}",
            "split": "dev",
            "trajectory_id": None,
        }
        for index in range(1534)
    ]
    write_jsonl(derived, input_rows)
    rows = []
    for source in input_rows:
        sample = {
            "sample_index": 0,
            "correct": True,
            "legal": True,
            "steps": 1,
            "errors": 0,
            "failure_type": None,
            "turns": [],
        }
        rows.append(
            {
                "example_index": source["example_index"],
                "db_id": source["db_id"],
                "question": source["question"],
                "gold_sql": source["gold_sql"],
                "dataset_split": "dev",
                "trajectory_id": None,
                "protocol_version": "version26",
                "protocol_hash": "4da19387399bd3a5",
                "temperature": 0.0,
                "top_p": 1.0,
                "denotation_comparison": "bird-set",
                "n_samples": 1,
                "pass_k": [1],
                "max_steps": 30,
                "max_tokens": 2048,
                "correct": True,
                "pass_at": {"1": True},
                "samples": [sample],
            }
        )
    result = tmp_path / "result"
    write_jsonl(result / "all.jsonl", rows)
    manifest = {
        "runner": "tool_rollout_passk",
        "tool_scheme": "atomic",
        "assistant_carrier": "think-json-v1",
        "protocol_version": "version26",
        "protocol_hash": "4da19387399bd3a5",
        "model": "formal-final",
        "base_url": "http://127.0.0.1:8087/v1",
        "dataset": str(derived),
        "requested_size": 1534,
        "n_samples": 1,
        "sample_workers": 1,
        "first_sample_workers": 0,
        "max_inflight_requests": 24,
        "stop_on_success": False,
        "pass_k": [1],
        "sample_detail": "full",
        "summary_every": 10,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "max_steps": 30,
        "api_retries": 3,
        "few_shot": 0,
        "enable_thinking": "1",
        "context_mode": "rolling-legal-history",
        "history_turns": 4,
        "rolling_prompt_variant": "full",
        "rolling_observation_style": "resident",
        "denotation_comparison": "bird-set",
        "system_prompt": prompt,
    }
    manifest["config_sha256"] = hashlib.sha256(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest["created_at_utc"] = "test"
    write_json(result / "manifest.json", manifest)
    write_json(
        result / "summary.json",
        {
            "total": 1534,
            "correct": 1534,
            "completed_example_indices": list(range(1534)),
            "pass_at": {"1": {"correct": 1534, "total": 1534, "rate": 1.0}},
            "average_legal_samples": 1.0,
        },
    )
    return contract, result, input_rows


def test_result_validation_is_exact_and_source_bound(tmp_path: Path) -> None:
    contract, result, input_rows = make_result_fixture(tmp_path)
    report = formal.validate_result(
        contract,
        result,
        input_rows,
        derived=tmp_path / "input.jsonl",
        served_model="formal-final",
        port=8087,
    )
    assert report["records"] == report["correct"] == report["legal"] == 1534

    rows = formal.load_jsonl(result / "all.jsonl")
    rows[0]["question"] = "drift"
    write_jsonl(result / "all.jsonl", rows)
    with pytest.raises(ValueError, match="question mismatch"):
        formal.validate_result(
            contract,
            result,
            input_rows,
            derived=tmp_path / "input.jsonl",
            served_model="formal-final",
            port=8087,
        )


def test_result_validation_rejects_extra_rows_and_api_errors(tmp_path: Path) -> None:
    contract, result, input_rows = make_result_fixture(tmp_path)
    rows = formal.load_jsonl(result / "all.jsonl")
    extra = copy.deepcopy(rows[0])
    extra["example_index"] = 1534
    write_jsonl(result / "all.jsonl", rows + [extra])
    with pytest.raises(ValueError, match="result count"):
        formal.validate_result(
            contract,
            result,
            input_rows,
            derived=tmp_path / "input.jsonl",
            served_model="formal-final",
            port=8087,
        )

    rows[0]["samples"][0]["failure_type"] = "api_error"
    rows[0]["samples"][0]["error"] = "ChatAPIError: timeout"
    write_jsonl(result / "all.jsonl", rows)
    with pytest.raises(ValueError, match="API errors"):
        formal.validate_result(
            contract,
            result,
            input_rows,
            derived=tmp_path / "input.jsonl",
            served_model="formal-final",
            port=8087,
        )


def make_training_fixture(tmp_path: Path) -> tuple[dict, Path, Path, dict]:
    contract = load_contract()
    training = tmp_path / "training"
    base = tmp_path / "base"
    base.mkdir()
    adapter_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "base_model_name_or_path": str(base),
        "bias": "none",
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["v_proj", "q_proj", "k_proj", "o_proj"],
        "eva_config": None,
        "trainable_token_indices": None,
    }
    sft1_config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "base_model_name_or_path": str(base),
        "bias": "none",
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "o_proj", "v_proj"],
    }
    config_payload = json.dumps(adapter_config) + "\n"
    for directory in (training / "checkpoint-20", training / "final"):
        directory.mkdir(parents=True)
        (directory / "adapter_model.safetensors").write_bytes(b"final weights")
        (directory / "adapter_config.json").write_text(config_payload)
    write_json(training / "checkpoint-20/trainer_state.json", {"global_step": 20})

    implementation_payload = b"pinned source\n"
    implementation_hash = hashlib.sha256(implementation_payload).hexdigest()
    snapshot = training / "implementation_source_snapshot/src/rl/pinned.py"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_bytes(implementation_payload)
    reference = {
        "enabled": False,
        "kl_beta": 0.0,
        "adapter_name": None,
        "adapter_path": None,
        "adapter_sha256": None,
        "expected_adapter_sha256": contract["sft1"]["files_sha256"][
            "adapter_model.safetensors"
        ],
        "equals_initial_adapter": True,
        "load_audit": None,
        "after_trainer_init_audit": None,
    }
    runtime_root = contract["host_paths"]["runtime"]
    manifest = {
        **contract["training_final"]["expected_manifest"],
        "experiment_config_sha256": contract["training_final"][
            "experiment_config_sha256"
        ],
        "runtime_identity_audit": {
            "expected": {
                "runtime_content_tree_sha256": contract["runtime"][
                    "content_tree_sha256"
                ]
            },
            "actual": {
                "runtime_content_tree_sha256": contract["runtime"][
                    "content_tree_sha256"
                ]
            },
        },
        "runtime_module_audit": {
            "runtime_root": runtime_root,
            "tool_environment_factory_module": "tool_environment_v26",
            "module_paths": {
                "protocol": runtime_root + "/src/sft/protocol.py",
                "rollout": runtime_root + "/src/eval/rollout.py",
                "executor": runtime_root + "/src/harness/executor.py",
                "tool_schemes": runtime_root + "/src/sft/tool_schemes.py",
            },
        },
        "rollout_settings": {
            "max_steps": 30,
            "max_new_tokens": 2048,
            "max_context_tokens": 16384,
            "history_turns": 4,
            "enable_thinking": True,
            "tool_execution_timeout_seconds": 10.0,
        },
        "reference_policy": reference,
        "base_model_identity": contract["model"]["base_model_identity"],
        "implementation_source_sha256": {"src/rl/pinned.py": implementation_hash},
    }
    write_json(training / "run_manifest.json", manifest)
    write_json(
        training / "implementation_lock.json",
        {
            "files": manifest["implementation_source_sha256"],
            "initial_adapter_sha256": manifest["initial_adapter_sha256"],
            "protocol_version": manifest["protocol_version"],
            "protocol_hash": manifest["protocol_hash"],
            "student_prompt_sha256": manifest["student_prompt_sha256"],
            "reference_policy": reference,
            "base_model_identity": manifest["base_model_identity"],
            "experiment_config_sha256": manifest["experiment_config_sha256"],
        },
    )
    write_json(
        training / "training_precision.json",
        {
            "schema_version": "table-agent-trl-training-precision-v1",
            "optimizer_name": "adamw_torch",
            "trainable_parameters": {
                "trainable_tensors": 2,
                "non_fp32_tensors": [],
            },
            "optimizer_state": {"moment_tensors": 4, "non_fp32_moments": []},
        },
    )
    sft1 = {
        "adapter_sha256": contract["sft1"]["files_sha256"][
            "adapter_model.safetensors"
        ],
        "adapter_config_sha256": hashlib.sha256(
            (json.dumps(sft1_config) + "\n").encode()
        ).hexdigest(),
        "adapter_config_semantic_sha256": formal.adapter_config_semantic_sha256(
            sft1_config
        ),
    }
    return contract, training, base, sft1


def test_final_is_bound_to_checkpoint20_and_training_locks(tmp_path: Path) -> None:
    contract, training, base, sft1 = make_training_fixture(tmp_path)
    report = formal.verify_training_final(contract, training, base, sft1)
    assert report["checkpoint_global_step"] == 20
    assert report["adapter_sha256"] != sft1["adapter_sha256"]
    assert report["adapter_config_sha256"] != sft1["adapter_config_sha256"]
    assert (
        report["adapter_config_semantic_sha256"]
        == sft1["adapter_config_semantic_sha256"]
    )

    (training / "final/adapter_model.safetensors").write_bytes(b"drift")
    with pytest.raises(ValueError, match="final/checkpoint-20 adapter weights"):
        formal.verify_training_final(contract, training, base, sft1)


def test_adapter_config_normalization_is_narrow_and_semantic() -> None:
    sft1 = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "loftq_config": {"bits": None},
    }
    resaved = {
        **sft1,
        "target_modules": ["o_proj", "v_proj", "k_proj", "q_proj"],
        "eva_config": None,
        "trainable_token_indices": None,
        "loftq_config": {"bits": None, "group_size": None},
    }
    assert formal.adapter_config_semantic_sha256(
        sft1
    ) == formal.adapter_config_semantic_sha256(resaved)

    changed_value = {**resaved, "lora_alpha": 64}
    assert formal.adapter_config_semantic_sha256(
        sft1
    ) != formal.adapter_config_semantic_sha256(changed_value)

    changed_target_set = {
        **resaved,
        "target_modules": ["q_proj", "k_proj", "v_proj"],
    }
    assert formal.adapter_config_semantic_sha256(
        sft1
    ) != formal.adapter_config_semantic_sha256(changed_target_set)

    nonnull_new_field = {**resaved, "eva_config": {"rho": 0.5}}
    assert formal.adapter_config_semantic_sha256(
        sft1
    ) != formal.adapter_config_semantic_sha256(nonnull_new_field)


def test_training_rejects_real_adapter_semantic_drift(tmp_path: Path) -> None:
    contract, training, base, sft1 = make_training_fixture(tmp_path)
    changed = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 16,
        "base_model_name_or_path": str(base),
        "bias": "none",
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "o_proj", "v_proj"],
    }
    sft1["adapter_config_semantic_sha256"] = (
        formal.adapter_config_semantic_sha256(changed)
    )
    with pytest.raises(ValueError, match="adapter config semantics"):
        formal.verify_training_final(contract, training, base, sft1)


def test_train600_contract_rejects_boundary300_training_arm(tmp_path: Path) -> None:
    contract, training, base, sft1 = make_training_fixture(tmp_path)
    manifest = formal.load_object(training / "run_manifest.json")
    manifest["records"] = 300
    manifest["expected_records"] = 300
    manifest["prompts_per_update"] = 15
    write_json(training / "run_manifest.json", manifest)

    with pytest.raises(ValueError, match="training manifest records mismatch"):
        formal.verify_training_final(contract, training, base, sft1)


def test_identities_match_every_field_except_adapter_and_operations() -> None:
    contract = load_contract()
    implementation = contract["implementation_sha256"]
    common = {
        "contract": contract,
        "runtime": Path(contract["host_paths"]["runtime"]),
        "model": Path(contract["host_paths"]["base_model"]),
        "derived": Path("/frozen/input.jsonl"),
        "versions": {"python": "3.11", "vllm": "0.19.1"},
        "implementation": implementation,
        "gpu_id": 7,
        "port": 8087,
    }
    config_hash = contract["sft1"]["files_sha256"]["adapter_config.json"]
    semantic_hash = "c" * 64
    sft = formal.evaluation_identity(
        **common,
        arm="sft1",
        adapter={
            "path": "/adapter/sft1",
            "adapter_sha256": "a" * 64,
            "adapter_config_sha256": config_hash,
            "adapter_config_semantic_sha256": semantic_hash,
        },
        served_model="sft1-served",
    )
    final = formal.evaluation_identity(
        **common,
        arm="final",
        adapter={
            "path": "/adapter/final",
            "adapter_sha256": "b" * 64,
            "adapter_config_sha256": "d" * 64,
            "adapter_config_semantic_sha256": semantic_hash,
        },
        served_model="final-served",
    )
    assert sft["adapter_sha256"] != final["adapter_sha256"]
    assert sft["adapter_config_sha256"] != final["adapter_config_sha256"]
    for field in formal.MATCHED_IDENTITY_FIELDS:
        assert sft[field] == final[field], field
    assert (
        sft["base_model_identity_sha256"]
        == contract["model"]["base_model_identity"]["aggregate_sha256"]
    )


def test_busy_compute_process_or_memory_fails_without_waiting() -> None:
    compute_busy = [
        subprocess.CompletedProcess([], 0, stdout="100\n", stderr=""),
        subprocess.CompletedProcess([], 0, stdout="4242\n", stderr=""),
    ]
    with patch.object(formal.subprocess, "run", side_effect=compute_busy):
        with pytest.raises(ValueError, match="compute processes"):
            formal.require_idle_gpu(0, 512)

    memory_busy = [
        subprocess.CompletedProcess([], 0, stdout="1024\n", stderr=""),
        subprocess.CompletedProcess([], 0, stdout="", stderr=""),
    ]
    with patch.object(formal.subprocess, "run", side_effect=memory_busy):
        with pytest.raises(ValueError, match="not safely idle"):
            formal.require_idle_gpu(0, 512)


def test_inter_arm_port_release_waits_for_transient_loopback_release() -> None:
    with (
        patch.object(
            formal,
            "require_free_port",
            side_effect=[
                ValueError("localhost port 8087 is already in use"),
                ValueError("localhost port 8087 is already in use"),
                None,
            ],
        ) as require_port,
        patch.object(
            formal.time,
            "monotonic",
            side_effect=[100.0, 100.1, 101.1, 102.1],
        ),
        patch.object(formal.time, "sleep") as sleep,
    ):
        report = formal.wait_for_released_port(
            8087, timeout_seconds=10.0, poll_interval_seconds=1.0
        )

    assert report["attempts"] == 3
    assert report["waited_seconds"] == pytest.approx(2.1)
    assert report["policy"] == (
        "bounded_wait_after_owned_cleanup_no_signal_no_preemption"
    )
    assert require_port.call_count == 3
    assert sleep.call_count == 2


def test_inter_arm_port_release_fails_closed_at_deadline() -> None:
    with (
        patch.object(
            formal,
            "require_free_port",
            side_effect=ValueError("localhost port 8087 is already in use"),
        ),
        patch.object(formal.time, "monotonic", side_effect=[100.0, 101.0]),
        patch.object(formal.time, "sleep") as sleep,
    ):
        with pytest.raises(TimeoutError, match="no process was signaled or preempted"):
            formal.wait_for_released_port(
                8087, timeout_seconds=1.0, poll_interval_seconds=0.25
            )
    sleep.assert_not_called()


def test_inter_arm_port_barrier_precedes_the_next_arm_launch() -> None:
    source = inspect.getsource(formal.execute_run)
    barrier = source.index("wait_for_released_port(port)")
    next_arm = source.index("arm_results[arm] = run_arm(")
    assert "if arm_index:" in source
    assert barrier < next_arm
    assert "inter_arm_port_release_before_{arm}.json" in source


def test_shell_launcher_is_plan_by_default_and_never_preempts() -> None:
    launcher = (
        Path(__file__).resolve().parent
        / "run_qwen3_8b_v26_vanilla_formal_matched_eval.sh"
    ).read_text(encoding="utf-8")
    assert "MODE=${1:---plan}" in launcher
    assert "--run" in launcher and "EVAL_GPU_ID" in launcher
    for forbidden in ("ssh ", "pkill", "pgrep", "--resume"):
        assert forbidden not in launcher

    source = Path(formal.__file__).read_text(encoding="utf-8")
    assert '"src.rl.diagnostics.analyze_evaluation_results"' in source
    assert '"src.rl.diagnostics.audit_vanilla_grpo_matched_gate"' in source
    assert '"--candidate",\n        "final"' in source
    assert "checkpoint2" not in source and "checkpoint4" not in source
