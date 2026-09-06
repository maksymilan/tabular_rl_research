from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from rl.experiments import (
    supervise_qwen3_8b_atomic_v26_representative600_onepass as supervisor,
)


ROOT = Path(__file__).resolve().parents[3]
FROZEN_CONTRACT = (
    ROOT
    / "src/rl/experiments/"
    "qwen3_8b_atomic_v26_representative600_onepass_contract.json"
)


def _contract() -> dict:
    source_hashes = {
        relative: supervisor.sha256_file(ROOT / relative)
        for relative in sorted(supervisor.REQUIRED_RUNTIME_SOURCES)
    }
    contract = {
        "schema_version": supervisor.SCHEMA_VERSION,
        "status": "frozen_before_launch",
        "experiment": {
            "name": supervisor.EXPECTED_EXPERIMENT_NAME,
            "algorithm": "vanilla_grpo",
            "arm": "baseline_result_only",
            "reward": "binary_result_only",
            "process_credit": "disabled",
            "custom_credit": "disabled",
            "records": 600,
            "passes": 1,
            "optimizer_steps": 20,
            "prompts_per_update": 30,
            "group_size": 8,
            "ppo_iterations": 1,
            "learning_rate": 8e-7,
            "kl_beta": 0.0,
            "seed": 20260812,
            "save_steps": 1,
            "save_total_limit": 20,
            "checkpoint_gate_step": 5,
            "evaluation_policy": "final_byte_identical_to_checkpoint_20_only",
            "posthoc_checkpoint_selection": False,
        },
        "paths": supervisor.EXPECTED_PATHS,
        "resources": {
            "gpus": supervisor.EXPECTED_GPUS,
            "vllm_port": 8076,
            "vllm_group_port": 51276,
            "stable_samples": 2,
            "stable_interval_seconds": 5.0,
            "max_idle_memory_mib": 512,
            "gpu_wait_poll_seconds": 30.0,
            "gpu_wait_timeout_seconds": 172800,
            "min_free_disk_bytes": 100 * 1024**3,
        },
        "identities": {
            "tasks_sha256": supervisor.EXPECTED_TASK_SHA256,
            "tasks_manifest_sha256": supervisor.EXPECTED_TASK_MANIFEST_SHA256,
            "config_sha256": supervisor.EXPECTED_CONFIG_SHA256,
            "protocol_runtime_tree_sha256": supervisor.EXPECTED_PROTOCOL_TREE_SHA256,
            "initial_adapter_sha256": supervisor.EXPECTED_INITIAL_ADAPTER_SHA256,
            "initial_adapter_config_sha256": (
                supervisor.EXPECTED_INITIAL_ADAPTER_CONFIG_SHA256
            ),
            "initial_adapter_state_sha256": (
                supervisor.EXPECTED_INITIAL_ADAPTER_STATE_SHA256
            ),
            "base_model_identity": supervisor.EXPECTED_BASE_MODEL_IDENTITY,
            "python_packages": supervisor.EXPECTED_PYTHON_PACKAGES,
            "supervisor_source_sha256": supervisor.sha256_file(
                Path(supervisor.__file__).resolve()
            ),
            "runtime_source_sha256": source_hashes,
        },
        "expected_implementation_lock_sha256": "0" * 64,
        "runtime": {
            "max_oom_restarts": 2,
            "monitor_poll_seconds": 20.0,
            "vllm_ready_timeout_seconds": 480.0,
            "vllm_gpu_memory_utilization": 0.82,
            "max_model_len": 16384,
            "allocator_conf": "expandable_segments:True",
            "import_smoke_cwd": "/tmp",
        },
    }
    contract["expected_implementation_lock_sha256"] = (
        supervisor.implementation_lock_sha256(contract)
    )
    return contract


def _write_contract(path: Path) -> str:
    path.write_text(json.dumps(_contract(), ensure_ascii=False, indent=2) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gpu_records(*, trainer_pids=(), vllm_pids=(), trainer_memory=2, vllm_memory=2):
    return {
        0: supervisor.GpuRecord(
            index=0,
            uuid=supervisor.EXPECTED_GPUS["trainer"]["uuid"],
            name=supervisor.EXPECTED_GPUS["trainer"]["name"],
            memory_used_mib=trainer_memory,
            compute_pids=tuple(trainer_pids),
        ),
        1: supervisor.GpuRecord(
            index=1,
            uuid=supervisor.EXPECTED_GPUS["vllm"]["uuid"],
            name=supervisor.EXPECTED_GPUS["vllm"]["name"],
            memory_used_mib=vllm_memory,
            compute_pids=tuple(vllm_pids),
        ),
    }


def test_real_contract_shape_and_preregistered_implementation_lock() -> None:
    contract = _contract()
    supervisor.validate_contract_shape(contract)
    lock = supervisor.implementation_lock_payload(contract)
    assert list(lock["files"]) == [
        *supervisor.RUNNER_IMPLEMENTATION_SOURCES,
        supervisor.GATE_AUDITOR,
    ]
    assert lock["checkpoint_gate"] == {
        "schema_version": "trl-synchronous-checkpoint-gate-v1",
        "step": 5,
        "script_relative_path": supervisor.GATE_AUDITOR,
        "script_sha256": contract["identities"]["runtime_source_sha256"][
            supervisor.GATE_AUDITOR
        ],
        "receipt": supervisor.EXPECTED_PATHS["train_output"] + "/step5_gate.json",
        "tasks_manifest": supervisor.EXPECTED_PATHS["tasks_manifest"],
        "tasks_manifest_sha256": supervisor.EXPECTED_TASK_MANIFEST_SHA256,
        "execution": "synchronous_on_save_before_next_optimizer_step",
        "resume_policy": "checkpoint_at_or_after_gate_requires_verified_receipt",
    }


def test_supervisor_source_order_exactly_matches_runner_constant() -> None:
    tree = ast.parse(
        (ROOT / "src/rl/frameworks/trl/run_transition_grpo.py").read_text(
            encoding="utf-8"
        )
    )
    runner_sources = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "IMPLEMENTATION_SOURCE_FILES"
            for target in node.targets
        ):
            runner_sources = ast.literal_eval(node.value)
            break
    assert runner_sources is not None
    assert supervisor.RUNNER_IMPLEMENTATION_SOURCES == runner_sources


def test_concrete_contract_preserves_v1_bytes_after_v2_runtime_evidence_upgrade() -> None:
    contract_sha = supervisor.sha256_file(FROZEN_CONTRACT)
    contract = supervisor.load_contract(FROZEN_CONTRACT, contract_sha)
    assert contract["paths"]["python"] == (
        "/home/dengyan/miniconda3/envs/trl-table/bin/python3.11"
    )
    identities = contract["identities"]
    assert identities["supervisor_source_sha256"] == supervisor.sha256_file(
        Path(supervisor.__file__).resolve()
    )
    v2_upgraded_sources = {
        "src/rl/frameworks/trl/rollout.py": (
            "87c36224ba0f01954db0d58db3bc3e5685bc5c86736c0679299c76872b6c821f"
        ),
        "src/rl/rollout_scoring.py": (
            "a11233e7a04a9efa34393e7d77a4c4aca34151144bf531978235ef6d644c1ac5"
        ),
    }
    for relative, expected in identities["runtime_source_sha256"].items():
        if relative in v2_upgraded_sources:
            assert expected == v2_upgraded_sources[relative]
            assert supervisor.sha256_file(ROOT / relative) != expected
        else:
            assert supervisor.sha256_file(ROOT / relative) == expected
    assert contract["expected_implementation_lock_sha256"] == (
        supervisor.implementation_lock_sha256(contract)
    )


def test_contract_rejects_source_or_lock_drift() -> None:
    contract = _contract()
    contract["identities"]["runtime_source_sha256"][supervisor.GATE_AUDITOR] = "f" * 64
    with pytest.raises(supervisor.ContractError, match="implementation lock SHA mismatch"):
        supervisor.validate_contract_shape(contract)


def test_default_plan_is_read_only_and_does_not_inspect_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    contract_path = tmp_path / "contract.json"
    _write_contract(contract_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("read-only plan called a mutating/runtime operation")

    monkeypatch.setattr(supervisor, "sample_gpus", forbidden)
    monkeypatch.setattr(supervisor, "_publish_immutable_json", forbidden)
    monkeypatch.setattr(supervisor, "_acquire_lock", forbidden)
    before = sorted(path.name for path in tmp_path.iterdir())
    assert supervisor.main(["--contract", str(contract_path)]) == 0
    after = sorted(path.name for path in tmp_path.iterdir())
    assert after == before
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "plan_read_only"
    assert "Arm B or custom/process credit" in payload["prohibited"]
    assert "old first32 probe admission or reuse" in payload["prohibited"]


def test_non_plan_requires_explicit_contract_hash(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    _write_contract(contract_path)
    with pytest.raises(supervisor.ContractError, match="explicit contract SHA"):
        supervisor.main(["status", "--contract", str(contract_path)])


def test_stable_idle_pair_uses_two_exact_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    samples = [_gpu_records(), _gpu_records()]
    monkeypatch.setattr(supervisor, "sample_gpus", lambda: samples.pop(0))
    sleeps = []

    class Journal:
        events = []

        def record(self, state, detail):
            self.events.append((state, detail))

    journal = Journal()
    result = supervisor.wait_for_stable_idle_pair(
        contract, journal, sleep=sleeps.append
    )
    assert len(result) == 2
    assert sleeps == [5.0]
    assert journal.events[-1][0] == "gpu_pair_stably_idle"


def test_gpu_monitor_rejects_compute_pid_outside_owned_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        supervisor,
        "sample_gpus",
        lambda: _gpu_records(trainer_pids=(1234,), vllm_pids=(5678,)),
    )
    monkeypatch.setattr(supervisor.os, "getpgid", lambda pid: {1234: 100, 5678: 999}[pid])
    valid, detail = supervisor._gpu_processes_owned_by({100})
    assert valid is False
    assert detail["vllm"]["foreign"] == [{"pid": 5678, "pgid": 999}]


def test_vllm_listener_must_belong_to_owned_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(supervisor, "_listener_pids", lambda _port: {200, 201})
    monkeypatch.setattr(supervisor.os, "getpgid", lambda pid: {200: 77, 201: 88}[pid])
    with pytest.raises(supervisor.ContractError, match="unowned listener"):
        supervisor.require_listener_owned(8076, {77})
    monkeypatch.setattr(supervisor.os, "getpgid", lambda _pid: 77)
    assert supervisor.require_listener_owned(8076, {77}) == {
        "port": 8076,
        "listeners": [{"pid": 200, "pgid": 77}, {"pid": 201, "pgid": 77}],
    }


def test_trainer_command_is_basic_grpo_with_synchronous_gate_and_no_custom_credit() -> None:
    contract = _contract()
    command, environment = supervisor.trainer_command_and_environment(contract, None)
    joined = " ".join(command)
    for required in (
        "--optimizer-steps 20",
        "--ppo-iterations 1",
        "--prompts-per-update 30",
        "--group-size 8",
        "--kl-beta 0",
        "--save-steps 1",
        "--save-total-limit 20",
        "--checkpoint-gate-step 5",
        "--checkpoint-gate-script",
        "--checkpoint-gate-receipt",
        "--checkpoint-gate-tasks-manifest",
    ):
        assert required in joined
    for forbidden in (
        "arm_b",
        "process-reward",
        "rank-loss",
        "custom-credit",
        "fixed-rollout",
        "probe_first32",
    ):
        assert forbidden not in joined.lower()
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["PYTHONPATH"] == supervisor.EXPECTED_PATHS["project_root"]
    assert environment["PYTORCH_CUDA_ALLOC_CONF"] == "expandable_segments:True"
    vllm_command, vllm_environment = supervisor.vllm_command_and_environment(contract)
    assert vllm_environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert vllm_environment["VLLM_PORT"] == "8076"
    assert vllm_command[-1].endswith("start_vllm_server.sh")


def test_hostile_parent_experiment_controls_are_not_inherited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key, value in {
        "FIXED_ROLLOUT_POOL": "/tmp/foreign-pool",
        "FIXED_POOL_MANIFEST": "/tmp/foreign-manifest",
        "COUNTERFACTUAL_SUITE_MANIFEST": "/tmp/foreign-counterfactual",
        "PROCESS_REWARD_CONFIG": "/tmp/foreign-process-credit",
        "REWARD_MODE": "process",
        "VLLM_HOST": "0.0.0.0",
        "MODEL_PATH": "/tmp/foreign-model",
        "OUTPUT_DIR": "/tmp/foreign-output",
    }.items():
        monkeypatch.setenv(key, value)
    contract = _contract()
    _, trainer_environment = supervisor.trainer_command_and_environment(contract, None)
    _, vllm_environment = supervisor.vllm_command_and_environment(contract)
    for environment in (trainer_environment, vllm_environment):
        assert environment["VLLM_HOST"] == "127.0.0.1"
        for forbidden in (
            "FIXED_ROLLOUT_POOL",
            "FIXED_POOL_MANIFEST",
            "COUNTERFACTUAL_SUITE_MANIFEST",
            "PROCESS_REWARD_CONFIG",
            "REWARD_MODE",
        ):
            assert forbidden not in environment
    assert trainer_environment["MODEL_PATH"] == supervisor.EXPECTED_PATHS["model"]
    assert trainer_environment["OUTPUT_DIR"] == supervisor.EXPECTED_PATHS["train_output"]


def test_preflight_import_smoke_is_outside_project_and_has_no_visible_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    paths = {key: Path(value) for key, value in contract["paths"].items()}
    expected_paths = supervisor._expected_import_smoke_paths(paths)
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        stdout = ""
        if "-c" in command:
            stdout = "IMPORT_SMOKE_JSON=" + json.dumps(
                {
                    **expected_paths,
                    "implementation_sources": list(
                        supervisor.RUNNER_IMPLEMENTATION_SOURCES
                    ),
                },
                sort_keys=True,
            )
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(supervisor.subprocess, "run", fake_run)
    audit = supervisor._validate_no_gpu_import_smoke(contract, paths)
    assert len(calls) == 5
    assert len(audit["help_entry_points"]) == 4
    assert audit["working_directory"] == str(Path("/tmp").resolve())
    assert audit["import_paths"] == expected_paths
    for _command, kwargs in calls:
        assert kwargs["cwd"] == Path("/tmp").resolve()
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["PYTHONPATH"] == ""
        assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
    help_commands = [call[0] for call in calls[:4]]
    assert [Path(command[1]).name for command in help_commands] == [
        "run_transition_grpo.py",
        "prepare_vanilla_grpo_resume.py",
        "audit_vanilla_grpo_train600_step5.py",
        "audit_representative600_onepass_training.py",
    ]
    assert all("--help" in command for command in help_commands)
    runner_help = help_commands[0]
    assert runner_help[runner_help.index("--experiment-config") + 1] == str(
        paths["experiment_config"]
    )
    assert runner_help[runner_help.index("--protocol-runtime-root") + 1] == str(
        paths["protocol_runtime"]
    )


def test_resume_and_completion_commands_are_exact_and_source_locked(tmp_path: Path) -> None:
    contract = _contract()
    resume = supervisor.resume_command(contract, tmp_path / "resume.json")
    assert resume[1].endswith("prepare_vanilla_grpo_resume.py")
    assert resume[resume.index("--save-steps") + 1] == "1"
    assert resume[resume.index("--retain-checkpoint-step") + 1] == "5"
    assert "--expected-records" in resume and "--group-size" in resume
    completion = supervisor.completion_command(contract, tmp_path / "complete.json")
    assert completion[1].endswith("audit_representative600_onepass_training.py")
    assert completion[
        completion.index("--expected-implementation-lock-sha256") + 1
    ] == contract["expected_implementation_lock_sha256"]


def test_launch_logs_are_globally_unique_and_stale_oom_cannot_carry_over(
    tmp_path: Path,
) -> None:
    (tmp_path / "trainer_attempt1.log").write_text("CUDA out of memory\n")
    (tmp_path / "vllm_attempt1.log").write_text("ready\n")
    assert supervisor._next_launch_number(tmp_path) == 2
    current = tmp_path / "trainer_attempt2.log"
    current.write_text("unrelated trainer failure\n")
    assert supervisor._confirmed_oom(tmp_path / "trainer_attempt1.log") is True
    assert supervisor._confirmed_oom(current) is False


def test_cleanup_refuses_any_process_group_not_proved_owned(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeProcess:
        pid = 42

    owned = supervisor.OwnedProcess(
        role="trainer",
        process=FakeProcess(),  # type: ignore[arg-type]
        pgid=43,
        start_time_ticks=123,
        log_path=tmp_path / "trainer.log",
    )
    monkeypatch.setattr(
        supervisor.os,
        "killpg",
        lambda *_args: (_ for _ in ()).throw(AssertionError("killpg called")),
    )
    with pytest.raises(supervisor.ContractError, match="unproved process group"):
        supervisor.terminate_owned_process(owned)


def test_launch_journal_failure_cleans_the_just_created_group(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeLog:
        def close(self) -> None:
            pass

    class FakeProcess:
        pid = 4242

    class FailingJournal:
        def record(self, *_args, **_kwargs) -> None:
            raise RuntimeError("journal publication failed")

    process = FakeProcess()
    cleaned = []
    monkeypatch.setattr(supervisor, "_open_new_no_follow", lambda _path: FakeLog())
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(supervisor.os, "getpgid", lambda _pid: process.pid)
    monkeypatch.setattr(supervisor, "_process_start_time_ticks", lambda _pid: 91)
    monkeypatch.setattr(supervisor, "terminate_owned_process", cleaned.append)
    with pytest.raises(RuntimeError, match="journal publication failed"):
        supervisor.start_owned_process(
            role="trainer",
            command=["trainer"],
            environment={},
            log_path=tmp_path / "trainer_attempt1.log",
            journal=FailingJournal(),  # type: ignore[arg-type]
        )
    assert len(cleaned) == 1
    assert cleaned[0].process is process
    assert cleaned[0].pgid == process.pid
    assert cleaned[0].start_time_ticks == 91


def test_source_drift_after_gpu_wait_prevents_resume_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _contract()
    calls = []

    class Journal:
        def record(self, state, _detail) -> None:
            calls.append(f"journal:{state}")

    monkeypatch.setattr(supervisor, "_has_final_candidate", lambda _contract: False)
    monkeypatch.setattr(supervisor, "_prior_confirmed_ooms", lambda _journal: 0)
    monkeypatch.setattr(supervisor, "_next_launch_number", lambda _root: 1)
    monkeypatch.setattr(
        supervisor,
        "wait_for_stable_idle_pair",
        lambda _contract, _journal: calls.append("stable_wait"),
    )

    def reject_drift(_contract):
        calls.append("static_validation")
        raise supervisor.ContractError("source drift after wait")

    def forbidden_resume(*_args, **_kwargs):
        calls.append("resume_mutation")
        raise AssertionError("resume helper ran before post-wait source validation")

    monkeypatch.setattr(supervisor, "validate_static_inputs", reject_drift)
    monkeypatch.setattr(supervisor, "prepare_resume", forbidden_resume)
    with pytest.raises(supervisor.ContractError, match="source drift after wait"):
        supervisor.run_training(contract, Journal())  # type: ignore[arg-type]
    assert calls == ["stable_wait", "static_validation"]


def test_busy_vllm_http_health_is_not_a_training_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class TrainerProcess:
        pid = 101

        def __init__(self) -> None:
            self.polls = iter((None, 0))

        def poll(self):
            return next(self.polls)

    class VllmProcess:
        pid = 202

        def poll(self):
            return None

    class Journal:
        def __init__(self) -> None:
            self.events = []

        def record(self, state, detail) -> None:
            self.events.append((state, detail))

    trainer = supervisor.OwnedProcess(
        role="trainer",
        process=TrainerProcess(),  # type: ignore[arg-type]
        pgid=101,
        start_time_ticks=1,
        log_path=tmp_path / "trainer.log",
    )
    vllm = supervisor.OwnedProcess(
        role="vllm",
        process=VllmProcess(),  # type: ignore[arg-type]
        pgid=202,
        start_time_ticks=2,
        log_path=tmp_path / "vllm.log",
    )
    journal = Journal()
    monkeypatch.setattr(
        supervisor,
        "_health",
        lambda _port: (_ for _ in ()).throw(
            AssertionError("HTTP health must not gate an in-flight rollout")
        ),
    )
    monkeypatch.setattr(
        supervisor,
        "require_listener_owned",
        lambda port, groups: {"port": port, "groups": sorted(groups)},
    )
    monkeypatch.setattr(
        supervisor,
        "_gpu_processes_owned_by",
        lambda groups: (True, {"owned_groups": sorted(groups)}),
    )
    monkeypatch.setattr(supervisor.time, "sleep", lambda _seconds: None)

    assert supervisor.wait_for_trainer(
        _contract(), trainer, vllm, journal  # type: ignore[arg-type]
    ) == 0
    assert journal.events == [("trainer_exited", {"returncode": 0})]


def test_supervisor_has_no_broad_process_kill_or_legacy_probe_dependency() -> None:
    text = Path(supervisor.__file__).read_text(encoding="utf-8")
    assert "pkill" not in text
    assert "killall" not in text
    assert "probe_first32" not in text
    assert "start_new_session=True" in text
    assert "os.killpg(owned.pgid" in text
