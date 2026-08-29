from __future__ import annotations

import re
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = (
    ROOT
    / "src"
    / "rl"
    / "experiments"
    / "run_qwen3_8b_atomic_v26_vanilla_grpo_table_rl.sh"
)


def test_launcher_is_valid_bash_and_has_no_unfilled_identity_hashes() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    assert LAUNCHER.stat().st_mode & stat.S_IXUSR
    text = LAUNCHER.read_text()
    assert re.search(r"__[A-Z0-9_]+__", text) is None


def test_probe_is_default_and_binds_the_no_update_gate() -> None:
    text = LAUNCHER.read_text()
    assert "MODE=${1:-probe}" in text
    assert "generate_fixed_rollout_pool.py" in text
    assert "--limit 32" in text
    assert "--group-size 8" in text
    assert "--max-new-tokens 2048" in text
    assert "--max-context-tokens 16384" in text
    assert "--enable-thinking" in text
    assert 'TABLE_AGENT_PROTOCOL_RUNTIME_ROOT="$PROTOCOL_RUNTIME"' in text
    assert "audit_vanilla_grpo_rollout_probe.py" in text
    assert "resume_groups=true" in text
    assert "! -name groups" in text
    assert "! -name '*.json'" in text
    assert 'audit["status"]["passes"] is True' in text
    assert 'audit["status"]["probe_admitted"] is True' in text
    assert 'audit["inputs"]["manifest_sha256"] == sha(manifest_path)' in text
    assert 'audit["inputs"]["trajectories_sha256"] == sha(trajectories_path)' in text
    assert 'manifest["protocol_runtime_root"]' in text
    assert 'manifest["protocol_runtime_content_tree_sha256"]' in text
    train_body = text.split("run_train() {", 1)[1]
    assert train_body.lstrip().startswith("verify_probe_gate")


def test_training_contract_is_isolated_resumable_and_owns_only_its_sessions() -> None:
    text = LAUNCHER.read_text()
    assert "rl_runtime_qwen3_8b_v26_vanilla_grpo_20260812" in text
    assert "rl_runtime_qwen3_8b_version26_trustsql_20260811" not in text
    assert "/home/dengyan/models/Qwen3-8B-TrustSQL-baseline" in text
    assert "qwen3_8b_atomic_v26_vanilla_grpo_train600_v1.jsonl" in text
    assert "qwen3_8b_atomic_v26_vanilla_grpo.yaml" in text
    assert "--optimizer-steps 20" in text
    assert "--prompts-per-update 30" in text
    assert "--group-size 8" in text
    assert "--save-steps 2" in text
    assert '--protocol-runtime-root "$PROTOCOL_RUNTIME"' in text
    assert "CUDA_VISIBLE_DEVICES=0" in text
    assert "CUDA_VISIBLE_DEVICES=1" in text
    assert "MAX_MODEL_LEN=16384" in text
    assert "--resume-from-checkpoint" in text
    for checkpoint_file in (
        "adapter_model.safetensors",
        "adapter_config.json",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "rng_state.pth",
        "training_args.bin",
    ):
        assert checkpoint_file in text
    assert "setsid env" in text
    assert 'kill -TERM -- "-$pgid"' in text
    assert "pgrep" not in text
    assert "pkill" not in text


def test_runtime_and_policy_identities_are_fail_closed() -> None:
    text = LAUNCHER.read_text()
    assert "EXPECTED_PROTOCOL_VERSION=version26" in text
    assert "EXPECTED_PROTOCOL_HASH=4da19387399bd3a5" in text
    assert (
        "EXPECTED_SFT1_SHA256="
        "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5"
    ) in text
    assert "v26_tool_environment" in text
    assert "EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256" in text
    assert "runtime_content_tree_sha256(protocol_runtime)" in text
    assert 'runtime_modules["module_paths"] == {' in text
    assert '"protocol": expected_runtime + "/src/sft/protocol.py"' in text
    assert 'manifest["result_reward_profile"] == "binary"' in text
    assert 'manifest["kl_beta"] == 0.0' in text
    assert 'reference["enabled"] is False' in text
    assert 'reference["adapter_name"] is None' in text
    assert 'reference["load_audit"] is None' in text
