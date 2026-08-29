from __future__ import annotations

import os
import re
import stat
import subprocess
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = (
    ROOT
    / "src"
    / "rl"
    / "experiments"
    / "run_qwen3_8b_atomic_v26_boundary_screen_table_rl.sh"
)
AUDITOR = ROOT / "src" / "rl" / "diagnostics" / "audit_vanilla_grpo_boundary_screen.py"
SELECTOR = ROOT / "src" / "rl" / "select_policy_boundary_grpo_tasks.py"


def _run(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args],
        text=True,
        capture_output=True,
        env=merged,
        check=False,
    )


def test_launcher_is_executable_valid_bash_and_default_is_read_only_dry_run(
    tmp_path: Path,
) -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    assert LAUNCHER.stat().st_mode & stat.S_IXUSR
    run_root = tmp_path / "must-not-exist"
    result = _run(env={"RUN_ROOT": str(run_root)})
    assert result.returncode == 0, result.stderr
    assert "boundary-screen dry-run" in result.stdout
    assert "no files written, no GPU inspected" in result.stdout
    assert not run_root.exists()


def test_only_explicit_s1_lifecycle_can_allocate_work() -> None:
    text = LAUNCHER.read_text()
    assert "MODE=${1:-dry-run}" in text
    assert "dry-run|prepare|worker|reuse-first32|finalize" in text
    assert "prepare -> worker[0..shards-1] -> reuse-first32 -> finalize" in text
    assert "SCREEN_STAGE=${SCREEN_STAGE:-S1}" in text
    assert "ENABLE_S2=${ENABLE_S2:-0}" in text
    assert "S2 is reserved" in text
    assert 'SCREEN_STAGE" == S2' in text

    invalid = _run("unknown")
    assert invalid.returncode == 2
    s2 = _run("dry-run", env={"SCREEN_STAGE": "S2", "ENABLE_S2": "0"})
    assert s2.returncode == 3
    assert "requires explicit ENABLE_S2=1" in s2.stderr


def test_current600_remaining568_shards_and_explicit_gpu_mapping_are_frozen() -> None:
    text = LAUNCHER.read_text()
    assert "EXPECTED_TASKS_SHA256=b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e" in text
    assert "EXPECTED_TASKS_MANIFEST_SHA256=9212d1f7ca4fb63576eb7e846a9b05f7d159e131fe992495b99e350b6aae8dd6" in text
    assert "EXPECTED_SFT1_SHA256=3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5" in text
    assert "SCREEN_SHARDS=${SCREEN_SHARDS:-1}" in text
    assert "SCREEN_GPU0=${SCREEN_GPU0:-0}" in text
    assert "SCREEN_GPU1=${SCREEN_GPU1:-1}" in text
    assert 'ids[32:]' in text
    assert 'offset % shards == shard' in text
    assert '"generated_remaining_tasks": 568' in text
    assert '"reused_prefix_tasks": 32' in text
    assert '--task-id-file "$assignment"' in text
    assert "--no-finalize" in text
    assert 'CUDA_VISIBLE_DEVICES="$gpu"' in text

    collision = _run(
        "dry-run",
        env={"SCREEN_SHARDS": "2", "SCREEN_GPU0": "0", "SCREEN_GPU1": "0"},
    )
    assert collision.returncode == 3
    assert "two shards require distinct GPU indices" in collision.stderr


def test_worker_is_resumable_and_owns_only_its_spawned_process_group() -> None:
    text = LAUNCHER.read_text()
    assert "resume_groups=true" in text
    assert "worker_complete.json" in text
    assert 'flock -n 8' in text
    assert "setsid env" in text
    assert 'kill -TERM -- "-$worker_pgid"' in text
    assert 'kill -KILL -- "-$worker_pgid"' in text
    assert "wait_for_gpu" in text
    assert "no process will be stopped" in text
    assert "pgrep" not in text
    assert "pkill" not in text
    assert "killall" not in text
    assert re.search(r"\bssh\b", text) is None


def test_first32_reuse_is_identity_checked_and_byte_exact() -> None:
    text = LAUNCHER.read_text()
    for contract in (
        '"tasks": 32',
        '"group_size": 8',
        '"trajectories": 256',
        '"temperature": 0.8',
        '"max_new_tokens": 2048',
        '"max_context_tokens": 16384',
        '"enable_thinking": True',
        '"seed": 20260812',
        '"result_reward_profile": "binary"',
    ):
        assert contract in text
    assert 'manifest["tasks_sha256"] == expected_tasks' in text
    assert 'manifest["adapter_sha256"] == expected_adapter' in text
    assert 'manifest["protocol_runtime_content_tree_sha256"] == expected_tree' in text
    assert 'target.exists() and target.read_bytes() != payload' in text
    assert 'assert target.read_bytes() == payload' in text
    assert 'assert reconstructed == trajectories_path.read_bytes()' in text
    assert '"status": "byte_identical_reuse_complete"' in text


def test_finalize_only_merges_complete_groups_then_calls_hash_pinned_hook() -> None:
    text = LAUNCHER.read_text()
    finalize = text.split("finalize() {", 1)[1].split("\n}\n\ncase", 1)[0]
    assert 'merge_all_groups' in finalize
    assert '--finalize-only' in finalize
    assert 'verify_final_pool_identity' in finalize
    assert 'require_sha "$BOUNDARY_AUDITOR" "$EXPECTED_BOUNDARY_AUDITOR_SHA256"' in finalize
    assert 'require_sha "$BOUNDARY_SELECTOR" "$EXPECTED_BOUNDARY_SELECTOR_SHA256"' in finalize
    assert '--manifest "$POOL_OUT/manifest.pending.json"' in finalize
    assert '--trajectories "$POOL_OUT/trajectories.jsonl"' in finalize
    assert '--tasks "$TASKS_JSONL"' in finalize
    assert '--output-dir "$BOUNDARY_AUDIT_DIR"' in finalize
    assert '--overwrite' in finalize
    assert 'boundary_screen_audit.json' in finalize
    assert 'assert [int(row["sequence"]) for row in trajectory_rows] == list(range(4800))' in text
    # Selection policy belongs only to the external reviewed hook.  The launcher
    # must not duplicate or relax its mixed/core admission thresholds.
    assert "mixed>=360" not in text
    assert "core>=240" not in text


def test_finalize_branches_on_frozen_audit_status_and_never_auto_starts_s2() -> None:
    text = LAUNCHER.read_text()
    finalize = text.split("finalize() {", 1)[1].split("\n}\n\ncase", 1)[0]
    assert 'status["next_stage"]' in text
    assert 'status["selection_ready_without_s2"]' in text
    assert 'if [[ "$next_stage" == screen_s2 ]]' in finalize
    assert 'write_status "$FINALIZE_STATUS" requires_s2' in finalize
    assert 'selector was not called' in finalize
    assert 'return 4' in finalize
    assert '[[ "$next_stage" == select_s1 ]]' in finalize
    selector_call = 'env PYTHONPATH= "$PYTHON_BIN" "$BOUNDARY_SELECTOR"'
    assert selector_call in finalize
    assert finalize.index('if [[ "$next_stage" == screen_s2 ]]') < finalize.index(selector_call)
    assert finalize.index('if [[ "$next_stage" == screen_s2 ]]') < finalize.index(
        'require_sha "$BOUNDARY_SELECTOR"'
    )
    assert '--screen-audit "$audit_path"' in finalize
    assert '--tasks "$TASKS_JSONL"' in finalize


def test_auditor_and_selector_implementations_are_exactly_hash_pinned() -> None:
    text = LAUNCHER.read_text()
    auditor_sha = hashlib.sha256(AUDITOR.read_bytes()).hexdigest()
    selector_sha = hashlib.sha256(SELECTOR.read_bytes()).hexdigest()
    assert auditor_sha == "b983bd5d8083de49328fb25b98fba6105b2ffc773faceb7a5d0c8eea7c9181e6"
    assert selector_sha == "2849de8109b74cce59ca590b94b478edc5cf1a4c120ebda0e1d71902791c01d8"
    assert f"EXPECTED_BOUNDARY_AUDITOR_SHA256={auditor_sha}" in text
    assert f"EXPECTED_BOUNDARY_SELECTOR_SHA256={selector_sha}" in text


def test_select_s1_outputs_and_manifest_hashes_are_reverified() -> None:
    text = LAUNCHER.read_text()
    for name, count in (("boundary", 332), ("train", 300), ("validation", 32)):
        assert f'"{name}": {count}' in text
    assert 'manifest["schema_version"] == "policy-boundary-grpo-cohort-v1"' in text
    assert 'manifest["status"] == "frozen_boundary_training_cohort"' in text
    assert 'pool["audit_sha256"] == hashlib.sha256(audit_path.read_bytes()).hexdigest()' in text
    assert 'declared["sha256"] == actual_sha' in text
    assert 'manifest["task_ids"][name] == ids' in text
    assert 'set(observed_ids["train"]).isdisjoint(observed_ids["validation"])' in text
    assert 'boundary_selection_verification.json' in text
    assert 'manifest_sha256=$selection_manifest_sha' in text


def test_frozen_generator_and_protocol_runtime_are_verified_before_gpu_use() -> None:
    text = LAUNCHER.read_text()
    assert "EXPECTED_GENERATOR_SHA256=db934c05cbf4f2d9beef3e01e8b42ff74312ef00834e681a9ae65647de12e191" in text
    assert "EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256=5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab" in text
    assert "EXPECTED_PROTOCOL_VERSION=version26" in text
    assert "EXPECTED_PROTOCOL_HASH=4da19387399bd3a5" in text
    assert "training overlay shadows frozen protocol source" in text
    assert text.index("validate_static_inputs") < text.index("wait_for_gpu \"$gpu\"")
