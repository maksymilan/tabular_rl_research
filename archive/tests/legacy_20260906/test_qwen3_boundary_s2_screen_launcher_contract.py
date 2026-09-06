from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = (
    ROOT
    / "src/rl/experiments/run_qwen3_8b_atomic_v26_boundary_screen_s2_table_rl.sh"
)
PREPARER = ROOT / "src/rl/prepare_policy_boundary_s2_tasks.py"
PREPARATION_RECOVERY_HELPER = (
    ROOT / "src/rl/diagnostics/recover_policy_boundary_s2_preparation.py"
)
GROUP_RESUME_HELPER = (
    ROOT / "src/rl/diagnostics/prepare_boundary_screen_group_resume.py"
)
SELECTION_RECOVERY_HELPER = (
    ROOT / "src/rl/diagnostics/recover_policy_boundary_selection.py"
)
AUDITOR = ROOT / "src/rl/diagnostics/audit_vanilla_grpo_boundary_screen.py"
SELECTOR = ROOT / "src/rl/select_policy_boundary_grpo_tasks.py"


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


def test_default_is_read_only_dry_run_and_all_embedded_python_compiles(
    tmp_path: Path,
) -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    assert LAUNCHER.stat().st_mode & stat.S_IXUSR
    run_root = tmp_path / "must-not-exist"
    result = _run(env={"RUN_ROOT": str(run_root)})
    assert result.returncode == 0, result.stderr
    assert "S2 dry-run" in result.stdout
    assert "no files written, no GPU inspected" in result.stdout
    assert "no S1 reuse" in result.stdout
    assert "no S3" in result.stdout
    assert not run_root.exists()

    source = LAUNCHER.read_text()
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", source, flags=re.DOTALL)
    assert len(blocks) >= 8
    for index, block in enumerate(blocks):
        compile(block, f"{LAUNCHER}:heredoc-{index}", "exec")


def test_only_explicit_s2_lifecycle_and_frozen_activation_can_allocate() -> None:
    text = LAUNCHER.read_text()
    assert "MODE=${1:-dry-run}" in text
    assert "dry-run|prepare|worker|finalize" in text
    assert "prepare -> worker[0] + worker[1] -> finalize" in text
    invalid = _run("unknown")
    assert invalid.returncode == 2

    active = _run("prepare")
    assert active.returncode == 3
    assert "FROZEN_S1_AUDIT_SHA256" in active.stderr
    assert "SFT1_INDEX and OLD_MIXED60" in text
    assert '[[ -n "$SFT1_INDEX" && -n "$OLD_MIXED60" ]]' in text
    assert "S1_AUDIT=${S1_AUDIT:-$S1_SCREEN_ROOT/boundary_selection/boundary_screen_audit.json}" in text
    assert 'require_sha "$S1_AUDIT" "$FROZEN_S1_AUDIT_SHA256"' in text
    assert 'verify_screen_audit "$S1_AUDIT" "$S1_TASKS" "$FROZEN_S1_AUDIT_SHA256" S1' in text
    assert '"selection_ready_without_s2": False, "next_stage": "screen_s2"' in text


def test_frozen_preparer_cli_and_all_input_hashes_are_fail_closed() -> None:
    text = LAUNCHER.read_text()
    assert hashlib.sha256(PREPARER.read_bytes()).hexdigest() == (
        "069f17a32c39aa28777432a664ead17a1d90eab68c62ff3811f4fc9bc19ccc18"
    )
    assert (
        "EXPECTED_PREPARER_SHA256=" + hashlib.sha256(PREPARER.read_bytes()).hexdigest()
    ) in text
    for option, variable in (
        ("--s1-audit", "$S1_AUDIT"),
        ("--expected-s1-audit-sha256", "$FROZEN_S1_AUDIT_SHA256"),
        ("--reference", "$REFERENCE_TASKS"),
        ("--eligible", "$ELIGIBLE_TASKS"),
        ("--s1-tasks", "$S1_TASKS"),
        ("--s1-cohort-manifest", "$S1_COHORT_MANIFEST"),
        ("--baseline-eval300", "$BASELINE_EVAL300"),
        ("--sft1-index", "$SFT1_INDEX"),
        ("--old-mixed60", "$OLD_MIXED60"),
        ("--output", "$S2_TASKS"),
        ("--manifest", "$S2_COHORT_MANIFEST"),
        ("--remote-db-root", "$REMOTE_DB_ROOT"),
    ):
        assert f'{option} "{variable}"' in text
    for digest in (
        "24d21a349c430df549f92724dd07de1c59b2365cd72802957120afba13c7a2e1",
        "c8ea3c6419ac57f05021eca470c7519843456e8ac02b04cca9af39e779ec7545",
        "b5a83c373e9be094ea7355c0bc23c9212249491491f44dfdcb3b09ea49b2457e",
        "9212d1f7ca4fb63576eb7e846a9b05f7d159e131fe992495b99e350b6aae8dd6",
        "87b37b307ea2710acdff7d03bdc474a6ba2cdb8ed51b005cff0fe8fa54c67c03",
        "aa6b8a72a1cfbcf44e37702e0498c9a85f615edc201ba3f4172b65f701e0cda5",
        "ef97b6977735996fde505aaad4e00215570963b37a1c82d890c34c4b414f30ae",
    ):
        assert digest in text
    recovery_sha = hashlib.sha256(PREPARATION_RECOVERY_HELPER.read_bytes()).hexdigest()
    assert f"EXPECTED_S2_PREPARATION_RECOVERY_HELPER_SHA256={recovery_sha}" in text
    assert '"$PYTHON_BIN" "$S2_PREPARATION_RECOVERY_HELPER"' in text
    assert '--preparer "$S2_PREPARER"' in text
    assert '--quarantine-dir "$QUARANTINE_DIR/s2-preparation"' in text
    assert '"schema_version"] == "bird-train-policy-boundary-s2-extra600-v1"' in text
    assert '"status"] == "frozen_s2_screening_cohort"' in text
    assert '"gold_used_for_selection_or_order"] is False' in text
    assert '"merge_rule": "S1+S2 audits only; no S3 and no threshold relaxation"' in text


def test_s2_is_fresh_extra600_k8_two_gpu_resumable_without_prefix_reuse() -> None:
    text = LAUNCHER.read_text()
    assert "SCREEN_SHARDS=2" in text
    assert "SCREEN_GPU0=${SCREEN_GPU0:-0}" in text
    assert "SCREEN_GPU1=${SCREEN_GPU1:-1}" in text
    assert 'position % 2 == shard' in text
    assert '"records": 300' in text
    assert '"reused_tasks": 0' in text
    assert '"generated_tasks": 600' in text
    assert '"stage": "S2"' in text
    assert '--task-id-file "$assignment"' in text
    assert "--no-finalize" in text
    assert "resume_groups=true" in text
    assert "worker_complete.json" in text
    helper_sha = hashlib.sha256(GROUP_RESUME_HELPER.read_bytes()).hexdigest()
    assert f"EXPECTED_GROUP_RESUME_HELPER_SHA256={helper_sha}" in text
    assert '--quarantine-dir "$QUARANTINE_DIR/$shard_name/groups"' in text
    assert 'marker_quarantine=$4' in text
    assert '--group-size 8' in text
    assert '--temperature 0.8' in text
    assert '--seed 20260812' in text
    assert 'CUDA_VISIBLE_DEVICES="$gpu"' in text
    assert "reuse-first32" not in text
    assert "FIRST32_SOURCE" not in text

    collision = _run(
        env={"SCREEN_GPU0": "0", "SCREEN_GPU1": "0", "RUN_ROOT": "/tmp/s2-dry"}
    )
    assert collision.returncode == 3
    assert "requires distinct GPUs" in collision.stderr


def test_worker_owns_only_its_process_group_and_never_touches_other_runs() -> None:
    text = LAUNCHER.read_text()
    assert "setsid env" in text
    assert 'kill -TERM -- "-$worker_pgid"' in text
    assert 'kill -KILL -- "-$worker_pgid"' in text
    assert "wait_for_gpu" in text
    assert "no process will be stopped" in text
    assert "pgrep" not in text
    assert "pkill" not in text
    assert "killall" not in text
    assert re.search(r"\bssh\b", text) is None
    assert "qwen3_8b_atomic_v26_boundary_screen_s2_2gpu_20260812" in text
    assert 'case "$POOL_OUT" in "$RUN_ROOT"/*)' in text


def test_finalize_strictly_merges_finalizes_and_rehashes_audit_inputs() -> None:
    text = LAUNCHER.read_text()
    finalize = text.split("finalize() {", 1)[1].split("\n}\n\ncase", 1)[0]
    assert "merge_worker_groups" in finalize
    assert "--finalize-only" in finalize
    assert "verify_final_pool" in finalize
    assert "--scheduler static" in finalize
    assert "--task-batch-size 1" in finalize
    assert 'manifest["generation_scheduler"] == "static"' in text
    assert 'manifest["generation_task_batch_size"] == 1' in text
    assert 'manifest["generation_question_window"] is None' in text
    assert 'assert set(sources) == set(ids) and len(sources) == 600' in text
    assert 'len(rows) == 4800' in text
    assert 'list(range(4800))' in text
    assert 'Counter({task_id: 8 for task_id in ids})' in text
    assert '--manifest "$POOL_OUT/manifest.pending.json"' in finalize
    assert '--trajectories "$POOL_OUT/trajectories.jsonl"' in finalize
    assert '--tasks "$S2_TASKS"' in finalize
    assert 'verify_screen_audit "$s2_audit" "$S2_TASKS" - S2' in finalize
    assert 'for key in ("manifest", "trajectories")' in text
    assert 'inputs[f"{key}_sha256"] == sha(path)' in text


def test_selector_is_called_once_in_strict_s1_then_s2_order_and_no_s3() -> None:
    text = LAUNCHER.read_text()
    finalize = text.split("finalize() {", 1)[1].split("\n}\n\ncase", 1)[0]
    ordered = (
        '--screen-audit "$S1_AUDIT"',
        '--screen-audit "$s2_audit"',
        '--tasks "$S1_TASKS"',
        '--tasks "$S2_TASKS"',
    )
    recovery_call = finalize.split('"$PYTHON_BIN" "$SELECTION_RECOVERY_HELPER"', 1)[1]
    positions = [recovery_call.index(value) for value in ordered]
    assert positions == sorted(positions)
    assert '--output-dir "$BOUNDARY_SELECTION_DIR"' in finalize
    assert '--seed qwen3-v26-boundary332-v1-20260812' in finalize
    recovery_sha = hashlib.sha256(SELECTION_RECOVERY_HELPER.read_bytes()).hexdigest()
    assert f"EXPECTED_SELECTION_RECOVERY_HELPER_SHA256={recovery_sha}" in text
    assert '"$PYTHON_BIN" "$SELECTION_RECOVERY_HELPER"' in finalize
    assert '--selector "$BOUNDARY_SELECTOR"' in finalize
    assert '--quarantine-dir "$QUARANTINE_DIR/boundary-selection"' in finalize
    assert "recoverable S1+S2 publication" in finalize
    assert "no S3 and no threshold relaxation" in finalize
    assert "S3_PREPARER" not in text
    assert "screen_s3" not in text


def test_combined_selection_manifest_outputs_and_verification_are_rechecked() -> None:
    text = LAUNCHER.read_text()
    assert 'assert len(pools) == 2' in text
    assert 'zip(pools, (s1_audit_path, s2_audit_path)' in text
    assert 'selection["screen_stage"] == "S1+S2"' in text
    assert 'selection["pool_eligible_counts"] == pool_counts' in text
    assert 'selection["eligible_mixed_groups"] == sum' in text
    assert 'selection["eligible_core_groups"] == sum' in text
    assert '"boundary": "boundary332.jsonl"' in text
    assert '("boundary", 332), ("train", 300), ("validation", 32)' in text
    assert 'set(observed["train"]).isdisjoint(observed["validation"])' in text
    assert 'set(observed["train"]) | set(observed["validation"]) == set(observed["boundary"])' in text
    assert 'manifest["outputs"][name] == {"path": str(paths[name]), "sha256": sha(paths[name]), "records": count}' in text
    assert "boundary_selection_verification.json" in text
    assert "FROZEN_BOUNDARY_MANIFEST_SHA256=%s" in text


def test_external_hooks_and_runtime_sources_are_exactly_pinned() -> None:
    text = LAUNCHER.read_text()
    auditor_sha = hashlib.sha256(AUDITOR.read_bytes()).hexdigest()
    selector_sha = hashlib.sha256(SELECTOR.read_bytes()).hexdigest()
    assert auditor_sha == "b983bd5d8083de49328fb25b98fba6105b2ffc773faceb7a5d0c8eea7c9181e6"
    assert selector_sha == "2849de8109b74cce59ca590b94b478edc5cf1a4c120ebda0e1d71902791c01d8"
    assert f"EXPECTED_BOUNDARY_AUDITOR_SHA256={auditor_sha}" in text
    assert f"EXPECTED_BOUNDARY_SELECTOR_SHA256={selector_sha}" in text
    assert "EXPECTED_GENERATOR_SHA256=db934c05cbf4f2d9beef3e01e8b42ff74312ef00834e681a9ae65647de12e191" in text
    assert "EXPECTED_FIXED_POOL_RUNTIME_SHA256=e0be4c4c830ca2e87172175553ab47d1b19561510f8fa6f77410e96846cf200d" in text
    assert "EXPECTED_TRANSITION_BATCH_SHA256=c6477393589581209ad581230eb598160e0674469103ce197b7d7d7f26a0bcd8" in text
    assert "EXPECTED_TERMINAL_REWARD_SHA256=2b45d45f562a903fe6e83589bc3aa9b9cc8d22036be6d0544c562ced82757b7b" in text
    assert "EXPECTED_PROTOCOL_RUNTIME_TREE_SHA256=5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab" in text
    assert "training overlay shadows frozen protocol source" in text
    assert 'require_sha "$BOUNDARY_AUDITOR" "$EXPECTED_BOUNDARY_AUDITOR_SHA256"' in text
    assert 'require_sha "$BOUNDARY_SELECTOR" "$EXPECTED_BOUNDARY_SELECTOR_SHA256"' in text
    for name, digest in (
        ("config.json", "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30"),
        ("model.safetensors.index.json", "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc"),
        ("tokenizer_config.json", "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"),
        ("tokenizer.json", "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"),
    ):
        assert f'require_sha "$MODEL_PATH/{name}" {digest}' in text


def test_python_assertions_cannot_be_disabled_and_nested_symlink_escape_blocks(
    tmp_path: Path,
) -> None:
    text = LAUNCHER.read_text()
    assert "unset PYTHONOPTIMIZE" in text
    optimized = _run("prepare", env={"PYTHONOPTIMIZE": "1"})
    assert optimized.returncode == 3
    assert "FROZEN_S1_AUDIT_SHA256" in optimized.stderr

    run_root = tmp_path / "run"
    run_root.mkdir()
    escape = tmp_path / "escape"
    escape.mkdir()
    (run_root / "link").symlink_to(escape, target_is_directory=True)
    result = _run(
        "prepare",
        env={
            "RUN_ROOT": str(run_root),
            "S2_INPUT_DIR": str(run_root / "link/inputs"),
            "FROZEN_S1_AUDIT_SHA256": "a" * 64,
            "SFT1_INDEX": str(tmp_path / "sft.jsonl"),
            "OLD_MIXED60": str(tmp_path / "mixed.jsonl"),
            "PYTHON_BIN": str(ROOT / ".venv/bin/python"),
        },
    )
    assert result.returncode == 3
    assert "resolved S2 path ownership/isolation check failed" in result.stderr
