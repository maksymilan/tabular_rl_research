from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WATCHER = ROOT / "rl/experiments/watch_earlystop_mixed180_post_training_table_rl.sh"
RUNTIME_FILES = {
    "EXPECTED_AUDITOR_SHA256": ROOT / "rl/diagnostics/audit_earlystop_mixed180_training.py",
    "EXPECTED_PAIRING_SHA256": ROOT / "rl/diagnostics/summarize_earlystop_pass_pairing.py",
    "EXPECTED_EVAL_LAUNCHER_SHA256": ROOT / "rl/evaluation/run_qwen3_8b_v26_earlystop_mixed180_formal_matched_eval.sh",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_watcher_is_fail_closed_and_never_starts_followup_training() -> None:
    subprocess.run(["bash", "-n", str(WATCHER)], check=True)
    source = WATCHER.read_text(encoding="utf-8")
    plan = subprocess.run(
        [str(WATCHER), "--plan"], check=True, capture_output=True, text=True
    )
    assert '"status": "plan_only_no_mutation"' in plan.stdout
    assert "strict_training_audit_failed" in source
    assert "summarize_earlystop_pass_pairing.py" in source
    assert "fresh_sft1_then_final12" in source
    assert "formal_run_dir_exists_no_resume" in source
    assert "training-cohort" in source
    assert 'EVAL_GPU_ID=${EVAL_GPU_ID:-0}' in source
    assert '[[ "$EVAL_GPU_ID" == 0 ]]' in source
    assert "status.json" in source
    assert "partial formal run is rejected" in source
    assert "earlystop-mixed180-post-training-queue-status-v1" in source
    assert "watcher sends no signals" in source
    assert "qwen3-v26-formal-matched-eval-status-v1" in source
    assert 'status.get("run_dir") == expected_run_dir' in source
    assert 'type(status.get("physical_gpu_id")) is int' in source
    assert 'status.get("physical_gpu_id") == 0' in source
    assert 'status.get("decision") in {"promote_final", "do_not_promote_final"}' in source
    assert source.index('if [[ -e "$RUN_DIR" || -L "$RUN_DIR" ]]') < source.index(
        'while ! gpu_idle "$EVAL_GPU_ID"'
    )
    assert source.index("formal_run_dir_exists_no_resume") < source.index(
        'while ! gpu_idle "$EVAL_GPU_ID"'
    )
    assert "formal_matched_eval_returned_without_strict_completion" in source
    for forbidden in ("pkill", "pgrep", "kill ", "--resume", "arm_b_train", "boundary300_vanilla_grpo"):
        assert forbidden not in source


def test_watcher_literal_runtime_pins_are_current() -> None:
    source = WATCHER.read_text(encoding="utf-8")
    for variable, path in RUNTIME_FILES.items():
        assert f"{variable}={sha(path)}" in source
    assert re.search(r"__PIN_[A-Z0-9_]+__", source) is None
