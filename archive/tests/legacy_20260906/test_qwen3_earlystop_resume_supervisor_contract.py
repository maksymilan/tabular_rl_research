from __future__ import annotations

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = (
    ROOT
    / "src/rl/experiments/"
    "resume_qwen3_8b_atomic_v26_earlystop_mixed180_grpo_table_rl.sh"
)


def test_resume_supervisor_is_bound_to_the_interrupted_arm() -> None:
    text = SCRIPT.read_text()
    assert "train180_two_pass_seed20260812" in text
    assert "EXPECTED_TASKS_SHA256=a015c651" in text
    assert "EXPECTED_COHORT_MANIFEST_SHA256=e940c996" in text
    assert "EXPECTED_IMPLEMENTATION_LOCK_SHA256=f3e52d1e" in text
    assert "EXPECTED_RUNNER_SHA256=2990f15e" in text
    assert "--optimizer-steps 12" in text
    assert "--prompts-per-update 30" in text
    assert "--group-size 8" in text
    assert "--resume-from-checkpoint" in text


def test_resume_supervisor_preserves_and_trims_only_with_pinned_helper() -> None:
    text = SCRIPT.read_text()
    assert "EXPECTED_RESUME_HELPER_SHA256=0b1b254e" in text
    assert 'pretrim_sha=$(sha256_file "$TRAIN_OUT/rollouts.jsonl")' in text
    assert 'rows-${rows}.sha-${pretrim_sha}.jsonl' in text
    assert '[[ "$archive_sha" == "$pretrim_sha" ]]' in text
    assert "--save-steps 2" in text
    assert "committed_rollouts_per_step\": 240" in text


def test_resume_supervisor_waits_for_both_gpus_without_touching_external_jobs() -> None:
    text = SCRIPT.read_text()
    assert "wait_for_gpu_pair()" in text
    assert 'gpu_idle "$TRAIN_GPU" && gpu_idle "$VLLM_GPU"' in text
    assert text.count('gpu_idle "$TRAIN_GPU" && gpu_idle "$VLLM_GPU"') == 2
    assert "owner=external_process" in text
    assert "pkill" not in text
    assert "killall" not in text
    # TERM/KILL are scoped only to PGIDs captured from this script's own setsid.
    assert 'stop_group "$trainer_pgid"' in text
    assert 'stop_group "$vllm_pgid"' in text
    assert 'kill -TERM -- "-$pgid"' in text


def test_resume_supervisor_uses_fixed_fragmentation_mitigation_and_oom_only_retry() -> None:
    text = SCRIPT.read_text()
    assert "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" in text
    assert "export PYTORCH_ALLOC_CONF=expandable_segments:True" in text
    assert "torch.OutOfMemoryError: CUDA out of memory" in text
    assert "failed with non-OOM" in text
    assert "MAX_OOM_ATTEMPTS" in text


def test_resume_supervisor_has_an_exact_final_gate() -> None:
    text = SCRIPT.read_text()
    assert 'state["global_step"] == state["max_steps"] == 12' in text
    assert "assert len(rows) == 2880" in text
    assert "for step in range(12)" in text
    assert "for pass_index in range(2)" in text
    assert 'sha(out / "checkpoint-12" / "adapter_model.safetensors")' in text
    assert '(out / "training_precision.json").is_file()' in text


def test_deployed_source_hash_is_explicitly_reviewable() -> None:
    digest = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert len(digest) == 64

