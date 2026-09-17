from pathlib import Path


LAUNCHER = Path(__file__).parents[1] / "scenarios" / "rl_main" / (
    "run_qwen3_8b_atomic_v26_saam_fourlevel_spanbalanced_700_single_gpu_a100.sh"
)
CONFIG = Path(__file__).parents[1] / "configs" / "experiments" / (
    "qwen3_8b_atomic_v26_saam_screened500_single_gpu.yaml"
)


def test_single_gpu_launcher_is_screened500_and_allowlisted():
    text = LAUNCHER.read_text()
    assert 'source "$SCRIPT_DIR/../../frameworks/launcher/launch_common.sh"' in text
    assert "PROMPTS_PER_UPDATE=${PROMPTS_PER_UPDATE:-30}" in text
    assert "EXPECTED_RECORDS=${EXPECTED_RECORDS:-500}" in text
    assert "gpu_ids=(0 1 2 3)" in text
    assert "A100 launcher only permits GPU ids 0-3" in text
    assert "PROTOCOL_RUNTIME=${PROTOCOL_RUNTIME:-$OUTPUT_ROOT/runtime/version26-" in text


def test_screened500_config_matches_launcher_contract():
    text = CONFIG.read_text()
    assert "expected_records: 500" in text
    assert "prompts_per_update: 30" in text
    assert "max_new_tokens: 4096" in text
    assert "protocol_version: version26" in text
    assert "protocol_hash: 4da19387399bd3a5" in text
