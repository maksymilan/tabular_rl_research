from pathlib import Path

path = Path("/home/dengyan/tabular_rl_outputs/rl_runtime_qwen3_8b_v26_smc_audit_gate60_20260906/src/rl/frameworks/trl/transition_batch.py")
source = path.read_text(encoding="utf-8")

old = """_LAST_SMC_SELECTION_AUDIT: list[dict[str, Any]] = []

def pop_smc_selection_audit() -> list[dict[str, Any]]:
"""
new = """_LAST_SMC_SELECTION_AUDIT: list[dict[str, Any]] = []

def clear_smc_selection_audit() -> None:
    global _LAST_SMC_SELECTION_AUDIT
    _LAST_SMC_SELECTION_AUDIT = []

def pop_smc_selection_audit() -> list[dict[str, Any]]:
"""
if old not in source:
    raise SystemExit("audit helper anchor not found")
source = source.replace(old, new, 1)

old = """    global _LAST_SMC_SELECTION_AUDIT
    _LAST_SMC_SELECTION_AUDIT = []

    result = [0.0] * len(episodes)
"""
if old not in source:
    raise SystemExit("per-group reset anchor not found")
source = source.replace(old, "    result = [0.0] * len(episodes)\\n", 1)

old = '''    """Flatten causal turns while preserving trajectory- or step-local credit."""
    if reward_mode not in {"result-only", "process"}:
'''
new = '''    """Flatten causal turns while preserving trajectory- or step-local credit."""
    if result_advantage_profile == "smc-mode-concentration":
        clear_smc_selection_audit()
    if reward_mode not in {"result-only", "process"}:
'''
if old not in source:
    raise SystemExit("builder anchor not found")
source = source.replace(old, new, 1)
path.write_text(source, encoding="utf-8")
