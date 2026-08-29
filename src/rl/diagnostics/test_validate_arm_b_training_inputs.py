from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rl.diagnostics import validate_arm_b_training_inputs as admission
from src.rl.tests.test_vanilla_grpo_arm_b import build_real_arm_b_pipeline
from src.rl.vanilla_grpo_arm_b import sha256_bytes


def _sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _validate(paths: dict[str, Path]) -> dict:
    return admission.validate(
        selection_manifest_path=paths["selection"],
        expected_selection_manifest_sha256=_sha(paths["selection"]),
        train320_path=paths["train"],
        expected_train320_sha256=_sha(paths["train"]),
        validation64_audit_path=paths["validation_audit"],
        expected_validation64_audit_sha256=_sha(paths["validation_audit"]),
    )


def _fixture(tmp_path: Path) -> dict[str, Path]:
    """Compatibility fixture consumed by the matched-eval contract suite."""

    return build_real_arm_b_pipeline(tmp_path)


def test_admits_real_preparer_to_f3_builder_artifacts(tmp_path: Path) -> None:
    paths = build_real_arm_b_pipeline(tmp_path)
    report = _validate(paths)
    assert report["status"] == "training_inputs_admitted"
    assert report["train320"]["records"] == 320
    assert report["training_contract"] == admission.FORMAL_TRAINING_CONTRACT
    assert report["primary_checkpoint"] == "final-step32-only"


def test_rejects_validation_gate_drift_even_when_new_audit_hash_is_bound(
    tmp_path: Path,
) -> None:
    paths = build_real_arm_b_pipeline(tmp_path)
    audit = json.loads(paths["validation_audit"].read_bytes())
    audit["observed"]["mixed_boundary_groups"] = 55
    paths["validation_audit"].write_text(
        json.dumps(audit, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="mixed gate failed"):
        _validate(paths)


def test_rejects_changed_inline_arm_a_failure_evidence_source(tmp_path: Path) -> None:
    paths = build_real_arm_b_pipeline(tmp_path)
    cohort = json.loads(paths["cohort"].read_bytes())
    cohort["activation"]["combined"]["mixed"] = 999
    paths["cohort"].write_text(
        json.dumps(cohort, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    # The selection binds the cohort itself as immutable inline evidence.
    with pytest.raises(ValueError, match="source cohort manifest: SHA-256 mismatch"):
        _validate(paths)
