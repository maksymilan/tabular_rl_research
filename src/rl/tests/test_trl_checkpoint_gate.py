from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from rl.frameworks.trl.checkpoint_gate import (
    build_checkpoint_gate_spec,
    run_checkpoint_gate,
    sha256_file,
)


FAKE_GATE = r'''#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--tasks", type=Path, required=True)
parser.add_argument("--tasks-manifest", type=Path, required=True)
parser.add_argument("--initial-adapter", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--verify-existing", action="store_true")
args = parser.parse_args()
payload = {"schema_version": "fake-gate-v1", "status": {"gate_passed": True}}
encoded = json.dumps(payload, sort_keys=True) + "\n"
if args.verify_existing:
    if args.output.read_text() != encoded:
        raise SystemExit(2)
else:
    args.output.write_text(encoded)
'''


def make_spec(tmp_path: Path):
    root = tmp_path / "project"
    output = tmp_path / "run"
    script = root / "src/rl/scenarios/diagnostics/gate.py"
    tasks = root / "tasks.jsonl"
    tasks_manifest = root / "tasks.manifest.json"
    adapter = root / "adapter"
    script.parent.mkdir(parents=True)
    output.mkdir()
    adapter.mkdir()
    script.write_text(FAKE_GATE)
    tasks.write_text("{}\n")
    tasks_manifest.write_text("{}\n")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    spec = build_checkpoint_gate_spec(
        project_root=root,
        output_dir=output,
        optimizer_steps=20,
        save_steps=1,
        step=5,
        script=script,
        receipt=output / "step5_gate.json",
        tasks=tasks,
        tasks_manifest=tasks_manifest,
        initial_adapter=adapter,
    )
    assert spec is not None
    return spec


def test_gate_runs_and_reverifies_immutable_receipt(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    normal = run_checkpoint_gate(spec, verify_existing=False)
    verification = run_checkpoint_gate(spec, verify_existing=True)
    assert normal["returncode"] == 0
    assert verification["returncode"] == 0
    assert normal["receipt_sha256"] == sha256_file(spec.receipt)
    assert verification["receipt_sha256"] == normal["receipt_sha256"]
    assert json.loads(spec.receipt.read_text())["status"]["gate_passed"] is True
    assert (spec.output_dir / "checkpoint_gate_step5_invocation.json").is_file()
    assert (
        spec.output_dir / "checkpoint_gate_step5_resume_verification.json"
    ).is_file()


def test_gate_fails_closed_and_preserves_subprocess_evidence(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    spec.script.write_text("raise SystemExit(1)\n")
    spec = replace(spec, script_sha256=sha256_file(spec.script))
    with pytest.raises(RuntimeError, match="checkpoint gate failed"):
        run_checkpoint_gate(spec, verify_existing=False)
    evidence = json.loads(
        (spec.output_dir / "checkpoint_gate_step5_invocation.json").read_text()
    )
    assert evidence["returncode"] == 1
    assert not spec.receipt.exists()


@pytest.mark.parametrize(
    ("step", "save_steps", "message"),
    [(0, 1, "positive"), (5, 2, "coincide"), (20, 1, "precede")],
)
def test_gate_contract_rejects_invalid_step_shapes(
    tmp_path: Path,
    step: int,
    save_steps: int,
    message: str,
) -> None:
    spec = make_spec(tmp_path)
    with pytest.raises(ValueError, match=message):
        build_checkpoint_gate_spec(
            project_root=tmp_path / "project",
            output_dir=tmp_path / "run",
            optimizer_steps=20,
            save_steps=save_steps,
            step=step,
            script=spec.script,
            receipt=spec.receipt,
            tasks=spec.tasks,
            tasks_manifest=spec.tasks_manifest,
            initial_adapter=spec.initial_adapter,
        )


def test_runner_places_gate_between_checkpoint_save_and_next_step() -> None:
    source = Path("src/rl/frameworks/trl/run_transition_grpo.py").read_text()
    callback = source.index("class SynchronousCheckpointGateCallback")
    on_save = source.index("def on_save", callback)
    invocation = source.index("run_checkpoint_gate(self.spec", on_save)
    snapshot = source.index("snapshot_implementation_sources(args.output_dir")
    registration = source.index("trainer.add_callback(SynchronousCheckpointGateCallback")
    training = source.index("trainer.train(", registration)
    assert callback < on_save < invocation
    assert snapshot < registration < training
