from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import pytest

from src.rl.diagnostics.analyze_evaluation_results import (
    analyze_results,
    evaluation_identity,
    exact_mcnemar_p,
    js_divergence_bits,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _row(
    index: int,
    *,
    correct: bool,
    legal: bool,
    steps: int,
    calls: list[tuple[str, dict]],
    temperature: float = 0.0,
) -> dict:
    return {
        "example_index": index,
        "protocol_version": "version36",
        "protocol_hash": "20a8d3b4356d883c",
        "temperature": temperature,
        "top_p": 1.0,
        "denotation_comparison": "bird-set",
        "samples": [
            {
                "correct": correct,
                "legal": legal,
                "steps": steps,
                "turns": [
                    {"parsed": {"tool": tool, "arguments": arguments}}
                    for tool, arguments in calls
                ],
            }
        ],
    }


def _identity(
    adapter_sha256: str,
    *,
    runtime: str = "/frozen/runtime",
    concurrency: int = 24,
) -> dict:
    return {
        "schema_version": "qwen3-grpo-checkpoint-evaluation-v1",
        "adapter_sha256": adapter_sha256,
        "base_model": "/models/Qwen3-8B",
        "dataset": "BIRD-dev1534",
        "input_sha256": "input-hash",
        "task_identity_sha256_without_db_path": "task-hash",
        "protocol_version": "version36",
        "protocol_hash": "20a8d3b4356d883c",
        "runtime": runtime,
        "concurrency": {
            "workers": concurrency,
            "max_inflight_requests": concurrency,
            "max_num_seqs": concurrency,
        },
        "decode": {
            "enable_thinking": True,
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "agent": {
            "denotation_comparison": "bird-set",
            "history_turns": 4,
            "max_steps": 30,
        },
        "tool_execution_timeout_seconds": 10.0,
    }


def test_exact_mcnemar_matches_known_small_cases() -> None:
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(2, 0) == pytest.approx(0.5)
    assert exact_mcnemar_p(4, 0) == pytest.approx(0.125)
    assert exact_mcnemar_p(3, 3) == 1.0


def test_js_divergence_does_not_depend_on_counter_insertion_order() -> None:
    first_left = Counter({"z": 3, "a": 1})
    second_left = Counter({"a": 1, "z": 3})
    right = Counter({"m": 2, "a": 2})
    assert js_divergence_bits(first_left, right) == js_divergence_bits(
        second_left, right
    )


def test_evaluation_identity_is_loaded_from_result_directory(tmp_path: Path) -> None:
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    result = result_dir / "all.jsonl"
    result.write_text("", encoding="utf-8")
    identity = {"schema_version": "evaluation-model-identity-v1", "x": 1}
    (result_dir / "evaluation_identity.json").write_text(json.dumps(identity))
    assert evaluation_identity(result) == identity


def test_unified_arm_pair_and_policy_metrics(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    indices = tmp_path / "indices.json"
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(
        examples,
        [
            {"example_index": 0, "metadata": {"difficulty": "simple"}},
            {"example_index": 1, "metadata": {"difficulty": "moderate"}},
            {"example_index": 2, "metadata": {"difficulty": "challenging"}},
        ],
    )
    indices.write_text(json.dumps({"indices": [2, 0]}), encoding="utf-8")
    _write_jsonl(
        baseline,
        [
            _row(
                0,
                correct=True,
                legal=True,
                steps=2,
                calls=[("describe", {"table": "a"}), ("project", {"x": 1})],
            ),
            _row(1, correct=True, legal=True, steps=1, calls=[("unused", {})]),
            _row(
                2,
                correct=False,
                legal=False,
                steps=1,
                calls=[("filter", {"value": 1})],
            ),
        ],
    )
    _write_jsonl(
        candidate,
        [
            _row(
                0,
                correct=False,
                legal=False,
                steps=3,
                calls=[("describe", {"table": "a"}), ("join", {"x": 1})],
            ),
            _row(1, correct=False, legal=False, steps=1, calls=[("unused", {})]),
            _row(
                2,
                correct=True,
                legal=True,
                steps=1,
                calls=[("filter", {"value": 2})],
            ),
        ],
    )

    result = analyze_results(
        examples_path=examples,
        indices_path=indices,
        arm_paths={"sft2": baseline, "rl": candidate},
        comparisons=[("rl", "sft2")],
        expected_count=2,
        protocol_version="version36",
        protocol_hash="20a8d3b4356d883c",
        temperature=0.0,
        top_p=1.0,
        denotation_comparison="bird-set",
        include_per_example=True,
    )

    assert result["cohort"]["indices"] == [2, 0]
    assert result["cohort"]["difficulty_counts"] == {
        "challenging": 1,
        "simple": 1,
    }
    assert result["arms"]["sft2"]["correct"] == 1
    assert result["arms"]["rl"]["legal"] == 1
    comparison = result["comparisons"]["rl_vs_sft2"]
    assert comparison["accuracy"]["gains"] == 1
    assert comparison["accuracy"]["regressions"] == 1
    assert comparison["accuracy"]["baseline_rate"] == 0.5
    assert comparison["accuracy"]["candidate_rate"] == 0.5
    assert comparison["accuracy"]["net_percentage_points"] == 0.0
    assert comparison["legal"]["net"] == 0
    assert comparison["sequence_change"]["exact_action_sequence_changed"] == 2
    assert comparison["sequence_change"]["first_exact_action_changed"] == 1
    assert comparison["sequence_change"]["tool_sequence_changed"] == 1
    assert comparison["sequence_change"]["arguments_only_changed"] == 1
    assert len(comparison["per_example"]) == 2


def test_contract_mismatch_is_rejected(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(
        examples,
        [{"example_index": 0, "metadata": {"difficulty": "simple"}}],
    )
    _write_jsonl(
        baseline,
        [_row(0, correct=True, legal=True, steps=1, calls=[("x", {})])],
    )
    _write_jsonl(
        candidate,
        [
            _row(
                0,
                correct=True,
                legal=True,
                steps=1,
                calls=[("x", {})],
                temperature=0.7,
            )
        ],
    )
    with pytest.raises(ValueError, match="evaluation contract mismatch"):
        analyze_results(
            examples_path=examples,
            arm_paths={"base": baseline, "candidate": candidate},
            comparisons=[("candidate", "base")],
        )


def test_protocol_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    _write_jsonl(
        examples,
        [{"example_index": 0, "metadata": {"difficulty": "simple"}}],
    )
    baseline_row = _row(
        0, correct=True, legal=True, steps=1, calls=[("x", {})]
    )
    candidate_row = _row(
        0, correct=True, legal=True, steps=1, calls=[("x", {})]
    )
    candidate_row["protocol_hash"] = "wrong-hash"
    _write_jsonl(baseline, [baseline_row])
    _write_jsonl(candidate, [candidate_row])
    with pytest.raises(ValueError, match="evaluation contract mismatch"):
        analyze_results(
            examples_path=examples,
            arm_paths={"base": baseline, "candidate": candidate},
            comparisons=[("candidate", "base")],
        )


def test_strict_identity_guard_accepts_explicit_parent_sidecars(
    tmp_path: Path,
) -> None:
    examples = tmp_path / "examples.jsonl"
    _write_jsonl(
        examples,
        [{"example_index": 0, "metadata": {"difficulty": "simple"}}],
    )
    arm_paths = {}
    identity_paths = {}
    for name, adapter_sha in (("sft1", "sft-sha"), ("checkpoint", "rl-sha")):
        run_dir = tmp_path / name
        result_dir = run_dir / "result"
        result_dir.mkdir(parents=True)
        result = result_dir / "all.jsonl"
        _write_jsonl(
            result,
            [_row(0, correct=True, legal=True, steps=1, calls=[("x", {})])],
        )
        identity = run_dir / "evaluation_identity.json"
        identity.write_text(json.dumps(_identity(adapter_sha)), encoding="utf-8")
        arm_paths[name] = result
        identity_paths[name] = identity

    result = analyze_results(
        examples_path=examples,
        arm_paths=arm_paths,
        comparisons=[("checkpoint", "sft1")],
        identity_paths=identity_paths,
        require_identities=True,
        matched_identity_fields=(
            "base_model",
            "runtime",
            "concurrency",
            "decode",
            "agent",
            "input_sha256",
            "task_identity_sha256_without_db_path",
        ),
        expected_adapter_sha256={"sft1": "sft-sha", "checkpoint": "rl-sha"},
        require_distinct_adapters=True,
    )

    assert result["identity_contract"]["all_present"]
    assert result["identity_contract"]["matched_fields"]["runtime"] == (
        "/frozen/runtime"
    )
    assert result["arms"]["checkpoint"]["evaluation_identity"][
        "adapter_sha256"
    ] == "rl-sha"


def test_strict_identity_guard_rejects_missing_sidecar(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    result = tmp_path / "result" / "all.jsonl"
    result.parent.mkdir()
    _write_jsonl(
        examples,
        [{"example_index": 0, "metadata": {"difficulty": "simple"}}],
    )
    _write_jsonl(
        result,
        [_row(0, correct=True, legal=True, steps=1, calls=[("x", {})])],
    )
    with pytest.raises(ValueError, match="identity is required"):
        analyze_results(
            examples_path=examples,
            arm_paths={"sft1": result},
            comparisons=[],
            require_identities=True,
        )


def test_strict_identity_guard_rejects_runtime_mismatch(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    _write_jsonl(
        examples,
        [{"example_index": 0, "metadata": {"difficulty": "simple"}}],
    )
    arm_paths = {}
    identity_paths = {}
    for name, runtime in (("sft1", "/runtime/a"), ("checkpoint", "/runtime/b")):
        run_dir = tmp_path / name
        result_dir = run_dir / "result"
        result_dir.mkdir(parents=True)
        result = result_dir / "all.jsonl"
        _write_jsonl(
            result,
            [_row(0, correct=True, legal=True, steps=1, calls=[("x", {})])],
        )
        identity = run_dir / "evaluation_identity.json"
        identity.write_text(
            json.dumps(_identity(f"{name}-sha", runtime=runtime)),
            encoding="utf-8",
        )
        arm_paths[name] = result
        identity_paths[name] = identity
    with pytest.raises(ValueError, match="field mismatch for 'runtime'"):
        analyze_results(
            examples_path=examples,
            arm_paths=arm_paths,
            comparisons=[("checkpoint", "sft1")],
            identity_paths=identity_paths,
            require_identities=True,
            matched_identity_fields=("runtime",),
        )


def test_identity_protocol_must_match_result_rows(tmp_path: Path) -> None:
    examples = tmp_path / "examples.jsonl"
    result_dir = tmp_path / "result"
    result_dir.mkdir()
    result = result_dir / "all.jsonl"
    identity = result_dir / "evaluation_identity.json"
    _write_jsonl(
        examples,
        [{"example_index": 0, "metadata": {"difficulty": "simple"}}],
    )
    _write_jsonl(
        result,
        [_row(0, correct=True, legal=True, steps=1, calls=[("x", {})])],
    )
    payload = _identity("adapter-sha")
    payload["protocol_hash"] = "wrong-hash"
    identity.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity/row contract mismatch"):
        analyze_results(
            examples_path=examples,
            arm_paths={"candidate": result},
            comparisons=[],
        )
