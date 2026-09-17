from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import hashlib
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from rl.configuration.experiment_config import (  # noqa: E402
    BASE_MODEL_IDENTITY_SCHEMA_VERSION,
    QWEN3_8B_BASE_MODEL_FILES,
    RLExperimentConfig,
    base_model_aggregate_sha256,
    require_resume_base_model_identity,
    runtime_content_tree_sha256,
    verify_base_model_identity,
    validate_runtime_identity,
)
from rl.evaluation.runners.formal_v26_matched_eval import (  # noqa: E402
    content_tree as formal_runtime_content_tree,
)
from rl.frameworks.trl.transition_grpo import use_frozen_reference_adapter  # noqa: E402


V26_PROTOCOL_HASH = "4da19387399bd3a5"
V26_PROMPT_SHA256 = (
    "848598074e653648d52b58dfa1a54dd777beabae16cb91d83c4ba1a54b5d7316"
)
SFT1_SHA256 = (
    "3ecbbe3dbb65bb26d0308b09d20496c0023b3090ecc44a36c98bb51024efbab5"
)
V26_RUNTIME_TREE_SHA256 = (
    "5fecf5b40447c956a470957022ca4eff8ba9ea0804a4e070b9742959edc00bab"
)
QWEN3_8B_BASE_MODEL_AGGREGATE_SHA256 = (
    "85bd3b7d908acb3a9b9c7ec57b98d6b9e3b2fb427685ae808d1c43173279cecc"
)


def _runtime_kwargs(**overrides):
    values = {
        "expected_protocol_version": "version26",
        "expected_protocol_hash": V26_PROTOCOL_HASH,
        "expected_student_prompt_sha256": V26_PROMPT_SHA256,
        "expected_initial_adapter_sha256": SFT1_SHA256,
        "expected_reference_adapter_sha256": SFT1_SHA256,
        "actual_protocol_version": "version26",
        "actual_protocol_hash": V26_PROTOCOL_HASH,
        "actual_student_prompt_sha256": V26_PROMPT_SHA256,
        "actual_initial_adapter_sha256": SFT1_SHA256,
        "kl_beta": 0.001,
    }
    values.update(overrides)
    return values


def test_qwen3_v26_vanilla_grpo_contract_is_binary_pinned_and_length_safe() -> None:
    path = (
        ROOT
        / "src"
        / "rl"
        / "configs"
        / "experiments"
        / "qwen3_8b_atomic_v26_vanilla_grpo.yaml"
    )
    config = RLExperimentConfig.load(path)
    defaults = config.argparse_defaults(ROOT)

    assert defaults["reward_mode"] == "result-only"
    assert defaults["result_reward_profile"] == "binary"
    assert defaults["rank_loss_coefficient"] == 0.0
    assert defaults["policy_reduction"] == "trajectory_token_mean"
    assert defaults["kl_beta"] == 0.0
    assert defaults["expected_protocol_version"] == "version26"
    assert defaults["expected_runtime_content_tree_sha256"] == (
        V26_RUNTIME_TREE_SHA256
    )
    assert defaults["expected_protocol_hash"] == V26_PROTOCOL_HASH
    assert defaults["expected_student_prompt_sha256"] == V26_PROMPT_SHA256
    assert defaults["expected_initial_adapter_sha256"] == SFT1_SHA256
    assert defaults["expected_reference_adapter_sha256"] == SFT1_SHA256
    assert defaults["expected_records"] == 600
    assert defaults["optimizer_steps"] == 20
    assert defaults["prompts_per_update"] == 30
    assert defaults["group_size"] == 8
    assert defaults["gradient_accumulation_steps"] == 1
    assert (
        defaults["optimizer_steps"]
        * defaults["prompts_per_update"]
        * defaults["group_size"]
        == 4800
    )
    assert defaults["max_new_tokens"] == 2048
    assert defaults["max_context_tokens"] == 16384
    assert defaults["enable_thinking"] is True


def test_qwen3_v26_boundary300_vanilla_grpo_is_exactly_two_passes() -> None:
    path = (
        ROOT
        / "src/rl/configs/experiments/qwen3_8b_atomic_v26_vanilla_grpo_boundary300.yaml"
    )
    config = RLExperimentConfig.load(path)
    defaults = config.argparse_defaults(ROOT)
    assert defaults["reward_mode"] == "result-only"
    assert defaults["result_reward_profile"] == "binary"
    assert defaults["policy_reduction"] == "trajectory_token_mean"
    assert defaults["expected_records"] == 300
    assert defaults["optimizer_steps"] == 20
    assert defaults["prompts_per_update"] == 30
    assert defaults["group_size"] == 8
    assert defaults["learning_rate"] == 8e-7
    assert defaults["kl_beta"] == 0.0
    assert defaults["optimizer_steps"] * defaults["prompts_per_update"] == 600
    assert 600 // defaults["expected_records"] == 2
    assert (
        defaults["optimizer_steps"]
        * defaults["prompts_per_update"]
        * defaults["group_size"]
        == 4800
    )
    assert defaults["max_new_tokens"] == 2048
    assert defaults["max_context_tokens"] == 16384
    assert defaults["enable_thinking"] is True


def _write_tiny_base_model(model_root: Path) -> dict[str, object]:
    files_sha256 = {}
    for index, filename in enumerate(QWEN3_8B_BASE_MODEL_FILES):
        content = f"frozen-model-file-{index}:{filename}\n".encode("utf-8")
        (model_root / filename).write_bytes(content)
        files_sha256[filename] = hashlib.sha256(content).hexdigest()
    canonical = b"".join(
        f"{filename}\t{files_sha256[filename]}\n".encode("utf-8")
        for filename in sorted(files_sha256, key=lambda value: value.encode("utf-8"))
    )
    return {
        "schema_version": BASE_MODEL_IDENTITY_SCHEMA_VERSION,
        "aggregate_sha256": hashlib.sha256(canonical).hexdigest(),
        "files_sha256": files_sha256,
    }


def test_base_model_identity_hashes_every_required_file_and_detects_drift(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    expected = _write_tiny_base_model(model_root)
    assert base_model_aggregate_sha256(expected["files_sha256"]) == expected[
        "aggregate_sha256"
    ]
    assert verify_base_model_identity(model_root, expected) == expected

    (model_root / "generation_config.json").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="generation_config.json"):
        verify_base_model_identity(model_root, expected)


def test_resume_requires_manifest_and_lock_to_bind_the_same_base_model(
    tmp_path: Path,
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    identity = _write_tiny_base_model(model_root)
    output = tmp_path / "train"
    checkpoint = output / "checkpoint-4"
    checkpoint.mkdir(parents=True)

    with pytest.raises(ValueError, match="missing identity artifact"):
        require_resume_base_model_identity(
            output_dir=output,
            resume_checkpoint=checkpoint,
            base_model_identity=identity,
        )

    for filename in ("run_manifest.json", "implementation_lock.json"):
        (output / filename).write_text(
            json.dumps({"base_model_identity": identity}),
            encoding="utf-8",
        )
    require_resume_base_model_identity(
        output_dir=output,
        resume_checkpoint=checkpoint,
        base_model_identity=identity,
    )

    changed = json.loads(json.dumps(identity))
    changed["files_sha256"]["config.json"] = "0" * 64
    changed["aggregate_sha256"] = base_model_aggregate_sha256(
        changed["files_sha256"]
    )
    (output / "implementation_lock.json").write_text(
        json.dumps({"base_model_identity": changed}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="changed since implementation_lock.json"):
        require_resume_base_model_identity(
            output_dir=output,
            resume_checkpoint=checkpoint,
            base_model_identity=identity,
        )


def test_isolated_v26_environment_imports_only_the_exact_frozen_runtime() -> None:
    runtime_root = (
        ROOT
        / "tmp"
        / "version26-runtime-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de"
    )
    if not runtime_root.is_dir():
        pytest.skip("local frozen version26 runtime is not available")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "src" / "rl"),
            str(ROOT / "src"),
            str(runtime_root / "src" / "eval"),
            str(runtime_root / "src" / "harness"),
            str(runtime_root / "src" / "sft"),
        ]
    )
    code = """
import json
import inspect
import sys
import protocol
import rollout as evaluator
import executor
from tool_modules import registry as tool_registry
from rl.runtime import tool_environment_v26 as environment
from rl.frameworks.trl import rollout
print(json.dumps({
    "protocol_version": protocol.PROTOCOL_VERSION,
    "factory_module": rollout.TOOL_ENVIRONMENT_FACTORY_MODULE,
    "environment_implementation": environment.ENVIRONMENT_IMPLEMENTATION,
    "protocol_root": str(environment._PROTOCOL_RUNTIME_ROOT),
    "eval_root": str(environment._EVAL_RUNTIME_ROOT),
    "process_credit_loaded": "process_credit" in sys.modules,
    "trajectory_replay_loaded": "trajectory_replay" in sys.modules,
    "module_paths": {
        "protocol": inspect.getfile(protocol),
        "rollout": inspect.getfile(evaluator),
        "executor": inspect.getfile(executor),
        "tool_registry": inspect.getfile(tool_registry),
    },
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    audit = json.loads(completed.stdout)
    assert audit["protocol_version"] == "version26"
    assert audit["factory_module"] == "rl.runtime.tool_environment_v26"
    assert audit["environment_implementation"] == "atomic-v26-isolated-v1"
    assert audit["protocol_root"] == str(runtime_root)
    assert audit["eval_root"] == str(runtime_root)
    assert audit["process_credit_loaded"] is False
    assert audit["trajectory_replay_loaded"] is False
    assert audit["module_paths"] == {
        "protocol": str(runtime_root / "src" / "sft" / "protocol.py"),
        "rollout": str(runtime_root / "src" / "eval" / "rollout.py"),
        "executor": str(runtime_root / "src" / "harness" / "executor.py"),
        "tool_registry": str(ROOT / "src" / "tool_modules" / "registry.py"),
    }


def test_runtime_tree_identity_is_stable_across_python_cache_and_resume() -> None:
    with tempfile.TemporaryDirectory() as directory:
        runtime_root = Path(directory)
        exported = ["src/eval", "src/sft", "src/harness"]
        files = {
            "src/eval/rollout.py": b"EVAL\n",
            "src/sft/protocol.py": b"SFT\n",
            "src/harness/executor.py": b"HARNESS\n",
        }
        for relative, content in files.items():
            path = runtime_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        digest = hashlib.sha256()
        for relative in sorted(files):
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(files[relative])
            digest.update(b"\0")
        expected = digest.hexdigest()
        (runtime_root / "runtime_lock.json").write_text(
            json.dumps(
                {
                    "exported_paths": exported,
                    "exported_file_count": len(files),
                    "content_tree_sha256": expected,
                }
            ),
            encoding="utf-8",
        )

        assert runtime_content_tree_sha256(runtime_root) == expected
        cache = runtime_root / "src" / "sft" / "__pycache__" / "protocol.pyc"
        cache.parent.mkdir()
        cache.write_bytes(b"import side effect")
        (runtime_root / "src" / "eval" / "standalone.pyc").write_bytes(b"bytecode")
        (runtime_root / "src" / "eval" / "standalone.pyo").write_bytes(b"optimized")
        pytest_cache = runtime_root / "src" / "harness" / ".pytest_cache" / "README.md"
        pytest_cache.parent.mkdir()
        pytest_cache.write_bytes(b"pytest state")
        (runtime_root / "src" / "harness" / ".DS_Store").write_bytes(b"finder")
        assert runtime_content_tree_sha256(runtime_root) == expected
        count, formal_digest = formal_runtime_content_tree(runtime_root, exported)
        assert count == len(files)
        assert formal_digest == expected

        extra_source = runtime_root / "src" / "sft" / "unexpected.py"
        extra_source.write_bytes(b"SOURCE DRIFT\n")
        with pytest.raises(ValueError, match="content tree mismatch"):
            runtime_content_tree_sha256(runtime_root)
        _, formal_drift = formal_runtime_content_tree(runtime_root, exported)
        assert formal_drift != expected
        extra_source.unlink()

        (runtime_root / "src" / "sft" / "protocol.py").write_bytes(b"CHANGED\n")
        with pytest.raises(ValueError, match="content tree mismatch"):
            runtime_content_tree_sha256(runtime_root)


def test_runner_preloads_identity_modules_before_path_mutating_rl_imports() -> None:
    source = (
        ROOT / "src" / "rl" / "frameworks" / "trl" / "run_transition_grpo.py"
    ).read_text(encoding="utf-8")
    boundary = source.index("from rl.runtime.counterfactual_suite import")
    for statement in (
        "import protocol as protocol_runtime",
        "import rollout as evaluator_runtime",
        "import executor as executor_runtime",
        "from tool_modules import registry as tool_schemes_runtime",
    ):
        assert source.index(statement) < boundary


def test_implementation_lock_covers_every_eager_rl_policy_module() -> None:
    runner = ROOT / "src/rl/frameworks/trl/run_transition_grpo.py"
    tree = ast.parse(runner.read_text(encoding="utf-8"))
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "IMPLEMENTATION_SOURCE_FILES"
            for target in node.targets
        )
    )
    locked = set(ast.literal_eval(assignment.value))
    assert {
        "src/rl/runtime/counterfactual_suite.py",
        "src/rl/runtime/reference_result_filter.py",
        "src/rl/frameworks/trl/serving_contract.py",
        "src/rl/frameworks/trl/fixed_rollout_pool.py",
        "src/rl/scenarios/diagnostics/prepare_vanilla_grpo_resume.py",
        "src/rl/frameworks/trl/run_atomic_transition_grpo.sh",
        "src/rl/frameworks/trl/start_vllm_server.sh",
        "src/rl/frameworks/trl/tool_loss_mask.py",
        "src/rl/frameworks/trl/trajectory_ranking.py",
    } <= locked


def test_factory_audit_accepts_only_canonical_environment_path(tmp_path):
    import inspect
    from types import ModuleType

    runner = ROOT / "src/rl/frameworks/trl/run_transition_grpo.py"
    tree = ast.parse(runner.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "protocol_module_path_audit")
    scope = {"ROOT": tmp_path, "Path": Path, "Any": object, "inspect": inspect,
             "TOOL_ENVIRONMENT_FACTORY_MODULE": "rl.runtime.tool_environment_v26"}
    paths = {"protocol_runtime": "src/sft/protocol.py", "evaluator_runtime": "src/eval/rollout.py",
             "executor_runtime": "src/harness/executor.py", "tool_schemes_runtime": "src/tool_modules/registry.py",
             "tool_environment_runtime": "src/rl/runtime/tool_environment_v26.py"}
    for name, path in paths.items():
        module = ModuleType(name)
        module.__file__ = str(tmp_path / path)
        scope[name] = module
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(runner), "exec"), scope)
    audit = scope["protocol_module_path_audit"](tmp_path)
    assert audit["module_paths"]["tool_environment"] == str(tmp_path / paths["tool_environment_runtime"])
    scope["tool_environment_runtime"].__file__ = str(tmp_path / "wrong/tool_environment_v26.py")
    with pytest.raises(RuntimeError, match="tool_environment import escaped"):
        scope["protocol_module_path_audit"](tmp_path)


def test_constructor_preserves_span_alpha_and_cpu_preflight_precedes_model_load():
    trainer_tree = ast.parse((ROOT / "src/rl/frameworks/trl/transition_grpo.py").read_text())
    constructor = next(n for n in ast.walk(trainer_tree) if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    call = next(n for n in ast.walk(constructor) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "from_legacy_args")
    assert {"span_balance_alpha", "span_routing"} <= {k.arg for k in call.keywords}
    runner_tree = ast.parse((ROOT / "src/rl/frameworks/trl/run_transition_grpo.py").read_text())
    main = next(n for n in runner_tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    preflight_return = next(n for n in main.body if isinstance(n, ast.If)
                            and ast.unparse(n.test) == "args.preflight_only")
    assert any(isinstance(n, ast.Return) for n in preflight_return.body)
    model_load = next(n for n in ast.walk(main) if isinstance(n, ast.Call)
                      and isinstance(n.func, ast.Name) and n.func.id == "load_qlora_model")
    assert preflight_return.lineno < model_load.lineno


def test_runner_exposes_checkpoint_retention_without_changing_grpo_objective() -> None:
    source = (
        ROOT / "src/rl/frameworks/trl/run_transition_grpo.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--save-total-limit", type=int, default=3)' in source
    assert 'if args.save_steps < 1 or args.save_total_limit < 1:' in source
    assert 'save_total_limit=args.save_total_limit' in source
    assert '"save_steps": args.save_steps' in source
    assert '"save_total_limit": args.save_total_limit' in source


def test_runner_embeds_one_base_model_identity_and_checks_it_before_resume_load() -> None:
    source = (
        ROOT / "src/rl/frameworks/trl/run_transition_grpo.py"
    ).read_text(encoding="utf-8")
    assert '"base_model_identity": base_model_identity' in source
    assert '"base_model_identity": manifest["base_model_identity"]' in source
    resume_check = source.index("require_resume_base_model_identity(")
    model_load = source.index("load_qlora_model(", resume_check)
    assert resume_check < model_load


def test_nonzero_kl_accepts_only_the_exact_initial_sft_reference() -> None:
    audit = validate_runtime_identity(**_runtime_kwargs())
    assert audit["pinned"] is True
    assert audit["reference"] == {
        "enabled": True,
        "expected_adapter_sha256": SFT1_SHA256,
        "actual_adapter_sha256": SFT1_SHA256,
        "equals_initial_adapter": True,
    }

    with pytest.raises(ValueError, match="frozen KL reference must equal"):
        validate_runtime_identity(
            **_runtime_kwargs(expected_reference_adapter_sha256="a" * 64)
        )
    with pytest.raises(ValueError, match="runtime identity mismatch for protocol_hash"):
        validate_runtime_identity(**_runtime_kwargs(actual_protocol_hash="b" * 16))
    with pytest.raises(ValueError, match="requires an expected frozen reference"):
        validate_runtime_identity(
            **_runtime_kwargs(expected_reference_adapter_sha256=None)
        )


def test_zero_kl_remains_available_without_a_pinned_reference() -> None:
    audit = validate_runtime_identity(
        **_runtime_kwargs(
            expected_protocol_version=None,
            expected_protocol_hash=None,
            expected_student_prompt_sha256=None,
            expected_initial_adapter_sha256=None,
            expected_reference_adapter_sha256=None,
            kl_beta=0.0,
        )
    )
    assert audit["pinned"] is False
    assert audit["reference"]["enabled"] is False


class _FakeParameter:
    def __init__(self, requires_grad: bool):
        self.requires_grad = requires_grad


class _FakePeftModel:
    def __init__(self):
        self.active_adapter = "default"
        self._parameters = {
            "layer.default.weight": _FakeParameter(True),
            "layer.frozen_sft_reference.weight": _FakeParameter(False),
        }

    def named_parameters(self):
        return list(self._parameters.items())

    def set_adapter(self, name: str, *, inference_mode: bool = False):
        for parameter in self._parameters.values():
            parameter.requires_grad = False
        self._parameters[f"layer.{name}.weight"].requires_grad = not inference_mode
        self.active_adapter = name


def test_frozen_reference_context_never_makes_reference_trainable() -> None:
    model = _FakePeftModel()
    reference = model._parameters["layer.frozen_sft_reference.weight"]
    policy = model._parameters["layer.default.weight"]

    with use_frozen_reference_adapter(model, "frozen_sft_reference"):
        assert model.active_adapter == "frozen_sft_reference"
        assert reference.requires_grad is False
        assert policy.requires_grad is False

    assert model.active_adapter == "default"
    assert reference.requires_grad is False
    assert policy.requires_grad is True
