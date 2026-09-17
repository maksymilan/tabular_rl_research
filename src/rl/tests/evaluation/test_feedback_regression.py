from pathlib import Path
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from rl.evaluation.runners.feedback_regression import serving_library_environment
from rl.evaluation.runners.formal_v26_matched_eval import sha256_file


def test_library_environment_is_opt_in(monkeypatch):
    monkeypatch.delenv("TRITON_LIBCUDA_PATH", raising=False)
    assert serving_library_environment() == {"environment": {}, "driver": None}


def test_library_environment_records_existing_link_without_modification(tmp_path, monkeypatch):
    driver = tmp_path / "libcuda.so.1"
    driver.write_bytes(b"test fixture, not a CUDA library")
    link = tmp_path / "libcuda.so"
    link.symlink_to(driver)
    monkeypatch.setenv("TRITON_LIBCUDA_PATH", str(tmp_path))
    audit = serving_library_environment()
    assert audit["environment"] == {"TRITON_LIBCUDA_PATH": str(tmp_path)}
    assert audit["driver"] == {"link": str(link), "resolved": str(driver), "sha256": sha256_file(driver)}
    assert link.is_symlink() and Path(audit["driver"]["resolved"]) == driver


def test_library_environment_rejects_missing_link(tmp_path, monkeypatch):
    monkeypatch.setenv("TRITON_LIBCUDA_PATH", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no usable libcuda.so"):
        serving_library_environment()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("shard_count", [3, 5])
def test_model_identity_covers_pinned_shards_for_both_model_sizes(tmp_path, monkeypatch, shard_count):
    from rl.evaluation.runners.formal_v26_matched_eval import verify_model
    shards = {f"model-{i:05d}-of-{shard_count:05d}.safetensors" for i in range(1, shard_count + 1)}
    names = shards | {"config.json", "generation_config.json", "merges.txt",
                      "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"}
    for name in names:
        (tmp_path / name).write_text(name)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}))
    index = {"weight_map": {str(i): name for i, name in enumerate(sorted(shards))}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    files = {name: sha256_file(tmp_path / name) for name in sorted(names)}
    aggregate = hashlib.sha256("".join(f"{name}\t{files[name]}\n" for name in sorted(files)).encode()).hexdigest()
    tokenizer = SimpleNamespace(chat_template="template", apply_chat_template=lambda *a, **k: "<|im_start|>assistant\n")
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)))
    model = {"repository": "test/qwen3", "revision": "test", "config_sha256": files["config.json"],
             "model_index_sha256": files["model.safetensors.index.json"],
             "tokenizer_config_sha256": files["tokenizer_config.json"],
             "chat_template_sha256": hashlib.sha256(b"template").hexdigest(),
             "shards_sha256": {n: files[n] for n in shards},
             "base_model_identity": {"schema_version": "trl-base-model-identity-v1", "files_sha256": files, "aggregate_sha256": aggregate}}
    assert verify_model({"model": model}, tmp_path)["base_model_identity_sha256"] == aggregate
    (tmp_path / sorted(shards)[0]).write_text("changed weights")
    with pytest.raises(ValueError, match="file hashes"):
        verify_model({"model": model}, tmp_path)
    index["weight_map"].pop("0")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="shard coverage"):
        verify_model({"model": model}, tmp_path)


def test_sft_preflight_rejects_wrong_checkpoint_step_before_asset_loading(tmp_path):
    from rl.evaluation.runners.feedback_regression import prepare_feedback_regression
    (tmp_path / "trainer_state.json").write_text(json.dumps({"global_step": 4785, "epoch": 3.0}))
    with pytest.raises(ValueError, match="SFT checkpoint step"):
        prepare_feedback_regression(run_root=tmp_path, control_root=tmp_path, contract_path=tmp_path,
                                    runtime=tmp_path, model=tmp_path, adapter=tmp_path,
                                    source=tmp_path, databases=tmp_path, baseline=tmp_path,
                                    baseline_manifest=tmp_path, adapter_sha256="", baseline_sha256="",
                                    max_tokens=2048, gpu_ids=[0, 1], ports=[18330, 18331],
                                    checkpoint_global_step=6380)
