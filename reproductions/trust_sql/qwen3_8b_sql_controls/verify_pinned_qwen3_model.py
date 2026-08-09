#!/usr/bin/env python3
"""Fail-closed verifier for a pinned Qwen3-4B or Qwen3-8B snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def verify(model_root: Path, size: str, specs_path: Path) -> dict[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError("transformers is unavailable") from exc

    specs = load_object(specs_path)
    if specs.get("schema_version") != "pinned-qwen3-model-specs-v1":
        raise ValueError("unsupported model-spec schema")
    model_spec = (specs.get("models") or {}).get(size)
    if not isinstance(model_spec, dict):
        raise ValueError(f"model size absent from spec: {size}")
    model_root = model_root.resolve()
    config_path = model_root / "config.json"
    index_path = model_root / "model.safetensors.index.json"
    tokenizer_config_path = model_root / "tokenizer_config.json"
    for required in (config_path, index_path, tokenizer_config_path):
        if not required.is_file():
            raise ValueError(f"missing model artifact: {required}")

    actual_fixed = {
        "config_sha256": sha256(config_path),
        "model_index_sha256": sha256(index_path),
        "tokenizer_config_sha256": sha256(tokenizer_config_path),
    }
    expected_fixed = {
        "config_sha256": model_spec["config_sha256"],
        "model_index_sha256": model_spec["model_index_sha256"],
        "tokenizer_config_sha256": specs["tokenizer_config_sha256"],
    }
    if actual_fixed != expected_fixed:
        raise ValueError(
            f"fixed-file hash drift: expected={expected_fixed}, actual={actual_fixed}"
        )

    config = load_object(config_path)
    expectations = model_spec.get("config_expectations") or {}
    mismatches = {
        key: {"expected": expected, "actual": config.get(key)}
        for key, expected in expectations.items()
        if config.get(key) != expected
    }
    if config.get("architectures") != ["Qwen3ForCausalLM"]:
        mismatches["architectures"] = {
            "expected": ["Qwen3ForCausalLM"],
            "actual": config.get("architectures"),
        }
    if mismatches:
        raise ValueError(f"architecture drift: {mismatches}")

    shard_expectations = model_spec.get("shards_sha256") or {}
    if not shard_expectations:
        raise ValueError("model spec contains no weight shards")
    shard_hashes: dict[str, str] = {}
    for name, expected in sorted(shard_expectations.items()):
        path = model_root / name
        if not path.is_file():
            raise ValueError(f"missing model shard: {path}")
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"weight hash drift for {name}: {actual} != {expected}")
        shard_hashes[name] = actual

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_root), local_files_only=True, trust_remote_code=False
    )
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("tokenizer has no chat template")
    template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if template_hash != specs["chat_template_sha256"]:
        raise ValueError(f"chat-template drift: {template_hash}")
    messages = [
        {"role": "system", "content": "system contract"},
        {"role": "user", "content": "test question"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    if not rendered.endswith("<|im_start|>assistant\n"):
        raise ValueError("enable_thinking=true generation boundary drift")
    if rendered.endswith("<think>\n\n</think>\n\n"):
        raise ValueError("thinking was unexpectedly suppressed")

    return {
        "status": "ok",
        "model_size": size,
        "model_root": str(model_root),
        "repository": model_spec["repository"],
        "model_revision": model_spec["revision"],
        **actual_fixed,
        "chat_template_sha256": template_hash,
        "model_shards_sha256": shard_hashes,
        "enable_thinking": True,
        "reasoning_parser": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--model-size", choices=("4b", "8b"), required=True)
    parser.add_argument("--specs", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = verify(args.model_root, args.model_size, args.specs)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"pinned Qwen3 model gate failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
