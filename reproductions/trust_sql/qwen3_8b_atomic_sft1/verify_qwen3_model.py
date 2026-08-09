#!/usr/bin/env python3
"""Read-only gate for the pinned official Qwen3-8B tokenizer and an optional LoRA adapter."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXPECTED_MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_TOKENIZER_CONFIG_SHA256 = "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101"
EXPECTED_CHAT_TEMPLATE_SHA256 = "a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def verify_model(model_root: Path, adapter: Path | None, max_lora_rank: int) -> dict:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ValueError("transformers is not installed in the selected vLLM environment") from exc

    model_root = model_root.resolve()
    config_path = model_root / "config.json"
    tokenizer_config_path = model_root / "tokenizer_config.json"
    if not config_path.is_file() or not tokenizer_config_path.is_file():
        raise ValueError(f"model is incomplete: {model_root}")

    config = load_json(config_path)
    if config.get("model_type") != "qwen3":
        raise ValueError(f"expected model_type=qwen3, got {config.get('model_type')!r}")
    architectures = config.get("architectures") or []
    if "Qwen3ForCausalLM" not in architectures:
        raise ValueError(f"expected Qwen3ForCausalLM architecture, got {architectures!r}")

    tokenizer_config_hash = sha256(tokenizer_config_path)
    if tokenizer_config_hash != EXPECTED_TOKENIZER_CONFIG_SHA256:
        raise ValueError(
            "tokenizer_config.json does not match the pinned official Qwen3-8B revision: "
            f"{tokenizer_config_hash}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_root),
        local_files_only=True,
        trust_remote_code=False,
    )
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str) or not template:
        raise ValueError("tokenizer has no chat_template")
    template_hash = hashlib.sha256(template.encode("utf-8")).hexdigest()
    if template_hash != EXPECTED_CHAT_TEMPLATE_SHA256:
        raise ValueError(
            f"official Qwen3 chat-template hash mismatch: expected "
            f"{EXPECTED_CHAT_TEMPLATE_SHA256}, got {template_hash}"
        )

    messages = [
        {"role": "system", "content": "system contract"},
        {"role": "user", "content": "one test question"},
    ]
    thinking_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    no_thinking_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assistant_marker = "<|im_start|>assistant\n"
    empty_think = "<think>\n\n</think>\n\n"
    if not thinking_text.endswith(assistant_marker):
        raise ValueError("enable_thinking=true did not leave the official assistant generation boundary")
    if thinking_text.endswith(empty_think):
        raise ValueError("enable_thinking=true unexpectedly injected the empty-think suppression prefix")
    if not no_thinking_text.endswith(empty_think):
        raise ValueError("enable_thinking=false did not inject the expected empty-think prefix")

    adapter_report = None
    if adapter is not None:
        adapter = adapter.resolve()
        adapter_config_path = adapter / "adapter_config.json"
        if not adapter_config_path.is_file():
            raise ValueError(f"missing adapter_config.json: {adapter}")
        if not any((adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
            raise ValueError(f"missing adapter weights: {adapter}")
        adapter_config = load_json(adapter_config_path)
        peft_type = str(adapter_config.get("peft_type", "")).upper()
        if peft_type != "LORA":
            raise ValueError(f"expected a LoRA/QLoRA adapter, got peft_type={peft_type!r}")
        if str(adapter_config.get("task_type", "")).upper() != "CAUSAL_LM":
            raise ValueError(
                f"expected task_type=CAUSAL_LM, got {adapter_config.get('task_type')!r}"
            )
        base_reference = adapter_config.get("base_model_name_or_path")
        if not isinstance(base_reference, str) or not base_reference:
            raise ValueError("adapter does not record base_model_name_or_path")
        if Path(base_reference).is_absolute():
            if Path(base_reference).resolve() != model_root:
                raise ValueError(
                    f"adapter was trained against a different local base: {base_reference}"
                )
        elif base_reference != "Qwen/Qwen3-8B":
            raise ValueError(f"unexpected adapter base model: {base_reference}")
        rank = int(adapter_config.get("r", 0))
        if rank <= 0 or rank > max_lora_rank:
            raise ValueError(f"adapter rank {rank} is outside vLLM max_lora_rank={max_lora_rank}")
        adapter_report = {
            "path": str(adapter),
            "peft_type": peft_type,
            "rank": rank,
            "base_model_name_or_path": base_reference,
        }

    return {
        "status": "ok",
        "model_root": str(model_root),
        "assumed_huggingface_revision": EXPECTED_MODEL_REVISION,
        "model_type": config["model_type"],
        "architecture": "Qwen3ForCausalLM",
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_config_sha256": tokenizer_config_hash,
        "chat_template_sha256": template_hash,
        "official_chat_template": True,
        "enable_thinking": True,
        "empty_think_suppression_injected": False,
        "reasoning_parser": None,
        "adapter": adapter_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    args = parser.parse_args()
    try:
        report = verify_model(args.model_root, args.adapter, args.max_lora_rank)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"Qwen3 model gate failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
