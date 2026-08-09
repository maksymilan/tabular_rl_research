#!/usr/bin/env python3
"""Build an auditable Qwen3 view of rolling SFT JSONL records.

The canonical dataset remains untouched.  In the derived view, the only model-visible
mutation is removal of one leading ``<think>...</think>`` block (and the newline
separator immediately after it) from *historical* assistant messages.  The final
assistant target, action JSON text, all non-assistant messages, root fields, and
metadata are retained exactly.

An optional parity audit compares the complete training prefix produced by the
installed LLaMA-Factory Qwen3 template from the derived view with the inference
prefix produced by Hugging Face's Qwen3 chat template from the canonical view.
Heavy training dependencies are imported only when ``--audit-model`` is supplied.

The audited training invariant for this view is ``mask_history=true``,
``preserve_thinking=false``, and ``enable_thinking=true``.  In particular, setting
``preserve_thinking=true`` after thoughts have already been projected out makes the
LLaMA-Factory reasoning template inject empty thought blocks into historical turns,
so it no longer matches Qwen3 inference history.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator


PROJECTION_NAME = "qwen3-history-think-strip-v1"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_id(row: dict[str, Any], line_number: int) -> str:
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"line {line_number}: metadata must be an object")
    record_id = metadata.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise ValueError(f"line {line_number}: metadata.record_id must be a non-empty string")
    return record_id


def _message_role_and_content_key(
    message: dict[str, Any], *, record_id: str, message_index: int
) -> tuple[str, str, str]:
    """Return normalized role, content key, and source schema."""
    has_sharegpt = "from" in message or "value" in message
    has_role_content = "role" in message or "content" in message
    if has_sharegpt and has_role_content:
        raise ValueError(
            f"{record_id}: conversation {message_index} mixes from/value and role/content"
        )
    if has_sharegpt:
        if "from" not in message or "value" not in message:
            raise ValueError(
                f"{record_id}: conversation {message_index} needs both from and value"
            )
        raw_role = message["from"]
        content_key = "value"
        schema = "sharegpt"
    elif has_role_content:
        if "role" not in message or "content" not in message:
            raise ValueError(
                f"{record_id}: conversation {message_index} needs both role and content"
            )
        raw_role = message["role"]
        content_key = "content"
        schema = "role-content"
    else:
        raise ValueError(
            f"{record_id}: conversation {message_index} has no supported role/content fields"
        )

    if not isinstance(raw_role, str) or not raw_role:
        raise ValueError(f"{record_id}: conversation {message_index} has an invalid role")
    normalized = {
        "gpt": "assistant",
        "assistant": "assistant",
        "human": "user",
        "user": "user",
        "tool": "tool",
        "observation": "tool",
        "system": "system",
        "function": "function",
    }.get(raw_role, raw_role)
    return normalized, content_key, schema


def _validate_action_json(text: str, *, location: str) -> None:
    """Validate the retained suffix without normalizing a single character of it."""
    first_non_whitespace = len(text) - len(text.lstrip())
    try:
        action, end = json.JSONDecoder().raw_decode(text, first_non_whitespace)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location}: assistant suffix is not one JSON action: {exc}") from exc
    if text[end:].strip():
        raise ValueError(f"{location}: non-whitespace text follows the JSON action")
    if not isinstance(action, dict):
        raise ValueError(f"{location}: assistant action must be a JSON object")
    if not isinstance(action.get("tool"), str) or not action["tool"]:
        raise ValueError(f"{location}: assistant action needs a non-empty string tool")
    if not isinstance(action.get("arguments"), dict):
        raise ValueError(f"{location}: assistant action needs an object arguments field")


def _thought_suffix(content: str, *, location: str) -> tuple[str, int]:
    """Return the byte-for-byte action suffix after one strict leading thought block.

    Qwen3's Hugging Face chat template applies ``.lstrip("\\n")`` to the content
    after ``</think>``.  Mirroring only that separator rule is necessary for exact
    prefix parity; spaces and every character belonging to the JSON action remain
    untouched.
    """
    if content.count(THINK_OPEN) != 1 or content.count(THINK_CLOSE) != 1:
        raise ValueError(f"{location}: expected exactly one <think>...</think> block")
    if not content.startswith(THINK_OPEN):
        raise ValueError(f"{location}: the think block must begin at character zero")
    close_at = content.find(THINK_CLOSE, len(THINK_OPEN))
    if close_at < 0:
        raise ValueError(f"{location}: malformed think block")
    raw_suffix = content[close_at + len(THINK_CLOSE) :]
    suffix = raw_suffix.lstrip("\n")
    _validate_action_json(suffix, location=location)
    return suffix, len(raw_suffix) - len(suffix)


def project_row(
    row: dict[str, Any], *, line_number: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Project one row while proving every out-of-scope field is unchanged."""
    if not isinstance(row, dict):
        raise ValueError(f"line {line_number}: each JSONL record must be an object")
    record_id = _record_id(row, line_number)
    system = row.get("system")
    if not isinstance(system, str) or not system:
        raise ValueError(f"{record_id}: system must be a non-empty string")
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"{record_id}: conversations must be a non-empty list")
    if not all(isinstance(message, dict) for message in conversations):
        raise ValueError(f"{record_id}: every conversation item must be an object")

    projected = copy.deepcopy(row)
    schemas: Counter[str] = Counter()
    historical_assistant_messages = 0
    removed_separator_newlines = 0

    final_role, final_key, final_schema = _message_role_and_content_key(
        conversations[-1], record_id=record_id, message_index=len(conversations) - 1
    )
    schemas[final_schema] += 1
    if final_role != "assistant":
        raise ValueError(f"{record_id}: final conversation item must be an assistant target")
    final_content = conversations[-1].get(final_key)
    if not isinstance(final_content, str):
        raise ValueError(f"{record_id}: final assistant target must be a string")
    # Validate, but never assign, normalize, or otherwise mutate the supervised target.
    _thought_suffix(final_content, location=f"{record_id}: final target")

    for index, message in enumerate(conversations[:-1]):
        role, content_key, schema = _message_role_and_content_key(
            message, record_id=record_id, message_index=index
        )
        schemas[schema] += 1
        if role != "assistant":
            continue
        content = message.get(content_key)
        if not isinstance(content, str):
            raise ValueError(f"{record_id}: historical assistant {index} must be a string")
        suffix, removed_newlines = _thought_suffix(
            content, location=f"{record_id}: historical assistant {index}"
        )
        projected["conversations"][index][content_key] = suffix
        historical_assistant_messages += 1
        removed_separator_newlines += removed_newlines

    if projected.get("system") != row.get("system"):
        raise AssertionError(f"{record_id}: system changed during projection")
    if projected.get("metadata") != row.get("metadata"):
        raise AssertionError(f"{record_id}: metadata changed during projection")
    if projected["conversations"][-1] != conversations[-1]:
        raise AssertionError(f"{record_id}: final target changed during projection")
    for index, message in enumerate(conversations[:-1]):
        role, content_key, _ = _message_role_and_content_key(
            message, record_id=record_id, message_index=index
        )
        if role != "assistant" and projected["conversations"][index] != message:
            raise AssertionError(f"{record_id}: non-assistant message {index} changed")
        if role == "assistant":
            original_without_content = copy.deepcopy(message)
            projected_without_content = copy.deepcopy(projected["conversations"][index])
            original_without_content.pop(content_key)
            projected_without_content.pop(content_key)
            if projected_without_content != original_without_content:
                raise AssertionError(
                    f"{record_id}: historical assistant fields other than content changed"
                )

    return projected, {
        "record_id": record_id,
        "historical_assistant_messages": historical_assistant_messages,
        "removed_separator_newlines": removed_separator_newlines,
        "message_schemas": dict(schemas),
    }


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_number}: invalid JSON: {exc}") from exc
            yield line_number, row


def _message_for_template(
    message: dict[str, Any],
    *,
    record_id: str,
    message_index: int,
    target: str,
) -> dict[str, str]:
    role, content_key, _ = _message_role_and_content_key(
        message, record_id=record_id, message_index=message_index
    )
    content = message.get(content_key)
    if not isinstance(content, str):
        raise ValueError(f"{record_id}: conversation {message_index} content must be a string")
    if target == "hf":
        mapped_role = {"user": "user", "assistant": "assistant", "tool": "tool"}.get(role)
    elif target == "llamafactory":
        mapped_role = {
            "user": "user",
            "assistant": "assistant",
            "tool": "observation",
        }.get(role)
    else:  # pragma: no cover - internal programming error
        raise AssertionError(target)
    if mapped_role is None:
        raise ValueError(
            f"{record_id}: role {role!r} is unsupported by the {target} parity audit"
        )
    return {"role": mapped_role, "content": content}


def _as_token_ids(value: Any, *, location: str) -> list[int]:
    # Transformers 5.x returns BatchEncoding, which implements Mapping but is
    # deliberately not a dict subclass.
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise TypeError(f"{location}: tokenizer did not return one flat token-id list")
    return value


def _length_summary(values: list[int]) -> dict[str, int]:
    if not values:
        return {"min": 0, "p50": 0, "p90": 0, "p99": 0, "max": 0}
    ordered = sorted(values)

    def percentile(fraction: float) -> int:
        return ordered[math.ceil(fraction * len(ordered)) - 1]

    return {
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def _first_token_difference(left: list[int], right: list[int]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))


def audit_prefix_parity(
    canonical_path: Path,
    projected_path: Path,
    *,
    model_path: str,
    template_name: str = "qwen3",
    cutoff_len: int = 6400,
    local_files_only: bool = True,
) -> dict[str, Any]:
    """Audit every row against the actual HF and LLaMA-Factory templates."""
    if cutoff_len <= 0:
        raise ValueError("cutoff_len must be positive")
    try:
        from transformers import AutoTokenizer
        from llamafactory.data.template import get_template_and_fix_tokenizer
        from llamafactory.hparams.data_args import DataArguments
    except ImportError as exc:  # pragma: no cover - depends on the training environment
        raise RuntimeError(
            "prefix parity requires Transformers and LLaMA-Factory in the active environment"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    data_args = DataArguments(
        template=template_name,
        cutoff_len=cutoff_len,
        mask_history=True,
        preserve_thinking=False,
        enable_thinking=True,
    )
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    if getattr(template, "preserve_thinking", None) is not False:
        raise ValueError("parity audit requires LLaMA-Factory preserve_thinking=false")
    if getattr(template, "enable_thinking", None) is not True:
        raise ValueError("parity audit requires LLaMA-Factory enable_thinking=true")

    records = 0
    prefix_lengths: list[int] = []
    target_lengths: list[int] = []
    full_lengths: list[int] = []
    over_cutoff = 0
    for pair_index, pair in enumerate(
        zip_longest(_iter_jsonl(canonical_path), _iter_jsonl(projected_path)), start=1
    ):
        canonical_item, projected_item = pair
        if canonical_item is None or projected_item is None:
            raise ValueError("canonical and projected JSONL files contain different row counts")
        canonical_line, canonical = canonical_item
        projected_line, projected = projected_item
        expected, detail = project_row(canonical, line_number=canonical_line)
        record_id = detail["record_id"]
        if projected != expected:
            raise ValueError(
                f"{record_id}: projected row {projected_line} is not the exact deterministic view"
            )
        if canonical["conversations"][-1] != projected["conversations"][-1]:
            raise AssertionError(f"{record_id}: final target differs before tokenization")

        lf_messages = [
            _message_for_template(
                message,
                record_id=record_id,
                message_index=index,
                target="llamafactory",
            )
            for index, message in enumerate(projected["conversations"])
        ]
        processed = template.mm_plugin.process_messages(lf_messages, [], [], [], None)
        pairs = template.encode_multiturn(
            tokenizer,
            processed,
            projected["system"],
            None,
            True,
        )
        if not pairs:
            raise ValueError(f"{record_id}: LLaMA-Factory produced no training pair")
        lf_prefix: list[int] = []
        for source_ids, response_ids in pairs[:-1]:
            lf_prefix.extend(source_ids)
            lf_prefix.extend(response_ids)
        lf_prefix.extend(pairs[-1][0])
        final_target_ids = list(pairs[-1][1])

        hf_history = [
            _message_for_template(
                message,
                record_id=record_id,
                message_index=index,
                target="hf",
            )
            for index, message in enumerate(canonical["conversations"][:-1])
        ]
        hf_messages = [{"role": "system", "content": canonical["system"]}, *hf_history]
        hf_prefix = _as_token_ids(
            tokenizer.apply_chat_template(
                hf_messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            ),
            location=f"{record_id}: Hugging Face prefix",
        )
        if lf_prefix != hf_prefix:
            first_difference = _first_token_difference(lf_prefix, hf_prefix)
            raise ValueError(
                f"{record_id}: prefix parity failed at token {first_difference} "
                f"(LLaMA-Factory={len(lf_prefix)}, HuggingFace={len(hf_prefix)})"
            )

        records += 1
        prefix_lengths.append(len(lf_prefix))
        target_lengths.append(len(final_target_ids))
        full_length = len(lf_prefix) + len(final_target_ids)
        full_lengths.append(full_length)
        over_cutoff += int(full_length > cutoff_len)

    chat_template = getattr(tokenizer, "chat_template", None)
    chat_template_hash = (
        hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
        if isinstance(chat_template, str)
        else None
    )
    return {
        "audit": "hf-llamafactory-qwen3-full-prefix-parity-v1",
        "canonical_input": str(canonical_path),
        "projected_input": str(projected_path),
        "model": model_path,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "tokenizer_chat_template_sha256": chat_template_hash,
        "llamafactory_template": template_name,
        "mask_history": True,
        "preserve_thinking": False,
        "enable_thinking": True,
        "hf_apply_chat_template_enable_thinking": True,
        "records": records,
        "exact_projected_rows": records,
        "exact_final_target_fields": records,
        "exact_prefix_token_sequences": records,
        "prefix_token_lengths": _length_summary(prefix_lengths),
        "target_token_lengths": _length_summary(target_lengths),
        "full_sequence_token_lengths": _length_summary(full_lengths),
        "cutoff_len": cutoff_len,
        "full_sequences_over_cutoff": over_cutoff,
        "local_files_only": local_files_only,
    }


def project(
    input_path: Path,
    output_path: Path,
    *,
    manifest_path: Path | None = None,
    overwrite: bool = False,
    audit_model: str | None = None,
    audit_template: str = "qwen3",
    audit_cutoff_len: int = 6400,
    audit_local_files_only: bool = True,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("output must differ from the canonical input")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists (pass --overwrite to replace it): {output_path}"
        )
    manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else output_path.with_name(output_path.name + ".manifest.json")
    )
    if manifest_path in {input_path, output_path}:
        raise ValueError("manifest path must differ from input and output")
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"manifest already exists (pass --overwrite to replace it): {manifest_path}"
        )

    input_hash_before = sha256(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary path already exists: {temporary}")

    records = 0
    record_ids: set[str] = set()
    records_with_history = 0
    historical_assistant_messages = 0
    removed_separator_newlines = 0
    message_schemas: Counter[str] = Counter()
    parity: dict[str, Any] | None = None
    try:
        with temporary.open("w", encoding="utf-8") as target:
            for line_number, row in _iter_jsonl(input_path):
                projected, detail = project_row(row, line_number=line_number)
                record_id = detail["record_id"]
                if record_id in record_ids:
                    raise ValueError(f"duplicate metadata.record_id: {record_id}")
                record_ids.add(record_id)
                count = detail["historical_assistant_messages"]
                records += 1
                records_with_history += int(count > 0)
                historical_assistant_messages += count
                removed_separator_newlines += detail["removed_separator_newlines"]
                message_schemas.update(detail["message_schemas"])
                target.write(json.dumps(projected, ensure_ascii=False) + "\n")
        if not records:
            raise ValueError("canonical input contains no records")
        if sha256(input_path) != input_hash_before:
            raise RuntimeError("canonical input changed while the projection was being built")
        if audit_model is not None:
            parity = audit_prefix_parity(
                input_path,
                temporary,
                model_path=audit_model,
                template_name=audit_template,
                cutoff_len=audit_cutoff_len,
                local_files_only=audit_local_files_only,
            )
            # The audit reads the unpublished temporary file so a failure cannot
            # expose a partial view.  Record the stable published path in the
            # manifest rather than that implementation-detail temporary name.
            parity["projected_input"] = str(output_path)
        output_hash = sha256(temporary)
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()

    manifest: dict[str, Any] = {
        "projection": PROJECTION_NAME,
        "mutation": (
            "Only historical assistant content is replaced by the exact JSON suffix after "
            "one leading <think>...</think> block; newline separators after </think> follow "
            "Qwen3 HF lstrip('\\n') behavior. Final targets, action JSON text, non-assistant "
            "messages, system, metadata, and all other fields are unchanged."
        ),
        "canonical_input": str(input_path),
        "canonical_input_sha256": input_hash_before,
        "output": str(output_path),
        "output_sha256": output_hash,
        "records": records,
        "unique_record_ids": len(record_ids),
        "records_with_historical_assistant": records_with_history,
        "records_without_historical_assistant": records - records_with_history,
        "historical_assistant_messages_projected": historical_assistant_messages,
        "separator_newlines_removed_after_think": removed_separator_newlines,
        "final_targets_exactly_preserved": records,
        "metadata_exactly_preserved": records,
        "message_schema_occurrences": dict(sorted(message_schemas.items())),
    }
    if parity is not None:
        manifest["prefix_parity_audit"] = parity

    manifest_temporary = manifest_path.with_name(manifest_path.name + f".{os.getpid()}.tmp")
    try:
        manifest_temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(manifest_temporary, manifest_path)
    finally:
        if manifest_temporary.exists():
            manifest_temporary.unlink()
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a metadata-preserving Qwen3 rolling-SFT history view."
    )
    parser.add_argument("--input", type=Path, required=True, help="canonical JSONL")
    parser.add_argument("--out", type=Path, required=True, help="derived JSONL")
    parser.add_argument("--manifest", type=Path, help="default: <out>.manifest.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--audit-model",
        help="run full HF/LLaMA-Factory prefix parity with this local model/tokenizer",
    )
    parser.add_argument("--audit-template", default="qwen3")
    parser.add_argument("--audit-cutoff-len", type=int, default=6400)
    parser.add_argument(
        "--allow-remote-model",
        action="store_true",
        help="allow AutoTokenizer to access the Hub during parity audit (default: local only)",
    )
    args = parser.parse_args()
    manifest = project(
        args.input,
        args.out,
        manifest_path=args.manifest,
        overwrite=args.overwrite,
        audit_model=args.audit_model,
        audit_template=args.audit_template,
        audit_cutoff_len=args.audit_cutoff_len,
        audit_local_files_only=not args.allow_remote_model,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
