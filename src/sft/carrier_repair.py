"""Data and tokenizer invariants for exact tool-carrier repair."""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Iterator


TOOL_NAME_PATTERN = re.compile(
    r"<tool_call>\s*\{\s*\"tool\"\s*:\s*\"([^\"]+)\"",
    re.DOTALL,
)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc


def assistant_tool_name(record: dict[str, Any]) -> str:
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("record must contain non-empty conversations")
    final_turn = conversations[-1]
    if final_turn.get("from") != "gpt":
        raise ValueError("record must end with a gpt target")
    match = TOOL_NAME_PATTERN.search(str(final_turn.get("value", "")))
    if match is None:
        raise ValueError("final gpt target lacks a canonical tool_call")
    return match.group(1)


def select_tool_balanced_records(
    records: Iterable[dict[str, Any]],
    *,
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select records round-robin across tools with deterministic shuffling."""
    if limit <= 0:
        raise ValueError("limit must be positive")
    by_tool: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_tool[assistant_tool_name(record)].append(record)

    rng = random.Random(seed)
    queues: dict[str, deque[dict[str, Any]]] = {}
    for tool_name, tool_records in by_tool.items():
        rng.shuffle(tool_records)
        queues[tool_name] = deque(tool_records)

    selected: list[dict[str, Any]] = []
    tool_names = sorted(queues)
    while len(selected) < limit:
        added = False
        for tool_name in tool_names:
            if queues[tool_name]:
                selected.append(queues[tool_name].popleft())
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
    return selected


def canonical_boundary_token_ids(
    tokenizer: Any,
    boundaries: tuple[str, ...] = ("<tool_call>", "</tool_call>"),
) -> dict[str, int]:
    result: dict[str, int] = {}
    for boundary in boundaries:
        token_ids = tokenizer.encode(boundary, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(
                f"carrier boundary must be one dedicated token: {boundary!r} -> {token_ids}"
            )
        result[boundary] = int(token_ids[0])
    if len(set(result.values())) != len(result):
        raise ValueError(f"carrier boundaries resolve to duplicate token ids: {result}")
    return result


def sharegpt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    role_map = {"human": "user", "gpt": "assistant"}
    messages: list[dict[str, str]] = [
        {"role": "system", "content": str(record["system"])}
    ]
    for turn in record["conversations"]:
        try:
            role = role_map[turn["from"]]
        except KeyError as exc:
            raise ValueError(f"unsupported ShareGPT role: {turn.get('from')!r}") from exc
        messages.append({"role": role, "content": str(turn["value"])})
    if messages[-1]["role"] != "assistant":
        raise ValueError("record must end with an assistant target")
    return messages


def encode_last_assistant_target(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    cutoff_len: int,
    required_token_ids: Iterable[int],
) -> dict[str, list[int]]:
    """Encode a ShareGPT record while supervising only its final assistant turn."""
    messages = sharegpt_messages(record)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full_text.startswith(prompt_text):
        raise ValueError("chat template does not preserve the generation prompt as a prefix")

    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("tokenized full conversation does not preserve prompt token prefix")
    if len(full_ids) > cutoff_len:
        raise ValueError(
            f"record has {len(full_ids)} tokens, exceeding cutoff_len={cutoff_len}"
        )

    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    supervised = labels[len(prompt_ids) :]
    for token_id in required_token_ids:
        count = supervised.count(int(token_id))
        if count != 1:
            raise ValueError(
                f"required carrier token {token_id} occurs {count} times in final target"
            )
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }
