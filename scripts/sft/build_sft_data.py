#!/usr/bin/env python3
"""Convert verified harness trajectories into LLaMA-Factory ShareGPT SFT data.

Record shape: {"system": SYSTEM_PROMPT, "conversations": [human, gpt, observation, gpt, ...]}.
Loss is computed on "gpt" turns only; "observation" (tool outputs) and "human" are masked by
LLaMA-Factory. The terminal answer_from_context call is the last gpt turn (no observation after).

Examples:
  # Backward-compatible v0 build.
  .venv/bin/python scripts/sft/build_sft_data.py both

  # Build v1 from grounded, think-filled trajectories without editing this script.
  .venv/bin/python scripts/sft/build_sft_data.py both \
    --input-pattern 'data/trajectories/spider_{split}_think.jsonl' \
    --output-prefix spider_v1 --dataset-name spider_tools_v1 \
    --max-est-tokens 10240
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import statistics
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)

from protocol import (SYSTEM_PROMPT, PROTOCOL_VERSION, assistant_message,  # noqa: E402
                      first_user_message, protocol_hash, tool_output_message)

CHARS_PER_TOKEN = 3.5  # rough for English+JSON; manifest reports char counts too
DEFAULT_INPUT_PATTERN = "data/trajectories/spider_{split}.jsonl"
DEFAULT_OUTPUT_DIR = "data/sft"
DEFAULT_OUTPUT_PREFIX = "spider_v0"
DEFAULT_DATASET_NAME = "spider_tools_v0"
SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def convert(traj: dict) -> dict:
    conv = [{"from": "human",
             "value": first_user_message(traj["initial_state"]["dataset_overview"], traj["question"])}]
    steps = traj["steps"]
    for i, s in enumerate(steps):
        tc = s["tool_call"]
        conv.append({"from": "gpt",
                     "value": assistant_message(s.get("think", ""), tc["tool"], tc["arguments"])})
        if i < len(steps) - 1:  # terminal call has no observation
            conv.append({"from": "observation",
                         "value": tool_output_message(s["step_id"], s["tool_output"])})
    return {"system": SYSTEM_PROMPT, "conversations": conv}


def est_tokens(rec: dict) -> int:
    chars = len(rec["system"]) + sum(len(m["value"]) for m in rec["conversations"])
    return int(chars / CHARS_PER_TOKEN)


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(ROOT) / p


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_trajectory(traj: dict, source: Path, line_no: int) -> None:
    where = f"{source}:{line_no}"
    if traj.get("label_status") != "verified":
        raise ValueError(f"{where}: trajectory is not execution-verified")
    steps = traj.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{where}: trajectory has no steps")
    if steps[-1].get("tool_call", {}).get("tool") != "answer_from_context":
        raise ValueError(f"{where}: final step is not answer_from_context")
    for i, step in enumerate(steps, 1):
        if not str(step.get("think", "")).strip():
            raise ValueError(f"{where}: step {i} has an empty think field")
        tool_call = step.get("tool_call")
        if not isinstance(tool_call, dict) or not isinstance(tool_call.get("arguments"), dict):
            raise ValueError(f"{where}: step {i} has an invalid tool_call")


def build(split: str, max_est_tokens: int, source: Path, out_path: Path) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    source_count = kept = dropped = 0
    dropped_ids: list[str] = []
    toks: list[int] = []
    chars: list[int] = []
    len_hist = collections.Counter()
    tool_hist = collections.Counter()
    seen_ids: set[str] = set()
    try:
        with source.open(encoding="utf-8") as f, tmp_path.open("w", encoding="utf-8") as out:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                source_count += 1
                traj = json.loads(line)
                validate_trajectory(traj, source, line_no)
                trajectory_id = traj.get("trajectory_id")
                if not trajectory_id:
                    raise ValueError(f"{source}:{line_no}: missing trajectory_id")
                if trajectory_id in seen_ids:
                    raise ValueError(f"{source}:{line_no}: duplicate trajectory_id {trajectory_id!r}")
                seen_ids.add(trajectory_id)

                rec = convert(traj)
                t = est_tokens(rec)
                if t > max_est_tokens:
                    dropped += 1
                    dropped_ids.append(trajectory_id)
                    continue
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                toks.append(t)
                chars.append(
                    len(rec["system"]) + sum(len(message["value"]) for message in rec["conversations"])
                )
                len_hist[len(traj["steps"])] += 1
                for step in traj["steps"]:
                    tool_hist[step["tool_call"]["tool"]] += 1
        os.replace(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    toks.sort()
    chars.sort()
    pct = lambda p: toks[min(len(toks) - 1, int(p * len(toks)))] if toks else 0  # noqa: E731
    char_pct = lambda p: chars[min(len(chars) - 1, int(p * len(chars)))] if chars else 0  # noqa: E731
    manifest = {
        "split": split,
        "input": os.path.relpath(source, ROOT),
        "input_sha256": file_sha256(source),
        "protocol_version": PROTOCOL_VERSION,
        "protocol_hash": protocol_hash(),
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "source_trajectories": source_count,
        "kept": kept,
        "dropped_overlong": dropped,
        "dropped_trajectory_ids": dropped_ids,
        "max_est_tokens": max_est_tokens,
        "token_estimate_method": f"characters / {CHARS_PER_TOKEN}",
        "est_tokens": {"p50": pct(.5), "p90": pct(.9), "p95": pct(.95),
                       "max": toks[-1] if toks else 0,
                       "mean": int(statistics.mean(toks)) if toks else 0},
        "characters": {"p50": char_pct(.5), "p95": char_pct(.95),
                       "max": chars[-1] if chars else 0},
        "trajectory_length_hist": dict(sorted(len_hist.items())),
        "tool_hist": dict(tool_hist.most_common()),
        "output": os.path.relpath(out_path, ROOT),
        "note": "dev file is for eval-loss / inspection only — dev is the held-out eval set, never train on it",
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def dataset_info_entry(dataset_name: str, output_prefix: str) -> dict:
    return {
        dataset_name: {
            "file_name": f"{output_prefix}_train.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system"},
            "tags": {"role_tag": "from", "content_tag": "value",
                     "user_tag": "human", "assistant_tag": "gpt",
                     "observation_tag": "observation"},
        }
    }


def write_dataset_info(out_dir: Path, dataset_name: str, output_prefix: str) -> tuple[Path, Path]:
    entry = dataset_info_entry(dataset_name, output_prefix)
    snippet_path = out_dir / f"dataset_info.{dataset_name}.snippet.json"
    snippet_path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")

    registry_path = out_dir / "dataset_info.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.exists() else {}
    registry.update(entry)
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return snippet_path, registry_path


def validate_name(value: str, flag: str) -> str:
    if not SAFE_NAME.fullmatch(value):
        raise ValueError(f"{flag} must contain only letters, digits, '.', '_' or '-'")
    return value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("split", nargs="?", default="both", choices=["train", "dev", "both"])
    ap.add_argument("--input-pattern", default=DEFAULT_INPUT_PATTERN,
                    help="trajectory JSONL path; may contain {split}")
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    ap.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    ap.add_argument("--max-est-tokens", type=int, default=12000)
    args = ap.parse_args()

    validate_name(args.output_prefix, "--output-prefix")
    validate_name(args.dataset_name, "--dataset-name")
    if args.max_est_tokens <= 0:
        ap.error("--max-est-tokens must be positive")
    if args.split == "both" and "{split}" not in args.input_pattern:
        ap.error("--input-pattern must contain {split} when building both splits")

    out_dir = resolve_path(args.output_dir)
    splits = ["train", "dev"] if args.split == "both" else [args.split]
    for split in splits:
        source = resolve_path(args.input_pattern.format(split=split))
        if not source.is_file():
            ap.error(f"input trajectory file not found: {source}")
        out_path = out_dir / f"{args.output_prefix}_{split}.jsonl"
        manifest = build(split, args.max_est_tokens, source, out_path)
        print(f"{split}: source {manifest['source_trajectories']}  kept {manifest['kept']}  "
              f"dropped_overlong {manifest['dropped_overlong']}  "
              f"est_tokens p50/p95/max = {manifest['est_tokens']['p50']}/"
              f"{manifest['est_tokens']['p95']}/{manifest['est_tokens']['max']}")
        print(f"  tools: {manifest['tool_hist']}")
        print(f"  -> {manifest['output']}")

    snippet_path, registry_path = write_dataset_info(
        out_dir, args.dataset_name, args.output_prefix
    )
    print(f"dataset_info snippet -> {os.path.relpath(snippet_path, ROOT)}")
    print(f"dataset_info registry -> {os.path.relpath(registry_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
