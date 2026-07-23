#!/usr/bin/env python3
"""Build the frozen 1,024-record SFT-1/SFT-2 tool-use ablation mixture.

The source corpora contain one supervised assistant action per JSONL row.  SFT-2 already includes
an SFT-1 demonstration-replay lane, so that lane is excluded before selecting an equal number of
records from each stage.  Selection and final ordering are hash-priority based and deterministic.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


HISTORICAL_COMMIT = "2c603ffa01a00061e4d32bc157a27b1bed4eea83"
PROTOCOL_VERSION = "v2i-state-only-join-feedback-r2"
ACTION_RE = re.compile(
    r"^<think>.*?</think>\s*<tool_call>(\{.*\})</tool_call>\s*$",
    flags=re.DOTALL,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            rows.append(row)
    return rows


def validate_and_annotate(rows: Iterable[dict], source: str) -> list[dict]:
    annotated = []
    for row_index, row in enumerate(rows):
        if not isinstance(row.get("system"), str) or not row["system"].strip():
            raise ValueError(f"{source} row {row_index}: missing system prompt")
        conversations = row.get("conversations")
        if not isinstance(conversations, list) or len(conversations) < 2:
            raise ValueError(f"{source} row {row_index}: invalid conversations")
        if conversations[-1].get("from") != "gpt":
            raise ValueError(f"{source} row {row_index}: final turn is not gpt")
        for turn_index, turn in enumerate(conversations):
            expected = "human" if turn_index % 2 == 0 else "gpt"
            if turn.get("from") != expected or not isinstance(turn.get("value"), str):
                raise ValueError(
                    f"{source} row {row_index}: invalid turn {turn_index}, expected {expected}"
                )

        match = ACTION_RE.fullmatch(conversations[-1]["value"])
        if not match:
            raise ValueError(f"{source} row {row_index}: non-canonical final assistant action")
        action = json.loads(match.group(1))
        if set(action) != {"tool", "arguments"}:
            raise ValueError(f"{source} row {row_index}: invalid tool-call keys")
        if not isinstance(action["tool"], str) or not isinstance(action["arguments"], dict):
            raise ValueError(f"{source} row {row_index}: invalid tool-call values")

        fingerprint = canonical_sha256(
            {"system": row["system"], "conversations": conversations}
        )
        annotated.append(
            {
                "row": row,
                "fingerprint": fingerprint,
                "tool": action["tool"],
                "system_sha256": hashlib.sha256(row["system"].encode("utf-8")).hexdigest(),
            }
        )
    return annotated


def stable_priority(seed: int, stage: str, fingerprint: str) -> str:
    return hashlib.sha256(f"{seed}:{stage}:{fingerprint}".encode("utf-8")).hexdigest()


def unique_by_fingerprint(rows: Iterable[dict]) -> tuple[list[dict], int]:
    seen = set()
    unique = []
    duplicate_count = 0
    for item in rows:
        if item["fingerprint"] in seen:
            duplicate_count += 1
            continue
        seen.add(item["fingerprint"])
        unique.append(item)
    return unique, duplicate_count


def select_stage(
    rows: list[dict],
    *,
    count: int,
    seed: int,
    stage: str,
    force_transition_types: set[str] | None = None,
    force_tools: set[str] | None = None,
) -> list[dict]:
    forced = []
    remaining = []
    force_transition_types = force_transition_types or set()
    force_tools = force_tools or set()
    for item in rows:
        transition_type = (item["row"].get("metadata") or {}).get("transition_type")
        if transition_type in force_transition_types or item["tool"] in force_tools:
            forced.append(item)
        else:
            remaining.append(item)
    if len(forced) > count:
        raise ValueError(f"{stage}: {len(forced)} forced rows exceed quota {count}")
    remaining.sort(key=lambda item: stable_priority(seed, stage, item["fingerprint"]))
    selected = forced + remaining[: count - len(forced)]
    if len(selected) != count:
        raise ValueError(f"{stage}: only {len(selected)} rows available for quota {count}")
    return selected


def counts(items: Iterable[dict]) -> dict:
    items = list(items)
    return {
        "records": len(items),
        "source_episodes": len(
            {
                (item["row"].get("metadata") or {}).get("source_episode_id")
                for item in items
            }
        ),
        "feedback_recovery": sum(
            bool((item["row"].get("metadata") or {}).get("feedback_recovery"))
            for item in items
        ),
        "tool": dict(sorted(Counter(item["tool"] for item in items).items())),
        "lane": dict(
            sorted(
                Counter(
                    (item["row"].get("metadata") or {}).get("sft2_lane", "not-applicable")
                    for item in items
                ).items()
            )
        ),
        "transition_type": dict(
            sorted(
                Counter(
                    (item["row"].get("metadata") or {}).get(
                        "transition_type", "not-applicable"
                    )
                    for item in items
                ).items()
            )
        ),
        "system_sha256": dict(
            sorted(Counter(item["system_sha256"] for item in items).items())
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft1", type=Path, required=True)
    parser.add_argument("--sft2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--per-stage", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260723)
    args = parser.parse_args()
    if args.per_stage <= 0:
        parser.error("--per-stage must be positive")

    sft1_raw = read_jsonl(args.sft1)
    sft2_raw = read_jsonl(args.sft2)
    sft1 = validate_and_annotate(sft1_raw, "sft1")
    sft2 = validate_and_annotate(sft2_raw, "sft2")
    sft1, sft1_internal_duplicates = unique_by_fingerprint(sft1)
    sft2, sft2_internal_duplicates = unique_by_fingerprint(sft2)

    excluded_replay = [
        item
        for item in sft2
        if (item["row"].get("metadata") or {}).get("sft2_lane") == "sft1_demo_replay"
    ]
    sft2_non_replay = [
        item
        for item in sft2
        if (item["row"].get("metadata") or {}).get("sft2_lane") != "sft1_demo_replay"
    ]
    sft1_fingerprints = {item["fingerprint"] for item in sft1}
    cross_stage_duplicates = [
        item for item in sft2_non_replay if item["fingerprint"] in sft1_fingerprints
    ]
    sft2_eligible = [
        item for item in sft2_non_replay if item["fingerprint"] not in sft1_fingerprints
    ]

    selected_sft1 = select_stage(
        sft1,
        count=args.per_stage,
        seed=args.seed,
        stage="sft1",
        force_tools={"set_op"},
    )
    selected_sft2 = select_stage(
        sft2_eligible,
        count=args.per_stage,
        seed=args.seed,
        stage="sft2",
        force_transition_types={"decision_correction"},
    )
    selected = [("sft1", item) for item in selected_sft1]
    selected.extend(("sft2", item) for item in selected_sft2)
    selected.sort(
        key=lambda pair: stable_priority(args.seed, "combined", pair[1]["fingerprint"])
    )

    output_rows = []
    for source_stage, item in selected:
        row = copy.deepcopy(item["row"])
        metadata = row.setdefault("metadata", {})
        metadata["ablation_source_stage"] = source_stage
        metadata["ablation_source_fingerprint"] = item["fingerprint"]
        output_rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    output_fingerprints = [
        canonical_sha256({"system": row["system"], "conversations": row["conversations"]})
        for row in output_rows
    ]
    if len(output_fingerprints) != len(set(output_fingerprints)):
        raise AssertionError("output contains duplicate model-visible records")

    manifest_path = args.manifest or args.out.with_suffix(".manifest.json")
    manifest = {
        "experiment": "omnisql-vs-qwen25-coder-sft12-tool-ablation",
        "historical_commit": HISTORICAL_COMMIT,
        "protocol_version": PROTOCOL_VERSION,
        "seed": args.seed,
        "selection": {
            "per_stage": args.per_stage,
            "total": len(output_rows),
            "method": (
                "sha256-priority without replacement; both SFT-1 set_op rows and both SFT-2 "
                "decision-correction rows forced"
            ),
            "sft2_excluded_lane": "sft1_demo_replay",
        },
        "sources": {
            "sft1": {
                "path": str(args.sft1),
                "sha256": file_sha256(args.sft1),
                "raw_records": len(sft1_raw),
                "unique_records": len(sft1),
                "internal_duplicates": sft1_internal_duplicates,
            },
            "sft2": {
                "path": str(args.sft2),
                "sha256": file_sha256(args.sft2),
                "raw_records": len(sft2_raw),
                "unique_records": len(sft2),
                "internal_duplicates": sft2_internal_duplicates,
                "excluded_sft1_demo_replay": len(excluded_replay),
                "excluded_cross_stage_duplicates": len(cross_stage_duplicates),
                "eligible_records": len(sft2_eligible),
            },
        },
        "selected": {
            "sft1": counts(selected_sft1),
            "sft2": counts(selected_sft2),
            "combined": counts([item for _, item in selected]),
        },
        "output": {
            "path": str(args.out),
            "records": len(output_rows),
            "sha256": file_sha256(args.out),
            "unique_model_visible_fingerprints": len(set(output_fingerprints)),
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
