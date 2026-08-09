#!/usr/bin/env python3
"""Prepare BIRD-dev inputs for TRUST-SQL's released async evaluator.

The released evaluator treats ``prompt`` as model-visible and ``reward_model`` as
environment-owned metadata.  This adapter deliberately keeps the gold SQL only in
``reward_model.ground_truth.target`` and reproduces the released training JSONL's
two-message prompt shape without adding a full schema or other privileged context.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


FORMAT_VERSION = "trustsql-qwen3-8b-base-bird-dev-v1"
USER_PROMPT_STYLE = "trustsql-released-train-user-v1"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _load_json_array(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"BIRD dev input must be a JSON array: {path}")
    if any(not isinstance(record, dict) for record in payload):
        raise ValueError(f"every BIRD dev record must be an object: {path}")
    return payload, raw


def _select_indices(
    record_count: int,
    *,
    limit: int | None,
    indices: Sequence[int] | None,
) -> tuple[list[int], dict[str, Any]]:
    if limit is not None and indices is not None:
        raise ValueError("limit and indices are mutually exclusive")
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be non-negative")
        if limit > record_count:
            raise ValueError(
                f"limit {limit} exceeds BIRD dev record count {record_count}"
            )
        selected = list(range(limit))
        return selected, {"mode": "limit", "limit": limit}
    if indices is not None:
        requested = list(indices)
        if len(requested) != len(set(requested)):
            raise ValueError("indices must not contain duplicates")
        invalid = sorted(index for index in requested if not 0 <= index < record_count)
        if invalid:
            raise ValueError(
                f"indices outside BIRD dev range 0..{record_count - 1}: {invalid}"
            )
        selected = sorted(requested)
        return selected, {"mode": "indices", "indices": selected}
    selected = list(range(record_count))
    return selected, {"mode": "all"}


def released_train_user_prompt(record: dict[str, Any]) -> str:
    """Render the exact user-message layout used by the released RL JSONL."""
    db_id = record.get("db_id")
    question = record.get("question")
    evidence = record.get("evidence", "")
    if not isinstance(db_id, str) or not db_id:
        raise ValueError("BIRD dev record has a missing or invalid db_id")
    if not isinstance(question, str) or not question:
        raise ValueError("BIRD dev record has a missing or invalid question")
    if evidence is None:
        evidence = ""
    if not isinstance(evidence, str):
        raise ValueError("BIRD dev record has a non-string evidence field")
    return (
        "\n**Task Configuration**\n"
        "**Database Engine:** SQLite\n"
        f"**Database:** {db_id}\n"
        f"**External Knowledge:** {evidence}\n"
        f"**User Question:** {question}?\n"
    )


def _build_record(
    source: dict[str, Any],
    *,
    source_index: int,
    system_prompt: str,
    database_root: Path,
) -> dict[str, Any]:
    question_id = source.get("question_id")
    if isinstance(question_id, bool) or not isinstance(question_id, (int, str)):
        raise ValueError(
            f"BIRD dev record {source_index} has a missing or invalid question_id"
        )
    db_id = source.get("db_id")
    question = source.get("question")
    evidence = source.get("evidence", "")
    gold_sql = source.get("SQL")
    if not isinstance(db_id, str) or not db_id:
        raise ValueError(f"BIRD dev record {source_index} has an invalid db_id")
    if not isinstance(question, str) or not question:
        raise ValueError(f"BIRD dev record {source_index} has an invalid question")
    if evidence is None:
        evidence = ""
    if not isinstance(evidence, str):
        raise ValueError(f"BIRD dev record {source_index} has invalid evidence")
    if not isinstance(gold_sql, str) or not gold_sql.strip():
        raise ValueError(f"BIRD dev record {source_index} has invalid gold SQL")

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": released_train_user_prompt(source)},
    ]
    model_visible = json.dumps(prompt, ensure_ascii=False, separators=(",", ":"))
    if gold_sql in model_visible:
        raise ValueError(
            f"BIRD dev record {source_index} would expose its gold SQL in prompt"
        )

    return {
        "id": f"dev_{question_id}",
        "question": question,
        "db_id": db_id,
        "prompt": prompt,
        "reward_model": {
            "ground_truth": {"target": [gold_sql]},
            "style": "rule",
            "data_source": db_id,
            "database": str(database_root),
        },
        "extra_info": {
            "index": source_index,
            "question_id": question_id,
            "split": "dev",
            "db_id": db_id,
            "evidence": evidence,
        },
    }


def prepare_dataset(
    *,
    bird_dev_path: Path,
    prompt_template_path: Path,
    database_root: Path,
    output_path: Path,
    manifest_path: Path | None = None,
    limit: int | None = None,
    indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    bird_dev_path = bird_dev_path.resolve()
    prompt_template_path = prompt_template_path.resolve()
    database_root = database_root.resolve()
    output_path = output_path.resolve()
    manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else Path(f"{output_path}.manifest.json")
    )

    if not bird_dev_path.is_file():
        raise ValueError(f"BIRD dev JSON does not exist: {bird_dev_path}")
    if not prompt_template_path.is_file():
        raise ValueError(f"TRUST-SQL prompt template does not exist: {prompt_template_path}")
    if not database_root.is_dir():
        raise ValueError(f"BIRD dev database root does not exist: {database_root}")
    protected_inputs = {bird_dev_path, prompt_template_path, database_root}
    if output_path in protected_inputs or manifest_path in protected_inputs:
        raise ValueError("output and manifest paths must not overwrite an input path")
    if output_path == manifest_path:
        raise ValueError("output and manifest paths must be different")

    source_records, source_bytes = _load_json_array(bird_dev_path)
    prompt_bytes = prompt_template_path.read_bytes()
    system_prompt = prompt_bytes.decode("utf-8")
    selected_indices, selection = _select_indices(
        len(source_records), limit=limit, indices=indices
    )

    records = [
        _build_record(
            source_records[index],
            source_index=index,
            system_prompt=system_prompt,
            database_root=database_root,
        )
        for index in selected_indices
    ]
    instance_ids = [record["id"] for record in records]
    if len(instance_ids) != len(set(instance_ids)):
        raise ValueError("selected BIRD dev records contain duplicate question_id values")

    jsonl = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in records
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(jsonl, encoding="utf-8")

    manifest = {
        "format_version": FORMAT_VERSION,
        "source": {
            "bird_dev_path": str(bird_dev_path),
            "bird_dev_sha256": sha256_bytes(source_bytes),
            "record_count": len(source_records),
        },
        "prompt": {
            "template_path": str(prompt_template_path),
            "template_sha256": sha256_bytes(prompt_bytes),
            "system_content_sha256": sha256_bytes(system_prompt.encode("utf-8")),
            "user_prompt_style": USER_PROMPT_STYLE,
        },
        "database_root": str(database_root),
        "selection": {
            **selection,
            "source_indices": selected_indices,
            "question_ids": [
                source_records[index]["question_id"] for index in selected_indices
            ],
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_bytes(jsonl.encode("utf-8")),
            "record_count": len(records),
        },
        "gold_sql_visibility": "reward_model.ground_truth.target-only",
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic BIRD-dev JSONL for TRUST-SQL's async evaluator."
    )
    parser.add_argument(
        "--bird-dev", "--bird-json", dest="bird_dev", type=Path, required=True
    )
    parser.add_argument(
        "--prompt-template",
        "--system-prompt",
        dest="prompt_template",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dev-databases",
        "--database-root",
        dest="dev_databases",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        help="default: <output>.manifest.json",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument(
        "--indices",
        type=int,
        nargs="+",
        help="explicit BIRD dev array offsets; output is normalized to source order",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = prepare_dataset(
            bird_dev_path=args.bird_dev,
            prompt_template_path=args.prompt_template,
            database_root=args.dev_databases,
            output_path=args.output,
            manifest_path=args.manifest,
            limit=args.limit,
            indices=args.indices,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
