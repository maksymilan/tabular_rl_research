#!/usr/bin/env python3
"""Preflight and resume a pinned evaluator; never bypass its manifest comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sqlite3
import sys
from datetime import datetime, timezone


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: dict) -> str:
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--examples-json", required=True, type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    manifest_path = args.result_dir / "manifest.json"
    journal_path = args.result_dir / "all.jsonl"
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    config = {k: v for k, v in manifest.items()
              if k not in {"config_sha256", "created_at_utc"}}
    if canonical_hash(config) != manifest["config_sha256"]:
        raise ValueError("existing manifest hash is invalid")
    if str(args.examples_json) != manifest["dataset"]:
        raise ValueError("input path differs from the immutable manifest")
    if args.base_url != manifest["base_url"]:
        raise ValueError("serving endpoint differs from the immutable manifest")
    raw = journal_path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError("journal tail requires explicit recovery before resume")
    records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    ids = [r["example_index"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate completed IDs")
    for record in records:
        if record.get("attempted_samples") != manifest["n_samples"]:
            raise ValueError(f"incomplete K group: {record['example_index']}")
        if record.get("protocol_hash") != manifest["protocol_hash"]:
            raise ValueError("completed record protocol differs")
    tasks = [json.loads(line) for line in args.examples_json.read_text().splitlines()
             if line.strip()][:manifest["requested_size"]]
    task_ids = [r["example_index"] for r in tasks]
    if len(set(task_ids)) != len(task_ids) or not set(ids).issubset(task_ids):
        raise ValueError("task IDs do not match completed results")
    pending = [r for r in tasks if r["example_index"] not in set(ids)]
    for path in sorted({r["db_path"] for r in pending}):
        db = Path(path)
        if not db.is_file():
            raise FileNotFoundError(db)
        conn = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=2)
        try:
            conn.execute("SELECT name FROM sqlite_master LIMIT 1").fetchone()
        finally:
            conn.close()

    evaluator = args.runtime_root / "src/eval/rollout_passk.py"
    namespace = runpy.run_path(str(evaluator), run_name="pinned_passk_recovery")
    entry = namespace["main"]
    globals_ = entry.__globals__
    writer_class = globals_["ArtifactWriter"]
    argv = [str(evaluator), "--resume", "--examples-json", str(args.examples_json),
            "--result-dir", str(args.result_dir), "--base-url", args.base_url,
            "--model", manifest["model"], "--n", str(manifest["requested_size"]),
            "--workers", str(args.workers)]
    for field in ("n_samples", "sample_workers", "first_sample_workers",
                  "max_inflight_requests", "sample_detail", "summary_every",
                  "temperature", "top_p", "max_tokens", "max_steps", "api_retries",
                  "few_shot", "context_mode", "history_turns", "rolling_prompt_variant",
                  "rolling_observation_style", "denotation_comparison"):
        argv.extend(["--" + field.replace("_", "-"), str(manifest[field])])
    argv.extend(["--pass-k", ",".join(map(str, manifest["pass_k"]))])
    if manifest["stop_on_success"]:
        argv.append("--stop-on-success")
    thinking = manifest.get("enable_thinking")
    if thinking is None:
        os.environ.pop("EVAL_ENABLE_THINKING", None)
    else:
        os.environ["EVAL_ENABLE_THINKING"] = thinking
    variant = manifest.get("system_prompt_variant", "default")
    if variant == "default":
        os.environ.pop("EVAL_SYSTEM_PROMPT_VARIANT", None)
    else:
        os.environ["EVAL_SYSTEM_PROMPT_VARIANT"] = variant

    class Verified(Exception):
        pass

    def compare_only(result_dir: str, actual: dict, resume: bool):
        if not resume or result_dir != str(args.result_dir):
            raise ValueError("unexpected evaluator target")
        differences = [k for k in sorted(set(actual) | set(config))
                       if actual.get(k) != config.get(k)]
        if differences or canonical_hash(actual) != manifest["config_sha256"]:
            raise ValueError(f"actual evaluator manifest differs: {differences}")
        raise Verified

    sys.argv = argv
    globals_["ArtifactWriter"] = compare_only
    try:
        entry()
        raise RuntimeError("evaluator did not reach manifest validation")
    except Verified:
        pass
    finally:
        globals_["ArtifactWriter"] = writer_class
    receipt = {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "result_dir": str(args.result_dir), "completed_before_resume": len(ids),
        "completed_ids": ids, "pending": len(pending),
        "journal_prefix_bytes": len(raw), "journal_prefix_sha256": digest(raw),
        "manifest_sha256": digest(manifest_bytes),
        "input_sha256": digest(args.examples_json.read_bytes()),
        "evaluator_sha256": digest(evaluator.read_bytes()),
        "exact_manifest_match": True, "read_only_database_preflight": "passed",
        "argv": argv,
    }
    print(json.dumps(receipt, ensure_ascii=False), flush=True)
    if args.preflight_only:
        return 0
    receipt_path = args.result_dir.parent / "resume_preflight_20260906.json"
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    # Invoke the unmodified pinned evaluator with its real ArtifactWriter and real argv.
    return entry()


if __name__ == "__main__":
    raise SystemExit(main())
