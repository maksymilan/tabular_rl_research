#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


BASE = Path("/home/dengyan/tabular_rl_outputs/qwen3_8b_v26_rl_two_phase_diag_20260902")
SRC = BASE / "pool2_k8"
OUT = BASE / "pool1_k8"
if OUT.exists():
    raise SystemExit(f"refusing existing directory: {OUT}")
OUT.mkdir(parents=True)

rows = json.loads((SRC / "groups" / "bird_train_05239.json").read_text(encoding="utf-8"))
if len(rows) != 8 or [int(row["sequence"]) for row in rows] != list(range(8)):
    raise RuntimeError("unexpected first-task K8 group")
trajectory_path = OUT / "trajectories.jsonl"
trajectory_path.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
    + "\n",
    encoding="utf-8",
)

task_source = SRC / "tasks2.jsonl"
task_lines = task_source.read_text(encoding="utf-8").splitlines()
if len(task_lines) != 2:
    raise RuntimeError("unexpected tasks2 line count")
(OUT / "tasks1.jsonl").write_text(task_lines[0] + "\n", encoding="utf-8")

manifest = json.loads((SRC / "diagnostic_manifest.json").read_text(encoding="utf-8"))
tasks_path = OUT / "tasks1.jsonl"
pool_sha = hashlib.sha256(trajectory_path.read_bytes()).hexdigest()
manifest.update(
    {
        "tasks_path": str(tasks_path),
        "tasks_sha256": hashlib.sha256(tasks_path.read_bytes()).hexdigest(),
        "tasks": 1,
        "trajectories": 8,
        "trajectories_sha256": pool_sha,
        "correct_trajectories": sum(bool(row["sample"]["correct"]) for row in rows),
        "files": {"validated_trajectories": {"path": str(trajectory_path), "sha256": pool_sha}},
        "generated_at_epoch": time.time(),
        "diagnostic_note": "Exact first-task K8 subset of fresh non-eager vLLM tokenized pool; four-level reward.",
    }
)
(OUT / "diagnostic_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
json.loads((OUT / "diagnostic_manifest.json").read_text(encoding="utf-8"))
print(json.dumps({"rows": len(rows), "pool_sha": pool_sha, "manifest": str(OUT / "diagnostic_manifest.json")}))
