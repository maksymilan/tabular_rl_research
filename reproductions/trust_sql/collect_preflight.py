#!/usr/bin/env python3
"""Collect a non-mutating TRUST-SQL host readiness snapshot."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def command_output(command: list[str]) -> str:
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else ""


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def disk(path: str) -> dict[str, int] | None:
    target = Path(path)
    if not target.exists():
        return None
    usage = shutil.disk_usage(target)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_lines = command_output(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    gpus = []
    for line in gpu_lines:
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 6:
            gpus.append(
                {
                    "index": int(fields[0]),
                    "name": fields[1],
                    "memory_total_mib": int(fields[2]),
                    "memory_used_mib": int(fields[3]),
                    "utilization_percent": int(fields[4]),
                    "driver_version": fields[5],
                }
            )
    compute_processes = command_output(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    ).splitlines()
    packages = {
        name: package_version(name)
        for name in (
            "torch",
            "transformers",
            "ray",
            "sglang",
            "sgl-kernel",
            "megatron-core",
            "apex",
            "transformer-engine",
            "vllm",
        )
    }
    paper_scale_ready = (
        len(gpus) >= 8
        and all("A100" in gpu["name"] for gpu in gpus[:8])
        and packages["sglang"] is not None
        and packages["megatron-core"] is not None
    )
    audit = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "gpus": gpus,
        "gpu_topology": command_output(["nvidia-smi", "topo", "-m"]),
        "active_compute_processes": compute_processes,
        "disk": {"root": disk("/"), "data": disk("/data")},
        "commands": {
            name: shutil.which(name)
            for name in ("git", "ray", "nvcc", "docker", "podman", "apptainer")
        },
        "packages": packages,
        "paper_minimum_qwen3_4b_rl": {
            "gpus": 8,
            "gpu_type": "NVIDIA A100",
            "paper_hours": 60,
        },
        "paper_scale_ready": paper_scale_ready,
        "allowed_scope": "paper-scale" if paper_scale_ready else "contract-data-reward-smoke-only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
