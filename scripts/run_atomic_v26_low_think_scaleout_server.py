#!/usr/bin/env python3
"""Run one frozen Atomic-v26 low-think rollout cohort on table_rl.

This supervisor keeps shard-level journals, caps aggregate provider usage, and fails closed on
identity drift or repeated provider failures.  It never changes the teacher/student protocol.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path


RUNTIME = Path("/home/dengyan/tabular_rl_outputs/runtime/sft1_low_think_ae3_20260826")
PYTHON = Path("/home/dengyan/miniconda3/envs/sft/bin/python")
GENERATOR = RUNTIME / "src/sft/generate_teacher_rollouts.py"
PROVIDER = RUNTIME / "src/sft/provider_adapter.py"
EXPECTED_PROVIDER_SHA256 = "98c698930f06a3aeef6c857526a283542eed8bed0fee60832379d74a64953fdf"
EXPECTED_GENERATOR_SHA256 = "8715b5790c0f090a8d551d738fecd4d02ebd878c40a274fdbbc1b42ea4b8fbc9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.tasks = args.tasks.resolve()
        self.selection_manifest = args.selection_manifest.resolve()
        self.root = args.result_root.resolve()

    def shard_name(self, start: int) -> str:
        end = min(start + self.args.shard_size, self.args.episodes) - 1
        return f"shard_{start:04d}_{end:04d}"

    def shard_paths(self, start: int) -> dict[str, Path]:
        directory = self.root / self.shard_name(start)
        return {
            "dir": directory,
            "verified": directory / "verified.jsonl",
            "failures": directory / "verified.failures.jsonl",
            "all": directory / "verified.all.jsonl",
            "manifest": directory / "verified.manifest.json",
            "log": directory / "run.log",
        }

    def completed_manifest(self, start: int) -> dict | None:
        paths = self.shard_paths(start)
        expected = min(self.args.shard_size, self.args.episodes - start)
        if not paths["manifest"].is_file() or not paths["all"].is_file():
            return None
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if int(manifest.get("unique_examples") or 0) != expected:
            return None
        final_ids = {str(row.get("trajectory_id")) for row in read_jsonl(paths["all"])}
        return manifest if len(final_ids) == expected else None

    def starts(self) -> range:
        return range(0, self.args.episodes, self.args.shard_size)

    def summarize(self) -> dict:
        totals: collections.Counter[str] = collections.Counter()
        completed: list[int] = []
        provider_failures = 0
        for start in self.starts():
            manifest = self.completed_manifest(start)
            if manifest is None:
                continue
            completed.append(start)
            usage = manifest.get("usage_total") or {}
            totals["tokens"] += int(usage.get("total_tokens") or 0)
            totals["api_request_attempts"] += int(usage.get("api_request_attempts") or 0)
            counts = manifest.get("counts") or {}
            expected = min(self.args.shard_size, self.args.episodes - start)
            totals["correct"] += int(counts.get("examples_correct") or counts.get("correct") or 0)
            totals["failed"] += int(counts.get("examples_failed") or counts.get("failed") or 0)
            totals["attempted"] += expected
            for row in read_jsonl(self.shard_paths(start)["all"]):
                failure = str(row.get("failure_type") or "").casefold()
                if any(marker in failure for marker in ("provider", "api_error", "transport", "http")):
                    provider_failures += 1
        return {
            "completed_shard_starts": completed,
            "completed_shards": len(completed),
            "attempted_episodes": totals["attempted"],
            "correct_episodes": totals["correct"],
            "failed_episodes": totals["failed"],
            "provider_failures": provider_failures,
            "provider_total_tokens": totals["tokens"],
            "api_request_attempts": totals["api_request_attempts"],
        }

    def verify(self) -> dict:
        if sha256(self.tasks) != self.args.expected_tasks_sha256:
            raise RuntimeError("task SHA mismatch")
        if sha256(self.selection_manifest) != self.args.expected_manifest_sha256:
            raise RuntimeError("selection manifest SHA mismatch")
        if sha256(PROVIDER) != EXPECTED_PROVIDER_SHA256:
            raise RuntimeError("provider adapter SHA mismatch")
        if sha256(GENERATOR) != EXPECTED_GENERATOR_SHA256:
            raise RuntimeError("generator SHA mismatch")
        manifest = json.loads(self.selection_manifest.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_generation_candidates_not_sft_admitted":
            raise RuntimeError("selection manifest status mismatch")
        if int((manifest.get("selection") or {}).get("records") or 0) != self.args.episodes:
            raise RuntimeError("selection count mismatch")
        if str((manifest.get("outputs") or {}).get("table_rl_tasks_sha256")) != self.args.expected_tasks_sha256:
            raise RuntimeError("manifest does not bind table_rl task SHA")
        teacher = manifest.get("teacher_identity") or {}
        expected_teacher = {
            "endpoint": "https://api.deepseek.com",
            "model": "deepseek-v4-flash",
            "thinking": "enabled",
            "reasoning_effort": "low",
            "response_format": "json_object",
            "atomic_protocol_version": "version24",
            "history": "rolling-legal-history-recent4-full",
        }
        for key, expected in expected_teacher.items():
            if teacher.get(key) != expected:
                raise RuntimeError(f"teacher identity mismatch: {key}")
        tasks = read_jsonl(self.tasks)
        ids = {str(row.get("example_id")) for row in tasks}
        if len(tasks) != self.args.episodes or len(ids) != self.args.episodes:
            raise RuntimeError("task count or identity mismatch")
        missing = sum(not Path(str(row.get("db_path"))).is_file() for row in tasks)
        if missing:
            raise RuntimeError(f"missing database paths: {missing}")
        api_key = base_url = ""
        for line in (RUNTIME / "api.md").read_text(encoding="utf-8").splitlines():
            if line.startswith("API_KEY="):
                api_key = line.split("=", 1)[1].strip()
            elif line.startswith("BASE_URL="):
                base_url = line.split("=", 1)[1].strip()
        if not api_key or base_url != "https://api.deepseek.com":
            raise RuntimeError("official provider configuration missing")
        return {
            "tasks_sha256": self.args.expected_tasks_sha256,
            "selection_manifest_sha256": self.args.expected_manifest_sha256,
            "provider_adapter_sha256": EXPECTED_PROVIDER_SHA256,
            "generator_sha256": EXPECTED_GENERATOR_SHA256,
            "official_base_url": True,
            "episodes": self.args.episodes,
            "unique_database_ids": len({str(row.get("db_id")) for row in tasks}),
            "source_histogram": dict(sorted(collections.Counter(str(row.get("dataset")) for row in tasks).items())),
            "database_paths_present": self.args.episodes,
        }

    def command(self, start: int, *, resume: bool) -> list[str]:
        paths = self.shard_paths(start)
        limit = min(self.args.shard_size, self.args.episodes - start)
        command = [
            str(PYTHON), "-u", str(GENERATOR),
            "--split", "train", "--examples-file", str(self.tasks),
            "--start", str(start), "--limit", str(limit),
            "--model", "deepseek-v4-flash",
            "--out", str(paths["verified"]), "--failures-out", str(paths["failures"]),
            "--all-out", str(paths["all"]), "--workers", str(self.args.workers),
            "--max-steps", "30", "--max-errors-per-type", "3", "--attempts-per-example", "1",
            "--max-tokens", "2048", "--api-timeout", "300", "--api-retries", "3",
            "--table-output-rows", "0", "--context-mode", "rolling-legal-history",
            "--history-turns", "4", "--rolling-prompt-variant", "full",
            "--policy-prompt-variant", "canonical", "--plan-policy", "optional",
            "--deepseek-carrier", "json-output", "--denotation-comparison", "bird-set",
        ]
        if resume:
            command.append("--resume")
        return command

    def run(self, *, dry_run: bool) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        identity = self.verify()
        if dry_run:
            print(json.dumps({
                "identity": identity, "episodes": self.args.episodes,
                "shards": len(list(self.starts())), "max_processes": self.args.max_processes,
                "workers_per_process": self.args.workers,
                "max_inflight_episodes": self.args.max_processes * self.args.workers,
                "total_token_cap": self.args.total_token_cap,
                "first_command": self.command(0, resume=False),
            }, ensure_ascii=False, indent=2, sort_keys=True))
            return 0
        pid_path = self.root / "supervisor.pid"
        if pid_path.exists():
            try:
                old_pid = int(pid_path.read_text().strip())
                os.kill(old_pid, 0)
                raise RuntimeError(f"supervisor already running: {old_pid}")
            except ProcessLookupError:
                pass
        pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
        stop_requested = False
        stop_reason: str | None = None

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stop_requested, stop_reason
            stop_requested = True
            stop_reason = stop_reason or "signal_requested"

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        active: dict[int, tuple[subprocess.Popen, object]] = {}
        status_path = self.root / "status.json"
        try:
            while True:
                for start, (process, log_handle) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    log_handle.close()
                    del active[start]
                    if code != 0:
                        stop_reason = f"shard_process_exit_{code}:{self.shard_name(start)}"
                        stop_requested = True
                    elif self.completed_manifest(start) is None:
                        stop_reason = f"shard_incomplete:{self.shard_name(start)}"
                        stop_requested = True
                summary = self.summarize()
                if summary["provider_failures"] >= self.args.max_provider_failures:
                    stop_reason, stop_requested = "provider_failure_circuit_breaker", True
                if summary["provider_total_tokens"] >= self.args.total_token_cap:
                    stop_reason, stop_requested = "global_token_cap", True
                remaining = [
                    start for start in self.starts()
                    if start not in summary["completed_shard_starts"] and start not in active
                ]
                while not stop_requested and remaining and len(active) < self.args.max_processes:
                    reserved = self.args.token_reservation_per_shard * (len(active) + 1)
                    if summary["provider_total_tokens"] + reserved > self.args.total_token_cap:
                        stop_reason, stop_requested = "token_reservation_gate", True
                        break
                    start = remaining.pop(0)
                    paths = self.shard_paths(start)
                    paths["dir"].mkdir(parents=True, exist_ok=True)
                    log_handle = paths["log"].open("ab")
                    environment = os.environ.copy()
                    environment.update({
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "PYTHONPATH": ":".join(str(RUNTIME / name) for name in ("src/sft", "src/eval", "src/harness")),
                    })
                    process = subprocess.Popen(
                        self.command(start, resume=paths["all"].exists()), cwd=RUNTIME,
                        env=environment, stdout=log_handle, stderr=subprocess.STDOUT,
                    )
                    active[start] = (process, log_handle)
                summary = self.summarize()
                state = "running"
                if summary["attempted_episodes"] == self.args.episodes and not active:
                    state = "completed"
                elif stop_requested and not active:
                    state = "stopped"
                write_json_atomic(status_path, {
                    "schema_version": "atomic-v26-low-think-scaleout-supervisor-status-v1",
                    "state": state, "stop_reason": stop_reason, "identity": identity,
                    "configuration": {
                        "model": "deepseek-v4-flash", "thinking": "enabled",
                        "reasoning_effort": "low", "episodes": self.args.episodes,
                        "shard_size": self.args.shard_size, "max_processes": self.args.max_processes,
                        "workers_per_process": self.args.workers,
                        "max_inflight_episodes": self.args.max_processes * self.args.workers,
                        "total_token_cap": self.args.total_token_cap,
                        "token_reservation_per_shard": self.args.token_reservation_per_shard,
                    },
                    "progress": summary,
                    "active_shards": {self.shard_name(start): process.pid for start, (process, _) in active.items()},
                    "updated_at_unix": time.time(),
                })
                if state in {"completed", "stopped"}:
                    return 0 if state == "completed" else 2
                time.sleep(2)
        finally:
            for process, log_handle in active.values():
                if process.poll() is None:
                    process.wait()
                log_handle.close()
            pid_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--expected-tasks-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--total-token-cap", type=int, required=True)
    parser.add_argument("--shard-size", type=int, default=25)
    parser.add_argument("--max-processes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--token-reservation-per-shard", type=int, default=2_500_000)
    parser.add_argument("--max-provider-failures", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0 or args.shard_size <= 0 or args.max_processes <= 0 or args.workers <= 0:
        parser.error("episode/shard/process/worker counts must be positive")
    if args.max_processes * args.workers > 24:
        parser.error("maximum in-flight episodes is 24")
    if args.total_token_cap <= 0 or args.token_reservation_per_shard <= 0:
        parser.error("token caps must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    raise SystemExit(Supervisor(arguments).run(dry_run=arguments.dry_run))
