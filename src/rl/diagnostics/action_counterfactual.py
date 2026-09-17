"""Corpus selection, identity gates and bounded offline action replay reporting."""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import importlib
from pathlib import Path
import signal
import sys
import threading
import time

from rl.configuration.experiment_config import runtime_content_tree_sha256
from rl.diagnostics.io import iter_jsonl, read_json, sha256_file, sha256_json, write_json, write_jsonl
from rl.diagnostics.reporting import ManifestBuilder
from rl.diagnostics.rollout_corpus import compact_rollout, question_key
from rl.runtime.action_counterfactual import audit_trajectory


def rollout_identity(row: dict) -> tuple:
    return (row.get("policy_global_step"), row.get("policy_micro_step", 0),
            question_key(row), str(row.get("trajectory_id")))


def select_pairs(rollouts: Path, partitions: dict[str, str], groups: int,
                 protocol_hash: str) -> list[dict]:
    """Two streaming passes: only bounded selected records retain full transcripts."""
    if groups <= 0:
        raise ValueError("groups must be positive")
    grouped = defaultdict(list)
    for row in iter_jsonl(rollouts):
        if row.get("protocol_hash") != protocol_hash:
            raise ValueError("rollout protocol mismatch")
        key = rollout_identity(row)
        if partitions.get(key[2]) not in ("discovery", "reviewed_development"):
            continue
        if not isinstance(row.get("correct"), bool):
            raise ValueError("missing Boolean correctness")
        grouped[key[:3]].append((key, row["correct"]))
    selected_keys = set()
    for group_key in sorted(grouped, key=repr):
        rows = grouped[group_key]
        if len(rows) != 8 or len({r[0] for r in rows}) != 8:
            raise ValueError("selected corpus has incomplete or duplicate K=8 groups")
        positives = sorted((r[0] for r in rows if r[1]), key=lambda k: k[3])
        negatives = sorted((r[0] for r in rows if not r[1]), key=lambda k: k[3])
        if positives and negatives:
            selected_keys.update({positives[0], negatives[0]})
        if len(selected_keys) >= groups * 2:
            break
    if len(selected_keys) != groups * 2:
        raise ValueError("not enough mixed groups for the requested sample")
    selected = [row for row in iter_jsonl(rollouts) if rollout_identity(row) in selected_keys]
    if len(selected) != len(selected_keys):
        raise ValueError("two-pass selection identity/coverage mismatch")
    return sorted(selected, key=lambda row: tuple(map(str, rollout_identity(row))))


def resolve_database(db_id: str, roots: list[Path]) -> Path:
    if Path(db_id).name != db_id or db_id in (".", ".."):
        raise ValueError("invalid database identifier")
    candidates = {p.resolve() for root in roots
                  if (p := root / db_id / f"{db_id}.sqlite").is_file()}
    if len(candidates) != 1:
        raise ValueError(f"expected one explicit database for {db_id}, found {len(candidates)}")
    return next(iter(candidates))


def configure_runtime(runtime_root: Path, source_manifest: dict) -> str:
    """Explicit frozen-runtime selection; an already loaded different runtime fails."""
    runtime_root = runtime_root.resolve()
    digest = runtime_content_tree_sha256(runtime_root)
    expected = (source_manifest.get("runtime_identity_audit") or {}).get("expected") or {}
    if digest != expected.get("runtime_content_tree_sha256"):
        raise ValueError("replay runtime is not the recorded frozen source runtime")
    for name in ("protocol", "rollout", "executor"):
        module = sys.modules.get(name)
        if module is not None and not Path(module.__file__).resolve().is_relative_to(runtime_root):
            raise ValueError(f"{name} already loaded from another runtime; use a fresh process")
    sys.path[:0] = [str(runtime_root / "src" / name) for name in ("eval", "sft", "harness")]
    protocol = importlib.import_module("protocol")
    importlib.import_module("rollout")
    if protocol.PROTOCOL_VERSION != source_manifest.get("protocol_version"):
        raise ValueError("protocol version differs from source manifest")
    selected_prompt = protocol.student_runtime_system_prompt(
        context_mode="rolling-legal-history", compact=False
    )
    if protocol.protocol_hash(selected_prompt) != source_manifest.get("protocol_hash"):
        raise ValueError("protocol hash differs from source manifest")
    if hashlib.sha256(selected_prompt.encode("utf-8")).hexdigest() != source_manifest.get("student_prompt_sha256"):
        raise ValueError("student prompt hash differs from source manifest")
    return digest


class _AuditDeadline(BaseException):
    """Escapes the environment's policy-error handler; not a policy timeout."""


@contextmanager
def audit_deadline(seconds: float):
    if seconds <= 0 or threading.current_thread() is not threading.main_thread():
        raise ValueError("audit deadline requires a positive budget and main thread")
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("refusing to replace an existing process alarm")
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise _AuditDeadline()
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def run_audit(*, rollouts: Path, source_manifest: Path, partitions_path: Path,
              runtime_root: Path, db_roots: list[Path], output: Path,
              groups: int = 30, trajectory_seconds: float = 60.0) -> dict:
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    source = read_json(source_manifest, require_object=True)
    runtime_hash = configure_runtime(runtime_root, source)
    partitions = {r["task_key"]: r["partition"] for r in iter_jsonl(partitions_path)}
    selected = select_pairs(rollouts, partitions, groups, source["protocol_hash"])
    databases = {r["db_id"]: resolve_database(r["db_id"], db_roots) for r in selected}
    database_hashes = {name: sha256_file(path) for name, path in databases.items()}
    repo = Path(__file__).resolve().parents[3]
    builder = ManifestBuilder(
        "action-counterfactual-fixed-suffix-v2", diagnostic_only=True,
        actor_update_allowed=False, external_teacher=False, gpu_used=False,
        estimand="fixed_recorded_suffix_no_policy_regeneration",
        source_identity={key: source.get(key) for key in (
            "protocol_version", "protocol_hash", "student_prompt_sha256", "adapter_path",
            "initial_adapter_sha256", "base_model_identity",
        )}, runtime_content_tree_sha256=runtime_hash,
        trajectory_seconds=trajectory_seconds, groups=groups,
        database_hashes=database_hashes,
        selection_rule="lexicographic source-step/task/id order; one positive and one negative per mixed K=8 group",
    )
    builder.add_input("rollouts", rollouts, include_size=True)
    builder.add_input("source_run_manifest", source_manifest)
    builder.add_input("partitions", partitions_path)
    builder.add_input("runtime_lock", runtime_root / "runtime_lock.json")
    for name, path in databases.items():
        builder.add_input("db_" + name, path, include_size=True)
    for name, path in {
        "orchestration": Path(__file__),
        "replay": repo / "src/rl/runtime/action_counterfactual.py",
        "environment": repo / "src/rl/runtime/tool_environment_v26.py",
        "feedback": repo / "src/rl/runtime/error_feedback.py",
        "cli": repo / "src/rl/scenarios/diagnostics/audit_action_counterfactual.py",
    }.items():
        builder.add_input(name, path)
    output.mkdir(parents=True)
    builder.write(output / "preflight_manifest.json")
    write_jsonl(output / "selected_public_rows.jsonl", [compact_rollout(r) for r in selected])
    (output / "receipts").mkdir()
    all_cases, trajectory_rows = [], []
    for position, row in enumerate(selected, 1):
        started = time.monotonic()
        identity = rollout_identity(row)
        receipt = {
            "trajectory_identity": list(identity), "trajectory_id": row["trajectory_id"],
            "db_id": row["db_id"], "task_key": identity[2],
            "partition": partitions[identity[2]], "correct": row["correct"],
            "source_errors": row.get("errors"), "source_failure_type": row.get("failure_type"),
        }
        try:
            with audit_deadline(trajectory_seconds):
                cases = audit_trajectory(row, str(databases[row["db_id"]]))
            receipt["status"] = cases[0].label if cases and cases[0].skip_turn_index < 0 else "baseline_passed"
            case_rows = [{**c.as_dict(), "trajectory_identity": list(identity)} for c in cases]
        except _AuditDeadline:
            receipt["status"] = "audit_budget_exceeded"
            case_rows = []
        receipt["branches"] = sum(c["skip_turn_index"] >= 0 for c in case_rows)
        receipt["labels"] = dict(Counter(c["label"] for c in case_rows))
        receipt["seconds"] = round(time.monotonic() - started, 3)
        all_cases.extend(case_rows)
        trajectory_rows.append(receipt)
        write_json(output / "receipts" / (sha256_json(identity) + ".json"),
                   {"trajectory": receipt, "cases": case_rows})
        print(f"[{position}/{len(selected)}] {row['trajectory_id']} {receipt['status']} branches={receipt['branches']}", flush=True)
    after_hashes = {name: sha256_file(path) for name, path in databases.items()}
    if after_hashes != database_hashes:
        raise RuntimeError("database contents changed during read-only audit; results invalid")
    branches = [c for c in all_cases if c["skip_turn_index"] >= 0]
    summary = {
        "diagnostic_only": True, "selected_trajectories": len(selected),
        "unique_tasks": len({r["task_key"] for r in trajectory_rows}),
        "trajectory_statuses": dict(Counter(r["status"] for r in trajectory_rows)),
        "branches": len(branches), "branch_labels": dict(Counter(c["label"] for c in branches)),
        "by_original_correct": {
            str(value): dict(Counter(c["label"] for c in branches if c["original_correct"] is value))
            for value in (True, False)
        },
        "by_tool": {
            tool: dict(Counter(c["label"] for c in branches if c["skip_tool"] == tool))
            for tool in sorted({c["skip_tool"] for c in branches})
        },
        "database_hashes_unchanged": True,
        "policy_mediated_effect": "not_identified",
        "seconds": round(sum(r["seconds"] for r in trajectory_rows), 3),
    }
    write_jsonl(output / "cases.jsonl", all_cases)
    write_jsonl(output / "trajectory_summary.jsonl", trajectory_rows)
    write_json(output / "summary.json", summary)
    for filename in ("cases.jsonl", "trajectory_summary.jsonl", "summary.json", "selected_public_rows.jsonl", "preflight_manifest.json"):
        builder.add_output(filename, output / filename)
    builder.set("summary", summary).write(output / "manifest.json")
    return summary
