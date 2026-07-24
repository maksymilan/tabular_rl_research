#!/usr/bin/env python3
"""Audit harness-inferred grounding edges with an external structured reviewer."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT / "src" / "sft")]

from provider_client import load_api_config  # noqa: E402
from process_credit import replay_step_features  # noqa: E402


LABELS = frozenset({"valid", "invalid", "ambiguous"})
REVIEW_PROTOCOL_VERSION = "grounding-review-v4-visible-literal-copy"
SYSTEM = """You audit causal grounding edges produced by a deterministic table-tool harness.
Use ONLY tool calls, harness outputs, and final-table samples in the supplied JSON. Never use or
infer from hidden/model reasoning. An edge is valid when the source observation structurally
supports the exact table, column, or value consumed by the target action. Mark invalid for a wrong
table/column, an unrelated final branch, or an incidental common-value collision. Mark ambiguous
when the supplied environment facts are insufficient. Check for important missing observation
dependencies.

Protocol facts you MUST apply:
- dependency_edges are complete, trusted data/value lineage. Never repeat one in missing_edges.
- A relation-producing tool registers a resident relation with its schema and all rows. Downstream
  relational tools may consume it directly even when the compact output shows no row preview.
- answer_from_context with an evidence table consumes that resident relation directly. It does not
  require a preceding read_subtable or row_observation edge.
- A derived relation's columns are carried by its producer output and trusted dependency edge.
  Do not request a repeated describe_table/schema_observation edge for a derived handle.
- A filter literal stated directly in question or external_knowledge needs no separate domain
  observation. The same applies to deterministic renderings such as an attached ID, a normalized
  date, a schema-declared unit conversion, or a conventional acronym. Do not flag such literals.
- A row/domain grounding edge normally represents the model copying an exact scalar from an
  earlier visible harness output into a later literal argument. It does NOT require value_ref or
  any syntactic source pointer. For example, aggregate output [[0]] followed by a filter literal 0
  is a valid row observation when the target column is structurally compatible. value_ref is a
  separate trusted value dependency and therefore would not need a grounding edge.
- Grounding edges represent observations needed for a base-table schema choice or for copying a
  domain/row value into a later action. Report a missing edge only for such an observation.
- Separately mark the final dependency invalid when the executed relation path omits a question
  constraint, comparison, aggregation, ordering, negation, or entity-mapping step, even if its
  denotation happens to equal the reference answer.

The dependency_edges field is supplied only as context; do not include those edges in the output
edge list. Return exactly one JSON object and no markdown. Keep each reason under 30 words."""
SYSTEM_SHA256 = hashlib.sha256(SYSTEM.encode("utf-8")).hexdigest()


def _edge_id(
    from_step: str,
    to_step: str,
    edge_type: str,
    role: str,
    target: dict[str, Any],
) -> str:
    """Return a stable target-aware id so parallel edges cannot collapse."""
    payload = json.dumps(
        target,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    return f"{from_step}->{to_step}:{edge_type}:{role}:{suffix}"


def _flatten_values(value: Any) -> list[Any]:
    if isinstance(value, dict):
        return [scalar for item in value.values() for scalar in _flatten_values(item)]
    if isinstance(value, (list, tuple)):
        return [scalar for item in value for scalar in _flatten_values(item)]
    return [value]


def _evidence_rows(rows: list[Any], target: dict[str, Any], limit: int = 10) -> list[Any]:
    wanted = _flatten_values(target.get("values") or [])
    matches = [
        row for row in rows
        if wanted and any(cell == value for cell in _flatten_values(row) for value in wanted)
    ]
    selected = []
    seen = set()
    for row in list(rows[:3]) + matches:
        key = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _compact_output(tool: str | None, output: dict[str, Any], target: dict[str, Any]) -> dict[str, Any]:
    if tool == "describe_table":
        wanted = target.get("table")
        tables = [table for table in output.get("tables", []) if table.get("table_name") == wanted]
        return {"tables": tables}
    if tool == "inspect_column":
        return output
    if isinstance(output.get("rows"), list):
        rows = output.get("rows") or []
        compact = {
            key: output.get(key)
            for key in ("table", "table_name", "kind", "columns", "row_count", "result_sample")
            if output.get(key) is not None
        }
        compact["rows"] = _evidence_rows(rows, target)
        return compact
    return {
        key: output.get(key)
        for key in ("table", "table_name", "kind", "columns", "row_count", "rows", "result_sample")
        if output.get(key) is not None
    }


def _risk_flags(role: str, target: dict[str, Any], final_audit: dict[str, Any] | None) -> list[str]:
    flags = []
    if role == "row_observation":
        values = target.get("values") or []
        common_literals = {0, 1, True, False, None, "0", "1", "true", "false"}
        if any(
            value in common_literals
            for value in values
            if isinstance(value, (str, int, float, bool)) or value is None
        ):
            flags.append("common_literal")
    if role == "automatic_final_table" and final_audit and len(final_audit.get("columns") or []) > 1:
        flags.append("multi_column_final_table")
    return flags


def _validate_package_output_path(
    packages_input: Path | None,
    packages_output: Path,
    *,
    selected_ids: set[str] | None,
    limit: int,
) -> None:
    """Prevent a subset review from truncating its complete package source."""
    if (
        packages_input is not None
        and packages_input.resolve() == packages_output.resolve()
        and (selected_ids is not None or limit > 0)
    ):
        raise ValueError(
            "--packages-output must differ from --packages-input for a selected or limited review"
        )


def build_review_package(trajectory: dict[str, Any]) -> dict[str, Any]:
    features, diagnostics = replay_step_features(trajectory)
    if not diagnostics.get("replay_correct"):
        raise ValueError(f"{trajectory.get('trajectory_id')} is not replay-correct")
    steps = {step["step_id"]: step for step in trajectory.get("steps") or []}
    slice_ids = set(diagnostics.get("back_slice_step_ids") or [])
    final_step = next(
        (step for step in reversed(trajectory.get("steps") or [])
         if (step.get("tool_call") or {}).get("tool") == "answer_from_context"),
        None,
    )
    edges = []
    dependency_edges = []
    seen = set()
    seen_dependencies = set()

    def add_edge(ref: dict[str, Any], to_step: str) -> None:
        if ref.get("type") != "grounding" or not ref.get("step"):
            return
        from_step = str(ref["step"])
        role = str(ref.get("role") or "grounding")
        target = ref.get("target") or {}
        edge_id = _edge_id(from_step, to_step, "grounding", role, target)
        if edge_id in seen:
            return
        seen.add(edge_id)
        source = steps.get(from_step) or {}
        destination = steps.get(to_step) or {}
        source_call = source.get("tool_call") or {}
        destination_call = destination.get("tool_call") or {}
        edges.append({
            "edge_id": edge_id,
            "role": role,
            "from_step": from_step,
            "from_tool": source_call.get("tool"),
            "from_arguments": source_call.get("arguments"),
            "from_output": _compact_output(source_call.get("tool"), source.get("tool_output") or {}, target),
            "to_step": to_step,
            "to_tool": destination_call.get("tool"),
            "to_arguments": destination_call.get("arguments"),
            "target": target,
            "risk_flags": _risk_flags(role, target, diagnostics.get("grounding_table_audit")),
        })

    def add_dependency(ref: dict[str, Any], to_step: str) -> None:
        if ref.get("type") not in {"data", "value"} or ref.get("step") not in slice_ids:
            return
        from_step = str(ref["step"])
        target = ref.get("target") or {}
        edge_id = _edge_id(
            from_step,
            to_step,
            str(ref["type"]),
            str(ref.get("role") or "dependency"),
            target,
        )
        if edge_id in seen_dependencies:
            return
        seen_dependencies.add(edge_id)
        source = steps.get(from_step) or {}
        destination = steps.get(to_step) or {}
        source_call = source.get("tool_call") or {}
        destination_call = destination.get("tool_call") or {}
        dependency_edges.append({
            "edge_id": edge_id,
            "type": ref["type"],
            "role": ref.get("role"),
            "from_step": from_step,
            "from_tool": source_call.get("tool"),
            "from_output": _compact_output(
                source_call.get("tool"), source.get("tool_output") or {}, target
            ),
            "to_step": to_step,
            "to_tool": destination_call.get("tool"),
            "to_arguments": destination_call.get("arguments"),
            "target": target,
        })

    for feature in features:
        if feature.step_id not in slice_ids:
            continue
        for ref in feature.references:
            add_edge(ref, feature.step_id)
            add_dependency(ref, feature.step_id)
    final_step_id = final_step.get("step_id") if final_step else "final"
    for ref in diagnostics.get("final_references") or []:
        add_edge(ref, final_step_id)
        add_dependency(ref, final_step_id)

    final_audit = diagnostics.get("grounding_table_audit") or {}
    final_dependency = {
        "handle": diagnostics.get("grounding_handle"),
        "selected_from_step": diagnostics.get("grounding_step_id"),
        "selected_from_role": diagnostics.get("grounding_step_role"),
        "table_audit": final_audit,
        "final_arguments": (final_step.get("tool_call") or {}).get("arguments") if final_step else None,
        "risk_flags": _risk_flags("automatic_final_table", {}, final_audit),
    }
    return {
        "trajectory_id": trajectory.get("trajectory_id"),
        "difficulty": trajectory.get("difficulty"),
        "question": trajectory.get("question"),
        "external_knowledge": (trajectory.get("source") or {}).get("external_knowledge"),
        "final_dependency": final_dependency,
        "grounding_edges": edges,
        "dependency_edges": dependency_edges,
        "back_slice_step_ids": diagnostics.get("back_slice_step_ids"),
    }


def _request(
    base_url: str,
    api_key: str,
    model: str,
    package: dict[str, Any],
    timeout: int,
    max_tokens: int,
) -> tuple[str, dict, dict[str, Any]]:
    edge_ids = [edge["edge_id"] for edge in package["grounding_edges"]]
    output_shape = {
        "trajectory_id": package["trajectory_id"],
        "final_dependency": {"label": "valid|invalid|ambiguous", "reason": "under 30 words"},
        "edges": [
            {"edge_id": edge_id, "label": "valid|invalid|ambiguous", "reason": "under 30 words"}
            for edge_id in edge_ids
        ],
        "missing_edges": ["short description, or empty list"],
        "overall": "pass|fail|ambiguous",
    }
    instruction = (
        "Audit the package below. Return the completed object with EXACTLY the top-level keys shown "
        "in OUTPUT SHAPE. Do not wrap it in required_output, result, response, or any other key. "
        "Copy trajectory_id and every edge_id exactly; return each edge exactly once.\n"
        f"OUTPUT SHAPE:\n{json.dumps(output_shape, ensure_ascii=False)}\n"
        f"AUDIT PACKAGE:\n{json.dumps(package, ensure_ascii=False)}"
    )
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": instruction},
        ],
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:500]}") from exc
    choice = body["choices"][0]
    message = choice["message"]
    transport = {
        "finish_reason": choice.get("finish_reason"),
        "reasoning_content": message.get("reasoning_content"),
        "message_keys": sorted(message),
    }
    return str(message.get("content") or ""), body.get("usage") or {}, transport


def _validate_review(package: dict[str, Any], text: str) -> dict[str, Any]:
    review = json.loads(text)
    if review.get("trajectory_id") != package["trajectory_id"]:
        raise ValueError("trajectory_id mismatch")
    final_label = (review.get("final_dependency") or {}).get("label")
    if final_label not in LABELS:
        raise ValueError("invalid final_dependency label")
    expected = {edge["edge_id"] for edge in package["grounding_edges"]}
    returned = {edge.get("edge_id") for edge in review.get("edges") or []}
    if returned != expected:
        raise ValueError(f"edge ids mismatch: expected {sorted(expected)}, got {sorted(returned)}")
    if any(edge.get("label") not in LABELS for edge in review.get("edges") or []):
        raise ValueError("invalid edge label")
    if review.get("overall") not in {"pass", "fail", "ambiguous"}:
        raise ValueError("invalid overall label")
    if not isinstance(review.get("missing_edges"), list):
        raise ValueError("missing_edges must be a list")
    return review


def review_one(package: dict[str, Any], *, base_url: str, api_key: str, model: str,
               timeout: int, retries: int, max_tokens: int) -> dict[str, Any]:
    last_error = None
    attempt_audit = []
    for attempt in range(retries + 1):
        try:
            text, usage, transport = _request(
                base_url, api_key, model, package, timeout, max_tokens
            )
            attempt_audit.append({
                "attempt": attempt + 1,
                "raw_content": text,
                "usage": usage,
                "transport": transport,
            })
            review = _validate_review(package, text)
            return {"trajectory_id": package["trajectory_id"], "model": model, "review": review,
                    "usage": usage, "raw_content": text, "attempts": attempt + 1,
                    "attempt_audit": attempt_audit,
                    "review_protocol_version": REVIEW_PROTOCOL_VERSION,
                    "system_prompt_sha256": SYSTEM_SHA256}
        except Exception as exc:  # transport and strict format failures remain audited
            last_error = f"{type(exc).__name__}: {exc}"
            if not attempt_audit or attempt_audit[-1].get("attempt") != attempt + 1:
                attempt_audit.append({"attempt": attempt + 1, "request_error": last_error})
            else:
                attempt_audit[-1]["validation_error"] = last_error
            if attempt < retries:
                time.sleep(min(2 ** attempt, 4))
    return {
        "trajectory_id": package["trajectory_id"],
        "model": model,
        "review_error": last_error,
        "attempts": retries + 1,
        "attempt_audit": attempt_audit,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "system_prompt_sha256": SYSTEM_SHA256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", default=[])
    parser.add_argument("--packages-input", type=Path)
    parser.add_argument("--packages-output", type=Path, required=True)
    parser.add_argument("--reviews-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--trajectory-ids-file", type=Path)
    parser.add_argument("--sft-index", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=3000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--packages-only", action="store_true")
    args = parser.parse_args()
    if not args.input and not args.packages_input:
        parser.error("provide at least one --input or --packages-input")
    if args.input and args.packages_input:
        parser.error("--input and --packages-input are mutually exclusive")
    if args.trajectory_ids_file and args.sft_index:
        parser.error("--trajectory-ids-file and --sft-index are mutually exclusive")

    selected_ids: set[str] | None = None
    if args.trajectory_ids_file:
        selected_ids = {
            line.strip()
            for line in args.trajectory_ids_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    elif args.sft_index:
        selected_ids = {
            str(row.get("trajectory_id") or row.get("source_episode_id"))
            for row in (
                json.loads(line)
                for line in args.sft_index.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if row.get("trajectory_id") or row.get("source_episode_id")
        }
    try:
        _validate_package_output_path(
            args.packages_input,
            args.packages_output,
            selected_ids=selected_ids,
            limit=args.limit,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.packages_input:
        packages = [
            json.loads(line)
            for line in args.packages_input.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if selected_ids is not None:
            packages = [
                package
                for package in packages
                if package.get("trajectory_id") in selected_ids
            ]
            observed_ids = {
                str(package.get("trajectory_id"))
                for package in packages
            }
            missing_ids = sorted(selected_ids - observed_ids)
            if missing_ids:
                parser.error(f"{len(missing_ids)} selected packages are missing from input")
    else:
        trajectories = []
        for path in args.input:
            trajectories.extend(
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        if selected_ids is not None:
            trajectories = [
                trajectory for trajectory in trajectories
                if trajectory.get("trajectory_id") in selected_ids
            ]
            observed_ids = {
                str(trajectory.get("trajectory_id"))
                for trajectory in trajectories
            }
            missing_ids = sorted(selected_ids - observed_ids)
            if missing_ids:
                parser.error(f"{len(missing_ids)} selected trajectories are missing from inputs")
        if args.limit > 0:
            trajectories = trajectories[:args.limit]
        packages = [build_review_package(trajectory) for trajectory in trajectories]

    if args.limit > 0:
        packages = packages[:args.limit]
    for output_path in (args.packages_output, args.reviews_output, args.summary_output):
        output_path.parent.mkdir(parents=True, exist_ok=True)
    args.packages_output.write_text(
        "".join(json.dumps(package, ensure_ascii=False) + "\n" for package in packages), encoding="utf-8"
    )
    if args.packages_only:
        summary = {
            "model": None,
            "review_protocol_version": REVIEW_PROTOCOL_VERSION,
            "system_prompt_sha256": SYSTEM_SHA256,
            "trajectories": len(packages),
            "review_errors": 0,
            "external_review_only": True,
            "packages_only": True,
            "grounding_precision_audit_approved": False,
        }
        args.reviews_output.write_text("", encoding="utf-8")
        args.summary_output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    existing = {}
    if args.resume and args.reviews_output.exists():
        existing = {
            row["trajectory_id"]: row
            for row in (json.loads(line) for line in args.reviews_output.read_text(encoding="utf-8").splitlines() if line.strip())
            if row.get("review")
        }
    api_key, base_url = load_api_config()
    if not api_key or not base_url:
        parser.error("api.md must define API_KEY and BASE_URL")

    results = dict(existing)
    pending = [package for package in packages if package["trajectory_id"] not in results]
    if not args.resume:
        args.reviews_output.write_text("", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(review_one, package, base_url=base_url, api_key=api_key, model=args.model,
                        timeout=args.timeout, retries=args.retries, max_tokens=args.max_tokens): package["trajectory_id"]
            for package in pending
        }
        with args.reviews_output.open("a", encoding="utf-8") as sink:
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                results[result["trajectory_id"]] = result
                sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                sink.flush()
                print(f"[{index}/{len(pending)}] {result['trajectory_id']} "
                      f"{(result.get('review') or {}).get('overall', 'ERROR')}", flush=True)

    ordered = [results[package["trajectory_id"]] for package in packages]
    args.reviews_output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered), encoding="utf-8"
    )
    final_labels = collections.Counter()
    edge_labels = collections.Counter()
    overall = collections.Counter()
    errors = 0
    missing = 0
    usage = collections.Counter()
    for result in ordered:
        review = result.get("review")
        if not review:
            errors += 1
            continue
        final_labels[(review.get("final_dependency") or {}).get("label")] += 1
        overall[review.get("overall")] += 1
        missing += len(review.get("missing_edges") or [])
        for edge in review.get("edges") or []:
            edge_labels[edge.get("label")] += 1
        for key, value in (result.get("usage") or {}).items():
            if isinstance(value, (int, float)):
                usage[key] += value
    summary = {
        "model": args.model,
        "review_protocol_version": REVIEW_PROTOCOL_VERSION,
        "system_prompt_sha256": SYSTEM_SHA256,
        "trajectories": len(packages),
        "review_errors": errors,
        "final_dependency_labels": dict(sorted(final_labels.items())),
        "edge_labels": dict(sorted(edge_labels.items())),
        "overall_labels": dict(sorted(overall.items())),
        "reported_missing_edges": missing,
        "usage": dict(sorted(usage.items())),
        "final_dependency_decided_precision": (
            final_labels["valid"] / (final_labels["valid"] + final_labels["invalid"])
            if final_labels["valid"] + final_labels["invalid"] else None
        ),
        "edge_decided_precision": (
            edge_labels["valid"] / (edge_labels["valid"] + edge_labels["invalid"])
            if edge_labels["valid"] + edge_labels["invalid"] else None
        ),
        "external_review_only": True,
        "grounding_precision_audit_approved": False,
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
