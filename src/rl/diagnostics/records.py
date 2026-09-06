"""Common record extraction and indexing for JSONL diagnostic inputs."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

try:
    from rl.diagnostics.io import read_json, read_jsonl
except ModuleNotFoundError:  # direct execution with src/rl itself on sys.path
    from diagnostics.io import read_json, read_jsonl


Row = dict[str, Any]
K = TypeVar("K")


def load_jsonl(path: Path) -> list[Row]:
    """Load validated object rows through the canonical JSONL reader."""

    return read_jsonl(path)


def load_json(path: Path, *, require_object: bool = True) -> dict[str, Any] | Any:
    """Load one JSON artifact using the same strict boundary as JSONL inputs."""

    return read_json(path, require_object=require_object)


def samples(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the sample records attached to an evaluation row."""

    value = row.get("samples") or []
    if not isinstance(value, list):
        raise ValueError(f"evaluation row {row.get('example_index')!r} has non-list samples")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"evaluation row {row.get('example_index')!r} has non-object sample")
    return value


def only_sample(row: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the sole trajectory from a single-sample evaluation row."""

    observed = samples(row)
    if len(observed) != 1:
        raise ValueError(f"expected exactly one sample for example {row.get('example_index')}")
    return dict(observed[0])


def turns(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return turns from either a trajectory record or an evaluation row."""

    if "samples" in record:
        record = only_sample(record)
    value = record.get("turns") or []
    if not isinstance(value, list):
        raise ValueError(f"trajectory {record.get('trajectory_id')!r} has non-list turns")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"trajectory {record.get('trajectory_id')!r} has non-object turn")
    return value


def trajectory_actions(record: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Extract canonical ``(tool, arguments-json)`` actions from a trajectory."""

    sequence: list[tuple[str, str]] = []
    for turn in turns(record):
        parsed = turn.get("parsed") or {}
        if not isinstance(parsed, Mapping):
            continue
        tool, arguments = parsed.get("tool"), parsed.get("arguments")
        if tool is None or arguments is None:
            continue
        sequence.append(
            (
                str(tool),
                json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            )
        )
    return tuple(sequence)


def actions(record: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Compatibility spelling for :func:`trajectory_actions` in eval reports."""

    return trajectory_actions(record)


def trajectory_action_signature(record: Mapping[str, Any], *, length: int = 16) -> str:
    """Hash the parsed tool/action sequence into a stable compact signature."""

    payload = json.dumps(
        [{"tool": tool, "arguments": json.loads(arguments)} for tool, arguments in trajectory_actions(record)],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def group_rows(rows: Iterable[Row], key: Callable[[Row], K]) -> dict[K, list[Row]]:
    """Group rows while preserving input order within each key."""

    grouped: dict[K, list[Row]] = defaultdict(list)
    for row in rows:
        grouped[key(row)].append(row)
    return dict(grouped)


def group_by_example_index(rows: Iterable[Row]) -> dict[int, list[Row]]:
    """Group rollout rows by their required integer example index."""

    return group_rows(rows, lambda row: int(row["example_index"]))


def index_rows(
    rows: Iterable[Row],
    key: Callable[[Row], K],
    *,
    require_unique: bool = True,
) -> dict[K, Row]:
    """Index rows by a stable key, optionally rejecting duplicate records."""

    indexed: dict[K, Row] = {}
    for row in rows:
        row_key = key(row)
        if require_unique and row_key in indexed:
            raise ValueError(f"duplicate record key: {row_key!r}")
        indexed[row_key] = row
    return indexed


def selected_rows(path: Path, indices: Sequence[int]) -> dict[int, Row]:
    """Load exactly one row for each requested example index."""

    wanted = {int(index) for index in indices}
    if len(wanted) != len(indices):
        raise ValueError("indices contain duplicates")
    selected = index_rows(
        (row for row in load_jsonl(path) if int(row["example_index"]) in wanted),
        lambda row: int(row["example_index"]),
    )
    if set(selected) != wanted:
        raise ValueError(
            f"{path} does not contain exact selected cohort; "
            f"missing={sorted(wanted - set(selected))[:10]} "
            f"extra={sorted(set(selected) - wanted)[:10]}"
        )
    for row in selected.values():
        only_sample(row)
    return selected


__all__ = [
    "actions",
    "group_by_example_index",
    "group_rows",
    "index_rows",
    "load_json",
    "load_jsonl",
    "only_sample",
    "samples",
    "selected_rows",
    "trajectory_action_signature",
    "trajectory_actions",
    "turns",
]
