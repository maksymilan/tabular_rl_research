#!/usr/bin/env python3
"""Atomically admit exactly one vanilla-GRPO confirmatory arm.

The advisory lock only serializes admission operations. Exclusivity survives
lock release because the winner is recorded in a canonical, read-only JSON
marker that this helper never replaces or removes. A later claim for the same
arm is an idempotent verification; a claim for the other arm fails forever.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import secrets
import stat
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


SCHEMA_VERSION = "vanilla-grpo-one-confirmatory-arm-admission-v1"
MARKER_STATUS = "immutable_one_arm_admission"
POLICY = "exactly_one_confirmatory_arm"
ARMS = frozenset({"arm_a", "arm_b"})
MARKER_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "policy",
        "admitted_arm",
        "admission_path",
        "lock_path",
    }
)


class AdmissionError(ValueError):
    """The immutable one-arm admission contract was not satisfied."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdmissionError(message)


def _canonical_path(path: Path, label: str) -> Path:
    _require(path.is_absolute(), f"{label} must be an absolute path: {path}")
    _require(not path.is_symlink(), f"{label} must not be a symlink: {path}")
    return path.resolve(strict=False)


def resolve_paths(admission_path: Path, lock_path: Path | None) -> tuple[Path, Path]:
    admission = _canonical_path(admission_path, "admission marker")
    if lock_path is None:
        lock = admission.with_name(admission.name + ".lock")
    else:
        lock = _canonical_path(lock_path, "admission lock")
    _require(admission != lock, "admission marker and lock paths must differ")
    _require(
        admission.parent == lock.parent,
        "admission marker and lock must be siblings in one directory",
    )
    return admission, lock


def _record(admission: Path, lock: Path, arm: str) -> dict[str, str]:
    _require(arm in ARMS, f"unsupported confirmatory arm: {arm!r}")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": MARKER_STATUS,
        "policy": POLICY,
        "admitted_arm": arm,
        "admission_path": str(admission),
        "lock_path": str(lock),
    }


def _canonical_bytes(record: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _open_flags(base: int) -> int:
    flags = base | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return flags


def _regular_fd(fd: int, label: str) -> os.stat_result:
    metadata = os.fstat(fd)
    _require(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular file")
    return metadata


def _read_fd(fd: int) -> bytes:
    os.lseek(fd, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _read_marker(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        fd = os.open(path, _open_flags(os.O_RDONLY))
    except FileNotFoundError as exc:
        raise AdmissionError(f"admission marker does not exist: {path}") from exc
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise AdmissionError(f"admission marker must not be a symlink: {path}") from exc
        raise
    try:
        before = _regular_fd(fd, "admission marker")
        _require(
            before.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0,
            "admission marker must be read-only",
        )
        payload = _read_fd(fd)
        after = os.fstat(fd)
        _require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "admission marker changed while it was read",
        )
    finally:
        os.close(fd)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdmissionError("admission marker is not valid UTF-8 JSON") from exc
    _require(isinstance(value, dict), "admission marker must be a JSON object")
    return value, payload


def _validate_marker(
    admission: Path,
    lock: Path,
    expected_arm: str,
) -> dict[str, str]:
    expected = _record(admission, lock, expected_arm)
    value, payload = _read_marker(admission)
    _require(
        frozenset(value) == MARKER_KEYS,
        "admission marker has unexpected or missing fields",
    )
    actual_arm = value.get("admitted_arm")
    _require(actual_arm in ARMS, "admission marker contains an invalid arm")
    if actual_arm != expected_arm:
        raise AdmissionError(
            f"confirmatory arm is permanently admitted to {actual_arm}; "
            f"requested {expected_arm} is forbidden"
        )
    _require(value == expected, "admission marker schema/path contract mismatch")
    _require(payload == _canonical_bytes(expected), "admission marker bytes are not canonical")
    return expected


def _fsync_directory(directory: Path) -> None:
    fd = os.open(directory, _open_flags(os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def _exclusive_lock(lock: Path) -> Iterator[None]:
    _require(lock.parent.is_dir(), f"admission directory does not exist: {lock.parent}")
    _require(
        not lock.parent.is_symlink(),
        f"admission directory must not be a symlink: {lock.parent}",
    )
    try:
        fd = os.open(lock, _open_flags(os.O_RDWR | os.O_CREAT), 0o600)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise AdmissionError(f"admission lock must not be a symlink: {lock}") from exc
        raise
    try:
        _regular_fd(fd, "admission lock")
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _publish_marker_once(path: Path, payload: bytes) -> bool:
    """Publish without an overwrite primitive; return whether this call won."""

    temporary = path.with_name(
        f".{path.name}.claim-{os.getpid()}-{secrets.token_hex(8)}"
    )
    fd = os.open(
        temporary,
        _open_flags(os.O_WRONLY | os.O_CREAT | os.O_EXCL),
        0o600,
    )
    linked = False
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            _require(written > 0, "failed to write admission marker")
            view = view[written:]
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        try:
            os.link(temporary, path, follow_symlinks=False)
            linked = True
        except FileExistsError:
            linked = False
    finally:
        os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if linked:
        _fsync_directory(path.parent)
    return linked


def claim(
    *, admission_path: Path, arm: str, lock_path: Path | None = None
) -> dict[str, Any]:
    admission, lock = resolve_paths(admission_path, lock_path)
    expected = _record(admission, lock, arm)
    with _exclusive_lock(lock):
        if admission.exists() or admission.is_symlink():
            marker = _validate_marker(admission, lock, arm)
            result_status = "existing_admission_verified"
        else:
            won = _publish_marker_once(admission, _canonical_bytes(expected))
            marker = _validate_marker(admission, lock, arm)
            result_status = "admission_claimed" if won else "existing_admission_verified"
    return {"result_status": result_status, **marker}


def verify(
    *, admission_path: Path, arm: str, lock_path: Path | None = None
) -> dict[str, Any]:
    admission, lock = resolve_paths(admission_path, lock_path)
    _record(admission, lock, arm)  # validate arm before touching shared state
    with _exclusive_lock(lock):
        marker = _validate_marker(admission, lock, arm)
    return {"result_status": "admission_verified", **marker}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("claim", "verify"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--admission", type=Path, required=True)
        subparser.add_argument("--lock", type=Path)
        subparser.add_argument("--arm", choices=sorted(ARMS), required=True)
    args = parser.parse_args(argv)
    operation = claim if args.command == "claim" else verify
    try:
        result = operation(
            admission_path=args.admission,
            lock_path=args.lock,
            arm=args.arm,
        )
    except (AdmissionError, OSError) as exc:
        print(f"confirmatory admission blocked: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
