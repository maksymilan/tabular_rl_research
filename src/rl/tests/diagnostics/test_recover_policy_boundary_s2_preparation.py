from __future__ import annotations

from pathlib import Path

import pytest

from rl.scenarios.diagnostics.recover_policy_boundary_s2_preparation import recover_preparation


def _fake_preparer(path: Path) -> None:
    path.write_text(
        """
from pathlib import Path
import argparse
def write_atomic(path, data):
    path.write_bytes(data)
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args(argv)
    payload = args.source.read_bytes()
    if args.output.exists() or args.manifest.exists():
        return 2
    write_atomic(args.output, b'TASKS:' + payload)
    write_atomic(args.manifest, b'MANIFEST:' + payload)
    return 0
""".lstrip()
    )


def _setup(tmp_path: Path) -> tuple[Path, Path, Path, Path, list[str]]:
    preparer = tmp_path / "preparer.py"
    _fake_preparer(preparer)
    source = tmp_path / "source"
    source.write_bytes(b"frozen")
    root = tmp_path / "inputs"
    root.mkdir()
    output, manifest = root / "tasks.jsonl", root / "tasks.manifest.json"
    argv = ["--output", str(output), "--manifest", str(manifest), "--source", str(source)]
    return preparer, output, manifest, tmp_path / "quarantine", argv


@pytest.mark.parametrize("published_before_crash", [0, 1, 2])
def test_recovers_every_two_file_publication_prefix(
    tmp_path: Path, published_before_crash: int
) -> None:
    preparer, output, manifest, quarantine, argv = _setup(tmp_path)
    first = recover_preparation(
        preparer_path=preparer,
        preparer_argv=argv,
        output_path=output,
        manifest_path=manifest,
        quarantine_dir=quarantine,
    )
    assert first["status"] == "complete"
    expected = (output.read_bytes(), manifest.read_bytes())
    for path in (output, manifest)[published_before_crash:]:
        path.unlink()
    recovered = recover_preparation(
        preparer_path=preparer,
        preparer_argv=argv,
        output_path=output,
        manifest_path=manifest,
        quarantine_dir=quarantine,
    )
    assert len(recovered["published_missing"]) == 2 - published_before_crash
    assert (output.read_bytes(), manifest.read_bytes()) == expected


def test_matching_tmp_recovers_but_mismatch_and_unknown_fail_before_publish(
    tmp_path: Path,
) -> None:
    preparer, output, manifest, quarantine, argv = _setup(tmp_path)
    output.with_suffix(output.suffix + ".tmp").write_bytes(b"TASKS:frozen")
    recovered = recover_preparation(
        preparer_path=preparer,
        preparer_argv=argv,
        output_path=output,
        manifest_path=manifest,
        quarantine_dir=quarantine,
    )
    assert len(recovered["published_missing"]) == 2

    manifest.unlink()
    output.write_bytes(b"changed")
    with pytest.raises(ValueError, match="differs"):
        recover_preparation(
            preparer_path=preparer,
            preparer_argv=argv,
            output_path=output,
            manifest_path=manifest,
            quarantine_dir=quarantine,
        )
    assert not manifest.exists()

    output.unlink()
    (output.parent / "unknown").write_text("unknown")
    with pytest.raises(ValueError, match="unknown S2 preparation"):
        recover_preparation(
            preparer_path=preparer,
            preparer_argv=argv,
            output_path=output,
            manifest_path=manifest,
            quarantine_dir=quarantine,
        )
    assert not output.exists() and not manifest.exists()

