"""Keep temporary and remote snapshots out of the active source tree."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_no_remote_current_snapshots_in_active_tree() -> None:
    """Remote copies belong under archive/code/remote_snapshots with provenance."""
    offenders = [
        path.relative_to(REPO_ROOT)
        for path in REPO_ROOT.rglob("*.remote.current.py")
        if ".git" not in path.parts and "archive" not in path.parts
    ]
    assert offenders == [], f"remote snapshots must be archived: {offenders}"


def test_no_root_one_off_scripts() -> None:
    """The repository root is reserved for project metadata and entry docs."""
    offenders = sorted(
        path.name
        for path in REPO_ROOT.iterdir()
        if path.is_file() and (path.name.startswith("tmp_") or path.name.endswith(".remote.py"))
    )
    assert offenders == [], f"one-off scripts must live under scripts/ or archive/: {offenders}"
