"""Exercise cleanup after its installer scope has returned, without real PIDs."""
from pathlib import Path
import subprocess

import pytest


HELPER = Path(__file__).resolve().parents[1] / "frameworks/launcher/launch_common.sh"


@pytest.mark.parametrize("termination,code", [("exit 7", 7), ("kill -TERM $$", 143), ("kill -INT $$", 130)])
def test_cleanup_preserves_exit_and_run_root(tmp_path, termination, code):
    run_root = tmp_path / "run ' with spaces"
    run_root.mkdir()
    result = subprocess.run(
        ["bash", "-c", '''
source "$1"
stop_group() { printf 'stopped:%s\n' "$1"; }
trainer_pgid=owned-trainer
vllm_pgid=owned-server
install_process_cleanup_trap "$2"
eval "$3"
''', "cleanup-test", str(HELPER), str(run_root), termination],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == code, result.stderr
    assert result.stdout.splitlines() == ["stopped:owned-trainer", "stopped:owned-server"]
    assert (run_root / "status").read_text().endswith(f"\tfailed\texit={code}\n")


def test_success_cleanup_preserves_completed_status(tmp_path):
    result = subprocess.run(
        ["bash", "-c", '''
source "$1"
stop_group() { :; }
install_process_cleanup_trap "$2"
set_status "$2" trained_pending_audit "completed"
exit 0
''', "cleanup-test", str(HELPER), str(tmp_path)],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert "\ttrained_pending_audit\t" in (tmp_path / "status").read_text()
