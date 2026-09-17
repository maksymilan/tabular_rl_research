"""Historical-runner feedback-only entrypoint, with no new timeout policy."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rl.evaluation.runners.formal_v26_rollout_passk import import_frozen_runner  # noqa: E402
from rl.evaluation.runners.feedback_overlay import install_feedback_overlay  # noqa: E402


def main() -> int:
    runner = import_frozen_runner()
    install_feedback_overlay(runner, sys.modules["protocol"])
    return int(runner.main())


if __name__ == "__main__":
    raise SystemExit(main())
