#!/usr/bin/env python3
"""Run the active harness tests.

The harness predates unittest discovery and exposes a small ``run()`` function per module. Retired
compiler/emitter tests live under ``archive/`` and are intentionally excluded.
"""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE / "tests")]

from test_environment_state import run as run_environment_state  # noqa: E402
from test_executor import run as run_executor  # noqa: E402
from test_observation_binding import run as run_observation_binding  # noqa: E402
from test_plan import run as run_plan  # noqa: E402
from test_relation_derivation import run as run_relation_derivation  # noqa: E402


def main() -> int:
    results = [
        run_executor(),
        run_plan(),
        run_environment_state(),
        run_relation_derivation(),
        run_observation_binding(),
    ]
    passed = sum(module_passed for module_passed, _, _ in results)
    failed = sum(module_failed for _, module_failed, _ in results)
    for _, _, failures in results:
        for failure in failures:
            print(f"FAIL: {failure}")
    total = passed + failed
    print(f"ACTIVE HARNESS TOTAL: {passed}/{total}")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
