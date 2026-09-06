from __future__ import annotations

import pytest

from src.rl.diagnostics.analyze_evaluation_results import exact_mcnemar_p


def test_exact_mcnemar_matches_known_small_cases():
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(1, 0) == 1.0
    assert exact_mcnemar_p(2, 0) == pytest.approx(0.5)
    assert exact_mcnemar_p(4, 0) == pytest.approx(0.125)
    assert exact_mcnemar_p(3, 3) == 1.0
