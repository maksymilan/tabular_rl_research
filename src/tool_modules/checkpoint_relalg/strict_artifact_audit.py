"""Hidden, diagnostic-only strict comparison of terminal relation artifacts.

The functions in this module receive already executed answer/reference rows.
They never expose reference values or SQL and return only aggregate booleans
required by checkpoint-relalg-v1 section 21.2.
"""

from __future__ import annotations

import math
from typing import Any, Sequence


STRICT_AUDIT_VERSION = "strict-artifact-audit-v1"
DEFAULT_REL_TOL = 1e-9
DEFAULT_ABS_TOL = 1e-9


def _numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _cell_equal(left: Any, right: Any, *, rel_tol: float, abs_tol: float) -> bool:
    if _numeric(left) and _numeric(right):
        if not math.isfinite(float(left)) or not math.isfinite(float(right)):
            return False
        return math.isclose(float(left), float(right), rel_tol=rel_tol, abs_tol=abs_tol)
    if type(left) is not type(right):
        return False
    return bool(left == right)


def _row_equal(
    left: Sequence[Any],
    right: Sequence[Any],
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    return len(left) == len(right) and all(
        _cell_equal(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
        for a, b in zip(left, right)
    )


def _cell_sort_key(value: Any) -> tuple[int, Any]:
    if value is None:
        return (0, "")
    if isinstance(value, bool):
        return (1, int(value))
    if _numeric(value):
        numeric = float(value)
        return (2, numeric if math.isfinite(numeric) else repr(numeric))
    if isinstance(value, str):
        return (3, value)
    if isinstance(value, bytes):
        return (4, value)
    return (5, f"{type(value).__name__}:{value!r}")


def _row_sort_key(row: Sequence[Any]) -> tuple[tuple[int, Any], ...]:
    return tuple(_cell_sort_key(value) for value in row)


def compare_strict_artifacts(
    answer_columns: Sequence[str],
    answer_rows: Sequence[Sequence[Any]],
    reference_columns: Sequence[str],
    reference_rows: Sequence[Sequence[Any]],
    *,
    ordered: bool,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> dict[str, Any]:
    """Return strict schema/bag-or-sequence diagnostics without reference data."""

    schema_match = list(answer_columns) == list(reference_columns)
    column_count_match = len(answer_columns) == len(reference_columns)
    row_count_match = len(answer_rows) == len(reference_rows)
    answer_width_match = all(len(row) == len(answer_columns) for row in answer_rows)
    reference_width_match = all(
        len(row) == len(reference_columns) for row in reference_rows
    )

    if ordered:
        comparable_answer = list(answer_rows)
        comparable_reference = list(reference_rows)
    else:
        comparable_answer = sorted(answer_rows, key=_row_sort_key)
        comparable_reference = sorted(reference_rows, key=_row_sort_key)
    values_match = row_count_match and all(
        _row_equal(left, right, rel_tol=rel_tol, abs_tol=abs_tol)
        for left, right in zip(comparable_answer, comparable_reference)
    )
    strict = bool(
        schema_match
        and answer_width_match
        and reference_width_match
        and row_count_match
        and values_match
    )
    return {
        "version": STRICT_AUDIT_VERSION,
        "strict_artifact_accuracy": strict,
        "schema_match": schema_match,
        "column_count_match": column_count_match,
        "row_count_match": row_count_match,
        "row_width_match": answer_width_match and reference_width_match,
        "values_match": values_match,
        "comparison": "ordered-sequence" if ordered else "unordered-multiset",
        "numeric_rel_tol": rel_tol,
        "numeric_abs_tol": abs_tol,
    }


def has_top_level_order_by(sql: str) -> bool:
    """Conservatively detect semantic top-level ORDER BY outside quotes/parens."""

    from .executors.sqlite_compiler import SQLiteRelationalExecutor

    tokens = SQLiteRelationalExecutor._top_level_tokens(sql)
    return any(
        tokens[index][0] == "ORDER" and tokens[index + 1][0] == "BY"
        for index in range(len(tokens) - 1)
    )


__all__ = [
    "DEFAULT_ABS_TOL",
    "DEFAULT_REL_TOL",
    "STRICT_AUDIT_VERSION",
    "compare_strict_artifacts",
    "has_top_level_order_by",
]
