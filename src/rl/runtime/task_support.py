#!/usr/bin/env python3
"""Deterministic support checks for literals derived from model-visible task text.

This module is reward-side only.  It recognizes representation-preserving rewrites that a policy
may legitimately make from the question or external knowledge (SQL LIKE patterns, date
normalization, calendar boundaries, and small written numbers).  It does not inspect gold SQL,
database contents, later actions, or model reasoning.
"""
from __future__ import annotations

import calendar
import datetime as dt
import re
from typing import Any


_NUMBER_WORDS = {
    0: ("zero",),
    1: ("one", "single"),
    2: ("two", "both"),
    3: ("three",),
    4: ("four",),
    5: ("five",),
    6: ("six",),
    7: ("seven",),
    8: ("eight",),
    9: ("nine",),
    10: ("ten",),
}
_MONTHS = {
    name.casefold(): index
    for index, name in enumerate(calendar.month_name)
    if name
}
_MONTHS.update(
    {
        name.casefold(): index
        for index, name in enumerate(calendar.month_abbr)
        if name
    }
)
_DATE_PATTERN = re.compile(
    r"(?<!\d)(?P<year>\d{4})[-/](?P<month>\d{1,2})[-/](?P<day>\d{1,2})(?!\d)"
)


def _normalized_text(value: Any) -> str:
    return " ".join(str(value).casefold().replace("''", "'").split())


def _date(value: str) -> dt.date | None:
    match = _DATE_PATTERN.fullmatch(value.strip())
    if not match:
        return None
    try:
        return dt.date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError:
        return None


def _dates_in_text(text: str) -> set[dt.date]:
    dates: set[dt.date] = set()
    for match in _DATE_PATTERN.finditer(text):
        parsed = _date(match.group(0))
        if parsed is not None:
            dates.add(parsed)
    return dates


def _has_token(text: str, token: str) -> bool:
    return bool(re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", text))


def _calendar_boundary_supported(value: dt.date, text: str) -> bool:
    """Recognize deterministic year/month start or end boundaries named by the task."""
    year_token = str(value.year)
    if not _has_token(text, year_token):
        return False
    if (value.month, value.day) in {(1, 1), (12, 31)}:
        return True
    named_months = {
        month
        for name, month in _MONTHS.items()
        if _has_token(text, name)
    }
    if value.month not in named_months:
        return False
    last_day = calendar.monthrange(value.year, value.month)[1]
    return value.day in {1, last_day}


def _like_pattern_supported(pattern: str, text: str) -> bool:
    """Treat SQL LIKE wildcards as gaps while preserving literal fragment order."""
    if "%" not in pattern and "_" not in pattern:
        return False
    fragments = [
        re.escape(fragment)
        for fragment in re.split(r"[%_]+", pattern)
        if fragment
    ]
    return bool(fragments and re.search(r".*".join(fragments), text))


def _scaled_thousands_supported(value: Any, column: str | None, text: str) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if not isinstance(column, str) or not re.search(r"(?:^|_)k$", column.casefold()):
        return False
    scaled = float(value) * 1000
    rounded = round(scaled)
    return abs(scaled - rounded) < 1e-6 and _has_token(text, str(rounded))


def _initialism_supported(value: str, text: str) -> bool:
    words = re.findall(r"[a-z0-9]+", value.casefold())
    if len(words) < 2:
        return False
    initialism = "".join(word[0] for word in words)
    return len(initialism) >= 2 and _has_token(text, initialism)


def task_text_supports_literal(value: Any, text: str, *, column: str | None = None) -> bool:
    """Whether a literal is explicitly present or canonically derivable from visible task text."""
    if value is None:
        return True
    haystack = _normalized_text(text)
    rendered = _normalized_text(value)
    if not rendered:
        return True

    # Numeric token boundaries prevent id 7 from matching 2017.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if _has_token(haystack, rendered):
            return True
        if re.search(
            rf"(?<![a-z0-9_])id\s*[:#-]?\s*{re.escape(rendered)}(?!\d)",
            haystack,
        ):
            return True
        if _scaled_thousands_supported(value, column, haystack):
            return True
        if isinstance(value, int):
            return any(_has_token(haystack, word) for word in _NUMBER_WORDS.get(value, ()))
        return False

    if rendered in haystack:
        return True
    if _initialism_supported(rendered, haystack):
        return True
    if _like_pattern_supported(rendered, haystack):
        return True
    parsed_date = _date(rendered)
    if parsed_date is not None:
        return (
            parsed_date in _dates_in_text(haystack)
            or _calendar_boundary_supported(parsed_date, haystack)
        )
    return False
