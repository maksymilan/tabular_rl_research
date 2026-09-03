#!/usr/bin/env python3
"""Compatibility alias for :mod:`tool_modules.direct_sql_search.protocol`."""
from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tool_modules.direct_sql_search import protocol as _implementation  # noqa: E402

sys.modules[__name__] = _implementation
