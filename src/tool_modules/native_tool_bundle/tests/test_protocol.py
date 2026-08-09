#!/usr/bin/env python3
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src" / "sft")]

from tool_modules.native_tool_bundle import protocol as atomic_version51  # noqa: E402
import protocol  # noqa: E402
from tool_modules.native_tool_bundle.provider_tools import (  # noqa: E402
    MAX_NATIVE_BUNDLE_CALLS,
    NATIVE_BUNDLE_ASSISTANT_CARRIER,
)


class AtomicVersion51Tests(unittest.TestCase):
    def test_version51_changes_turn_carrier_not_primitive_surface(self):
        self.assertEqual(atomic_version51.PROTOCOL_VERSION, "version51")
        self.assertEqual(
            atomic_version51.PROVIDER_ASSISTANT_CARRIER,
            NATIVE_BUNDLE_ASSISTANT_CARRIER,
        )
        self.assertEqual(atomic_version51.MODEL_ARG_SCHEMA, protocol.MODEL_ARG_SCHEMA)
        self.assertEqual(MAX_NATIVE_BUNDLE_CALLS, 8)
        self.assertEqual(len(atomic_version51.protocol_hash("prompt")), 16)


if __name__ == "__main__":
    unittest.main()
