#!/usr/bin/env python3
from __future__ import annotations

import ast
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = SRC_ROOT / "tool_modules"
ARCHIVE_ROOT = SRC_ROOT.parent / "archive" / "code"


class ToolModuleBoundaryTests(unittest.TestCase):
    def test_active_registry_exposes_only_atomic(self):
        import sys

        sys.path.insert(0, str(SRC_ROOT))
        sys.path.insert(0, str(SRC_ROOT / "sft"))
        from tool_modules.registry import TOOL_SCHEME_NAMES

        self.assertEqual(TOOL_SCHEME_NAMES, ("atomic",))

    def test_active_modules_do_not_import_archived_scheme_packages(self):
        violations: list[str] = []
        archived_roots = {
            "action_block",
            "checkpoint_relalg",
            "direct_sql_search",
            "iterative_sql",
            "native_tool_bundle",
            "relational_program",
            "sql_common",
        }
        for path in SRC_ROOT.rglob("*.py"):
            if "tests" in path.parts or path == Path(__file__):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.Import):
                    module = node.names[0].name
                elif isinstance(node, ast.ImportFrom):
                    module = node.module
                if module and module.startswith("tool_modules."):
                    root = module.split(".", 2)[1]
                    if root in archived_roots:
                        violations.append(f"{path.relative_to(SRC_ROOT)} imports {module}")
        self.assertEqual(violations, [])

    def test_legacy_packages_and_aliases_are_archived(self):
        for relative in (
            "legacy_tool_modules/action_block",
            "legacy_tool_modules/checkpoint_relalg",
            "legacy_tool_modules/direct_sql_search",
            "legacy_tool_modules/iterative_sql",
            "legacy_tool_modules/native_tool_bundle",
            "legacy_tool_modules/relational_program",
            "legacy_tool_modules/sql_common",
            "legacy_compatibility/tool_schemes.py",
            "legacy_compatibility/deepseek_native_tools.py",
        ):
            with self.subTest(relative=relative):
                self.assertTrue((ARCHIVE_ROOT / relative).exists())


if __name__ == "__main__":
    unittest.main()
