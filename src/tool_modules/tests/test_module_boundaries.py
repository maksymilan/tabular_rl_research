#!/usr/bin/env python3
from __future__ import annotations

import ast
import unittest
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
TOOL_ROOT = SRC_ROOT / "tool_modules"

LEGACY_FLAT_MODULES = {
    "batch_plan_protocol",
    "evaluate_batch_plan",
    "relational_program_protocol",
    "evaluate_relational_program",
    "direct_sql_search_protocol",
    "iterative_sql_protocol",
    "iterative_sql",
    "build_action_block_sft_data",
    "tool_schemes",
    "atomic_version51",
    "deepseek_native_tools",
    "audit_native_tool_bundle",
}

COMPATIBILITY_FILES = (
    SRC_ROOT / "eval" / "batch_plan_protocol.py",
    SRC_ROOT / "eval" / "evaluate_batch_plan.py",
    SRC_ROOT / "eval" / "relational_program_protocol.py",
    SRC_ROOT / "eval" / "evaluate_relational_program.py",
    SRC_ROOT / "eval" / "iterative_sql.py",
    SRC_ROOT / "sft" / "direct_sql_search_protocol.py",
    SRC_ROOT / "sft" / "iterative_sql_protocol.py",
    SRC_ROOT / "sft" / "build_action_block_sft_data.py",
    SRC_ROOT / "sft" / "tool_schemes.py",
    SRC_ROOT / "sft" / "atomic_version51.py",
    SRC_ROOT / "sft" / "deepseek_native_tools.py",
    SRC_ROOT / "sft" / "audit_native_tool_bundle.py",
)


class ToolModuleBoundaryTests(unittest.TestCase):
    def test_scheme_modules_do_not_import_legacy_flat_aliases(self):
        violations: list[str] = []
        for path in TOOL_ROOT.rglob("*.py"):
            if "tests" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = {alias.name.split(".", 1)[0] for alias in node.names}
                elif isinstance(node, ast.ImportFrom) and node.module:
                    roots = {node.module.split(".", 1)[0]}
                else:
                    continue
                forbidden = roots & LEGACY_FLAT_MODULES
                if forbidden:
                    violations.append(
                        f"{path.relative_to(SRC_ROOT)} imports {sorted(forbidden)}"
                    )
        self.assertEqual(violations, [])

    def test_old_locations_are_thin_compatibility_aliases(self):
        for path in COMPATIBILITY_FILES:
            with self.subTest(path=path):
                source = path.read_text(encoding="utf-8")
                self.assertIn("Compatibility", source)
                self.assertLessEqual(len(source.splitlines()), 25)
                self.assertIn("tool_modules.", source)

    def test_each_non_atomic_scheme_has_an_owned_package(self):
        for name in (
            "action_block",
            "relational_program",
            "direct_sql_search",
            "iterative_sql",
            "native_tool_bundle",
            "sql_common",
        ):
            with self.subTest(name=name):
                package = TOOL_ROOT / name
                self.assertTrue((package / "__init__.py").is_file())


if __name__ == "__main__":
    unittest.main()
