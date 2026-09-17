"""CPU-only regression checks for trainer execution defaults."""
from __future__ import annotations

import argparse
import ast
import unittest
from pathlib import Path


RUNNER = Path(__file__).resolve().parents[1] / "frameworks/trl/run_transition_grpo.py"


class GradientCheckpointingDefaultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Exercise the real resolver without importing GPU-only launcher deps.
        tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "resolve_gradient_checkpointing"
        )
        scope = {"argparse": argparse}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(RUNNER), "exec"), scope)
        cls.resolve = staticmethod(scope["resolve_gradient_checkpointing"])

    def test_omitted_flag_preserves_both_replicated_storage_defaults(self):
        for storage in ("bf16", "4bit"):
            with self.subTest(storage=storage):
                self.assertTrue(self.resolve(
                    argparse.Namespace(gradient_checkpointing=None),
                    trainer_sharding="replicated", replicated_base_storage=storage,
                ))

    def test_fsdp_keeps_separate_activation_checkpointing_default(self):
        self.assertFalse(self.resolve(
            argparse.Namespace(gradient_checkpointing=None),
            trainer_sharding="fsdp", replicated_base_storage="bf16",
        ))

    def test_explicit_replicated_override_is_honored(self):
        for storage in ("bf16", "4bit"):
            for enabled in (False, True):
                with self.subTest(storage=storage, enabled=enabled):
                    self.assertEqual(enabled, self.resolve(
                        argparse.Namespace(gradient_checkpointing=enabled),
                        trainer_sharding="replicated", replicated_base_storage=storage,
                    ))


if __name__ == "__main__":
    unittest.main()
