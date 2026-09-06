from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src" / "rl"))

from rl.configuration.experiment_config import RLExperimentConfig  # noqa: E402


class SAAMExperimentConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        source = ROOT / "src/rl/configs/experiments/phase1_result_only.yaml"
        self.payload = yaml.safe_load(source.read_text(encoding="utf-8"))

    def _load(self, payload: dict) -> RLExperimentConfig:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "experiment.yaml"
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        return RLExperimentConfig.load(path)

    def test_saam_maps_to_an_explicit_trainer_argument(self):
        payload = copy.deepcopy(self.payload)
        payload["credit_assignment"] = "saam-strict"
        config = self._load(payload)
        defaults = config.argparse_defaults(ROOT)
        self.assertEqual(defaults["reward_mode"], "result-only")
        self.assertEqual(defaults["result_reward_profile"], "binary")
        self.assertEqual(defaults["credit_assignment"], "saam-strict")
        self.assertEqual(defaults["kl_beta"], 0.0)
        self.assertEqual(defaults["rank_loss_coefficient"], 0.0)

    def test_existing_configs_default_to_vanilla_trajectory_credit(self):
        config = self._load(copy.deepcopy(self.payload))
        self.assertEqual(
            config.argparse_defaults(ROOT)["credit_assignment"],
            "trajectory",
        )

    def test_saam_rejects_nonbinary_or_mixed_auxiliary_objectives(self):
        mutations = (
            ("nonbinary", lambda value: value.update(result_reward_profile="execution-ladder")),
            ("kl", lambda value: value["optimizer"].update(kl_beta=0.1)),
            (
                "rank",
                lambda value: value["rank_loss"].update(
                    enabled=True,
                    coefficient=0.5,
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                payload = copy.deepcopy(self.payload)
                payload["credit_assignment"] = "saam-strict"
                mutate(payload)
                with self.assertRaises(ValueError):
                    self._load(payload)

    def test_asymmetric_saam_maps_penalty_and_gradient_logging(self):
        payload = copy.deepcopy(self.payload)
        payload["credit_assignment"] = "saam-asymmetric-error"
        payload["error_penalty"] = 2.0
        payload["record_gradient_conflicts"] = True
        payload["gradient_conflict_save_vectors"] = False
        config = self._load(payload)
        defaults = config.argparse_defaults(ROOT)
        self.assertEqual(defaults["credit_assignment"], "saam-asymmetric-error")
        self.assertEqual(defaults["error_penalty"], 2.0)
        self.assertTrue(defaults["record_gradient_conflicts"])
        self.assertFalse(defaults["gradient_conflict_save_vectors"])

    def test_asymmetric_rejects_nonpositive_penalty(self):
        payload = copy.deepcopy(self.payload)
        payload["credit_assignment"] = "saam-asymmetric-error"
        payload["error_penalty"] = 0.0
        with self.assertRaises(ValueError):
            self._load(payload)

    def test_smc_mode_concentration_is_an_isolated_trajectory_arm(self):
        payload = copy.deepcopy(self.payload)
        payload["result_advantage_profile"] = "smc-mode-concentration"
        payload["credit_assignment"] = "trajectory"
        config = self._load(payload)
        defaults = config.argparse_defaults(ROOT)
        self.assertEqual(
            defaults["result_advantage_profile"],
            "smc-mode-concentration",
        )
        self.assertEqual(defaults["credit_assignment"], "trajectory")

    def test_smc_rejects_saam_composition(self):
        payload = copy.deepcopy(self.payload)
        payload["result_advantage_profile"] = "smc-mode-concentration"
        payload["credit_assignment"] = "saam-asymmetric-error"
        with self.assertRaises(ValueError):
            self._load(payload)

    def test_nested_mechanism_is_validated_and_maps_to_trainer(self):
        payload = copy.deepcopy(self.payload)
        payload["credit_assignment"] = "trajectory"
        payload["mechanism"] = {
            "reward_mode": "result-only",
            "result_advantage_profile": "stored",
            "clean_advantage_weight": 0.4,
            "policy_reduction": "trajectory_token_mean",
            "credit_assignment": "saam-asymmetric-error",
            "error_penalty": 1.25,
            "span_balance_alpha": 0.5,
            "kl_beta": 0.0,
        }
        config = self._load(payload)
        defaults = config.argparse_defaults(ROOT)
        self.assertEqual(defaults["credit_assignment"], "saam-asymmetric-error")
        self.assertEqual(defaults["error_penalty"], 1.25)
        self.assertEqual(defaults["span_balance_alpha"], 0.5)

    def test_nested_mechanism_rejects_invalid_reduction(self):
        payload = copy.deepcopy(self.payload)
        payload["mechanism"] = {"policy_reduction": "unknown"}
        with self.assertRaises(ValueError):
            self._load(payload)


if __name__ == "__main__":
    unittest.main()
