import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("server.py")
SPEC = importlib.util.spec_from_file_location("experiment_dashboard_server", MODULE_PATH)
server = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(server)


class DashboardDataTests(unittest.TestCase):
    def test_research_summary_covers_formal_results_and_artifacts(self):
        summary = server.load_research_summary()
        self.assertEqual(summary["headline"]["best_full_dev_correct"], 785)
        self.assertEqual(len(summary["full_dev_runs"]), 10)
        self.assertEqual(
            next(item for item in summary["full_dev_runs"] if item["id"] == "strict120")["correct"],
            765,
        )
        self.assertGreaterEqual(len(summary["diagnostic_runs"]), 16)
        self.assertGreaterEqual(summary["coverage"]["artifact_paths"], 50)
        self.assertGreater(summary["coverage"]["local_artifacts_available"], 20)
        self.assertGreater(summary["coverage"]["remote_artifacts"], 10)
        self.assertTrue(any(
            item["path"].endswith("BIRD_VERSION51_NATIVE_TOOL_BUNDLE_FIXED200_20260805_ZH.md")
            for item in summary["artifacts"]
        ))
        self.assertEqual(len(summary["rl_method_families"]), 7)
        self.assertEqual(len(summary["rl_experiments"]), 21)
        self.assertGreaterEqual(len(summary["rl_papers"]), 12)
        self.assertEqual(
            next(item for item in summary["rl_experiments"] if item["id"] == "Exp15")["result"],
            "776/1534",
        )
        self.assertEqual(
            next(item for item in summary["rl_experiments"] if item["id"] == "Exp15")["comparison"],
            "vs SFT2: +14",
        )
        self.assertEqual(
            next(item for item in summary["rl_experiments"] if item["id"] == "Exp15")["training"],
            "32题 / 36 pair / 32 step",
        )
        self.assertEqual(
            next(item for item in summary["full_dev_runs"] if item["id"] == "exp15")["training_tasks"],
            32,
        )
        self.assertEqual(
            next(item for item in summary["rl_experiments"] if item["id"] == "Strict120")["training"],
            "120题 · batch2 / 60 batch / 53 step",
        )
        self.assertIn("GRPO-style", summary["rl_implementation_note"]["exp1"])
        self.assertTrue(all(
            {"algorithm", "training", "optimization"}.issubset(item)
            for item in summary["rl_experiments"]
        ))
        self.assertTrue(all("decision" not in item for item in summary["rl_experiments"]))
        mopd = next(item for item in summary["rl_papers"] if item["id"] == "mopd")
        self.assertEqual(mopd["url"], "https://arxiv.org/abs/2606.30406")
        self.assertTrue(any(
            item["path"].endswith("RL_METHODS_AND_PAPER_PROVENANCE_20260805_ZH.md")
            for item in summary["artifacts"]
        ))
        omnisql = summary["omnisql_comparison"]
        self.assertEqual(omnisql["total"], 1534)
        self.assertEqual(
            [(item["id"], item["correct"]) for item in omnisql["greedy"]],
            [
                ("omnisql-before-sft", 983),
                ("omnisql-after-sft", 789),
                ("sft2-checkpoint-1682", 762),
            ],
        )
        self.assertEqual(
            omnisql["pass_k"][0]["pass_at"]["4"]["correct"],
            1033,
        )
        self.assertEqual(
            next(
                item for item in omnisql["paired"]
                if item["reference"] == "SFT2 checkpoint-1682"
            )["net_correct"],
            27,
        )
        self.assertEqual(
            [(item["name"], item["correct"]) for item in summary["coder_tool_sft_runs"]],
            [
                ("greedy", 734),
                ("pass@1", 778),
                ("pass@2", 924),
                ("pass@4", 1033),
            ],
        )

    def test_registry_experiments_have_metrics(self):
        experiments = server.load_registry()
        self.assertGreaterEqual(len(experiments), 5)
        for experiment in experiments:
            metrics = server.training_metrics(experiment)
            if (
                experiment.get("kind") in {"baseline", "data-construction"}
                or not experiment.get("training", {}).get("trainer_state")
            ):
                self.assertFalse(metrics["available"])
            else:
                self.assertTrue(metrics["available"])
                self.assertGreater(len(metrics["train"]), 0)
                self.assertIn("train_loss", metrics["summary"])

    def test_evaluation_summary_matches_artifacts(self):
        experiment = server.find_experiment("qwen25-7b-v1")
        directory = server.resolve_repo_path(experiment["evaluation"]["directory"])
        summary = server.evaluation_summary(directory)
        self.assertEqual(summary["total"], 1034)
        self.assertEqual(summary["correct"], 707)
        self.assertAlmostEqual(summary["accuracy"], 707 / 1034)

    def test_bird_and_rolling_artifacts_are_visible(self):
        bird = server.find_experiment("qwen25-7b-bird-dev-direct-sql")
        bird_summary = server.enrich_experiment(bird)["evaluation_summary"]
        self.assertEqual((bird_summary["correct"], bird_summary["total"]), (598, 1534))
        self.assertTrue(bird_summary["complete"])

        rolling = server.find_experiment("bird-flash-context100-rolling4")
        runs = server.enrich_experiment(rolling)["evaluation_runs"]
        self.assertEqual([(run["summary"]["correct"], run["summary"]["total"]) for run in runs],
                         [(47, 100), (35, 100)])

        compact = server.find_experiment("bird-flash-context100-rolling4-compactpilot100")
        compact_summary = server.enrich_experiment(compact)["evaluation_summary"]
        self.assertEqual((compact_summary["correct"], compact_summary["total"]), (10, 100))
        self.assertTrue(compact_summary["complete"])

    def test_review_suffix_is_visible_to_construction_browser(self):
        self.assertTrue(
            server.is_construction_review_file(
                "bird_ds_flash_v4_context100_rolling4_success_review.jsonl"
            )
        )
        self.assertFalse(server.is_construction_review_file("bird_ds_flash_v4_context100_rolling4_success.jsonl"))

    def test_jsonl_pagination(self):
        experiment = server.find_experiment("qwen25-7b-v2ctx")
        path = server.resolve_repo_path(experiment["dataset"]["train"])
        page = server.jsonl_page(path, page=2, page_size=3)
        self.assertEqual([item["index"] for item in page["records"]], [3, 4, 5])
        self.assertGreater(page["total"], 6000)

    def test_playground_uses_training_protocol(self):
        config = server.playground_config()
        self.assertEqual(config["system_prompt"], server.PROTOCOL.SYSTEM_PROMPT)
        self.assertEqual(config["protocol_hash"], server.PROTOCOL.protocol_hash())
        self.assertIn("describe_table", config["system_prompt"])

    def test_launchable_models_are_registry_backed(self):
        models = server.launchable_models()
        self.assertGreaterEqual(len(models), 3)
        self.assertEqual(
            {item["experiment_id"] for item in models},
            {
                item["id"]
                for item in server.load_registry()
                if item.get("training", {}).get("checkpoint")
            },
        )
        for model in models:
            self.assertTrue(
                model["adapter_path"].startswith(("/home/dengyan/", "/data/dengyan/"))
            )
            self.assertGreater(model["max_model_len"], 0)

    def test_baselines_and_all_artifacts_are_visible(self):
        sql_baseline = server.find_experiment("qwen25-7b-base-sql")
        summary = server.evaluation_summary(
            server.resolve_repo_path(sql_baseline["evaluation"]["directory"])
        )
        self.assertEqual(summary["correct"], 716)
        self.assertIsNone(summary["legal_rate"])
        self.assertIsNone(summary["average_steps"])
        source_ids = {item["id"] for item in server.data_sources(sql_baseline)}
        self.assertTrue({
            "eval_all",
            "eval_success",
            "eval_failure",
            "eval_manifest",
            "eval_summary",
            "related_1_all",
            "related_1_manifest",
            "related_1_summary",
        }.issubset(source_ids))

    def test_json_assets_are_browsable(self):
        experiment = server.find_experiment("qwen25-7b-v2ctx")
        source = next(
            item for item in server.data_sources(experiment) if item["id"] == "trainer_state"
        )
        page = server.json_page(server.resolve_repo_path(source["path"]))
        self.assertEqual(page["total"], 1)
        self.assertIn("log_history", page["records"][0]["record"])


if __name__ == "__main__":
    unittest.main()
