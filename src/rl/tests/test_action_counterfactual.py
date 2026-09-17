"""Regression tests for offline fixed-suffix audit semantics."""
from pathlib import Path
import os
import subprocess
import sys
import unittest

from rl.runtime.action_counterfactual import _branch_label, _turn_signature


class ActionCounterfactualUnitTests(unittest.TestCase):
    def test_json_roundtrip_is_not_a_semantic_mismatch(self):
        raw = {"parsed": {"tool": "answer_from_context"}, "pred_sample": [(1, "a")]}
        stored = {"parsed": {"tool": "answer_from_context"}, "pred_sample": [[1, "a"]]}
        self.assertEqual(_turn_signature(raw), _turn_signature(stored))
        stored["pred_sample"] = [[2, "a"]]
        self.assertNotEqual(_turn_signature(raw), _turn_signature(stored))

    def test_missing_evidence_is_not_semantic_result_change(self):
        self.assertEqual(_branch_label(
            {"correct": True}, {"correct": False, "replay_terminal_evidence_exists": False},
            had_new_error=False,
        ), "missing_terminal_evidence")

    def test_fixed_suffix_labels_do_not_claim_policy_effect(self):
        self.assertEqual(_branch_label(
            {"correct": True}, {"correct": True, "replay_terminal_evidence_exists": True},
            had_new_error=False,
        ), "executable_same_result")
        self.assertEqual(_branch_label({}, {}, had_new_error=False), "unknown_no_terminal")
        self.assertEqual(_branch_label({}, {}, had_new_error=True), "new_execution_error")

    def test_frozen_runtime_skip_and_readonly_integration(self):
        root = Path(__file__).resolve().parents[3]
        runtime = root / "tmp/version26-runtime-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de"
        if not runtime.is_dir():
            self.skipTest("explicit exported v26 runtime is not installed")
        program = r'''
import json, sqlite3, tempfile, runpy, sys
from pathlib import Path
runtime = Path("tmp/version26-runtime-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de/src")
sys.path[:0] = [str(runtime / name) for name in ("eval", "sft", "harness")]
from rl.runtime.tool_environment_v26 import ToolUseEnv
from rl.runtime.action_counterfactual import audit_trajectory, _env_from_record
with tempfile.TemporaryDirectory() as temp:
    db = Path(temp) / "items.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE items(id INTEGER, name TEXT)")
    con.executemany("INSERT INTO items VALUES (?,?)", [(1,"a"),(2,"b")])
    con.commit(); con.close()
    env = ToolUseEnv({"db_id":"items", "question":"Name of largest id?", "db_path":str(db),
                      "gold_sql":"SELECT name FROM items WHERE id=2"}, max_steps=30)
    calls = [
        ("describe_table", {"tables":["items"]}),
        ("condition_filter", {"table":"items", "conditions":{"column":"id","op":"=","value":1}}),
        ("group_aggregate", {"table":"items","group_by":[],"aggregations":[{"column":"id","op":"max","as":"id"}]}),
        ("condition_filter", {"table":"items", "conditions":{"column":"id","op":"=","value_ref":"step_3"}}),
        ("project", {"table":"filter_003","expressions":["name"]}),
        ("answer_from_context", {"evidence":{"table":"project_004"}}),
    ]
    for tool,args in calls:
        env.apply_model_output("<think>Use the visible facts.</think>" + json.dumps({"tool":tool,"arguments":args}))
    record = json.loads(json.dumps(env.record()))
    record["trajectory_id"] = "synthetic"
    env.close()
    assert record["correct"], record["failure_type"]
    cases = audit_trajectory(record,str(db))
    labels = {c.skip_turn_index:c.label for c in cases}
    assert labels[0] == "executable_same_result", labels
    assert labels[1] == "executable_same_result", labels
    assert labels[2] == "new_execution_error", labels
    assert labels[4] == "missing_terminal_evidence", labels
    guarded = _env_from_record(record,str(db))
    try:
        try:
            guarded.harness.conn.execute("CREATE TABLE forbidden(x INT)")
            raise AssertionError("persistent write was allowed")
        except sqlite3.DatabaseError:
            pass
        guarded.harness.conn.execute("CREATE TEMP TABLE allowed(x INT)")
    finally:
        guarded.close()
    record["correct"] = False
    assert audit_trajectory(record,str(db))[0].label == "baseline_mismatch"
    record["turns"][-1] = {"generation_truncation":True}
    assert audit_trajectory(record,str(db))[0].label == "unknown_incomplete"
'''
        result = subprocess.run(
            [sys.executable, "-c", program], cwd=root,
            env={**os.environ, "PYTHONPATH": str(root / "src")},
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
