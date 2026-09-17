from __future__ import annotations

import json
import io
import os
import sqlite3
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "src/eval"),
               str(ROOT / "src/harness"), str(ROOT / "src/sft")]

from rl.runtime.error_feedback import (  # noqa: E402
    FEEDBACK_VERSION, error_feedback_payload, merge_error_feedback,
)
from rl.evaluation.runners.feedback_overlay import install_feedback_overlay  # noqa: E402


def test_reference_feedback_is_factual_and_does_not_propose_invalid_fix():
    state = {"tables": {"filter_002": {
        "row_count": 2, "created_by": "step_4", "columns": ["Name", "Length"],
        "rows": [["SECRET_ROW", 3]], "gold_sql": "SECRET_GOLD",
    }}}
    history = {"step_5": {"tool": "read_subtable", "arguments": {"table": "filter_002"},
                           "output": {"result_sample": [["SECRET_ROW", 3]]}}}
    exc = ValueError("not a table-producing step")
    exc.details = {"gold_sql": "SECRET_GOLD", "future_answer": "SECRET_FUTURE"}
    parsed = {"tool": "scalar_compute", "arguments": {"operation": "subtract", "operands": [
        {"value_ref": "step_5", "column": "Length"}, {"value_ref": "step_5", "column": "Length"},
    ]}}
    before = json.dumps((state, history, parsed), sort_keys=True)
    payload = error_feedback_payload(exc, parsed=parsed, state=state, history=history)
    source = payload["error"]["details"]["operand_context"][0]["source"]
    assert source["tool"] == "read_subtable"
    assert source["created_by"] == "step_4" and source["row_count"] == 2
    assert "insufficient" in payload["error"]["details"]["hint"]
    assert "SECRET" not in json.dumps(payload)
    assert json.dumps((state, history, parsed), sort_keys=True) == before


def test_numeric_container_and_date_feedback_do_not_coerce():
    for operation, expected in (("add", "finite numeric"), ("date_diff_days", "ISO date")):
        parsed = {"tool": "scalar_compute", "arguments": {"operation": operation,
                  "operands": [{"value": {"value": 0}}, {"value": 2}]}}
        payload = error_feedback_payload(ValueError("bad operand"), parsed=parsed, state={}, history={})
        assert payload["attempted_action"] == parsed
        fact = payload["error"]["details"]["operand_context"][0]
        assert fact["argument_path"] == "scalar_compute.operands[0].value"
        assert fact["received_type"] == "dict" and expected in fact["expected"]


def test_join_does_not_strip_qualifiers_or_recommend_a_join_key():
    parsed = {"tool": "join_tables", "arguments": {"base": "orders", "joins": [
        {"table": "customers", "on": [{"left": "orders.customer_id", "right": "customers.id"}]},
    ]}}
    payload = error_feedback_payload(ValueError("right must be bare"), parsed=parsed, state={}, history={})
    assert payload["attempted_action"] == parsed
    assert payload["error"]["details"]["argument_constraints"][0]["received"] == "customers.id"
    assert "corrected_action" not in payload


def test_unparsed_error_never_fabricates_action_and_bookkeeping_preserved():
    payload = error_feedback_payload(ValueError("invalid JSON"), state={}, history={})
    assert "attempted_action" not in payload
    original = {"step_id": "step_3", "status": "error", "error": {"type": "protocol_error", "message": "old"}}
    merged = merge_error_feedback(original, payload)
    assert merged["step_id"] == "step_3" and merged["error"]["type"] == "protocol_error"
    assert original["error"]["message"] == "old"


def test_large_metadata_is_explicitly_bounded():
    exc = ValueError("bad column")
    exc.details = {"available_columns": [str(i) for i in range(100)]}
    payload = error_feedback_payload(exc, parsed={"tool": "project", "arguments": {"huge": "x" * 10000}}, state={}, history={})
    assert payload["attempted_action"]["arguments_omitted_due_to_size"]
    assert len(payload["error"]["details"]["available_columns"]) == 32
    assert payload["error"]["details"]["available_columns_truncated"]


def test_feedback_does_not_crash_on_malformed_operation_types():
    for operation in ([], {}, None, False):
        payload = error_feedback_payload(TypeError("invalid operation"),
            parsed={"tool": "scalar_compute", "arguments": {"operation": operation, "operands": []}},
            state={}, history={})
        assert payload["error"]["message"] == "TypeError: invalid operation"


def test_overlay_concurrent_samples_keep_their_own_rejected_action():
    def validator(tool, arguments):
        raise ValueError("rejected")
    protocol = SimpleNamespace(validate_model_arguments=validator)
    runner = SimpleNamespace(
        execute_tool=lambda *a, **kw: None,
        format_tool_error=lambda exc, *a: f"ValueError: {exc}",
        rolling_legal_history_messages=lambda s, o, q, state, error, *a, **kw: error,
        model_context_messages=lambda s, o, q, state, error, *a, **kw: error,
        ArtifactWriter=lambda root, manifest: manifest,
    )
    def sample(index):
        runner.rolling_legal_history_messages("", {}, "", {}, None)
        try:
            protocol.validate_model_arguments("read_subtable", {"table": f"table_{index}"})
        except ValueError as exc:
            message = runner.format_tool_error(exc, None, None, None)
        error = {"step_id": "step_1", "status": "error", "error": {"type": "execution_error", "message": message}}
        visible = runner.rolling_legal_history_messages("", {}, "", {}, error)
        return {"visible": visible, "error_events": [{"step_id": "step_1", "error_type": "execution_error", "message": message}]}
    runner.run_sample = sample
    install_feedback_overlay(runner, protocol)
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(runner.run_sample, range(40)))
    for index, record in enumerate(records):
        assert record["visible"]["attempted_action"]["arguments"]["table"] == f"table_{index}"
        assert record["error_feedback_version"] == FEEDBACK_VERSION
        assert record["error_events"][0]["message"] == "ValueError: rejected"
    assert runner.ArtifactWriter("unused", {})["error_feedback_version"] == FEEDBACK_VERSION


@pytest.fixture(scope="module")
def frozen_runtime(tmp_path_factory):
    runtime = tmp_path_factory.mktemp("frozen_feedback_runtime")
    exported = subprocess.run(["git", "archive", "4cd47c957fc6ae791e76a10594c8cd22f4d3b6de", "src"],
                              cwd=ROOT, check=True, capture_output=True)
    with tarfile.open(fileobj=io.BytesIO(exported.stdout)) as archive:
        archive.extractall(runtime, filter="data")
    return runtime


def _run_frozen(tmp_path, frozen_runtime, body):
    setup = '''
import json, sqlite3
from pathlib import Path
from rl.runtime.error_feedback import FEEDBACK_VERSION
from rl.runtime.tool_environment_v26 import ToolUseEnv
def environment(version):
    path = Path(version + '.sqlite')
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE items (id INTEGER, amount REAL)')
        connection.executemany('INSERT INTO items VALUES (?, ?)', [(1, 3), (2, 4)])
    return ToolUseEnv({'db_id':'test', 'db_path':str(path), 'question':'Sum amounts',
                      'gold_sql':'select sum(amount) from items'},
                     max_steps=30, error_feedback_version=version)
old = environment('legacy')
new = environment(FEEDBACK_VERSION)
'''
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"),
           *(str(frozen_runtime / "src" / part) for part in ("eval", "harness", "sft"))])}
    result = subprocess.run([sys.executable, "-c", setup + body], cwd=tmp_path,
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("action", [
    {"tool": "scalar_compute", "arguments": {"operation": "add", "operands": [{"value": {"value": 0}}, {"value": 2}]}},
    {"tool": "join_tables", "arguments": {"base": "items", "joins": [{"table": "items", "role": "other", "on": [{"left": "items.id", "right": "other.id"}]}]}},
])
def test_real_environment_rejects_same_actions_and_exposes_them_next_turn(tmp_path, frozen_runtime, action):
    _run_frozen(tmp_path, frozen_runtime, f"action = {action!r}\n" + '''
text = '<think>Test invalid action.</think>' + json.dumps(action)
old_step = old.apply_model_output(text)
new_step = new.apply_model_output(text)
assert old.errors == new.errors == 1
assert old_step.turn['execution_error'] == new_step.turn['execution_error']
assert old_step.turn['execution_error_type'] == new_step.turn['execution_error_type']
assert old.done == new.done
visible = json.loads(new_step.observation)
assert visible['attempted_action'] == action
assert 'attempted_action' in new.model_messages()[-1]['content']
assert 'Test invalid action.' not in new.model_messages()[-1]['content']
assert 'attempted_action' not in json.loads(old_step.observation)
old.close()
new.close()
''')


def test_successful_environment_inputs_outputs_are_unchanged(tmp_path, frozen_runtime):
    _run_frozen(tmp_path, frozen_runtime, '''
text = '<think>Inspect schema.</think>{"tool":"describe_table","arguments":{"tables":["items"]}}'
assert old.model_messages() == new.model_messages()
assert old.apply_model_output(text).observation == new.apply_model_output(text).observation
assert old.model_messages() == new.model_messages()
old.close()
new.close()
''')


def test_actual_frozen_evaluator_overlay_preserves_execution_and_records_feedback(tmp_path, frozen_runtime):
    _run_frozen(tmp_path, frozen_runtime, '''
import protocol
import rollout_passk as runner
from rl.evaluation.runners.feedback_overlay import install_feedback_overlay
ex = old.example
actions = [
    {'tool':'describe_table', 'arguments':{'tables':['items']}},
    {'tool':'scalar_compute', 'arguments':{'operation':'add', 'operands':[{'value':{'value':0}}, {'value':2}]}},
    {'tool':'scalar_compute', 'arguments':{'operation':'add', 'operands':[{'value':3}, {'value':4}]}},
]
def run():
    pending = iter(actions)
    def chat(*args, **kwargs):
        kwargs['retry_stats'].update(api_transport_retries=0, api_context_retries=0)
        return '<think>Test.</think>' + json.dumps(next(pending))
    runner.chat_sample = chat
    return runner.run_sample(ex, 0, 'unused', 'unused', old.system_prompt,
        max_steps=3, max_tokens=4096, temperature=0, top_p=1, api_retries=0,
        context_mode='rolling-legal-history', history_turns=4,
        compact_history_observations=True, denotation_comparison='bird-set')
baseline = run()
install_feedback_overlay(runner, protocol)
candidate = run()
for key in ['correct', 'legal', 'steps', 'errors', 'failure_type']:
    assert baseline[key] == candidate[key], key
for index in [0, 1]:
    assert baseline['turns'][index]['model_input'] == candidate['turns'][index]['model_input']
assert 'attempted_action' in candidate['turns'][2]['model_input'][-1]['content']
assert 'attempted_action' not in baseline['turns'][2]['model_input'][-1]['content']
assert baseline['turns'][2]['tool_output'] == candidate['turns'][2]['tool_output']
assert candidate['error_events'][0]['model_visible_feedback']['attempted_action'] == actions[1]
old.close()
new.close()
''')
