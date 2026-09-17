"""Runtime transport repair: conservative, auditable, and semantics-preserving."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SFT_ROOT = Path(__file__).resolve().parents[2] / "sft"
if str(SFT_ROOT) not in sys.path:
    sys.path.insert(0, str(SFT_ROOT))

import protocol as v26_protocol  # noqa: E402
from protocol import ProtocolError  # noqa: E402


ACTION = '{"tool": "describe_table", "arguments": {"tables": ["Visitors"]}}'
CANONICAL = f"<think>look up the visitors</think>\n{ACTION}"


def test_strict_parse_still_rejects_repairable_shapes() -> None:
    for text in (
        f"<think>look up the visitors\n{ACTION}",  # unclosed think
        f"<think>look up the visitors</think>\n```json\n{ACTION}\n```",  # code fence
        f"<think>look up the visitors</think>\n{ACTION}\nDone.",  # trailing prose
        f"Sure.\n<think>look up the visitors</think>\n{ACTION}",  # leading prose
    ):
        with pytest.raises(ProtocolError):
            v26_protocol.parse_assistant_strict(text)


def test_runtime_repair_recovers_the_same_action() -> None:
    cases = {
        "unclosed_think": f"<think>look up the visitors\n{ACTION}",
        "code_fence": f"<think>look up the visitors</think>\n```json\n{ACTION}\n```",
        "extra_text_outside_action": f"<think>look up the visitors</think>\n{ACTION}\nDone.",
    }
    for kind, text in cases.items():
        think, tool, arguments, repair = v26_protocol.parse_assistant_strict_with_repair(text)
        assert repair is not None and repair["kind"] == kind, (kind, repair)
        assert think == "look up the visitors"
        assert tool == "describe_table"
        assert arguments == {"tables": ["Visitors"]}


def test_canonical_carrier_needs_no_repair() -> None:
    think, tool, arguments, repair = v26_protocol.parse_assistant_strict_with_repair(CANONICAL)
    assert repair is None
    assert (think, tool, arguments) == (
        "look up the visitors",
        "describe_table",
        {"tables": ["Visitors"]},
    )


def test_repair_can_be_disabled_for_a_strict_control() -> None:
    with pytest.raises(ProtocolError):
        v26_protocol.parse_assistant_strict_with_repair(
            f"<think>look up the visitors</think>\n{ACTION}\nDone.",
            allow_repair=False,
        )


def test_retired_and_ambiguous_carriers_stay_hard_errors() -> None:
    for text in (
        f"<think>look up the visitors</think>\n<tool_call>{ACTION}</tool_call>",
        f"<think></think>\n{ACTION}",  # empty think
        f"<think>a</think><think>b</think>\n{ACTION}",  # duplicate think
        "<think>no action object here</think>\njust prose",
        f"<think>look up the visitors</think>\n{ACTION[:-1]}",  # malformed JSON
    ):
        with pytest.raises(ProtocolError):
            v26_protocol.parse_assistant_strict_with_repair(text)
        assert v26_protocol.repair_action_carrier(text) is None


def test_repair_does_not_change_tool_validation() -> None:
    bad = '{"tool": "drop_table", "arguments": {}}'
    with pytest.raises(ProtocolError):
        v26_protocol.parse_assistant_strict_with_repair(
            f"<think>try something</think>\n{bad}\nDone."
        )


FROZEN_RUNTIME = (
    Path(__file__).resolve().parents[3]
    / "tmp/version26-runtime-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de"
)


def test_pinned_runtime_stays_strict_and_refuses_unsupported_repair() -> None:
    """The repair hook must not become a hard dependency of the pinned v26 runtime."""

    if not FROZEN_RUNTIME.is_dir():
        pytest.skip("explicit exported v26 runtime is not installed")
    root = Path(__file__).resolve().parents[3]
    program = r'''
import json, sqlite3, sys, tempfile
from pathlib import Path
runtime = Path("tmp/version26-runtime-4cd47c957fc6ae791e76a10594c8cd22f4d3b6de/src")
sys.path[:0] = [str(runtime / name) for name in ("eval", "sft", "harness")]
from rl.runtime import tool_environment_v26 as environment

assert environment.carrier_repair_supported() is False
assert environment.carrier_repair_enabled() is False
try:
    environment.set_carrier_repair_enabled(True)
except RuntimeError as exc:
    assert "parse_assistant_strict_with_repair" in str(exc), exc
else:
    raise AssertionError("opt-in repair must fail closed without protocol support")
assert environment.carrier_repair_enabled() is False, "a refused opt-in must not stick"

action = json.dumps({"tool": "describe_table", "arguments": {"tables": ["items"]}})
with tempfile.TemporaryDirectory() as temp:
    db = Path(temp) / "items.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE items(id INTEGER, name TEXT)")
    con.executemany("INSERT INTO items VALUES (?, ?)", [(1, "a")])
    con.commit(); con.close()
    example = (
        {"db_id": "items", "question": "name for id 1?", "db_path": str(db),
         "gold_sql": "SELECT name FROM items WHERE id=1"}
    )
    env = environment.ToolUseEnv(example, max_steps=4)
    try:
        step = env.apply_model_output("<think>Inspect schema.</think>" + action)
        assert env.errors == 0, step.observation
        assert env.turns[-1]["parsed"]["tool"] == "describe_table"
        assert "carrier_repair" not in env.turns[-1]
    finally:
        env.close()

    # When the active protocol does implement the repair parser, the hook is used and the
    # repair record is kept on the turn.
    import protocol
    def supporting_parser(text, *, allow_repair=True, **kwargs):
        think, tool, arguments = protocol.parse_assistant_strict(text, **kwargs)
        return think, tool, arguments, {"kind": "stub", "original_error_code": None}
    protocol.parse_assistant_strict_with_repair = supporting_parser
    environment.set_carrier_repair_enabled(True)
    assert environment.carrier_repair_supported() is True
    try:
        env = environment.ToolUseEnv(example, max_steps=4)
        try:
            env.apply_model_output("<think>Inspect schema.</think>" + action)
            assert env.errors == 0
            assert env.turns[-1]["carrier_repair"] == {"kind": "stub", "original_error_code": None}
        finally:
            env.close()
    finally:
        environment.set_carrier_repair_enabled(False)
    assert environment.carrier_repair_enabled() is False
'''
    result = subprocess.run(
        [sys.executable, "-c", program], cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
