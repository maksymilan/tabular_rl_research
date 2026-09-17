"""Runtime transport repair: conservative, auditable, and semantics-preserving."""
from __future__ import annotations

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
