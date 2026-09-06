from __future__ import annotations

from rl.scenarios.diagnostics.score_action_candidates import json_token_indices


def test_json_token_indices_excludes_shared_think_tokens() -> None:
    response = '<think>shared</think>\n{"tool":"plan","arguments":{}}'
    offsets = []
    cursor = 0
    for token in ("<think>", "shared", "</think>", "\n", "{", '"tool"', ":", '"plan"', "}"):
        offsets.append((cursor, cursor + len(token)))
        cursor += len(token)

    selected = json_token_indices(response, offsets)

    assert selected == [4, 5, 6, 7, 8]
