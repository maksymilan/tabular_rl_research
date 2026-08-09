from __future__ import annotations

import unittest

from audit_verified_rollouts import episode_issues


def _native_step(number: int, turn: int, tool: str) -> dict:
    call_id = f"call_{number}"
    return {
        "step_id": f"step_{number}",
        "model_turn_index": turn,
        "native_tool_call_id": call_id,
        "think": "",
        "tool_call": {"tool": tool, "arguments": {}},
        "tool_output": {},
        "environment_state_before": {},
        "environment_state": {},
    }


def _native_history(turn: int, call_id: str) -> dict:
    return {
        "model_turn_index": turn,
        "assistant": {"tool_calls": [{"id": call_id}]},
        "tool_messages": [{"tool_call_id": call_id, "content": "{}"}],
    }


class AuditVerifiedNativeBundleTests(unittest.TestCase):
    def _episode(self) -> dict:
        return {
            "trajectory_id": "native-count-contract",
            "label_status": "verified",
            "tool_scheme": "native-tool-bundle",
            "source": {},
            "steps": [
                _native_step(1, 1, "describe_table"),
                _native_step(3, 4, "answer_from_context"),
            ],
            "provider_native_history": [
                _native_history(1, "call_1"),
                _native_history(4, "call_3"),
            ],
            "rollout_generation": {
                "context_mode": "rolling-legal-history",
                "history_turns": 4,
                "error_actions_are_sft_targets": False,
                "action_count": 4,
                "primitive_action_count": 3,
                "error_events": [
                    {
                        "model_turn_index": 2,
                        "action_index": None,
                        "error_type": "protocol_error",
                        "error_code": "missing_tool_calls",
                    },
                    {
                        "model_turn_index": 3,
                        "action_index": None,
                        "error_type": "protocol_error",
                        "error_code": "missing_tool_calls",
                    },
                    {
                        "model_turn_index": 3,
                        "action_index": 2,
                        "native_tool_call_id": "call_error",
                        "error_type": "execution_error",
                    },
                ],
            },
        }

    def test_provider_turn_errors_do_not_count_as_primitives(self) -> None:
        self.assertEqual(episode_issues(self._episode()), [])

    def test_primitive_count_still_detects_real_mismatch(self) -> None:
        episode = self._episode()
        episode["rollout_generation"]["primitive_action_count"] = 4
        issues = episode_issues(episode)
        self.assertTrue(any("primitive count 4" in issue for issue in issues))
        self.assertTrue(any("step-id gaps" in issue for issue in issues))

    def test_selected_empty_result_policy_requires_context_only_annotation(self) -> None:
        episode = self._episode()
        episode["rollout_generation"]["empty_result_policy"] = (
            "causal-empty-result-target-filter-v1"
        )
        episode["steps"][0]["tool_output"] = {
            "table": "filter_001",
            "row_count": 0,
        }
        issues = episode_issues(episode)
        self.assertTrue(any("empty result remains SFT-target eligible" in issue for issue in issues))
        episode["steps"][0]["sft_target_eligible"] = False
        self.assertEqual(episode_issues(episode), [])


if __name__ == "__main__":
    unittest.main()
