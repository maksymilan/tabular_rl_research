"""CPU contract checks for the isolated legal-reason hybrid diagnostic."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from rl.configuration.experiment_config import RLExperimentConfig
from rl.frameworks.trl.mechanism import RLMechanism
from rl.frameworks.trl.transition_batch import PolicyEpisode, PolicyTurn, TransitionUpdate
from rl.frameworks.trl.transition_grpo import TransitionGRPOTrainer


ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "src/rl/configs/experiments/qwen3_8b_atomic_v26_binary_saam_legal_reason_hybrid_balanced60_newgnn.yaml"


class PieceTokenizer:
    def decode(self, token_ids, **kwargs):
        pieces = {1: "<think>", 2: "reason", 3: "</think>", 4: "\n", 5: '{"tool":', 6: '"plan"}'}
        return "".join(pieces[int(token_id)] for token_id in token_ids)


def _trainer(routing="legal_reason_only_hybrid", alpha=0.5):
    trainer = object.__new__(TransitionGRPOTrainer)
    trainer.span_routing = routing
    trainer.span_balance_alpha = alpha
    trainer.processing_class = PieceTokenizer()
    return trainer


def _update(correct, error):
    return TransitionUpdate(
        prompt_ids=(1,), response_ids=(1, 2, 3, 4, 5, 6),
        sampling_logprobs=(-0.1,) * 6, advantage=1.0 if correct else -1.0,
        trajectory_id="t", turn_index=0, example_index=0,
        trajectory_correct=correct, turn_has_harness_error=error,
    )


@pytest.mark.parametrize("correct", [False, True])
def test_legal_turn_zeroes_tool_direct_loss_regardless_of_result(correct):
    assert _trainer()._loss_mask_for_update(_update(correct, False)) == (1, 1, 1, 1, 0, 0)


@pytest.mark.parametrize("correct", [False, True])
def test_error_turn_gives_each_span_half_mass_regardless_of_result(correct):
    weights = _trainer()._loss_mask_for_update(_update(correct, True))
    assert sum(weights[:4]) == pytest.approx(0.5)
    assert sum(weights[4:]) == pytest.approx(0.5)
    assert all(w > 0 for w in weights)


def test_missing_outcome_is_not_silently_legal():
    with pytest.raises(ValueError, match="per-turn Harness outcome"):
        _trainer()._loss_mask_for_update(_update(False, None))


def test_unparseable_boundary_retains_registered_full_response_fallback():
    update = replace(_update(False, True), response_ids=(1, 2), sampling_logprobs=(-0.1, -0.1))
    assert _trainer()._loss_mask_for_update(update) == (1.0, 1.0)


def test_uniform_none_is_not_alpha_zero():
    update = _update(False, None)
    assert _trainer("uniform", None)._loss_mask_for_update(update) == (1.0,) * 6
    assert _trainer("uniform", 0)._loss_mask_for_update(update) == (0.25,) * 4 + (0.0,) * 2


def _episode(correct=True):
    turns = [
        {"turn_index": 0, "execution_error_type": "timeout_error", "error_event": {"error_type": "timeout_error"}},
        {"turn_index": 1, "feedback_recovery": True, "recovered_from_error_type": "timeout_error",
         "parsed": {"tool": "describe_table", "arguments": {"tables": ["t"]}}},
    ]
    policy = [PolicyTurn((1,), (1, 2, 3, 4, 5, 6), (-0.1,) * 6) for _ in turns]
    sample = SimpleNamespace(
        reward=1.0 if correct else -1.0, correct=correct, process_update=True,
        step_rewards=None, turns=[(list(t.prompt_ids), list(t.response_ids)) for t in policy],
        audit_record={"trajectory_id": str(correct), "example_index": 0, "turns": turns,
                      "result_reward": {"profile": "signed-binary", "policy_failure_penalty_enabled": True}},
    )
    return PolicyEpisode(sample, policy)


def _mechanism():
    return RLMechanism(span_routing="legal_reason_only_hybrid", span_balance_alpha=0.5,
                       credit_assignment="saam-asymmetric-error")


def test_successful_recovery_is_not_a_current_error_and_timeout_credit_is_negative():
    mechanism = _mechanism()
    episodes = [_episode(True), _episode(False)]
    updates = mechanism.build_updates(episodes, train_turns="all")
    assert [u.turn_has_harness_error for u in updates] == [True, False, True, False]
    credited, audit = mechanism.apply_credit(episodes, updates)
    assert credited[0].advantage <= -1.0
    assert credited[2].advantage <= -1.0
    assert credited[1].advantage > 0
    assert audit.timeout_penalized_transitions == 2
    assert _trainer()._loss_mask_for_update(credited[1])[-2:] == (0, 0)


@pytest.mark.parametrize("audit", [None, [], [{}], [None, {}], [{"turn_index": 1}, {"turn_index": 0}]])
def test_hybrid_rejects_missing_or_misaligned_audit(audit):
    episode = _episode()
    episode.sample.audit_record["turns"] = audit
    with pytest.raises(ValueError, match="aligned per-turn Harness audit"):
        _mechanism().build_updates([episode], train_turns="all")


def test_all_wrong_group_never_produces_positive_result_advantage():
    episodes = [_episode(False) for _ in range(8)]
    updates = _mechanism().build_updates(episodes, train_turns="all")
    assert all(u.advantage == 0.0 for u in updates)


def test_legacy_constructor_arguments_preserve_alpha():
    mechanism = RLMechanism.from_legacy_args(span_routing="legal_reason_only_hybrid", span_balance_alpha=0.5)
    assert mechanism.span_balance_alpha == 0.5


def test_missing_alpha_rejected_before_gpu():
    with pytest.raises(ValueError, match="requires span_balance_alpha"):
        RLMechanism(span_routing="legal_reason_only_hybrid")


def test_registered_config_maps_to_exact_diagnostic():
    defaults = RLExperimentConfig.load(CONFIG).argparse_defaults(ROOT)
    for key, value in {"result_reward_profile": "signed-binary", "span_routing": "legal_reason_only_hybrid",
                       "span_balance_alpha": 0.5, "optimizer_steps": 4, "group_size": 8,
                       "prompts_per_update": 30, "max_new_tokens": 2048, "kl_beta": 0.0}.items():
        assert defaults[key] == value


@pytest.mark.parametrize("mutation", ["missing_alpha", "conflict", "tool_only"])
def test_invalid_hybrid_config_rejected(tmp_path, mutation):
    payload = yaml.safe_load(CONFIG.read_text())
    if mutation == "missing_alpha":
        del payload["mechanism"]["span_balance_alpha"]
    elif mutation == "conflict":
        payload["span_routing"] = "uniform"
    else:
        payload["trainable_part"] = "tool_only"
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match="hybrid|span_routing"):
        RLExperimentConfig.load(path)
