from __future__ import annotations

from collections import Counter

from rl.scenarios.diagnostics.audit_dev_incorrect_train_consistency import (
    _audit_signature,
    _broad_family,
    _consistency,
    _distance_band,
    _error_message_family,
    _guard_empty_terminal_overlap,
    _semantic_signature,
)


def test_semantic_signature_preserves_tail_combination() -> None:
    assert _semantic_signature("one_tail_near_miss", ["output"]) == (
        "semantic:one_tail_near_miss:output"
    )
    assert _semantic_signature("two_tail_near_miss", ["value", "output"]) == (
        "semantic:two_tail_near_miss:value+output"
    )
    assert _semantic_signature("wrong_source_or_join_route", []) == (
        "semantic:wrong_source_or_join_route"
    )


def test_nonlegal_signature_uses_terminal_failure() -> None:
    assert _audit_signature(
        failure_type="argument_validation_error",
        legal=False,
        cohort="no_legal_terminal",
        tail_mismatches=[],
    ) == "terminal:argument_validation_error"
    assert _broad_family(
        failure_type="argument_validation_error",
        legal=False,
        cohort="no_legal_terminal",
    ) == "argument_schema"


def test_consistency_distinguishes_exact_mechanism_and_unseen() -> None:
    train = {
        "signatures": Counter({"semantic:one_tail_near_miss:output": 12}),
        "families": Counter({"tail_semantics": 110, "argument_schema": 0}),
        "event_episodes": Counter({"argument_validation_error": 18}),
    }
    exact = _consistency(
        signature="semantic:one_tail_near_miss:output",
        family="tail_semantics",
        failure_type="wrong_answer",
        error_events=[],
        train=train,
    )
    mechanism = _consistency(
        signature="terminal:argument_validation_error",
        family="argument_schema",
        failure_type="argument_validation_error",
        error_events=[{"error_type": "argument_validation_error"}],
        train=train,
    )
    unseen = _consistency(
        signature="terminal:context_overflow",
        family="context_capacity",
        failure_type="context_overflow",
        error_events=[],
        train=train,
    )
    assert exact["label"] == "exact_signature_seen"
    assert mechanism["label"] == "contributing_event_seen_only"
    assert unseen["label"] == "not_observed_in_train"


def test_distance_band_is_explicit_not_human_severity() -> None:
    assert _distance_band(legal=False, cohort="no_legal_terminal") == "no_legal_answer"
    assert _distance_band(legal=True, cohort="one_tail_near_miss") == "local_one_family"
    assert _distance_band(legal=True, cohort="wrong_source_or_join_route") == "route_level"


def test_error_messages_receive_deterministic_families() -> None:
    assert _error_message_family(
        "ProtocolError: read_subtable: unexpected arguments ['offset']"
    ) == "unsupported_read_pagination"
    assert _error_message_family(
        "OperationalError: no such column: foo"
    ) == "unknown_column"
    assert _error_message_family(
        "ProtocolError: expected exactly one <think>...</think> block"
    ) == "carrier_shape"


def test_empty_zero_terminal_overlap_is_not_semantic_exact() -> None:
    semantic = {
        "semantic_eligible": True,
        "scoring_mode": "terminal",
        "semantic_score": 0.0,
        "answer_score": None,
        "source_score": 1.0,
        "join_score": 1.0,
        "tail_mismatches": [],
        "tail_mismatch_count": 0,
    }
    guarded = _guard_empty_terminal_overlap(
        semantic,
        {"terminal_overlap": {"score": 0.0, "per_category": {}}},
    )
    assert guarded["scoring_mode"] == "unresolved_terminal_artifact"
    assert guarded["source_score"] is None
    assert guarded["tail_mismatch_count"] is None
