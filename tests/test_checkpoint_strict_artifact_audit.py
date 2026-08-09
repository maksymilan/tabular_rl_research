from tool_modules.checkpoint_relalg.strict_artifact_audit import (
    compare_strict_artifacts,
    has_top_level_order_by,
)


def test_strict_audit_preserves_bag_multiplicity_null_and_empty_string():
    passed = compare_strict_artifacts(
        ["x"], [[None], [""], ["a"], ["a"]],
        ["x"], [["a"], [None], ["a"], [""]],
        ordered=False,
    )
    assert passed["strict_artifact_accuracy"]
    missing_duplicate = compare_strict_artifacts(
        ["x"], [[None], [""], ["a"]],
        ["x"], [[None], [""], ["a"], ["a"]],
        ordered=False,
    )
    assert not missing_duplicate["strict_artifact_accuracy"]
    confused_null = compare_strict_artifacts(
        ["x"], [[""]], ["x"], [[None]], ordered=False
    )
    assert not confused_null["values_match"]


def test_strict_audit_schema_order_sequence_and_numeric_equivalence():
    numeric = compare_strict_artifacts(
        ["a", "b"], [[1, 1.0], [2, 2.0000000001]],
        ["a", "b"], [[1.0, 1], [2.0, 2.0]],
        ordered=True,
    )
    assert numeric["strict_artifact_accuracy"]
    wrong_order = compare_strict_artifacts(
        ["a"], [[2], [1]], ["a"], [[1], [2]], ordered=True
    )
    assert not wrong_order["values_match"]
    wrong_schema = compare_strict_artifacts(
        ["b", "a"], [[1, 2]], ["a", "b"], [[1, 2]], ordered=False
    )
    assert not wrong_schema["schema_match"]
    assert not wrong_schema["strict_artifact_accuracy"]


def test_top_level_order_detection_ignores_nested_and_quoted_occurrences():
    assert has_top_level_order_by("SELECT x FROM t ORDER BY x")
    assert not has_top_level_order_by(
        "SELECT x FROM (SELECT x FROM t ORDER BY x) AS q"
    )
    assert not has_top_level_order_by("SELECT 'ORDER BY' AS x")
