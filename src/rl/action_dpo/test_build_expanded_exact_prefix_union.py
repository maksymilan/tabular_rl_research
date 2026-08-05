from build_expanded_exact_prefix_union import forbidden_visible_field, state_sha256


def test_state_hash_is_order_stable() -> None:
    assert state_sha256({"b": 2, "a": 1}) == state_sha256({"a": 1, "b": 2})


def test_forbidden_visible_field_only_checks_structured_keys() -> None:
    assert forbidden_visible_field([{"content": {"gold_sql": "select 1"}}]) == "gold_sql"
    assert forbidden_visible_field([{"content": "gold_sql is not included"}]) is None
