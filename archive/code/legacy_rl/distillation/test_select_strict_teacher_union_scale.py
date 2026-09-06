from src.rl.distillation.select_strict_teacher_union_scale import (
    CATEGORIES,
    LEVELS,
    allocate_cells,
)


def test_exact_ten_per_cell_is_preferred_when_available():
    availability = {(category, level): 12 for category in CATEGORIES for level in LEVELS}
    allocation = allocate_cells(availability, per_category=30, per_difficulty=40)
    assert set(allocation.values()) == {10}


def test_feasible_uneven_cells_preserve_exact_marginals():
    availability = {(category, level): 30 for category in CATEGORIES for level in LEVELS}
    availability[("branch_only", "simple")] = 8
    allocation = allocate_cells(availability, per_category=30, per_difficulty=40)
    assert allocation[("branch_only", "simple")] == 8
    assert all(
        sum(allocation[(category, level)] for level in LEVELS) == 30
        for category in CATEGORIES
    )
    assert all(
        sum(allocation[(category, level)] for category in CATEGORIES) == 40
        for level in LEVELS
    )
