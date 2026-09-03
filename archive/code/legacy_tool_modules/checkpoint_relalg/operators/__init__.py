"""Public operator-module facades for checkpoint-relalg-v1."""

from . import add_rank, aggregate, distinct, filter_rows, join, limit, project, set_operation, sort

OPERATOR_MODULES = {
    module.NAME: module
    for module in (
        filter_rows,
        project,
        join,
        aggregate,
        distinct,
        set_operation,
        sort,
        limit,
        add_rank,
    )
}

__all__ = ["OPERATOR_MODULES", *OPERATOR_MODULES]
