"""Public interface for fact-only, table-bound relation derivation metadata."""
from .builders import build_relation_derivation
from .lineage import expression_source_columns
from .schema import (
    DERIVATION_SCHEMA,
    SUPPORTED_TABLE_OPERATORS,
    RelationInspector,
    validate_relation_derivation,
)

__all__ = [
    "DERIVATION_SCHEMA",
    "SUPPORTED_TABLE_OPERATORS",
    "RelationInspector",
    "build_relation_derivation",
    "expression_source_columns",
    "validate_relation_derivation",
]
