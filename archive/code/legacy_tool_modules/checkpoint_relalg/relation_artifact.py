"""Immutable relation metadata shared by every checkpoint-relalg execution mode.

This module deliberately contains no SQL/backend behavior.  The objects here are
the canonical, replayable description of source relations and derived artifacts.
"""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field
import math
from typing import Any, Mapping


CANONICAL_TYPES = frozenset(
    {"NULL", "BOOLEAN", "INTEGER", "REAL", "TEXT", "DATE", "DATETIME", "BLOB"}
)
_ASCII_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ASCII_LOWER = "abcdefghijklmnopqrstuvwxyz"
_SQLITE_NOCASE_TRANSLATION = str.maketrans(_ASCII_UPPER, _ASCII_LOWER)


def sqlite_identifier_key(value: str) -> str:
    """Match SQLite's built-in ASCII identifier case equivalence."""

    return value.translate(_SQLITE_NOCASE_TRANSLATION)


class _FrozenMapping(Mapping[str, Any]):
    """Small recursively immutable mapping used inside frozen records."""

    __slots__ = ("__data",)

    def __init__(self, value: Mapping[Any, Any]) -> None:
        self.__data = {str(key): _freeze_value(item) for key, item in value.items()}

    def __getitem__(self, key: str) -> Any:
        return self.__data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __repr__(self) -> str:
        return repr(self.__data)


class _FrozenList(tuple[Any, ...]):
    """Tuple storage with JSON-list equality for compatibility."""

    def __new__(cls, value: Any) -> "_FrozenList":
        return super().__new__(cls, (_freeze_value(item) for item in value))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple.__eq__(self, tuple(other))
        return False

    __hash__ = tuple.__hash__


def _freeze_value(value: Any) -> Any:
    if isinstance(value, _FrozenMapping):
        return value
    if isinstance(value, Mapping):
        return _FrozenMapping(value)
    if isinstance(value, (list, tuple)):
        return _FrozenList(value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite REAL values cannot enter canonical EnvironmentState")
    return deepcopy(value)


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    if isinstance(value, frozenset):
        return [_thaw_value(item) for item in sorted(value, key=repr)]
    return deepcopy(value)


def _nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _row_count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("row_count must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class Column:
    name: str
    canonical_type: str

    def __post_init__(self) -> None:
        _nonempty(self.name, "column name")
        if not isinstance(self.canonical_type, str):
            raise ValueError("canonical_type must be a string")
        canonical_type = self.canonical_type.upper()
        if canonical_type not in CANONICAL_TYPES:
            raise ValueError(f"unsupported canonical_type {self.canonical_type!r}")
        object.__setattr__(self, "canonical_type", canonical_type)

    @property
    def type(self) -> str:
        """Model-facing compatibility alias."""

        return self.canonical_type

    def to_payload(self) -> dict[str, str]:
        return {"name": self.name, "canonical_type": self.canonical_type}


@dataclass(frozen=True, slots=True)
class OrderingKey:
    column: str
    direction: str = "asc"
    nulls: str = "last"

    def __post_init__(self) -> None:
        _nonempty(self.column, "ordering column")
        direction = self.direction.lower() if isinstance(self.direction, str) else ""
        nulls = self.nulls.lower() if isinstance(self.nulls, str) else ""
        if direction not in {"asc", "desc"}:
            raise ValueError("ordering direction must be 'asc' or 'desc'")
        if nulls not in {"first", "last"}:
            raise ValueError("ordering nulls must be 'first' or 'last'")
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "nulls", nulls)

    def to_payload(self) -> dict[str, str]:
        return {"column": self.column, "direction": self.direction, "nulls": self.nulls}


@dataclass(frozen=True, slots=True)
class ForeignKey:
    columns: tuple[str, ...]
    ref_table: str
    ref_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        columns = tuple(self.columns)
        ref_columns = tuple(self.ref_columns)
        if not columns or not ref_columns or len(columns) != len(ref_columns):
            raise ValueError("foreign-key columns must be non-empty and have equal arity")
        for column in columns:
            _nonempty(column, "foreign-key column")
        _nonempty(self.ref_table, "referenced table")
        for column in ref_columns:
            _nonempty(column, "referenced column")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "ref_columns", ref_columns)

    def to_payload(self) -> dict[str, Any]:
        return {
            "columns": list(self.columns),
            "ref_table": self.ref_table,
            "ref_columns": list(self.ref_columns),
        }


def _validate_columns(columns: tuple[Column, ...]) -> set[str]:
    if not all(isinstance(column, Column) for column in columns):
        raise ValueError("columns must contain only Column objects")
    names = [column.name for column in columns]
    if len(names) != len({sqlite_identifier_key(name) for name in names}):
        raise ValueError("relation column names must be SQLite-identifier unique")
    return set(names)


@dataclass(frozen=True, slots=True)
class SourceRelation:
    name: str
    columns: tuple[Column, ...]
    row_count: int
    primary_key: tuple[str, ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()

    def __post_init__(self) -> None:
        _nonempty(self.name, "source relation name")
        columns = tuple(self.columns)
        primary_key = tuple(self.primary_key)
        foreign_keys = tuple(self.foreign_keys)
        names = _validate_columns(columns)
        _row_count(self.row_count)
        if len(primary_key) != len(set(primary_key)) or any(key not in names for key in primary_key):
            raise ValueError("primary_key must contain unique columns from this relation")
        if not all(isinstance(key, ForeignKey) for key in foreign_keys):
            raise ValueError("foreign_keys must contain only ForeignKey objects")
        for key in foreign_keys:
            if any(column not in names for column in key.columns):
                raise ValueError("foreign-key columns must exist in this relation")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "primary_key", primary_key)
        object.__setattr__(self, "foreign_keys", foreign_keys)

    @property
    def table(self) -> str:
        return self.name

    @property
    def kind(self) -> str:
        return "source"

    @property
    def ordered_by(self) -> tuple[OrderingKey, ...]:
        return ()

    @property
    def derivation(self) -> dict[str, Any]:
        return {}

    @property
    def scalar_cell(self) -> None:
        return None

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "columns": [column.to_payload() for column in self.columns],
            "row_count": self.row_count,
            "primary_key": list(self.primary_key),
            "foreign_keys": [key.to_payload() for key in self.foreign_keys],
        }


@dataclass(frozen=True, slots=True)
class RelationArtifact:
    table: str
    kind: str
    columns: tuple[Column, ...]
    row_count: int
    ordered_by: tuple[OrderingKey, ...] = ()
    derivation: Mapping[str, Any] = field(default_factory=dict)
    scalar_cell: Any = None

    def __post_init__(self) -> None:
        _nonempty(self.table, "artifact table")
        _nonempty(self.kind, "artifact kind")
        columns = tuple(self.columns)
        ordered_by = tuple(self.ordered_by)
        _validate_columns(columns)
        _row_count(self.row_count)
        if not all(isinstance(key, OrderingKey) for key in ordered_by):
            raise ValueError("ordered_by must contain only OrderingKey objects")
        # Ordering metadata may name a semantic key that a later ``project``
        # hides.  The backend preserves the exact relative order with a hidden
        # ordinal; §12 explicitly keeps that ordinal out of visible columns and
        # the final answer table.
        if not isinstance(self.derivation, Mapping):
            raise ValueError("derivation must be a mapping")
        if self.scalar_cell is not None and (self.row_count != 1 or len(columns) != 1):
            raise ValueError("scalar_cell is only valid for a 1x1 relation")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "ordered_by", ordered_by)
        # Isolate immutable records from caller-owned mutable payloads.  The
        # dataclass itself is frozen; payload serialization never returns this
        # private copy directly.
        object.__setattr__(self, "derivation", _freeze_value(self.derivation))
        object.__setattr__(self, "scalar_cell", _freeze_value(self.scalar_cell))

    def to_payload(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "kind": self.kind,
            "columns": [column.to_payload() for column in self.columns],
            "row_count": self.row_count,
            "ordered_by": [key.to_payload() for key in self.ordered_by],
            "derivation": _thaw_value(self.derivation),
            "scalar_cell": _thaw_value(self.scalar_cell),
        }


Relation = SourceRelation | RelationArtifact


__all__ = [
    "CANONICAL_TYPES",
    "Column",
    "ForeignKey",
    "OrderingKey",
    "Relation",
    "RelationArtifact",
    "SourceRelation",
    "sqlite_identifier_key",
]
