"""SQLite compiler for the frozen checkpoint-relalg-v1 relational kernel.

Every atomic call follows the same transaction boundary: validate exact state
references, compile the closed typed IR, materialize a TEMP relation, construct
relation-derivation-v2, and only then publish the immutable artifact.  A failed
call rolls back its savepoint and verifies that EnvironmentState's logical hash
did not move.
"""

from __future__ import annotations

from datetime import date, datetime
import json
import math
import re
import sqlite3
import time
from typing import Any, Callable, Mapping, Sequence, TypeVar

from ..errors import CheckpointRelalgError
from ..expression import (
    CompiledExpression,
    ExpressionCompiler,
    NonrecoverableExecutionError,
    RelAlgExecutionError,
    RelAlgStateValidationError,
    RelAlgValidationError,
    column_type_map,
    merge_types,
    quote_identifier,
    require_simple_identifier,
)
from ..operator_registry import get_operator_spec
from ..predicate import PredicateCompiler
from ..relation_artifact import (
    Column,
    OrderingKey,
    RelationArtifact,
    SourceRelation,
    sqlite_identifier_key,
)
from ..environment_state import StateError


_HIDDEN_ORDINAL = "__relalg_ordinal"
_OPAQUE_DIRECT_ORDER = "<opaque-direct-order>"
_ARTIFACT_BACKING_PREFIX = "__checkpoint_relalg_data_"
_JOIN_TYPES = frozenset({"inner", "left", "semi", "anti", "cross"})
_JOIN_OPS = frozenset({"=", "!=", ">", ">=", "<", "<="})
_AGGREGATES = frozenset({"count", "sum", "avg", "min", "max"})
_SET_OPS = frozenset({"union", "union_all", "intersect", "except"})
_RANK_METHODS = frozenset({"row_number", "rank", "dense_rank"})
_READONLY_SQL_START = re.compile(r"^\s*(?:--[^\n]*\n\s*|/\*.*?\*/\s*)*(SELECT|WITH)\b", re.I | re.S)
_T = TypeVar("_T")

EXECUTOR_VERSION = "checkpoint-relalg-sqlite-executor-v1"
ARTIFACT_BYTE_ACCOUNTING_VERSION = "canonical-cell-storage-v1"


def canonical_cell_storage_bytes(value: Any) -> tuple[int, int]:
    """Return deterministic ``(payload_bytes, stored_bytes)`` for one cell.

    The accounting is intentionally independent of SQLite page layout: text is
    UTF-8, blobs use their raw length, numeric values use eight bytes, and each
    cell contributes one framing byte to the containing relation.  It is a
    reproducible Harness budget, not an estimate of an on-disk database file.
    """

    if value is None:
        return 0, 1
    if isinstance(value, bool):
        return 8, 9
    if isinstance(value, int):
        return 8, 9
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RelAlgExecutionError(
                "non_finite_result",
                "result contains a non-finite REAL value",
            )
        return 8, 9
    if isinstance(value, str):
        size = len(value.encode("utf-8"))
        return size, size + 1
    if isinstance(value, (bytes, bytearray, memoryview)):
        size = len(value)
        return size, size + 1
    if isinstance(value, (date, datetime)):
        size = len(value.isoformat().encode("utf-8"))
        return size, size + 1
    raise RelAlgExecutionError(
        "unsupported_result_value",
        "result contains a value outside the canonical storage domain",
        details={"python_type": type(value).__name__},
    )


class ResultSizeTracker:
    """Streaming row/cell/storage guard shared by execution and evaluation."""

    def __init__(
        self,
        *,
        max_rows: int,
        max_artifact_bytes: int,
        max_cell_bytes: int,
        operation: str,
    ) -> None:
        self.max_rows = _integer(max_rows, path="max_rows", minimum=1)
        self.max_artifact_bytes = _integer(
            max_artifact_bytes, path="max_artifact_bytes", minimum=1
        )
        self.max_cell_bytes = _integer(
            max_cell_bytes, path="max_cell_bytes", minimum=1
        )
        self.operation = operation
        self.row_count = 0
        self.artifact_bytes = 0

    def add_row(self, row: Sequence[Any]) -> None:
        self.row_count += 1
        if self.row_count > self.max_rows:
            raise CheckpointRelalgError(
                "resource_limit_error",
                "result_too_large",
                "result row count exceeds the configured limit",
                {
                    "operation": self.operation,
                    "limit_kind": "rows",
                    "max_rows": self.max_rows,
                    "observed_rows": self.row_count,
                },
            )
        for column_index, value in enumerate(row):
            payload_bytes, stored_bytes = canonical_cell_storage_bytes(value)
            if payload_bytes > self.max_cell_bytes:
                raise CheckpointRelalgError(
                    "resource_limit_error",
                    "result_too_large",
                    "result cell exceeds the configured byte limit",
                    {
                        "operation": self.operation,
                        "limit_kind": "cell_bytes",
                        "column_index": column_index,
                        "max_cell_bytes": self.max_cell_bytes,
                        "observed_cell_bytes": payload_bytes,
                        "byte_accounting": ARTIFACT_BYTE_ACCOUNTING_VERSION,
                    },
                )
            self.artifact_bytes += stored_bytes
            if self.artifact_bytes > self.max_artifact_bytes:
                raise CheckpointRelalgError(
                    "resource_limit_error",
                    "result_too_large",
                    "result storage exceeds the configured byte limit",
                    {
                        "operation": self.operation,
                        "limit_kind": "artifact_bytes",
                        "max_artifact_bytes": self.max_artifact_bytes,
                        "observed_artifact_bytes": self.artifact_bytes,
                        "byte_accounting": ARTIFACT_BYTE_ACCOUNTING_VERSION,
                    },
                )


def _positive_timeout(value: Any, *, path: str = "timeout_seconds") -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise RelAlgValidationError("invalid_arguments", "timeout_seconds must be positive", path=path)
    return float(value)


class _SQLiteDeadline:
    """One non-nesting SQLite progress-handler deadline."""

    def __init__(self, connection: sqlite3.Connection, timeout_seconds: float) -> None:
        self.connection = connection
        self.timeout_seconds = timeout_seconds
        self.timed_out = False
        self._deadline = 0.0

    def start(self) -> None:
        self.timed_out = False
        self._deadline = time.monotonic() + self.timeout_seconds
        self.connection.set_progress_handler(self._progress, 1_000)

    def stop(self) -> None:
        self.connection.set_progress_handler(None, 0)

    def _progress(self) -> int:
        self.timed_out = time.monotonic() >= self._deadline
        return 1 if self.timed_out else 0

    def check(self, *, operation: str) -> None:
        """Enforce the same wall deadline across non-SQL compiler work."""

        if time.monotonic() >= self._deadline:
            self.timed_out = True
            raise RelAlgExecutionError(
                "sql_timeout",
                f"{operation} exceeded its execution timeout",
            )


def _keys(
    value: Any,
    *,
    required: set[str],
    optional: set[str] = set(),
    path: str = "arguments",
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RelAlgValidationError("invalid_arguments", "arguments must be an object", path=path)
    present = set(value)
    missing = required - present
    extra = present - required - optional
    if missing:
        raise RelAlgValidationError(
            "invalid_arguments", f"missing fields: {', '.join(sorted(missing))}", path=path
        )
    if extra:
        raise RelAlgValidationError(
            "invalid_arguments", f"unknown fields: {', '.join(sorted(extra))}", path=path
        )
    return value


def _array(value: Any, *, path: str, minimum: int = 0) -> list[Any]:
    if not isinstance(value, list) or len(value) < minimum:
        raise RelAlgValidationError(
            "invalid_arguments", f"expected an array with at least {minimum} items", path=path
        )
    return value


def _integer(value: Any, *, path: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise RelAlgValidationError(
            "invalid_arguments", f"expected an integer >= {minimum}", path=path
        )
    return value


def _relation_name(relation: SourceRelation | RelationArtifact) -> str:
    return getattr(relation, "table", getattr(relation, "name", ""))


def artifact_backing_name(handle: str) -> str:
    return f"{_ARTIFACT_BACKING_PREFIX}{handle}"


def _relation_sql(relation: SourceRelation | RelationArtifact) -> str:
    if isinstance(relation, SourceRelation):
        return f"{quote_identifier('main')}.{quote_identifier(_relation_name(relation))}"
    return (
        f"{quote_identifier('temp')}."
        f"{quote_identifier(artifact_backing_name(_relation_name(relation)))}"
    )


def _columns(relation: SourceRelation | RelationArtifact) -> tuple[Column, ...]:
    return tuple(relation.columns)


def _ordered_by(relation: SourceRelation | RelationArtifact) -> tuple[OrderingKey, ...]:
    return tuple(getattr(relation, "ordered_by", ()) or ())


def _canonical_json(value: Any) -> Any:
    """Deep-copy JSON-like semantics into deterministic derivation payloads."""

    # Round-trip strips caller-owned mappings/lists and rejects non-JSON model
    # values before they can leak into provenance.
    try:
        return json.loads(json.dumps(value, sort_keys=True, separators=(",", ":")))
    except (TypeError, ValueError) as exc:
        raise RelAlgValidationError(
            "invalid_arguments", "operator semantics must contain JSON values only"
        ) from exc


class SQLiteRelationalExecutor:
    """Execute typed relational calls over one SQLite connection and state."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        state: Any,
        *,
        max_rows: int = 100_000,
        max_artifact_bytes: int = 64 * 1024 * 1024,
        max_cell_bytes: int = 4 * 1024 * 1024,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.connection = connection
        self.state = state
        self.max_rows = _integer(max_rows, path="max_rows", minimum=1)
        self.max_artifact_bytes = _integer(
            max_artifact_bytes, path="max_artifact_bytes", minimum=1
        )
        self.max_cell_bytes = _integer(
            max_cell_bytes, path="max_cell_bytes", minimum=1
        )
        if self.max_cell_bytes > self.max_artifact_bytes:
            raise RelAlgValidationError(
                "invalid_arguments",
                "max_cell_bytes must not exceed max_artifact_bytes",
                path="max_cell_bytes",
            )
        self.timeout_seconds = _positive_timeout(timeout_seconds)
        self._handle_counters: dict[str, int] = {}
        self._savepoint_counter = 0
        self._scalar_error: CheckpointRelalgError | None = None
        # SQLite's length limit is a coarse fail-fast guard (it also applies to
        # complete encoded rows).  The stricter public limits are enforced by
        # ResultSizeTracker, so retain a small implementation-safety floor.
        if hasattr(connection, "setlimit") and hasattr(sqlite3, "SQLITE_LIMIT_LENGTH"):
            current_limit = connection.getlimit(sqlite3.SQLITE_LIMIT_LENGTH)
            coarse_limit = max(self.max_artifact_bytes, 64 * 1024)
            connection.setlimit(
                sqlite3.SQLITE_LIMIT_LENGTH,
                min(current_limit, coarse_limit),
            )
        self._register_scalar_functions()

    # ------------------------------------------------------------------
    # Public unified entry points
    # ------------------------------------------------------------------
    def execute(
        self,
        tool: str,
        args: Mapping[str, Any],
        artifact_handle: str | None = None,
    ) -> RelationArtifact:
        """Execute one of the nine registered atomic operators."""

        spec = get_operator_spec(tool)
        method = getattr(self, f"_{tool}")
        handle = self._allocate_handle(spec.artifact_kind, artifact_handle)
        before_hash = self._state_hash()
        savepoint = self._new_savepoint()
        started = False
        deadline = _SQLiteDeadline(self.connection, self.timeout_seconds)
        self._scalar_error = None
        try:
            self.connection.execute(f"SAVEPOINT {quote_identifier(savepoint)}")
            started = True
            deadline.start()
            # SQLite affinity is deliberately permissive (for example, text
            # ``'abc'`` participates in numeric arithmetic as zero).  The
            # checkpoint-relalg IR is not: every operand must actually match
            # its canonical column type before an operator may consume it.
            # Validate all relation arguments inside the same deadline and
            # savepoint as the operation so dirty source data cannot silently
            # change typed semantics.
            seen_inputs: set[str] = set()
            for argument_name in ("table", "left", "right"):
                relation_name = args.get(argument_name)
                if not isinstance(relation_name, str) or relation_name in seen_inputs:
                    continue
                relation = self._relation(relation_name)
                self._validate_canonical_values(relation, _columns(relation))
                seen_inputs.add(relation_name)
            artifact = method(args, handle)
            deadline.stop()
            self._publish(artifact)
            self.connection.execute(f"RELEASE SAVEPOINT {quote_identifier(savepoint)}")
            self._scalar_error = None
            return artifact
        except BaseException as exc:
            deadline.stop()
            if started:
                try:
                    self.connection.execute(f"ROLLBACK TO SAVEPOINT {quote_identifier(savepoint)}")
                    self.connection.execute(f"RELEASE SAVEPOINT {quote_identifier(savepoint)}")
                except sqlite3.Error:
                    # State verification below decides whether this is fatal.
                    pass
            after_hash = self._state_hash()
            if after_hash != before_hash:
                raise NonrecoverableExecutionError(
                    "state_changed_after_failure",
                    "failed relational execution changed logical EnvironmentState",
                    details={"tool": tool, "before_hash": before_hash, "after_hash": after_hash},
                ) from exc
            if isinstance(exc, CheckpointRelalgError):
                raise
            if isinstance(exc, StateError):
                raise CheckpointRelalgError(
                    exc.type, exc.code, exc.message, dict(exc.details)
                ) from exc
            if isinstance(exc, sqlite3.Error):
                if self._scalar_error is not None:
                    scalar_error = self._scalar_error
                    self._scalar_error = None
                    raise scalar_error from exc
                if deadline.timed_out:
                    raise RelAlgExecutionError(
                        "sql_timeout",
                        "Atomic relational execution exceeded its SQL timeout",
                        details={"tool": tool},
                    ) from exc
                if isinstance(exc, sqlite3.DataError) and "too big" in str(exc).lower():
                    raise self._sqlite_result_too_large(tool) from exc
                raise RelAlgExecutionError(
                    "sqlite_execution_failed",
                    "SQLite could not execute the validated relational operation",
                    details={"tool": tool, "sqlite_error": type(exc).__name__},
                ) from exc
            raise RelAlgExecutionError(
                "execution_failed",
                "relational operation failed after validation",
                details={"tool": tool, "exception": type(exc).__name__},
            ) from exc

    def run_readonly_queries(
        self,
        operation: str,
        callback: Callable[[], _T],
        *,
        timeout_seconds: float | None = None,
    ) -> _T:
        """Run Harness-authored perception queries under the shared SQL guard."""

        timeout = self.timeout_seconds if timeout_seconds is None else _positive_timeout(timeout_seconds)
        before_hash = self._state_hash()
        deadline = _SQLiteDeadline(self.connection, timeout)
        self._scalar_error = None
        try:
            deadline.start()
            result = callback()
            deadline.stop()
            self._scalar_error = None
            return result
        except BaseException as exc:
            deadline.stop()
            after_hash = self._state_hash()
            if after_hash != before_hash:
                raise NonrecoverableExecutionError(
                    "state_changed_after_failure",
                    "failed perception query changed logical EnvironmentState",
                    details={
                        "operation": operation,
                        "before_hash": before_hash,
                        "after_hash": after_hash,
                    },
                ) from exc
            if isinstance(exc, CheckpointRelalgError):
                raise
            if isinstance(exc, StateError):
                raise CheckpointRelalgError(
                    exc.type, exc.code, exc.message, dict(exc.details)
                ) from exc
            if isinstance(exc, sqlite3.Error):
                if self._scalar_error is not None:
                    scalar_error = self._scalar_error
                    self._scalar_error = None
                    raise scalar_error from exc
                if deadline.timed_out:
                    raise RelAlgExecutionError(
                        "sql_timeout",
                        "perception query exceeded its SQL timeout",
                        details={"operation": operation},
                    ) from exc
                if isinstance(exc, sqlite3.DataError) and "too big" in str(exc).lower():
                    raise self._sqlite_result_too_large(operation) from exc
                raise RelAlgExecutionError(
                    "sqlite_execution_failed",
                    "SQLite could not execute the validated perception query",
                    details={"operation": operation, "sqlite_error": type(exc).__name__},
                ) from exc
            raise RelAlgExecutionError(
                "execution_failed",
                "perception query failed after validation",
                details={"operation": operation, "exception": type(exc).__name__},
            ) from exc

    def validate_relation_values(self, table: str) -> None:
        """Validate canonical values and all terminal relation resource limits."""

        relation = self._relation(table)
        self.run_readonly_queries(
            "answer_value_validation",
            lambda: (
                self._validate_canonical_values(relation, _columns(relation)),
                self._validate_relation_size(
                    relation,
                    _columns(relation),
                    operation="answer",
                ),
            ),
        )

    def validate_observation_rows(
        self,
        rows: Sequence[Sequence[Any]],
        *,
        operation: str,
    ) -> None:
        """Reject oversized perception payloads before Observation publication."""

        tracker = self._new_size_tracker(operation=operation)
        for row in rows:
            tracker.add_row(row)

    def collect_relation_rows(self, table: str) -> tuple[list[str], list[list[Any]]]:
        """Stream a bounded terminal relation under the shared SQL deadline."""

        relation = self._relation(table)
        columns = [column.name for column in _columns(relation)]
        selected = ", ".join(quote_identifier(name) for name in columns)
        sql = f"SELECT {selected} FROM {_relation_sql(relation)}"
        if isinstance(relation, RelationArtifact) and relation.ordered_by:
            physical = set(self._physical_columns(relation.table))
            if _HIDDEN_ORDINAL in physical:
                sql += f" ORDER BY {quote_identifier(_HIDDEN_ORDINAL)}"

        def collect() -> list[list[Any]]:
            cursor = self.connection.execute(sql)
            tracker = self._new_size_tracker(operation="answer")
            rows: list[list[Any]] = []
            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                tracker.add_row(row)
                rows.append(list(row))
            return rows

        rows = self.run_readonly_queries("answer_rows", collect)
        return columns, rows

    def filter_rows(
        self, table: str, conditions: Mapping[str, Any], *, artifact_handle: str | None = None
    ) -> RelationArtifact:
        return self.execute(
            "filter_rows", {"table": table, "conditions": conditions}, artifact_handle
        )

    def project(
        self, table: str, outputs: list[Mapping[str, Any]], *, artifact_handle: str | None = None
    ) -> RelationArtifact:
        return self.execute("project", {"table": table, "outputs": outputs}, artifact_handle)

    def join(
        self,
        left: str,
        right: str,
        on: list[Mapping[str, Any]],
        *,
        left_role: str | None = None,
        right_role: str | None = None,
        type: str = "inner",
        artifact_handle: str | None = None,
    ) -> RelationArtifact:
        args: dict[str, Any] = {"left": left, "right": right, "on": on, "type": type}
        if left_role is not None:
            args["left_role"] = left_role
        if right_role is not None:
            args["right_role"] = right_role
        return self.execute("join", args, artifact_handle)

    def aggregate(
        self,
        table: str,
        group_by: list[str],
        metrics: list[Mapping[str, Any]],
        *,
        artifact_handle: str | None = None,
    ) -> RelationArtifact:
        return self.execute(
            "aggregate", {"table": table, "group_by": group_by, "metrics": metrics}, artifact_handle
        )

    def distinct(self, table: str, *, artifact_handle: str | None = None) -> RelationArtifact:
        return self.execute("distinct", {"table": table}, artifact_handle)

    def set_operation(
        self,
        left: str,
        right: str,
        op: str,
        *,
        artifact_handle: str | None = None,
    ) -> RelationArtifact:
        return self.execute("set_operation", {"left": left, "right": right, "op": op}, artifact_handle)

    def sort(
        self, table: str, keys: list[Mapping[str, Any]], *, artifact_handle: str | None = None
    ) -> RelationArtifact:
        return self.execute("sort", {"table": table, "keys": keys}, artifact_handle)

    def limit(
        self,
        table: str,
        count: int,
        *,
        offset: int = 0,
        artifact_handle: str | None = None,
    ) -> RelationArtifact:
        return self.execute(
            "limit", {"table": table, "count": count, "offset": offset}, artifact_handle
        )

    def add_rank(
        self,
        table: str,
        order_by: list[Mapping[str, Any]],
        method: str,
        as_: str,
        *,
        partition_by: list[str] | None = None,
        artifact_handle: str | None = None,
    ) -> RelationArtifact:
        return self.execute(
            "add_rank",
            {
                "table": table,
                "partition_by": partition_by or [],
                "order_by": order_by,
                "method": method,
                "as": as_,
            },
            artifact_handle,
        )

    def execute_sql(
        self,
        sql: str,
        artifact_handle: str | None = None,
        *,
        timeout_seconds: float = 30.0,
        max_rows: int | None = None,
    ) -> RelationArtifact:
        """Execute one complete read-only SELECT/WITH and materialize its rows.

        ``max_rows`` is a rejection limit, never a truncation limit.  The query
        first runs with SQLite ``query_only`` enabled; only after all rows and a
        unique output schema have been validated does the harness create the
        TEMP artifact.  Top-level ORDER BY is captured with a hidden ordinal.
        """

        if not isinstance(sql, str) or not sql.strip():
            raise RelAlgValidationError("invalid_sql", "sql must be a non-empty string")
        if _READONLY_SQL_START.match(sql) is None:
            raise RelAlgValidationError(
                "unsafe_sql", "execute_sql accepts one read-only SELECT or WITH statement"
            )
        timeout = _positive_timeout(timeout_seconds)
        effective_max_rows = self.max_rows
        if max_rows is not None:
            effective_max_rows = min(
                effective_max_rows, _integer(max_rows, path="max_rows", minimum=1)
            )
        handle = self._allocate_handle("sql", artifact_handle)
        before_hash = self._state_hash()
        savepoint = self._new_savepoint()
        started = False
        deadline = _SQLiteDeadline(self.connection, timeout)
        query_only_before = 0
        authorized_inputs: set[str] = set()
        authorization_error: CheckpointRelalgError | StateError | None = None
        try:
            self.connection.execute(f"SAVEPOINT {quote_identifier(savepoint)}")
            started = True
            query_only_before = int(self.connection.execute("PRAGMA query_only").fetchone()[0])
            self.connection.execute("PRAGMA query_only=ON")
            temp_relation_keys = {
                sqlite_identifier_key(row[0])
                for row in self.connection.execute(
                    "SELECT name FROM sqlite_temp_master WHERE type IN ('table', 'view')"
                ).fetchall()
            }
            def authorize(
                action: int,
                arg1: str | None,
                arg2: str | None,
                database: str | None,
                trigger: str | None,
            ) -> int:
                nonlocal authorization_error
                forbidden_actions = {
                    sqlite3.SQLITE_ATTACH,
                    sqlite3.SQLITE_DETACH,
                    sqlite3.SQLITE_DELETE,
                    sqlite3.SQLITE_INSERT,
                    sqlite3.SQLITE_UPDATE,
                    sqlite3.SQLITE_ALTER_TABLE,
                    sqlite3.SQLITE_DROP_TABLE,
                    sqlite3.SQLITE_DROP_TEMP_TABLE,
                    sqlite3.SQLITE_DROP_VIEW,
                    sqlite3.SQLITE_DROP_TEMP_VIEW,
                    sqlite3.SQLITE_CREATE_TABLE,
                    sqlite3.SQLITE_CREATE_TEMP_TABLE,
                    sqlite3.SQLITE_CREATE_VIEW,
                    sqlite3.SQLITE_CREATE_TEMP_VIEW,
                    sqlite3.SQLITE_PRAGMA,
                }
                if action in forbidden_actions:
                    authorization_error = CheckpointRelalgError(
                        "argument_validation_error",
                        "unsafe_sql",
                        "Direct SQL attempted a non-read-only SQLite action",
                    )
                    return sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_FUNCTION:
                    function_name = str(arg2 or arg1 or "").lower()
                    if function_name in {"load_extension", "readfile", "writefile", "eval"} or function_name.startswith(("pragma_", "_relalg_")):
                        authorization_error = CheckpointRelalgError(
                            "argument_validation_error",
                            "unsafe_sql",
                            f"Direct SQL function {function_name!r} is not read-only",
                        )
                        return sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_READ and arg1 and arg1.startswith("sqlite_"):
                    authorization_error = CheckpointRelalgError(
                        "argument_validation_error",
                        "unsafe_sql",
                        "Direct SQL cannot inspect SQLite internal catalog relations",
                    )
                    return sqlite3.SQLITE_DENY
                if action == sqlite3.SQLITE_READ and arg1:
                    if isinstance(trigger, str):
                        try:
                            exposed = self.state.get_relation(trigger)
                        except StateError:
                            exposed = None
                        if (
                            isinstance(exposed, RelationArtifact)
                            and database == "temp"
                            and sqlite_identifier_key(arg1)
                            == sqlite_identifier_key(artifact_backing_name(exposed.table))
                        ):
                            authorized_inputs.add(exposed.table)
                            return sqlite3.SQLITE_OK
                    try:
                        relation = self.state.get_relation(arg1)
                    except StateError as exc:
                        authorization_error = exc
                        return sqlite3.SQLITE_DENY
                    expected_database = (
                        "main" if isinstance(relation, SourceRelation) else "temp"
                    )
                    unqualified_is_safe = database is None and (
                        isinstance(relation, RelationArtifact)
                        or sqlite_identifier_key(arg1) not in temp_relation_keys
                    )
                    if database != expected_database and not unqualified_is_safe:
                        authorization_error = CheckpointRelalgError(
                            "argument_validation_error",
                            "unsafe_sql",
                            "Direct SQL may read only registered source relations in main "
                            "and active artifact views in temp",
                            {
                                "table": arg1,
                                "database": database,
                                "expected_database": expected_database,
                            },
                        )
                        return sqlite3.SQLITE_DENY
                    authorized_inputs.add(_relation_name(relation))
                return sqlite3.SQLITE_OK

            deadline.start()
            self.connection.set_authorizer(authorize)
            try:
                cursor = self.connection.execute(sql)
                if cursor.description is None:
                    raise RelAlgValidationError(
                        "unsafe_sql", "execute_sql statement did not return a relation"
                    )
                names = [item[0] for item in cursor.description]
                if any(not isinstance(name, str) or not name for name in names):
                    raise RelAlgValidationError(
                        "invalid_sql_output", "Direct SQL output columns must be non-empty"
                    )
                if len(names) != len({sqlite_identifier_key(name) for name in names}):
                    raise RelAlgValidationError(
                        "duplicate_output_column",
                        "Direct SQL output column names must be unique",
                        details={"columns": names},
                    )
                reserved_names = [
                    name
                    for name in names
                    if sqlite_identifier_key(name).startswith("__")
                ]
                if reserved_names:
                    raise RelAlgValidationError(
                        "reserved_output_column",
                        "Direct SQL output uses a Harness-reserved column name",
                        details={"columns": reserved_names},
                    )
                rows: list[Sequence[Any]] = []
                size_tracker = self._new_size_tracker(
                    operation="execute_sql",
                    max_rows=effective_max_rows,
                )
                while True:
                    row = cursor.fetchone()
                    if row is None:
                        break
                    size_tracker.add_row(row)
                    rows.append(row)
                deadline.check(operation="Direct SQL")
            finally:
                self.connection.set_authorizer(None)
                self.connection.execute(f"PRAGMA query_only={1 if query_only_before else 0}")
            declared_types = self._direct_declared_types(sql, names, handle)
            deadline.check(operation="Direct SQL")
            columns = self._infer_direct_columns(names, rows, declared_types=declared_types)
            deadline.check(operation="Direct SQL")
            has_order, parsed_order = self._direct_ordering(sql, names)
            deadline.check(operation="Direct SQL")
            if has_order and not parsed_order:
                # SQL may order by an expression or a non-projected column.
                # The hidden ordinal is its exact semantic carrier; ordering
                # metadata is allowed to reference hidden ordering keys.
                parsed_order = (
                    OrderingKey(
                        column=_OPAQUE_DIRECT_ORDER, direction="asc", nulls="last"
                    ),
                )
            self._materialize_direct_rows(
                handle, columns, rows, ordered=has_order
            )
            deadline.check(operation="Direct SQL")
            self._validate_relation_size(
                handle,
                columns,
                operation="execute_sql",
                include_hidden=has_order,
            )
            self._validate_canonical_values(handle, columns)
            scalar_cell = rows[0][0] if len(rows) == 1 and len(columns) == 1 else None
            artifact = RelationArtifact(
                table=handle,
                kind="sql",
                columns=tuple(columns),
                row_count=len(rows),
                ordered_by=tuple(parsed_order),
                derivation={
                    "schema": "relation-derivation-v2",
                    "operator": "execute_sql",
                    "inputs": sorted(authorized_inputs),
                    "semantics": {
                        "read_only": True,
                        "top_level_order_by": has_order,
                        # SQL text is required replay provenance for Direct mode.
                        "sql": sql,
                    },
                },
                scalar_cell=scalar_cell,
            )
            self._publish(artifact)
            deadline.check(operation="Direct SQL")
            self.connection.execute(f"RELEASE SAVEPOINT {quote_identifier(savepoint)}")
            deadline.stop()
            return artifact
        except BaseException as exc:
            deadline.stop()
            self.connection.set_authorizer(None)
            try:
                self.connection.execute(f"PRAGMA query_only={1 if query_only_before else 0}")
            except sqlite3.Error:
                pass
            if started:
                try:
                    self.connection.execute(f"ROLLBACK TO SAVEPOINT {quote_identifier(savepoint)}")
                    self.connection.execute(f"RELEASE SAVEPOINT {quote_identifier(savepoint)}")
                except sqlite3.Error:
                    pass
            after_hash = self._state_hash()
            if after_hash != before_hash:
                raise NonrecoverableExecutionError(
                    "state_changed_after_failure",
                    "failed Direct SQL execution changed logical EnvironmentState",
                    details={"before_hash": before_hash, "after_hash": after_hash},
                ) from exc
            if authorization_error is not None:
                exc = authorization_error
            if isinstance(exc, CheckpointRelalgError):
                raise exc
            if isinstance(exc, StateError):
                raise CheckpointRelalgError(exc.type, exc.code, exc.message, dict(exc.details)) from exc
            if deadline.timed_out:
                raise RelAlgExecutionError(
                    "sql_timeout", "Direct SQL exceeded its execution timeout"
                ) from exc
            if isinstance(exc, sqlite3.ProgrammingError) and "one statement" in str(exc).lower():
                raise RelAlgValidationError(
                    "multiple_sql_statements", "execute_sql accepts exactly one statement"
                ) from exc
            if isinstance(exc, sqlite3.Error):
                if isinstance(exc, sqlite3.DataError) and "too big" in str(exc).lower():
                    raise self._sqlite_result_too_large("execute_sql") from exc
                raise RelAlgExecutionError(
                    "sqlite_execution_failed",
                    "SQLite could not execute the read-only Direct SQL statement",
                    details={"sqlite_error": type(exc).__name__},
                ) from exc
            raise RelAlgExecutionError(
                "execution_failed",
                "Direct SQL execution failed",
                details={"exception": type(exc).__name__},
            ) from exc

    def execute_direct_sql(
        self,
        sql: str,
        artifact_handle: str | None = None,
        *,
        timeout_seconds: float = 30.0,
        max_rows: int | None = None,
    ) -> RelationArtifact:
        """Compatibility spelling for integrations that name the Direct lane."""

        return self.execute_sql(
            sql,
            artifact_handle,
            timeout_seconds=timeout_seconds,
            max_rows=max_rows,
        )

    # ------------------------------------------------------------------
    # Atomic implementations
    # ------------------------------------------------------------------
    def _filter_rows(self, raw: Mapping[str, Any], handle: str) -> RelationArtifact:
        args = _keys(raw, required={"table", "conditions"})
        relation = self._relation(args["table"])
        columns = _columns(relation)
        predicate = PredicateCompiler(columns, table_alias="src").compile(args["conditions"])
        select = self._visible_select(columns, "src")
        ordinal = self._preserved_ordinal(relation, "src")
        if ordinal:
            select.append(f"{ordinal} AS {quote_identifier(_HIDDEN_ORDINAL)}")
        sql = (
            f"SELECT {', '.join(select)} FROM {_relation_sql(relation)} AS "
            f"{quote_identifier('src')} WHERE {predicate.sql}"
        )
        if ordinal:
            sql += f" ORDER BY {ordinal}"
        return self._materialize(
            handle,
            "filter",
            columns,
            _ordered_by(relation),
            sql,
            predicate.params,
            inputs=[_relation_name(relation)],
            semantics={"conditions": args["conditions"]},
        )

    def _project(self, raw: Mapping[str, Any], handle: str) -> RelationArtifact:
        args = _keys(raw, required={"table", "outputs"})
        relation = self._relation(args["table"])
        outputs = _array(args["outputs"], path="arguments.outputs", minimum=1)
        compiler = ExpressionCompiler(_columns(relation), table_alias="src")
        select: list[str] = []
        params: list[Any] = []
        result_columns: list[Column] = []
        names: set[str] = set()
        canonical_outputs: list[dict[str, Any]] = []
        for index, raw_output in enumerate(outputs):
            path = f"arguments.outputs[{index}]"
            output = _keys(raw_output, required={"expression"}, optional={"as"}, path=path)
            expression = compiler.compile(output["expression"], path=f"{path}.expression")
            if "as" in output:
                name = require_simple_identifier(output["as"], path=f"{path}.as")
            elif expression.column_name is not None:
                name = expression.column_name
            else:
                raise RelAlgValidationError(
                    "missing_output_alias",
                    "computed project expressions require an explicit as",
                    path=path,
                )
            name_key = sqlite_identifier_key(name)
            if name_key in names:
                raise RelAlgValidationError(
                    "duplicate_output_column", f"duplicate project output {name!r}", path=path
                )
            names.add(name_key)
            select.append(f"{expression.sql} AS {quote_identifier(name)}")
            params.extend(expression.params)
            result_columns.append(Column(name=name, canonical_type=expression.canonical_type))
            canonical_outputs.append(dict(output))
        ordinal = self._preserved_ordinal(relation, "src")
        if ordinal:
            select.append(f"{ordinal} AS {quote_identifier(_HIDDEN_ORDINAL)}")
        sql = (
            f"SELECT {', '.join(select)} FROM {_relation_sql(relation)} AS "
            f"{quote_identifier('src')}"
        )
        if ordinal:
            sql += f" ORDER BY {ordinal}"
        return self._materialize(
            handle,
            "project",
            result_columns,
            _ordered_by(relation),
            sql,
            params,
            inputs=[_relation_name(relation)],
            semantics={"outputs": canonical_outputs},
        )

    def _join(self, raw: Mapping[str, Any], handle: str) -> RelationArtifact:
        args = _keys(
            raw,
            required={"left", "right", "on"},
            optional={"left_role", "right_role", "type"},
        )
        left = self._relation(args["left"])
        right = self._relation(args["right"])
        left_name, right_name = _relation_name(left), _relation_name(right)
        join_type = args.get("type", "inner")
        if join_type not in _JOIN_TYPES:
            raise RelAlgValidationError(
                "invalid_join_type", f"unsupported join type {join_type!r}", path="arguments.type"
            )
        left_role = require_simple_identifier(
            args.get("left_role", left_name), path="arguments.left_role", code="invalid_role"
        )
        right_role = require_simple_identifier(
            args.get("right_role", right_name), path="arguments.right_role", code="invalid_role"
        )
        if left_name == right_name and ("left_role" not in args or "right_role" not in args):
            raise RelAlgValidationError(
                "self_join_roles_required", "self join requires two explicit roles"
            )
        if left_name == right_name and sqlite_identifier_key(left_role) == sqlite_identifier_key(right_role):
            raise RelAlgValidationError(
                "invalid_join_roles", "self join roles must be different"
            )
        if join_type in {"inner", "left", "cross"} and sqlite_identifier_key(left_role) == sqlite_identifier_key(right_role):
            raise RelAlgValidationError(
                "invalid_join_roles", "a combined join output requires different roles"
            )
        edges = _array(args["on"], path="arguments.on")
        if join_type == "cross" and edges:
            raise RelAlgValidationError(
                "invalid_join_condition", "cross join requires on=[]", path="arguments.on"
            )
        if join_type != "cross" and not edges:
            raise RelAlgValidationError(
                "invalid_join_condition", "non-cross join requires at least one edge", path="arguments.on"
            )
        left_types, right_types = column_type_map(_columns(left)), column_type_map(_columns(right))
        conditions: list[str] = []
        canonical_edges: list[dict[str, str]] = []
        for index, raw_edge in enumerate(edges):
            path = f"arguments.on[{index}]"
            edge = _keys(raw_edge, required={"left_column", "op", "right_column"}, path=path)
            left_column, right_column, op = edge["left_column"], edge["right_column"], edge["op"]
            if left_column not in left_types:
                raise RelAlgStateValidationError(
                    "unknown_column", f"unknown exact left column {left_column!r}", path=f"{path}.left_column"
                )
            if right_column not in right_types:
                raise RelAlgStateValidationError(
                    "unknown_column", f"unknown exact right column {right_column!r}", path=f"{path}.right_column"
                )
            if op not in _JOIN_OPS:
                raise RelAlgValidationError(
                    "invalid_join_condition", f"unsupported join comparison {op!r}", path=f"{path}.op"
                )
            common = merge_types(
                left_types[left_column], right_types[right_column], path=path, equality=op in {"=", "!="}
            )
            if common in {"BOOLEAN", "BLOB"} and op not in {"=", "!="}:
                raise RelAlgValidationError(
                    "type_mismatch", f"{common} join edges support equality only", path=path
                )
            sql_op = "<>" if op == "!=" else op
            left_operand = (
                f"{quote_identifier('l')}.{quote_identifier(left_column)}"
            )
            right_operand = (
                f"{quote_identifier('r')}.{quote_identifier(right_column)}"
            )
            if common in {"TEXT", "DATE", "DATETIME"}:
                left_operand = f"({left_operand} COLLATE BINARY)"
                right_operand = f"({right_operand} COLLATE BINARY)"
            conditions.append(
                f"{left_operand} {sql_op} {right_operand}"
            )
            canonical_edges.append(dict(edge))
        condition_sql = " AND ".join(f"({condition})" for condition in conditions)
        left_source = f"{_relation_sql(left)} AS {quote_identifier('l')}"
        right_source = f"{_relation_sql(right)} AS {quote_identifier('r')}"

        if join_type in {"semi", "anti"}:
            result_columns = list(_columns(left))
            select = self._visible_select(result_columns, "l")
            exists = (
                f"EXISTS (SELECT 1 FROM {right_source} WHERE {condition_sql})"
            )
            if join_type == "anti":
                exists = f"NOT {exists}"
            sql = f"SELECT {', '.join(select)} FROM {left_source} WHERE {exists}"
        else:
            result_columns = []
            select = []
            for side, role, columns in (
                ("l", left_role, _columns(left)),
                ("r", right_role, _columns(right)),
            ):
                for column in columns:
                    output_name = f"{role}.{column.name}"
                    result_columns.append(
                        Column(name=output_name, canonical_type=column.canonical_type)
                    )
                    select.append(
                        f"{quote_identifier(side)}.{quote_identifier(column.name)} AS "
                        f"{quote_identifier(output_name)}"
                    )
            join_keyword = {"inner": "INNER JOIN", "left": "LEFT JOIN", "cross": "CROSS JOIN"}[join_type]
            sql = f"SELECT {', '.join(select)} FROM {left_source} {join_keyword} {right_source}"
            if join_type != "cross":
                sql += f" ON {condition_sql}"
        semantics: dict[str, Any] = {
            "type": join_type,
            "left_role": left_role,
            "right_role": right_role,
            "on": canonical_edges,
        }
        return self._materialize(
            handle,
            "join",
            result_columns,
            (),
            sql,
            (),
            inputs=[left_name, right_name],
            semantics=semantics,
        )

    def _aggregate(self, raw: Mapping[str, Any], handle: str) -> RelationArtifact:
        args = _keys(raw, required={"table", "group_by", "metrics"})
        relation = self._relation(args["table"])
        types = column_type_map(_columns(relation))
        group_by = _array(args["group_by"], path="arguments.group_by")
        if any(not isinstance(name, str) for name in group_by) or len(set(group_by)) != len(group_by):
            raise RelAlgValidationError(
                "invalid_arguments", "group_by must contain unique exact column names", path="arguments.group_by"
            )
        for index, name in enumerate(group_by):
            if name not in types:
                raise RelAlgStateValidationError(
                    "unknown_column", f"unknown exact group column {name!r}", path=f"arguments.group_by[{index}]"
                )
        metrics = _array(args["metrics"], path="arguments.metrics", minimum=1)
        result_columns = [Column(name=name, canonical_type=types[name]) for name in group_by]
        output_names = {sqlite_identifier_key(name) for name in group_by}
        select = [f"{quote_identifier('src')}.{quote_identifier(name)}" for name in group_by]
        canonical_metrics: list[dict[str, Any]] = []
        for index, raw_metric in enumerate(metrics):
            path = f"arguments.metrics[{index}]"
            metric = _keys(
                raw_metric, required={"op", "column", "as"}, optional={"distinct"}, path=path
            )
            op, column = metric["op"], metric["column"]
            alias = require_simple_identifier(metric["as"], path=f"{path}.as")
            distinct = metric.get("distinct", False)
            if not isinstance(distinct, bool):
                raise RelAlgValidationError(
                    "invalid_arguments", "distinct must be boolean", path=f"{path}.distinct"
                )
            if op not in _AGGREGATES:
                raise RelAlgValidationError(
                    "unsupported_aggregate", f"unsupported aggregate {op!r}", path=f"{path}.op"
                )
            if not isinstance(column, str) or (column != "*" and column not in types):
                raise RelAlgStateValidationError(
                    "unknown_column", f"unknown exact metric column {column!r}", path=f"{path}.column"
                )
            if column == "*" and (op != "count" or distinct):
                raise RelAlgValidationError(
                    "invalid_aggregate", "only count(*) with distinct=false is valid", path=path
                )
            alias_key = sqlite_identifier_key(alias)
            if alias_key in output_names:
                raise RelAlgValidationError(
                    "duplicate_output_column", f"duplicate aggregate output {alias!r}", path=f"{path}.as"
                )
            output_names.add(alias_key)
            if op in {"sum", "avg"} and types[column] not in {"INTEGER", "REAL"}:
                raise RelAlgValidationError(
                    "type_mismatch", f"{op} requires an INTEGER or REAL column", path=f"{path}.column"
                )
            if op in {"min", "max"} and types[column] == "BLOB":
                raise RelAlgValidationError(
                    "type_mismatch", f"{op} does not support BLOB", path=f"{path}.column"
                )
            if op == "count":
                output_type = "INTEGER"
            elif op == "avg":
                output_type = "REAL"
            else:
                output_type = types[column]
            if column == "*":
                operand = "*"
            else:
                operand = f"{quote_identifier('src')}.{quote_identifier(column)}"
                if types[column] in {"TEXT", "DATE", "DATETIME"}:
                    operand = f"({operand} COLLATE BINARY)"
                if distinct:
                    operand = f"DISTINCT {operand}"
            select.append(f"{op.upper()}({operand}) AS {quote_identifier(alias)}")
            result_columns.append(Column(name=alias, canonical_type=output_type))
            canonical_metrics.append(
                {"op": op, "column": column, "distinct": distinct, "as": alias}
            )
        sql = (
            f"SELECT {', '.join(select)} FROM {_relation_sql(relation)} AS "
            f"{quote_identifier('src')}"
        )
        if group_by:
            sql += " GROUP BY " + ", ".join(
                self._typed_column_sql(
                    Column(name=name, canonical_type=types[name]), "src"
                )
                for name in group_by
            )
        return self._materialize(
            handle,
            "aggregate",
            result_columns,
            (),
            sql,
            (),
            inputs=[_relation_name(relation)],
            semantics={"group_by": group_by, "metrics": canonical_metrics},
        )

    def _distinct(self, raw: Mapping[str, Any], handle: str) -> RelationArtifact:
        args = _keys(raw, required={"table"})
        relation = self._relation(args["table"])
        columns = _columns(relation)
        sql = (
            f"SELECT DISTINCT {', '.join(self._visible_select(columns, 'src'))} FROM "
            f"{_relation_sql(relation)} AS {quote_identifier('src')}"
        )
        return self._materialize(
            handle,
            "distinct",
            columns,
            (),
            sql,
            (),
            inputs=[_relation_name(relation)],
            semantics={},
        )

    def _set_operation(self, raw: Mapping[str, Any], handle: str) -> RelationArtifact:
        args = _keys(raw, required={"left", "right", "op"})
        left, right = self._relation(args["left"]), self._relation(args["right"])
        op = args["op"]
        if op not in _SET_OPS:
            raise RelAlgValidationError(
                "invalid_set_operation", f"unsupported set operation {op!r}", path="arguments.op"
            )
        left_columns, right_columns = _columns(left), _columns(right)
        if len(left_columns) != len(right_columns):
            raise RelAlgValidationError(
                "incompatible_set_schema",
                "set inputs must have the same column count",
                details={"left_count": len(left_columns), "right_count": len(right_columns)},
            )
        output_columns: list[Column] = []
        left_select: list[str] = []
        right_select: list[str] = []
        for index, (left_column, right_column) in enumerate(zip(left_columns, right_columns)):
            try:
                common = merge_types(
                    left_column.canonical_type,
                    right_column.canonical_type,
                    path=f"arguments.columns[{index}]",
                    equality=True,
                )
            except RelAlgValidationError as exc:
                raise RelAlgValidationError(
                    "incompatible_set_schema",
                    "set inputs have incompatible corresponding column types",
                    path=f"arguments.columns[{index}]",
                    details={
                        "left_type": left_column.canonical_type,
                        "right_type": right_column.canonical_type,
                    },
                ) from exc
            output_columns.append(Column(name=left_column.name, canonical_type=common))
            left_expr = f"{quote_identifier('l')}.{quote_identifier(left_column.name)}"
            right_expr = f"{quote_identifier('r')}.{quote_identifier(right_column.name)}"
            if common == "REAL":
                left_expr = f"CAST({left_expr} AS REAL)"
                right_expr = f"CAST({right_expr} AS REAL)"
            elif common in {"TEXT", "DATE", "DATETIME"}:
                left_expr = f"({left_expr} COLLATE BINARY)"
                right_expr = f"({right_expr} COLLATE BINARY)"
            left_select.append(f"{left_expr} AS {quote_identifier(left_column.name)}")
            right_select.append(f"{right_expr} AS {quote_identifier(left_column.name)}")
        keyword = {"union": "UNION", "union_all": "UNION ALL", "intersect": "INTERSECT", "except": "EXCEPT"}[op]
        sql = (
            f"SELECT {', '.join(left_select)} FROM {_relation_sql(left)} AS {quote_identifier('l')} "
            f"{keyword} SELECT {', '.join(right_select)} FROM {_relation_sql(right)} AS {quote_identifier('r')}"
        )
        return self._materialize(
            handle,
            "set",
            output_columns,
            (),
            sql,
            (),
            inputs=[_relation_name(left), _relation_name(right)],
            semantics={"op": op},
        )

    def _sort(self, raw: Mapping[str, Any], handle: str) -> RelationArtifact:
        args = _keys(raw, required={"table", "keys"})
        relation = self._relation(args["table"])
        keys = self._validate_order_keys(
            args["keys"], _columns(relation), path="arguments.keys", minimum=1
        )
        semantic_order = tuple(
            OrderingKey(column=item["column"], direction=item["direction"], nulls=item["nulls"])
            for item in keys
        )
        order_terms = self._order_terms(keys, "src", _columns(relation))
        # Deterministic physical tie breaking is deliberately absent from
        # semantic ordered_by metadata.
        tie_keys = [
            {"column": column.name, "direction": "asc", "nulls": "last"}
            for column in _columns(relation)
        ]
        deterministic_terms = order_terms + self._order_terms(
            tie_keys, "src", _columns(relation)
        )
        visible = self._visible_select(_columns(relation), "src")
        window_order = ", ".join(deterministic_terms)
        sql = (
            f"SELECT {', '.join(visible)}, ROW_NUMBER() OVER (ORDER BY {window_order}) AS "
            f"{quote_identifier(_HIDDEN_ORDINAL)} FROM {_relation_sql(relation)} AS "
            f"{quote_identifier('src')} ORDER BY {window_order}"
        )
        return self._materialize(
            handle,
            "sort",
            _columns(relation),
            semantic_order,
            sql,
            (),
            inputs=[_relation_name(relation)],
            semantics={"keys": keys},
        )

    def _limit(self, raw: Mapping[str, Any], handle: str) -> RelationArtifact:
        args = _keys(raw, required={"table", "count"}, optional={"offset"})
        relation = self._relation(args["table"])
        count = _integer(args["count"], path="arguments.count", minimum=1)
        offset = _integer(args.get("offset", 0), path="arguments.offset", minimum=0)
        ordering = _ordered_by(relation)
        if not ordering:
            raise RelAlgValidationError(
                "unordered_limit_input", "limit requires an input with non-empty ordered_by"
            )
        ordinal = self._preserved_ordinal(relation, "src")
        if ordinal is None:
            raise RelAlgExecutionError(
                "missing_order_ordinal", "ordered input has no materialized deterministic ordinal"
            )
        select = self._visible_select(_columns(relation), "src")
        select.append(f"{ordinal} AS {quote_identifier(_HIDDEN_ORDINAL)}")
        sql = (
            f"SELECT {', '.join(select)} FROM {_relation_sql(relation)} AS "
            f"{quote_identifier('src')} ORDER BY {ordinal} LIMIT ? OFFSET ?"
        )
        return self._materialize(
            handle,
            "limit",
            _columns(relation),
            ordering,
            sql,
            (count, offset),
            inputs=[_relation_name(relation)],
            semantics={"count": count, "offset": offset},
        )

    def _add_rank(self, raw: Mapping[str, Any], handle: str) -> RelationArtifact:
        args = _keys(
            raw,
            required={"table", "order_by", "method", "as"},
            optional={"partition_by"},
        )
        relation = self._relation(args["table"])
        types = column_type_map(_columns(relation))
        partition_by = _array(args.get("partition_by", []), path="arguments.partition_by")
        if any(not isinstance(item, str) for item in partition_by) or len(set(partition_by)) != len(partition_by):
            raise RelAlgValidationError(
                "invalid_arguments", "partition_by must contain unique exact columns", path="arguments.partition_by"
            )
        for index, column in enumerate(partition_by):
            if column not in types:
                raise RelAlgStateValidationError(
                    "unknown_column", f"unknown exact partition column {column!r}", path=f"arguments.partition_by[{index}]"
                )
        keys = self._validate_order_keys(
            args["order_by"], _columns(relation), path="arguments.order_by", minimum=1
        )
        method = args["method"]
        if method not in _RANK_METHODS:
            raise RelAlgValidationError(
                "invalid_rank_method", f"unsupported rank method {method!r}", path="arguments.method"
            )
        alias = require_simple_identifier(args["as"], path="arguments.as")
        if sqlite_identifier_key(alias) in {
            sqlite_identifier_key(name) for name in types
        }:
            raise RelAlgValidationError(
                "duplicate_output_column", f"rank output {alias!r} already exists", path="arguments.as"
            )
        order_terms = self._order_terms(keys, "src", _columns(relation))
        if method == "row_number":
            tie_keys = [
                {"column": column.name, "direction": "asc", "nulls": "last"}
                for column in _columns(relation)
            ]
            order_terms += self._order_terms(tie_keys, "src", _columns(relation))
        window_parts: list[str] = []
        if partition_by:
            window_parts.append(
                "PARTITION BY "
                + ", ".join(
                    self._typed_column_sql(
                        Column(name=column, canonical_type=types[column]), "src"
                    )
                    for column in partition_by
                )
            )
        window_parts.append("ORDER BY " + ", ".join(order_terms))
        function = {"row_number": "ROW_NUMBER", "rank": "RANK", "dense_rank": "DENSE_RANK"}[method]
        select = self._visible_select(_columns(relation), "src")
        select.append(
            f"{function}() OVER ({' '.join(window_parts)}) AS {quote_identifier(alias)}"
        )
        sql = (
            f"SELECT {', '.join(select)} FROM {_relation_sql(relation)} AS "
            f"{quote_identifier('src')}"
        )
        result_columns = [*_columns(relation), Column(name=alias, canonical_type="INTEGER")]
        return self._materialize(
            handle,
            "rank",
            result_columns,
            (),
            sql,
            (),
            inputs=[_relation_name(relation)],
            semantics={
                "partition_by": partition_by,
                "order_by": keys,
                "method": method,
                "as": alias,
            },
        )

    # ------------------------------------------------------------------
    # Materialization, state integration, and deterministic order
    # ------------------------------------------------------------------
    def _materialize(
        self,
        handle: str,
        kind: str,
        columns: Sequence[Column],
        ordering: Sequence[OrderingKey],
        select_sql: str,
        params: Sequence[Any],
        *,
        inputs: Sequence[str],
        semantics: Mapping[str, Any],
    ) -> RelationArtifact:
        if not columns:
            raise RelAlgValidationError("invalid_schema", "relation artifacts require visible columns")
        names = [column.name for column in columns]
        if len({sqlite_identifier_key(name) for name in names}) != len(names):
            raise RelAlgValidationError(
                "invalid_schema", "artifact visible column names must be unique"
            )
        reserved_names = [
            name
            for name in names
            if sqlite_identifier_key(name).startswith("__")
        ]
        if reserved_names:
            raise RelAlgValidationError(
                "reserved_output_column",
                "artifact output uses a Harness-reserved column name",
                details={"columns": reserved_names},
            )
        # Bound physical materialization at max_rows+1.  The extra row proves
        # overflow without silently publishing a truncated relation.
        backing = artifact_backing_name(handle)
        self.connection.execute(
            f"CREATE TEMP TABLE {quote_identifier(backing)} AS "
            f"SELECT * FROM ({select_sql}) AS {quote_identifier('__relalg_bounded')} LIMIT ?",
            (*tuple(params), self.max_rows + 1),
        )
        physical_columns = self._physical_columns(handle)
        allowed_physical_schemas = {
            tuple(names),
            (*tuple(names), _HIDDEN_ORDINAL),
        }
        if physical_columns not in allowed_physical_schemas:
            raise RelAlgExecutionError(
                "invalid_materialized_schema",
                "SQLite materialized a schema different from the validated artifact schema",
                details={
                    "expected_visible_columns": names,
                    "physical_columns": list(physical_columns),
                },
            )
        row_count = int(
            self.connection.execute(
                f"SELECT COUNT(*) FROM {quote_identifier('temp')}.{quote_identifier(backing)}"
            ).fetchone()[0]
        )
        if row_count > self.max_rows:
            raise CheckpointRelalgError(
                "resource_limit_error",
                "result_too_large",
                "atomic result exceeds max_rows and was not published",
                {"max_rows": self.max_rows, "row_count": row_count, "operator": kind},
            )
        self._validate_relation_size(
            handle,
            columns,
            operation=kind,
            include_hidden=bool(ordering),
        )
        self._validate_canonical_values(handle, columns)
        scalar_cell = None
        if row_count == 1 and len(columns) == 1:
            scalar_cell = self.connection.execute(
                f"SELECT {quote_identifier(columns[0].name)} FROM "
                f"{quote_identifier('temp')}.{quote_identifier(backing)}"
            ).fetchone()[0]
        visible = ", ".join(quote_identifier(column.name) for column in columns)
        self.connection.execute(
            f"CREATE TEMP VIEW {quote_identifier(handle)} AS SELECT {visible} FROM "
            f"{quote_identifier('temp')}.{quote_identifier(backing)}"
        )
        derivation_operator = {
            "filter": "filter_rows",
            "set": "set_operation",
            "rank": "add_rank",
        }.get(kind, kind)
        derivation = {
            "schema": "relation-derivation-v2",
            "operator": derivation_operator,
            "inputs": list(inputs),
            "semantics": _canonical_json(semantics),
        }
        return RelationArtifact(
            table=handle,
            kind=kind,
            columns=tuple(columns),
            row_count=row_count,
            ordered_by=tuple(ordering),
            derivation=derivation,
            scalar_cell=scalar_cell,
        )

    def _validate_canonical_values(
        self,
        relation: str | SourceRelation | RelationArtifact,
        columns: Sequence[Column],
    ) -> None:
        """Reject SQLite values outside the protocol's closed type system.

        NULL remains a legal missing value for every non-NULL column type.
        REAL deliberately accepts INTEGER storage (the one lossless widening
        permitted by the IR); all other storage-class conversions are
        rejected before they can influence an operator or enter state.
        """

        if isinstance(relation, str):
            table_name = relation
            table_sql = (
                f"{quote_identifier('temp')}."
                f"{quote_identifier(artifact_backing_name(relation))}"
            )
        else:
            table_name = _relation_name(relation)
            table_sql = _relation_sql(relation)
        storage_classes = {
            "BOOLEAN": ("integer",),
            "INTEGER": ("integer",),
            "REAL": ("integer", "real"),
            "TEXT": ("text",),
            "DATE": ("text",),
            "DATETIME": ("text",),
            "BLOB": ("blob",),
        }
        for column in columns:
            expression = quote_identifier(column.name)
            if column.canonical_type == "NULL":
                invalid = self.connection.execute(
                    f"SELECT 1 FROM {table_sql} WHERE {expression} IS NOT NULL LIMIT 1"
                ).fetchone()
            else:
                allowed = storage_classes[column.canonical_type]
                placeholders = ", ".join("?" for _ in allowed)
                invalid = self.connection.execute(
                    f"SELECT typeof({expression}), {expression} FROM {table_sql} "
                    f"WHERE {expression} IS NOT NULL "
                    f"AND typeof({expression}) NOT IN ({placeholders}) LIMIT 1",
                    allowed,
                ).fetchone()
            if invalid is not None:
                raise RelAlgExecutionError(
                    "canonical_type_mismatch",
                    "stored relation value does not match its declared canonical type",
                    details={
                        "table": table_name,
                        "column": column.name,
                        "canonical_type": column.canonical_type,
                        "storage_type": invalid[0] if len(invalid) > 1 else None,
                    },
                )
            if column.canonical_type == "BOOLEAN":
                invalid_boolean = self.connection.execute(
                    f"SELECT 1 FROM {table_sql} WHERE {expression} IS NOT NULL "
                    f"AND {expression} NOT IN (0, 1) LIMIT 1"
                ).fetchone()
                if invalid_boolean is not None:
                    raise RelAlgExecutionError(
                        "canonical_type_mismatch",
                        "BOOLEAN values must be stored as integer 0 or 1",
                        details={"table": table_name, "column": column.name},
                    )
            if column.canonical_type == "REAL":
                maximum = self.connection.execute(
                    f"SELECT MAX(ABS({expression})) FROM {table_sql} "
                    f"WHERE typeof({expression}) = 'real'"
                ).fetchone()[0]
                if isinstance(maximum, float) and not math.isfinite(maximum):
                    raise RelAlgExecutionError(
                        "non_finite_result",
                        "relation contains a non-finite REAL value that cannot enter canonical state",
                        details={"table": table_name, "column": column.name},
                    )
            if column.canonical_type in {"DATE", "DATETIME"}:
                parser = date.fromisoformat if column.canonical_type == "DATE" else datetime.fromisoformat
                cursor = self.connection.execute(
                    f"SELECT {expression} FROM {table_sql} WHERE {expression} IS NOT NULL"
                )
                for (value,) in cursor:
                    try:
                        parsed = parser(value)
                    except (TypeError, ValueError) as exc:
                        raise RelAlgExecutionError(
                            "canonical_type_mismatch",
                            f"stored value is not a valid {column.canonical_type} literal",
                            details={"table": table_name, "column": column.name},
                        ) from exc
                    if column.canonical_type == "DATE" and isinstance(parsed, datetime):
                        raise RelAlgExecutionError(
                            "canonical_type_mismatch",
                            "DATE values must not contain a time component",
                            details={"table": table_name, "column": column.name},
                        )

    def _relation(self, table: Any) -> SourceRelation | RelationArtifact:
        if not isinstance(table, str) or not table:
            raise RelAlgValidationError(
                "invalid_arguments", "table handle must be a non-empty string", path="arguments.table"
            )
        # EnvironmentState is the sole authority for active/inactive handles.
        return self.state.get_relation(table)

    def _publish(self, artifact: RelationArtifact) -> None:
        self.state.add_artifact(artifact)

    def _new_size_tracker(
        self,
        *,
        operation: str,
        max_rows: int | None = None,
    ) -> ResultSizeTracker:
        return ResultSizeTracker(
            max_rows=self.max_rows if max_rows is None else max_rows,
            max_artifact_bytes=self.max_artifact_bytes,
            max_cell_bytes=self.max_cell_bytes,
            operation=operation,
        )

    def _validate_relation_size(
        self,
        relation: str | SourceRelation | RelationArtifact,
        columns: Sequence[Column],
        *,
        operation: str,
        include_hidden: bool = False,
    ) -> ResultSizeTracker:
        """Stream one relation through the deterministic row/storage guard."""

        if isinstance(relation, str):
            table_sql = (
                f"{quote_identifier('temp')}."
                f"{quote_identifier(artifact_backing_name(relation))}"
            )
        else:
            table_sql = _relation_sql(relation)
        selected = [quote_identifier(column.name) for column in columns]
        if include_hidden:
            selected.append(quote_identifier(_HIDDEN_ORDINAL))
        cursor = self.connection.execute(
            f"SELECT {', '.join(selected)} FROM {table_sql}"
        )
        tracker = self._new_size_tracker(operation=operation)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            tracker.add_row(row)
        return tracker

    def _sqlite_result_too_large(self, operation: str) -> CheckpointRelalgError:
        return CheckpointRelalgError(
            "resource_limit_error",
            "result_too_large",
            "SQLite rejected a value or encoded row above the configured storage guard",
            {
                "operation": operation,
                "max_artifact_bytes": self.max_artifact_bytes,
                "max_cell_bytes": self.max_cell_bytes,
                "byte_accounting": ARTIFACT_BYTE_ACCOUNTING_VERSION,
            },
        )

    def _state_hash(self) -> str:
        value = self.state.logical_hash()
        return str(value)

    def _new_savepoint(self) -> str:
        self._savepoint_counter += 1
        return f"relalg_call_{self._savepoint_counter}"

    def _allocate_handle(self, kind: str, requested: str | None) -> str:
        if requested is not None:
            handle = require_simple_identifier(requested, path="artifact_handle")
            if (
                self._physical_relation_exists(handle)
                or self._physical_relation_exists(artifact_backing_name(handle))
                or self._logical_relation_exists(handle)
            ):
                raise RelAlgStateValidationError(
                    "artifact_handle_exists", f"artifact handle {handle!r} already exists"
                )
            return handle
        while True:
            handle = self.state.allocate_artifact_handle(kind)
            if (
                not self._physical_relation_exists(handle)
                and not self._physical_relation_exists(artifact_backing_name(handle))
                and not self._logical_relation_exists(handle)
            ):
                return handle

    def _logical_relation_exists(self, handle: str) -> bool:
        target = sqlite_identifier_key(handle)
        for attribute in ("artifacts", "sources", "active_artifact_ids", "inactive_artifact_ids"):
            value = getattr(self.state, attribute, None)
            names = value.keys() if isinstance(value, Mapping) else value
            if isinstance(names, (set, frozenset, list, tuple)) or hasattr(names, "__iter__"):
                try:
                    if any(
                        isinstance(name, str) and sqlite_identifier_key(name) == target
                        for name in names
                    ):
                        return True
                except TypeError:
                    pass
        try:
            self.state.get_relation(handle)
            return True
        except Exception:
            # Includes unknown and inactive handles. Physical collision checking
            # still protects abandoned TEMP artifacts.
            for attribute in ("artifacts", "sources", "active_artifact_ids", "inactive_artifact_ids"):
                value = getattr(self.state, attribute, None)
            return False

    def _physical_relation_exists(self, handle: str) -> bool:
        row = self.connection.execute(
            "SELECT 1 FROM sqlite_temp_master WHERE name=? COLLATE NOCASE UNION ALL "
            "SELECT 1 FROM sqlite_master WHERE name=? COLLATE NOCASE LIMIT 1",
            (handle, handle),
        ).fetchone()
        return row is not None

    def _physical_columns(self, handle: str) -> tuple[str, ...]:
        backing = artifact_backing_name(handle)
        cursor = self.connection.execute(
            f"SELECT * FROM {quote_identifier('temp')}.{quote_identifier(backing)} LIMIT 0"
        )
        return tuple(item[0] for item in (cursor.description or ()))

    def _preserved_ordinal(
        self, relation: SourceRelation | RelationArtifact, table_alias: str
    ) -> str | None:
        if not _ordered_by(relation):
            return None
        if _HIDDEN_ORDINAL not in self._physical_columns(_relation_name(relation)):
            return None
        return f"{quote_identifier(table_alias)}.{quote_identifier(_HIDDEN_ORDINAL)}"

    @staticmethod
    def _typed_column_sql(column: Column, alias: str) -> str:
        expression = f"{quote_identifier(alias)}.{quote_identifier(column.name)}"
        if column.canonical_type in {"TEXT", "DATE", "DATETIME"}:
            expression = f"({expression} COLLATE BINARY)"
        return expression

    @classmethod
    def _visible_select(cls, columns: Sequence[Column], alias: str) -> list[str]:
        return [
            f"{cls._typed_column_sql(column, alias)} AS {quote_identifier(column.name)}"
            for column in columns
        ]

    def _validate_order_keys(
        self, raw: Any, columns: Sequence[Column], *, path: str, minimum: int
    ) -> list[dict[str, str]]:
        values = _array(raw, path=path, minimum=minimum)
        types = column_type_map(columns)
        result: list[dict[str, str]] = []
        for index, raw_key in enumerate(values):
            item_path = f"{path}[{index}]"
            key = _keys(
                raw_key, required={"column", "direction"}, optional={"nulls"}, path=item_path
            )
            column = key["column"]
            if column not in types:
                raise RelAlgStateValidationError(
                    "unknown_column", f"unknown exact ordering column {column!r}", path=f"{item_path}.column"
                )
            if types[column] == "BLOB":
                raise RelAlgValidationError(
                    "type_mismatch",
                    "BLOB columns support projection and equality, not ordering",
                    path=f"{item_path}.column",
                )
            direction, nulls = key["direction"], key.get("nulls", "last")
            if direction not in {"asc", "desc"}:
                raise RelAlgValidationError(
                    "invalid_ordering", "direction must be asc or desc", path=f"{item_path}.direction"
                )
            if nulls not in {"first", "last"}:
                raise RelAlgValidationError(
                    "invalid_ordering", "nulls must be first or last", path=f"{item_path}.nulls"
                )
            result.append({"column": column, "direction": direction, "nulls": nulls})
        return result

    @classmethod
    def _order_terms(
        cls,
        keys: Sequence[Mapping[str, str]],
        alias: str,
        columns: Sequence[Column],
    ) -> list[str]:
        terms: list[str] = []
        types = {column.name: column for column in columns}
        for key in keys:
            expression = cls._typed_column_sql(types[key["column"]], alias)
            null_rank = "0" if key["nulls"] == "first" else "1"
            nonnull_rank = "1" if key["nulls"] == "first" else "0"
            terms.append(
                f"CASE WHEN {expression} IS NULL THEN {null_rank} ELSE {nonnull_rank} END ASC"
            )
            terms.append(f"{expression} {key['direction'].upper()}")
        return terms

    @staticmethod
    def _infer_direct_columns(
        names: Sequence[str],
        rows: Sequence[Sequence[Any]],
        *,
        declared_types: Sequence[str | None] | None = None,
    ) -> list[Column]:
        from ..expression import literal_type

        inferred = list(declared_types or [None] * len(names))
        if len(inferred) != len(names):
            raise RelAlgExecutionError(
                "invalid_sql_output", "Direct SQL declared schema width is inconsistent"
            )
        for row_index, row in enumerate(rows):
            if len(row) != len(names):
                raise RelAlgExecutionError(
                    "invalid_sql_output", "Direct SQL returned inconsistent row width"
                )
            for index, value in enumerate(row):
                value_type = literal_type(value)
                declared = inferred[index]
                if value is None:
                    continue
                if declared == "DATE" and value_type == "TEXT":
                    from ..expression import validate_iso_literal

                    validate_iso_literal(value, "DATE", path=f"result[{row_index}][{index}]")
                    continue
                if declared == "DATETIME" and value_type == "TEXT":
                    from ..expression import validate_iso_literal

                    validate_iso_literal(value, "DATETIME", path=f"result[{row_index}][{index}]")
                    continue
                if declared == "BOOLEAN" and value_type == "INTEGER" and value in {0, 1}:
                    continue
                try:
                    inferred[index] = merge_types(
                        declared or "NULL", value_type, path=f"result[{row_index}][{index}]"
                    )
                except RelAlgValidationError as exc:
                    raise RelAlgExecutionError(
                        "direct_sql_type_mismatch",
                        "Direct SQL produced incompatible dynamic types in one output column",
                        details={
                            "column": names[index],
                            "existing_type": declared or "NULL",
                            "value_type": value_type,
                        },
                    ) from exc
        return [
            Column(name=name, canonical_type=canonical_type or "NULL")
            for name, canonical_type in zip(names, inferred)
        ]

    def _direct_declared_types(
        self, sql: str, names: Sequence[str], handle: str
    ) -> list[str | None]:
        """Recover SQLite-declared types for empty/all-NULL direct results."""

        view = f"__relalg_schema_{handle}"
        suffix = 0
        while self._physical_relation_exists(view):
            suffix += 1
            view = f"__relalg_schema_{handle}_{suffix}"
        try:
            self.connection.execute(
                f"CREATE TEMP VIEW {quote_identifier(view)} AS {sql}"
            )
            rows = self.connection.execute(
                f"PRAGMA temp.table_info({quote_identifier(view)})"
            ).fetchall()
        finally:
            self.connection.execute(f"DROP VIEW IF EXISTS {quote_identifier(view)}")
        if len(rows) != len(names):
            return [None] * len(names)
        return [self._sqlite_declared_to_canonical(row[2]) for row in rows]

    @staticmethod
    def _sqlite_declared_to_canonical(declared: Any) -> str | None:
        if not isinstance(declared, str) or not declared.strip():
            return None
        normalized = declared.strip().upper()
        base = normalized.split("(", 1)[0].strip()
        if "DATETIME" in base or "TIMESTAMP" in base:
            return "DATETIME"
        if base == "DATE":
            return "DATE"
        if "BOOL" in base:
            return "BOOLEAN"
        if "INT" in base:
            return "INTEGER"
        if any(token in base for token in ("CHAR", "CLOB", "TEXT")):
            return "TEXT"
        if "BLOB" in base:
            return "BLOB"
        if any(token in base for token in ("REAL", "FLOA", "DOUB", "NUMERIC", "DECIMAL")):
            return "REAL"
        return None

    def _materialize_direct_rows(
        self,
        handle: str,
        columns: Sequence[Column],
        rows: Sequence[Sequence[Any]],
        *,
        ordered: bool,
    ) -> None:
        sqlite_types = {
            "NULL": "",
            "BOOLEAN": "INTEGER",
            "INTEGER": "INTEGER",
            "REAL": "REAL",
            "TEXT": "TEXT",
            "DATE": "TEXT",
            "DATETIME": "TEXT",
            "BLOB": "BLOB",
        }
        declarations = [
            f"{quote_identifier(column.name)} {sqlite_types[column.canonical_type]}".rstrip()
            for column in columns
        ]
        if ordered:
            declarations.append(f"{quote_identifier(_HIDDEN_ORDINAL)} INTEGER NOT NULL")
        backing = artifact_backing_name(handle)
        self.connection.execute(
            f"CREATE TEMP TABLE {quote_identifier(backing)} ({', '.join(declarations)})"
        )
        if self._physical_columns(handle) != tuple(column.name for column in columns) + (
            (_HIDDEN_ORDINAL,) if ordered else ()
        ):
            raise RelAlgExecutionError(
                "invalid_materialized_schema",
                "SQLite materialized a Direct artifact schema different from its validated schema",
            )
        if rows:
            width = len(columns) + (1 if ordered else 0)
            placeholders = ", ".join("?" for _ in range(width))
            if ordered:
                values = [tuple(row) + (index,) for index, row in enumerate(rows, start=1)]
            else:
                values = [tuple(row) for row in rows]
            self.connection.executemany(
                f"INSERT INTO {quote_identifier('temp')}.{quote_identifier(backing)} "
                f"VALUES ({placeholders})",
                values,
            )
        visible = ", ".join(quote_identifier(column.name) for column in columns)
        self.connection.execute(
            f"CREATE TEMP VIEW {quote_identifier(handle)} AS SELECT {visible} FROM "
            f"{quote_identifier('temp')}.{quote_identifier(backing)}"
        )

    @classmethod
    def _direct_ordering(
        cls, sql: str, output_names: Sequence[str]
    ) -> tuple[bool, tuple[OrderingKey, ...]]:
        """Parse a conservative top-level ORDER BY into visible ordering keys."""

        tokens = cls._top_level_tokens(sql)
        order_index: int | None = None
        for index in range(len(tokens) - 1):
            if tokens[index][0] == "ORDER" and tokens[index + 1][0] == "BY":
                order_index = index
        if order_index is None:
            return False, ()
        clause_start = tokens[order_index + 1][2]
        clause_end = len(sql)
        for word, start, _ in tokens[order_index + 2 :]:
            if word in {"LIMIT", "OFFSET", "FETCH"}:
                clause_end = start
                break
        clause = sql[clause_start:clause_end].strip().rstrip(";").strip()
        terms = cls._split_top_level(clause)
        parsed: list[OrderingKey] = []
        for raw_term in terms:
            term = raw_term.strip()
            nulls_match = re.search(r"\s+NULLS\s+(FIRST|LAST)\s*$", term, re.I)
            nulls: str | None = None
            if nulls_match:
                nulls = nulls_match.group(1).lower()
                term = term[: nulls_match.start()].strip()
            direction_match = re.search(r"\s+(ASC|DESC)\s*$", term, re.I)
            direction = "asc"
            if direction_match:
                direction = direction_match.group(1).lower()
                term = term[: direction_match.start()].strip()
            if nulls is None:
                nulls = "first" if direction == "asc" else "last"
            if re.fullmatch(r"\d+", term):
                ordinal = int(term)
                if not 1 <= ordinal <= len(output_names):
                    return True, ()
                column = output_names[ordinal - 1]
            else:
                column = cls._unquote_sql_identifier(term)
                if column not in output_names:
                    return True, ()
            parsed.append(OrderingKey(column=column, direction=direction, nulls=nulls))
        return True, tuple(parsed) if parsed else ()

    @staticmethod
    def _unquote_sql_identifier(value: str) -> str:
        # A qualified ORDER BY reference is representable when its final exact
        # component is one of the unique output columns.
        if "." in value and not (len(value) >= 2 and value[0] in {'"', "`", "["}):
            parts = value.split(".")
            if all(parts):
                return SQLiteRelationalExecutor._unquote_sql_identifier(parts[-1])
        if len(value) >= 2 and value[0] == value[-1] == '"':
            return value[1:-1].replace('""', '"')
        if len(value) >= 2 and value[0] == value[-1] == "`":
            return value[1:-1].replace("``", "`")
        if len(value) >= 2 and value[0] == "[" and value[-1] == "]":
            return value[1:-1].replace("]]", "]")
        return value if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value) else ""

    @staticmethod
    def _top_level_tokens(sql: str) -> list[tuple[str, int, int]]:
        tokens: list[tuple[str, int, int]] = []
        index, depth = 0, 0
        length = len(sql)
        while index < length:
            char = sql[index]
            if char in {"'", '"', "`"}:
                quote = char
                index += 1
                while index < length:
                    if sql[index] == quote:
                        if index + 1 < length and sql[index + 1] == quote:
                            index += 2
                            continue
                        index += 1
                        break
                    index += 1
                continue
            if char == "[":
                closing = sql.find("]", index + 1)
                index = length if closing < 0 else closing + 1
                continue
            if sql.startswith("--", index):
                newline = sql.find("\n", index + 2)
                index = length if newline < 0 else newline + 1
                continue
            if sql.startswith("/*", index):
                closing = sql.find("*/", index + 2)
                index = length if closing < 0 else closing + 2
                continue
            if char == "(":
                depth += 1
                index += 1
                continue
            if char == ")":
                depth = max(0, depth - 1)
                index += 1
                continue
            if depth == 0 and (char.isalpha() or char == "_"):
                start = index
                index += 1
                while index < length and (sql[index].isalnum() or sql[index] == "_"):
                    index += 1
                tokens.append((sql[start:index].upper(), start, index))
                continue
            index += 1
        return tokens

    @staticmethod
    def _split_top_level(value: str) -> list[str]:
        parts: list[str] = []
        start = 0
        index, depth = 0, 0
        while index < len(value):
            char = value[index]
            if char in {"'", '"', "`"}:
                quote = char
                index += 1
                while index < len(value):
                    if value[index] == quote:
                        if index + 1 < len(value) and value[index + 1] == quote:
                            index += 2
                            continue
                        index += 1
                        break
                    index += 1
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth = max(0, depth - 1)
            elif char == "," and depth == 0:
                parts.append(value[start:index])
                start = index + 1
            index += 1
        parts.append(value[start:])
        return [part for part in parts if part.strip()]

    # ------------------------------------------------------------------
    # Canonical scalar functions (avoid SQLite's accidental coercions)
    # ------------------------------------------------------------------
    def _register_scalar_functions(self) -> None:
        self.connection.create_function("_relalg_divide", 2, self._divide, deterministic=True)
        self.connection.create_function("_relalg_cast_null", 1, lambda value: None, deterministic=True)
        self.connection.create_function("_relalg_cast_boolean", 1, self._cast_boolean, deterministic=True)
        self.connection.create_function("_relalg_cast_integer", 1, self._cast_integer, deterministic=True)
        self.connection.create_function("_relalg_cast_real", 1, self._cast_real, deterministic=True)
        self.connection.create_function("_relalg_cast_text", 1, self._cast_text, deterministic=True)
        self.connection.create_function("_relalg_cast_date", 1, self._cast_date, deterministic=True)
        self.connection.create_function("_relalg_cast_datetime", 1, self._cast_datetime, deterministic=True)
        self.connection.create_function("_relalg_cast_blob", 1, self._cast_blob, deterministic=True)
        self.connection.create_function("_relalg_extract_year", 1, self._extract_year, deterministic=True)
        self.connection.create_function("_relalg_extract_month", 1, self._extract_month, deterministic=True)
        self.connection.create_function("_relalg_extract_day", 1, self._extract_day, deterministic=True)
        self.connection.create_function("_relalg_date_diff_days", 2, self._date_diff_days, deterministic=True)

    def _divide(self, numerator: Any, denominator: Any) -> float | None:
        if numerator is None or denominator is None:
            return None
        if denominator == 0:
            self._scalar_error = CheckpointRelalgError(
                "execution_error",
                "divide_by_zero",
                "typed divide denominator evaluated to zero",
            )
            raise ValueError("typed divide denominator is zero")
        return float(numerator) / float(denominator)

    @staticmethod
    def _cast_boolean(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or value in (0, 1):
            return int(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
            return 1 if value.strip().lower() in {"true", "1"} else 0
        raise ValueError("value cannot be cast to BOOLEAN")

    @staticmethod
    def _cast_integer(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer() and math.isfinite(value):
            return int(value)
        if isinstance(value, str) and re.fullmatch(r"[+-]?\d+", value.strip()):
            return int(value.strip())
        raise ValueError("value cannot be cast to INTEGER")

    @staticmethod
    def _cast_real(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        if isinstance(value, str):
            try:
                result = float(value.strip())
            except ValueError as exc:
                raise ValueError("value cannot be cast to REAL") from exc
            if math.isfinite(result):
                return result
        raise ValueError("value cannot be cast to REAL")

    @staticmethod
    def _cast_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return str(value)
        raise ValueError("value cannot be cast to TEXT")

    @staticmethod
    def _cast_date(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str) or "T" in value or " " in value:
            raise ValueError("value cannot be cast to DATE")
        return date.fromisoformat(value).isoformat()

    @staticmethod
    def _cast_datetime(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("value cannot be cast to DATETIME")
        return datetime.fromisoformat(value.replace("Z", "+00:00")).isoformat()

    @staticmethod
    def _cast_blob(value: Any) -> bytes | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value
        if isinstance(value, str):
            return value.encode("utf-8")
        raise ValueError("value cannot be cast to BLOB")

    @staticmethod
    def _parse_temporal(value: Any) -> date | datetime | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("temporal value is not ISO-8601 text")
        if "T" in value or " " in value:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        return date.fromisoformat(value)

    @classmethod
    def _extract_year(cls, value: Any) -> int | None:
        parsed = cls._parse_temporal(value)
        return None if parsed is None else parsed.year

    @classmethod
    def _extract_month(cls, value: Any) -> int | None:
        parsed = cls._parse_temporal(value)
        return None if parsed is None else parsed.month

    @classmethod
    def _extract_day(cls, value: Any) -> int | None:
        parsed = cls._parse_temporal(value)
        return None if parsed is None else parsed.day

    @classmethod
    def _date_diff_days(cls, start: Any, end: Any) -> float | None:
        parsed_start, parsed_end = cls._parse_temporal(start), cls._parse_temporal(end)
        if parsed_start is None or parsed_end is None:
            return None
        if isinstance(parsed_start, date) and not isinstance(parsed_start, datetime):
            parsed_start = datetime.combine(parsed_start, datetime.min.time())
        if isinstance(parsed_end, date) and not isinstance(parsed_end, datetime):
            parsed_end = datetime.combine(parsed_end, datetime.min.time())
        return (parsed_end - parsed_start).total_seconds() / 86_400.0


__all__ = ["SQLiteRelationalExecutor", "artifact_backing_name", "quote_identifier"]
