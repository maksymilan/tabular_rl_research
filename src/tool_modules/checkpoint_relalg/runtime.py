"""Shared Direct/Atomic/Hybrid episode runtime for checkpoint-relalg-v1."""

from __future__ import annotations

import re
import sqlite3
import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .checkpoint_store import CheckpointStore
from .environment_renderer import EnvironmentRenderer
from .environment_state import EnvironmentState, Observation, StateError, StepRecord
from .errors import CheckpointRelalgError, structured_error
from .executors import SQLiteRelationalExecutor
from .expression import RelAlgValidationError, column_type_map, quote_identifier
from .predicate import PredicateCompiler
from .protocol import ATOMIC_TOOLS, ProtocolValidationError, normalize_mode, validate_tool_call
from .relation_artifact import (
    Column,
    ForeignKey,
    OrderingKey,
    Relation,
    RelationArtifact,
    SourceRelation,
    sqlite_identifier_key,
)


_DECLARED_TYPE_TOKEN = re.compile(r"^[A-Za-z]+")


def canonical_sqlite_type(declared_type: Any) -> str:
    """Map one declared SQLite type to the frozen canonical type family."""

    text = str(declared_type or "").strip().upper()
    token_match = _DECLARED_TYPE_TOKEN.match(text)
    token = token_match.group(0) if token_match else ""
    if "BOOL" in token:
        return "BOOLEAN"
    if token == "DATE":
        return "DATE"
    if "DATE" in token or "TIME" in token:
        return "DATETIME"
    if "INT" in token:
        return "INTEGER"
    if any(marker in token for marker in ("CHAR", "CLOB", "TEXT", "JSON")):
        return "TEXT"
    if any(marker in token for marker in ("REAL", "FLOA", "DOUB", "NUM", "DEC")):
        return "REAL"
    if not token or "BLOB" in token:
        return "BLOB"
    # SQLite assigns NUMERIC affinity to other declared types.  Canonical v1
    # does not expose NUMERIC, so treat it as REAL rather than guessing TEXT.
    return "REAL"


def load_source_catalog(connection: sqlite3.Connection) -> dict[str, SourceRelation]:
    """Read immutable source metadata without making column schemas model-visible."""

    table_rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name COLLATE BINARY"
    ).fetchall()
    metadata: dict[str, dict[str, Any]] = {}
    for (table,) in table_rows:
        info = connection.execute(
            f"PRAGMA main.table_info({quote_identifier(table)})"
        ).fetchall()
        columns = tuple(
            Column(name=row[1], canonical_type=canonical_sqlite_type(row[2])) for row in info
        )
        primary_key = tuple(
            row[1] for row in sorted((row for row in info if row[5]), key=lambda row: row[5])
        )
        fk_rows = connection.execute(
            f"PRAGMA main.foreign_key_list({quote_identifier(table)})"
        ).fetchall()
        row_count = int(
            connection.execute(
                f"SELECT COUNT(*) FROM {quote_identifier('main')}.{quote_identifier(table)}"
            ).fetchone()[0]
        )
        metadata[table] = {
            "columns": columns,
            "primary_key": primary_key,
            "foreign_key_rows": fk_rows,
            "row_count": row_count,
        }

    canonical_tables = {sqlite_identifier_key(name): name for name in metadata}
    sources: dict[str, SourceRelation] = {}
    for table, table_metadata in metadata.items():
        grouped: dict[int, list[Sequence[Any]]] = {}
        for row in table_metadata["foreign_key_rows"]:
            grouped.setdefault(int(row[0]), []).append(row)
        foreign_keys: list[ForeignKey] = []
        for fk_id in sorted(grouped):
            rows = sorted(grouped[fk_id], key=lambda row: row[1])
            declared_parent = rows[0][2]
            parent_name = (
                canonical_tables.get(sqlite_identifier_key(declared_parent))
                if isinstance(declared_parent, str)
                else None
            )
            if parent_name is None:
                raise CheckpointRelalgError(
                    "nonrecoverable_execution_error",
                    "invalid_catalog",
                    "foreign key references a source table absent from the catalog",
                    {"table": table, "foreign_key_id": fk_id},
                )
            child_columns = tuple(row[3] for row in rows)
            if any(not isinstance(name, str) or not name for name in child_columns):
                raise CheckpointRelalgError(
                    "nonrecoverable_execution_error",
                    "invalid_catalog",
                    "foreign key has an invalid child-column definition",
                    {"table": table, "foreign_key_id": fk_id},
                )
            raw_targets = tuple(row[4] for row in rows)
            if all(target is None for target in raw_targets):
                parent_key = tuple(metadata[parent_name]["primary_key"])
                if not parent_key or len(parent_key) != len(rows):
                    raise CheckpointRelalgError(
                        "nonrecoverable_execution_error",
                        "invalid_catalog",
                        "implicit foreign-key target cannot be resolved to the parent primary key",
                        {"table": table, "foreign_key_id": fk_id, "ref_table": parent_name},
                    )
                target_columns = parent_key
            elif any(target is None for target in raw_targets):
                raise CheckpointRelalgError(
                    "nonrecoverable_execution_error",
                    "invalid_catalog",
                    "foreign key mixes explicit and implicit target columns",
                    {"table": table, "foreign_key_id": fk_id},
                )
            else:
                target_columns = tuple(str(target) for target in raw_targets)
            foreign_keys.append(
                ForeignKey(
                    columns=tuple(str(name) for name in child_columns),
                    ref_table=parent_name,
                    ref_columns=target_columns,
                )
            )
        sources[table] = SourceRelation(
            name=table,
            columns=table_metadata["columns"],
            row_count=table_metadata["row_count"],
            primary_key=table_metadata["primary_key"],
            foreign_keys=tuple(foreign_keys),
        )
    return sources


def _relation_sql(relation: Relation) -> str:
    schema = "main" if isinstance(relation, SourceRelation) else "temp"
    return f"{quote_identifier(schema)}.{quote_identifier(relation.table)}"


@dataclass(frozen=True)
class RuntimeConfig:
    max_primitive_calls: int = 30
    max_checkpoints: int = 8
    max_restores: int = 3
    sql_timeout_seconds: float = 20.0
    max_artifact_rows: int = 100_000
    max_artifact_bytes: int = 64 * 1024 * 1024
    max_cell_bytes: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        integer_fields = (
            "max_primitive_calls",
            "max_checkpoints",
            "max_restores",
            "max_artifact_rows",
            "max_artifact_bytes",
            "max_cell_bytes",
        )
        for field_name in integer_fields:
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if (
            self.max_primitive_calls < 1
            or self.max_artifact_rows < 1
            or self.max_artifact_bytes < 1
            or self.max_cell_bytes < 1
        ):
            raise ValueError("primitive-call and artifact storage limits must be positive")
        if self.max_cell_bytes > self.max_artifact_bytes:
            raise ValueError("max_cell_bytes must not exceed max_artifact_bytes")
        if (
            isinstance(self.sql_timeout_seconds, bool)
            or not isinstance(self.sql_timeout_seconds, (int, float))
            or not math.isfinite(float(self.sql_timeout_seconds))
            or self.sql_timeout_seconds <= 0
        ):
            raise ValueError("sql_timeout_seconds must be a positive finite number")


class CheckpointRelalgRuntime:
    """One causal episode; all modes share this exact state/control implementation."""

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        mode: str,
        config: RuntimeConfig | None = None,
        current_targets: Sequence[str] = (),
    ) -> None:
        self.connection = connection
        self.mode = normalize_mode(mode)
        self.config = config or RuntimeConfig()
        self.state = EnvironmentState(
            load_source_catalog(connection),
            current_targets=current_targets,
        )
        self.checkpoints = CheckpointStore(
            self.state,
            max_checkpoints=self.config.max_checkpoints,
            max_restores=self.config.max_restores,
        )
        self.executor = SQLiteRelationalExecutor(
            connection,
            self.state,
            max_rows=self.config.max_artifact_rows,
            max_artifact_bytes=self.config.max_artifact_bytes,
            max_cell_bytes=self.config.max_cell_bytes,
            timeout_seconds=self.config.sql_timeout_seconds,
        )
        self.renderer = EnvironmentRenderer()
        self.primitive_calls = 0
        self.error_count = 0
        self.done = False
        self.terminal_table: str | None = None
        self.failure_type: str | None = None

    def render_context(self, question: str, external_knowledge: Any = None) -> str:
        return self.renderer.render(
            question,
            external_knowledge,
            self.state,
            self.checkpoints,
        )

    def apply(self, tool: Any, arguments: Any) -> dict[str, Any]:
        """Validate and execute exactly one canonical model call."""

        if self.done:
            raise RuntimeError("episode is already complete")
        step_id = self.state.allocate_step_id()
        before_hash = self.state.logical_hash()
        phase_before = self.state.phase_id
        checkpoint_before = self.state.checkpoint_id
        self.primitive_calls += 1
        if self.primitive_calls > self.config.max_primitive_calls:
            return self._record_error(
                step_id,
                tool,
                arguments,
                CheckpointRelalgError(
                    "resource_limit_error",
                    "primitive_call_limit_reached",
                    "primitive call budget exhausted",
                    {"max_primitive_calls": self.config.max_primitive_calls},
                ),
                before_hash,
                terminal=True,
            )
        try:
            validated = validate_tool_call(self.mode, tool, arguments)
            output, terminal = self._dispatch(str(tool), validated)
            artifact_payload = output.get("artifact") if isinstance(output, Mapping) else None
            artifact_table = (
                artifact_payload.get("table") if isinstance(artifact_payload, Mapping) else None
            )
            self.state.add_step(
                StepRecord(
                    step_id=step_id,
                    tool=str(tool),
                    status="success",
                    output=deepcopy(output),
                    phase_id=phase_before,
                    checkpoint_id=checkpoint_before,
                    produced_artifact_ids=(artifact_table,)
                    if isinstance(artifact_table, str)
                    else (),
                )
            )
            after_hash = self.state.logical_hash()
            envelope = {
                "step_id": step_id,
                "status": "success",
                **deepcopy(output),
                "environment_state_hash": after_hash,
            }
            if terminal:
                self.done = True
            return envelope
        except ProtocolValidationError as exc:
            error = CheckpointRelalgError(
                "argument_validation_error",
                exc.code,
                exc.message,
                {"argument_path": exc.path},
            )
            return self._record_error(step_id, tool, arguments, error, before_hash)
        except BaseException as exc:  # noqa: BLE001
            error = structured_error(exc)
            return self._record_error(
                step_id,
                tool,
                arguments,
                error,
                before_hash,
                terminal=error.error_type == "nonrecoverable_execution_error",
            )

    def reject_native_turn(
        self,
        *,
        code: str,
        message: str,
        details: Mapping[str, Any] | None = None,
        attempted_tool: Any = None,
        attempted_arguments: Any = None,
    ) -> dict[str, Any]:
        """Record one state-preserving native carrier/shape rejection."""

        if self.done:
            raise RuntimeError("episode is already complete")
        step_id = self.state.allocate_step_id()
        before_hash = self.state.logical_hash()
        self.primitive_calls += 1
        if self.primitive_calls > self.config.max_primitive_calls:
            return self._record_error(
                step_id,
                attempted_tool,
                attempted_arguments,
                CheckpointRelalgError(
                    "resource_limit_error",
                    "primitive_call_limit_reached",
                    "primitive call budget exhausted",
                    {"max_primitive_calls": self.config.max_primitive_calls},
                ),
                before_hash,
                terminal=True,
            )
        return self._record_error(
            step_id,
            attempted_tool,
            attempted_arguments,
            CheckpointRelalgError("protocol_error", code, message, dict(details or {})),
            before_hash,
        )

    def _record_error(
        self,
        step_id: str,
        tool: Any,
        arguments: Any,
        error: CheckpointRelalgError,
        before_hash: str,
        *,
        terminal: bool = False,
    ) -> dict[str, Any]:
        actual_hash = self.state.logical_hash()
        if actual_hash != before_hash and error.error_type != "nonrecoverable_execution_error":
            error = CheckpointRelalgError(
                "nonrecoverable_execution_error",
                "state_changed_after_failure",
                "failed call changed logical EnvironmentState",
                {"before_hash": before_hash, "after_hash": actual_hash},
            )
            terminal = True
        envelope = {
            "step_id": step_id,
            "status": "error",
            "error": error.as_dict(),
            "attempted_action": {
                "tool": tool,
                "arguments": deepcopy(arguments),
            },
            "environment_state_hash": before_hash,
        }
        self.state.record_step(
            str(tool or "native_tool_call"),
            "error",
            None,
            deepcopy(envelope["error"]),
            step_id=step_id,
        )
        if self.state.logical_hash() != before_hash:
            envelope["error"] = CheckpointRelalgError(
                "nonrecoverable_execution_error",
                "state_changed_after_failure",
                "recording a failed call changed logical EnvironmentState",
            ).as_dict()
            terminal = True
        self.error_count += 1
        self.state.set_last_error(envelope)
        if terminal:
            self.done = True
            self.failure_type = envelope["error"]["type"]
        return envelope

    def _dispatch(self, tool: str, args: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
        if tool == "describe_table":
            return self._describe_table(args), False
        if tool == "inspect_column":
            return self._inspect_column(args), False
        if tool == "read_rows":
            return self._read_rows(args), False
        if tool == "execute_sql":
            artifact = self.executor.execute_sql(
                args["sql"],
                timeout_seconds=self.config.sql_timeout_seconds,
                max_rows=self.config.max_artifact_rows,
            )
            return {"artifact": artifact.to_payload()}, False
        if tool in ATOMIC_TOOLS:
            artifact = self.executor.execute(tool, args)
            return {"artifact": artifact.to_payload()}, False
        if tool == "commit_checkpoint":
            node = self.checkpoints.commit(
                args["progress_summary"],
                args["remaining_uncertainties"],
                args["next_targets"],
            )
            return {
                "checkpoint_id": node.checkpoint_id,
                "next_phase_started": True,
                "phase_id": self.state.phase_id,
                "phase_transition": "commit",
            }, False
        if tool == "restore_checkpoint":
            node = self.checkpoints.restore(
                args["checkpoint_id"],
                args["reason"],
                args["next_targets"],
            )
            return {
                "restored_from": node.restored_from,
                "checkpoint_id": node.checkpoint_id,
                "abandoned_checkpoints": list(node.abandoned_checkpoints),
                "next_phase_started": True,
                "phase_id": self.state.phase_id,
                "phase_transition": "restore",
            }, False
        if tool == "answer":
            relation = self.state.get_relation(args["table"])
            self.executor.validate_relation_values(relation.table)
            self.terminal_table = relation.table
            return {
                "table": relation.table,
                "columns": [column.name for column in relation.columns],
                "row_count": relation.row_count,
                "episode_ended": True,
            }, True
        raise AssertionError(f"unhandled validated tool {tool!r}")

    def _describe_table(self, args: Mapping[str, Any]) -> dict[str, Any]:
        relations = [self.state.get_relation(table) for table in args["tables"]]
        for relation in relations:
            if isinstance(relation, SourceRelation):
                self.state.discover_schema(relation.name)
        return {
            "tables": [
                relation.to_payload()
                if isinstance(relation, RelationArtifact)
                else relation.to_payload()
                for relation in relations
            ]
        }

    def _require_column(self, relation: Relation, column: Any, *, path: str) -> Column:
        for candidate in relation.columns:
            if candidate.name == column:
                return candidate
        raise CheckpointRelalgError(
            "state_validation_error",
            "unknown_column",
            f"unknown exact column {column!r}",
            {
                "argument_path": path,
                "requested": column,
                "available_columns": [item.name for item in relation.columns],
            },
        )

    def _inspect_column(self, args: Mapping[str, Any]) -> dict[str, Any]:
        relation = self.state.get_relation(args["table"])
        column = self._require_column(relation, args["column"], path="column")
        top_k = int(args.get("top_k", 10))
        table_sql = _relation_sql(relation)
        raw_column_sql = quote_identifier(column.name)
        column_sql = raw_column_sql
        if column.canonical_type in {"TEXT", "DATE", "DATETIME"}:
            column_sql = f"({column_sql} COLLATE BINARY)"
        def query() -> tuple[Any, Any, list[tuple[Any, ...]]]:
            distinct_count, null_count = self.connection.execute(
                f"SELECT COUNT(DISTINCT {column_sql}), "
                f"SUM(CASE WHEN {raw_column_sql} IS NULL THEN 1 ELSE 0 END) FROM {table_sql}"
            ).fetchone()
            rows = self.connection.execute(
                f"SELECT {column_sql} AS {raw_column_sql}, COUNT(*) AS frequency FROM {table_sql} "
                f"WHERE {raw_column_sql} IS NOT NULL GROUP BY {column_sql} "
                f"ORDER BY frequency DESC, typeof({raw_column_sql}) ASC, {column_sql} ASC LIMIT ?",
                (top_k,),
            ).fetchall()
            return distinct_count, null_count, rows

        distinct_count, null_count, rows = self.executor.run_readonly_queries(
            "inspect_column", query
        )
        self.executor.validate_observation_rows(
            [(row[0],) for row in rows],
            operation="inspect_column",
        )
        payload = {
            "column": column.name,
            "canonical_type": column.canonical_type,
            "distinct_count": int(distinct_count or 0),
            "has_null": bool(null_count),
            "values": [row[0] for row in rows],
            "frequencies": [int(row[1]) for row in rows],
            "complete": int(distinct_count or 0) <= top_k,
        }
        observation = Observation(
            self.state.allocate_observation_id(),
            "inspect_column",
            relation.table,
            payload,
        )
        self.state.add_observation(observation)
        return {"observation_id": observation.observation_id, **payload}

    def _read_rows(self, args: Mapping[str, Any]) -> dict[str, Any]:
        relation = self.state.get_relation(args["table"])
        all_columns = [column.name for column in relation.columns]
        selected = list(args.get("columns", all_columns))
        for index, name in enumerate(selected):
            self._require_column(relation, name, path=f"columns[{index}]")
        limit, offset = int(args.get("limit", 20)), int(args.get("offset", 0))
        order_by = list(args.get("order_by", []))
        if offset > 0 and not order_by:
            raise CheckpointRelalgError(
                "argument_validation_error",
                "invalid_arguments",
                "positive read_rows offset requires order_by",
                {"argument_path": "offset"},
            )
        params: tuple[Any, ...] = ()
        where = ""
        if "conditions" in args:
            compiled = PredicateCompiler(relation.columns, table_alias="src").compile(
                args["conditions"]
            )
            where = f" WHERE {compiled.sql}"
            params = compiled.params
        if order_by:
            ordering = self._validate_ordering(relation, order_by, path="order_by")
            order_clause = (
                " ORDER BY "
                + self._ordering_sql(ordering, relation=relation, alias="src")
            )
        else:
            ordering = []
            order_clause = ""
        selected_sql = ", ".join(
            f"{quote_identifier('src')}.{quote_identifier(name)}" for name in selected
        )
        sql = (
            f"SELECT {selected_sql} FROM {_relation_sql(relation)} AS "
            f"{quote_identifier('src')}{where}{order_clause} LIMIT ? OFFSET ?"
        )
        rows = self.executor.run_readonly_queries(
            "read_rows",
            lambda: self.connection.execute(
                sql, (*params, limit + 1, offset)
            ).fetchall(),
        )
        shown = rows[:limit]
        self.executor.validate_observation_rows(shown, operation="read_rows")
        payload = {
            "columns": selected,
            "rows": [list(row) for row in shown],
            "returned_count": len(shown),
            "has_more": len(rows) > limit,
            "offset": offset,
        }
        observation = Observation(
            self.state.allocate_observation_id(),
            "read_rows",
            relation.table,
            payload,
        )
        self.state.add_observation(observation)
        return {"observation_id": observation.observation_id, **payload}

    def _validate_ordering(
        self,
        relation: Relation,
        raw: Sequence[Mapping[str, Any]],
        *,
        path: str,
    ) -> list[OrderingKey]:
        result: list[OrderingKey] = []
        for index, item in enumerate(raw):
            column = self._require_column(
                relation, item["column"], path=f"{path}[{index}].column"
            )
            if column.canonical_type == "BLOB":
                raise CheckpointRelalgError(
                    "argument_validation_error",
                    "type_mismatch",
                    "BLOB columns support projection and equality, not ordering",
                    {"argument_path": f"{path}[{index}].column"},
                )
            result.append(
                OrderingKey(
                    column=item["column"],
                    direction=item["direction"],
                    nulls=item.get("nulls", "last"),
                )
            )
        return result

    @staticmethod
    def _ordering_sql(
        ordering: Sequence[OrderingKey], *, relation: Relation, alias: str
    ) -> str:
        terms: list[str] = []
        types = {column.name: column.canonical_type for column in relation.columns}
        for key in ordering:
            expression = f"{quote_identifier(alias)}.{quote_identifier(key.column)}"
            if types[key.column] in {"TEXT", "DATE", "DATETIME"}:
                expression = f"({expression} COLLATE BINARY)"
            terms.append(
                f"CASE WHEN {expression} IS NULL THEN "
                f"{'0' if key.nulls == 'first' else '1'} ELSE "
                f"{'1' if key.nulls == 'first' else '0'} END ASC"
            )
            terms.append(f"{expression} {key.direction.upper()}")
        return ", ".join(terms)

    def answer_rows(self) -> tuple[list[str], list[list[Any]]]:
        if self.terminal_table is None:
            raise RuntimeError("the episode has no terminal answer relation")
        return self.executor.collect_relation_rows(self.terminal_table)

    def audit_state(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "primitive_calls": self.primitive_calls,
            "errors": self.error_count,
            "done": self.done,
            "terminal_table": self.terminal_table,
            "failure_type": self.failure_type,
            "phase_id": self.state.phase_id,
            "checkpoint_id": self.state.checkpoint_id,
            "environment_state_hash": self.state.logical_hash(),
            "active_checkpoint_path": list(self.checkpoints.active_checkpoint_path),
            "checkpoint_history": self.checkpoints.export_history(include_snapshots=True),
            "checkpoint_count": self.checkpoints.checkpoint_count,
            "restore_count": self.checkpoints.restore_count,
        }


__all__ = [
    "CheckpointRelalgRuntime",
    "RuntimeConfig",
    "canonical_sqlite_type",
    "load_source_catalog",
]
