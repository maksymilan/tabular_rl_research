#!/usr/bin/env python3
"""Shared provenance: harness-authored reference edges (V2b unified schema).

Every trajectory step carries a `references` array of harness-derived edges — the model never
authors them. Both the offline emitter and the online rollout build edges through `build_references`,
over the SAME model-facing argument shape (a table ref is a real handle or a source name; a
`value_ref` is the producing step_id), so SFT and eval can never drift.

Edge schema (one dict per edge):
  {"type": "data"|"value"|"grounding", "step"|"source": id, "role": ..., "target": {...}}
  - data:   a table input        role=table|in_table   target={"handle":h} | {"table":t}
  - value:  a scalar threshold   role=value_ref        target={"column":c}
  - grounding: a harness observation grounding a schema, column-domain value, or row value.
`target` is a structured object so the process reward reads fields directly and never parses a
string. (Cross-table column lineage — rendering `T2__x` back to its source `table.column` — lands
with the grounding edges in V2c; this round a `value` edge's column is just the prefix-stripped name.)

`resolve_step(ref) -> step_id | None`: maps a model-facing reference (a produced table handle, or an
already-known step_id for a value_ref) to its trajectory step_id; returns None for a base source
table. The emitter and rollout each supply their own resolver over the same arg shape.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from observation_binding import (
    action_literal_slots,
    base_column,
    same_scalar,
    singleton_visible_cell,
)
from plan import TABLE_REF_ARGS


PERCEPTION_TOOLS = frozenset({"describe_table", "inspect_column", "read_subtable"})


@dataclass(frozen=True)
class _RowGroundingCandidate:
    score: int
    step_id: str
    source_table: str
    detail: dict[str, Any]


def _cond_value_refs(cond):
    """Yield (kind, ref, column) for each value_ref / in_table found anywhere in a condition tree."""
    if isinstance(cond, list):
        for c in cond:
            yield from _cond_value_refs(c)
        return
    if not isinstance(cond, dict):
        return
    for k in ("and", "or"):
        if k in cond:
            for c in cond[k]:
                yield from _cond_value_refs(c)
            return
    if "not" in cond:
        yield from _cond_value_refs(cond["not"])
        return
    col = cond.get("column")
    if "value_ref" in cond:
        yield ("value_ref", cond["value_ref"], col)
    if "in_table" in cond:
        yield ("in_table", cond["in_table"], col)


def build_references(tool: str, args: dict, resolve_step) -> list[dict]:
    """Harness-authored consumption edges for one action, over the model-facing arg shape. The
    emitter and rollout call this with their own `resolve_step`, so the edge schema is built in ONE
    place and can never drift between SFT and eval."""
    refs: list[dict] = []
    for key in TABLE_REF_ARGS.get(tool, []):
        v = args.get(key)
        if v is None:
            continue
        # Historical N-way joins carry inputs as a list under `tables`; emit one edge per input.
        for item in (v if isinstance(v, list) else [v]):
            sid = resolve_step(item)
            if sid:
                refs.append({"type": "data", "step": sid, "role": "table", "target": {"handle": item}})
            else:
                refs.append({"type": "data", "source": item, "role": "table", "target": {"table": item}})
    if tool == "join_tables":
        for join in args.get("joins") or []:
            if not isinstance(join, dict):
                continue
            item = join.get("table")
            if not isinstance(item, str):
                continue
            sid = resolve_step(item)
            if sid:
                refs.append({"type": "data", "step": sid, "role": "table", "target": {"handle": item}})
            else:
                refs.append({"type": "data", "source": item, "role": "table", "target": {"table": item}})
    if tool == "condition_filter":
        for kind, ref, col in _cond_value_refs(args.get("conditions")):
            sid = resolve_step(ref)
            if not sid:
                continue
            if kind == "value_ref":
                refs.append({"type": "value", "step": sid, "role": "value_ref",
                             "target": {"column": base_column(col)}})
            else:  # an IN-subquery's set membership is a data input
                refs.append({"type": "data", "step": sid, "role": "in_table",
                             "target": {"handle": ref}})
    if tool == "group_aggregate":
        for aggregation_index, aggregation in enumerate(args.get("aggregations") or []):
            if not isinstance(aggregation, dict):
                continue
            for kind, ref, col in _cond_value_refs(aggregation.get("where")):
                sid = resolve_step(ref)
                if not sid:
                    continue
                if kind == "value_ref":
                    refs.append({
                        "type": "value",
                        "step": sid,
                        "role": "aggregate_where",
                        "target": {
                            "aggregation_index": aggregation_index,
                            "column": base_column(col),
                        },
                    })
                else:
                    refs.append({
                        "type": "data",
                        "step": sid,
                        "role": "in_table",
                        "target": {
                            "aggregation_index": aggregation_index,
                            "handle": ref,
                        },
                    })
    if tool == "scalar_compute":
        for index, operand in enumerate(args.get("operands") or []):
            if not isinstance(operand, dict) or not isinstance(operand.get("value_ref"), str):
                continue
            sid = resolve_step(operand["value_ref"])
            if sid:
                target = {"operand_index": index}
                if isinstance(operand.get("column"), str):
                    target["column"] = operand["column"]
                refs.append({
                    "type": "value",
                    "step": sid,
                    "role": "operand",
                    "target": target,
                })
    return refs


def _table_refs(tool: str, args: dict) -> set[str]:
    keys = list(TABLE_REF_ARGS.get(tool, []))
    if tool == "inspect_column":
        keys.append("table")
    if tool == "describe_table":
        keys.append("tables")
    refs: set[str] = set()
    for key in keys:
        value = args.get(key)
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, str):
                refs.add(item)
    if tool == "join_tables":
        for join in args.get("joins") or []:
            if isinstance(join, dict) and isinstance(join.get("table"), str):
                refs.add(join["table"])
    return refs


def _columns(value, parent: str | None = None) -> set[str]:
    column_keys = {
        "column", "columns", "group_by", "return_columns", "passthrough",
        "left", "right", "partition_by", "order_by", "key_column", "value_column",
    }
    if isinstance(value, list):
        out: set[str] = set()
        for item in value:
            out.update(_columns(item, parent))
        return out
    if not isinstance(value, dict):
        if parent in column_keys and isinstance(value, str):
            return {base_column(value)}
        return set()
    out: set[str] = set()
    for key, item in value.items():
        out.update(_columns(item, key))
    return out


def _matching_values(left: list, right: list) -> list:
    matches = []
    for value in left:
        if any(same_scalar(value, candidate) for candidate in right):
            if not any(same_scalar(value, existing) for existing in matches):
                matches.append(value)
    return matches


def _history_handle_producers(history: dict[str, dict]) -> dict[str, str]:
    producers: dict[str, str] = {}
    for step_id, record in history.items():
        if record.get("tool") in PERCEPTION_TOOLS or record.get("tool") == "plan":
            continue
        output = record.get("output") or {}
        handle = output.get("table")
        if isinstance(handle, str):
            producers[handle] = step_id
    return producers


def _root_tables_for_step(
    step_id: str,
    history: dict[str, dict],
    producers: dict[str, str],
    memo: dict[str, set[str]],
) -> set[str]:
    if step_id in memo:
        return memo[step_id]
    # Break malformed cycles conservatively.
    memo[step_id] = set()
    record = history.get(step_id) or {}
    roots: set[str] = set()
    for ref in record.get("references") or []:
        if ref.get("type", "data") != "data":
            continue
        source = ref.get("source")
        if isinstance(source, str):
            roots.add(source)
        parent = ref.get("step")
        if isinstance(parent, str):
            roots.update(_root_tables_for_step(parent, history, producers, memo))
    if not roots:
        for table in _table_refs(record.get("tool"), record.get("arguments") or {}):
            producer = producers.get(table)
            if producer and producer != step_id:
                roots.update(_root_tables_for_step(producer, history, producers, memo))
            elif not producer:
                roots.add(table)
    memo[step_id] = roots
    return roots


def _root_tables_for_handle(
    handle: str | None,
    history: dict[str, dict],
    producers: dict[str, str],
    memo: dict[str, set[str]],
) -> set[str]:
    if not isinstance(handle, str):
        return set()
    producer = producers.get(handle)
    return _root_tables_for_step(producer, history, producers, memo) if producer else {handle}


def _foreign_key_graph(history: dict[str, dict]) -> dict[tuple[str, str], set[tuple[str, str]]]:
    graph: dict[tuple[str, str], set[tuple[str, str]]] = {}
    described_tables: list[dict] = []
    for record in history.values():
        if record.get("tool") != "describe_table":
            continue
        described_tables.extend((record.get("output") or {}).get("tables", []) or [])
    primary_keys = {
        described.get("table_name"): [
            base_column(column.get("name"))
            for column in described.get("columns", []) or []
            if isinstance(column, dict) and column.get("pk") and isinstance(column.get("name"), str)
        ]
        for described in described_tables
        if isinstance(described, dict) and isinstance(described.get("table_name"), str)
    }
    for described in described_tables:
        if not isinstance(described, dict):
            continue
        table = described.get("table_name")
        if not isinstance(table, str):
            continue
        for foreign_key in described.get("foreign_keys", []) or []:
            column = base_column(foreign_key.get("column"))
            reference = foreign_key.get("references")
            if not isinstance(column, str) or not isinstance(reference, str) or "." not in reference:
                continue
            target_table, target_column = reference.rsplit(".", 1)
            if target_column.casefold() in {"", "none", "null"}:
                candidates = primary_keys.get(target_table) or []
                if len(candidates) != 1:
                    continue
                target_column = candidates[0]
            left = (table, column)
            right = (target_table, base_column(target_column))
            graph.setdefault(left, set()).add(right)
            graph.setdefault(right, set()).add(left)
    return graph


def _columns_equivalent(
    source_tables: set[str],
    source_column: str,
    target_tables: set[str],
    target_column: str,
    foreign_keys: dict[tuple[str, str], set[tuple[str, str]]],
) -> bool:
    source_nodes = {(table, base_column(source_column)) for table in source_tables}
    target_nodes = {(table, base_column(target_column)) for table in target_tables}
    if source_nodes & target_nodes:
        return True
    source_base = base_column(source_column)
    target_base = base_column(target_column)
    if source_base == target_base:
        if source_tables & target_tables:
            return True
        # BIRD schemas do not always declare their foreign keys. A distinctive same-named column
        # is useful evidence across tables; generic names still require a real FK edge.
        if str(source_base).casefold() not in {"id", "name", "value", "code", "type", "status"}:
            return True
    frontier = list(source_nodes)
    seen = set(source_nodes)
    while frontier:
        node = frontier.pop()
        for neighbor in foreign_keys.get(node, set()):
            if neighbor in target_nodes:
                return True
            if neighbor not in seen:
                seen.add(neighbor)
                frontier.append(neighbor)
    return False


def build_grounding_references(tool: str, args: dict, history: dict[str, dict]) -> list[dict]:
    """Infer perception dependencies from legal actions and harness-owned observations.

    This is intentionally model-independent: it never reads think/reason text and does not require
    the model to cite an evidence id. Structurally grounded row observations outrank untyped visible
    copies, and recency breaks ties, avoiding broad credit for every earlier matching scalar.
    """
    table_refs = _table_refs(tool, args)
    action_columns = _columns(args)
    literal_slots = action_literal_slots(tool, args)
    literals = [slot.value for slot in literal_slots]
    refs: list[dict] = []
    schema_linked: set[str] = set()
    domain_linked: set[tuple[str, str]] = set()
    # One best source per consumed literal slot.  Structural column/FK matches outrank a newer
    # pure visible copy; recency breaks ties within the same evidence class.
    row_candidates: dict[
        tuple[str, tuple[str | int, ...], str, str],
        _RowGroundingCandidate,
    ] = {}
    producers = _history_handle_producers(history)
    root_memo: dict[str, set[str]] = {}
    foreign_keys = _foreign_key_graph(history)
    target_table = args.get("table")
    target_roots = _root_tables_for_handle(target_table, history, producers, root_memo)

    for step_id, record in reversed(list(history.items())):
        prior_tool = record.get("tool")
        prior_args = record.get("arguments") or {}
        output = record.get("output") or {}

        if prior_tool == "describe_table":
            if tool == "describe_table":
                continue
            for described in output.get("tables", []) or []:
                table = described.get("table_name")
                if table not in table_refs or table in schema_linked:
                    continue
                available = {
                    base_column(column.get("name"))
                    for column in described.get("columns", []) or []
                    if isinstance(column, dict) and isinstance(column.get("name"), str)
                }
                used = sorted(action_columns & available)
                refs.append({
                    "type": "grounding",
                    "step": step_id,
                    "role": "schema_observation",
                    "target": {"table": table, "columns": used},
                })
                schema_linked.add(table)

        elif prior_tool == "inspect_column":
            table = prior_args.get("table")
            column = base_column(output.get("column") or prior_args.get("column"))
            key = (str(table), str(column))
            if key in domain_linked or table not in table_refs or column not in action_columns:
                continue
            observed = list(output.get("frequent_values", []) or [])
            matched = _matching_values(literals, observed)
            if not matched:
                continue
            domain_linked.add(key)
            refs.append({
                "type": "grounding",
                "step": step_id,
                "role": "domain_observation",
                "target": {"table": table, "column": column, "values": matched},
            })

        elif literal_slots:
            columns = record.get("observed_columns") or output.get("columns")
            rows = output.get("rows")
            if not isinstance(columns, list) or not isinstance(rows, list):
                continue
            # Any row actually rendered by the harness is model-visible evidence.  This includes
            # bounded previews emitted by relation-producing tools, not only an explicit
            # read_subtable.  Prefer the produced handle so its recorded lineage is preserved.
            source_table = output.get("table") or prior_args.get("table")
            if not isinstance(source_table, str):
                continue
            source_roots = _root_tables_for_handle(source_table, history, producers, root_memo)
            computed_columns: set[str] = set()
            if prior_tool == "scalar_compute":
                computed_columns.update(
                    base_column(column)
                    for column in columns
                    if isinstance(column, str)
                )
            elif prior_tool == "group_aggregate":
                computed_columns.update(
                    base_column(aggregation.get("as"))
                    for aggregation in prior_args.get("aggregations") or []
                    if isinstance(aggregation, dict)
                    and isinstance(aggregation.get("as"), str)
                )
            for slot in literal_slots:
                target_column = slot.column
                literal = slot.value
                argument_path = slot.argument_path
                if not isinstance(target_column, str):
                    continue
                literal_key = (
                    target_column,
                    argument_path,
                    type(literal).__name__,
                    repr(literal),
                )
                exact_cells: list[tuple[int, int, str, bool]] = []
                structural_cells: list[tuple[int, int, str, bool]] = []
                for column_index, source_column in enumerate(columns):
                    source_column = base_column(source_column)
                    if not isinstance(source_column, str):
                        continue
                    computed_result = source_column in computed_columns
                    structurally_compatible = computed_result or _columns_equivalent(
                        source_roots,
                        source_column,
                        target_roots,
                        target_column,
                        foreign_keys,
                    )
                    for row_index, row in enumerate(rows):
                        if (
                            not isinstance(row, (list, tuple))
                            or column_index >= len(row)
                            or not same_scalar(row[column_index], literal)
                        ):
                            continue
                        cell = (row_index, column_index, source_column, computed_result)
                        exact_cells.append(cell)
                        if structurally_compatible:
                            structural_cells.append(cell)

                if structural_cells:
                    score = 2
                    cells = structural_cells
                elif len(exact_cells) == 1:
                    # BIRD omits many FK declarations.  An exact scalar copied from one unique
                    # visible cell is still causal evidence even when source/target names differ.
                    score = 1
                    cells = exact_cells
                else:
                    continue

                existing = row_candidates.get(literal_key)
                if existing is not None and existing.score >= score:
                    continue
                row_index, column_index, source_column, computed_result = cells[0]
                detail = {
                    "value": literal,
                    "source_tables": sorted(source_roots),
                    "source_column": source_column,
                    "target_tables": sorted(target_roots),
                    "target_column": target_column,
                    "argument_path": list(argument_path),
                    "match_kind": (
                        "computed_result"
                        if computed_result
                        else "column_equivalent"
                        if score == 2
                        else "visible_literal_copy"
                    ),
                    "binding_ambiguous": len(cells) != 1,
                }
                if len(cells) == 1:
                    detail["source_row_index"] = row_index
                    detail["source_column_index"] = column_index
                replay_source = singleton_visible_cell(step_id, record, literal)
                if replay_source is not None:
                    detail["replay_binding"] = replay_source.replay_target()
                row_candidates[literal_key] = _RowGroundingCandidate(
                    score=score,
                    step_id=step_id,
                    source_table=source_table,
                    detail=detail,
                )

    details_by_step: dict[tuple[str, str], list[dict]] = {}
    for candidate in row_candidates.values():
        details_by_step.setdefault(
            (candidate.step_id, candidate.source_table),
            [],
        ).append(candidate.detail)
    for (step_id, source_table), matched_details in details_by_step.items():
        matched = []
        for detail in matched_details:
            if not any(same_scalar(detail["value"], value) for value in matched):
                matched.append(detail["value"])
        refs.append({
            "type": "grounding",
            "step": step_id,
            "role": "row_observation",
            "target": {
                "table": source_table,
                "values": matched,
                "column_matches": matched_details,
            },
        })

    return refs


def backward_slice(traj: dict, reference_type=("data", "value")) -> set[str]:
    """Walk `references` backward from the terminal answer to the set of step_ids it depends on.
    `reference_type` selects which edge types count (default = the answer-computation slice: data +
    value). Grounding edges are excluded by default so the data slice keeps its 'which computation
    steps produced the answer' meaning; include 'grounding' for the perception-aware slice."""
    types = {reference_type} if isinstance(reference_type, str) else set(reference_type)
    by_id = {s["step_id"]: s for s in traj.get("steps", [])}
    if not traj.get("steps"):
        return set()

    def edges(step):
        return [r["step"] for r in step.get("references", [])
                if "step" in r and r.get("type", "data") in types]

    frontier = edges(traj["steps"][-1])
    seen: set[str] = set()
    while frontier:
        x = frontier.pop()
        if x in seen or x not in by_id:
            continue
        seen.add(x)
        frontier.extend(edges(by_id[x]))
    return seen


if __name__ == "__main__":   # unit demo: data + value edges over the model-facing arg shape
    history_ids = {"step_1", "step_2"}
    handle_to_step = {"filter_002": "step_2"}

    def resolve(ref):
        if ref in handle_to_step:
            return handle_to_step[ref]
        if ref in history_ids:            # already a step_id (a value_ref)
            return ref
        return None

    refs = build_references("condition_filter", {
        "table": "filter_002",
        "conditions": {"column": "T2__age", "op": ">", "value_ref": "step_1"},
    }, resolve)
    import json
    print(json.dumps(refs, ensure_ascii=False, indent=2))
