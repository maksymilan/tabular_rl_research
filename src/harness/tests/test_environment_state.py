"""Tests for resident plan + table-context state."""
from common import T, employees_db
from environment_state import EnvironmentState, EnvironmentStateError


def run():
    t = T("environment_state")
    h = employees_db()
    catalog = {
        "tables": [{"table_name": "employees", "num_rows": 6}],
        "relations": [],
    }
    state = EnvironmentState(catalog)

    out = state.apply_plan_ops([
        {"op": "create", "id": "p1", "goal": "find relevant employee rows"},
        {"op": "update", "id": "p1", "status": "in_progress"},
    ], "step_1")
    t.check("plan creates one item", len(out["plan"]) == 1, str(out))
    t.check("plan updates status", out["plan"][0]["status"] == "in_progress", str(out))
    history = {
        "step_0": {
            "tool": "condition_filter",
            "output": {"table": "filter_001", "kind": "table", "columns": ["id", "dept"], "row_count": 2},
        }
    }
    out = state.apply_plan_ops([
        {"op": "update", "id": "p1", "status": "done", "evidence": "step_0"},
    ], "step_1b", history)
    t.check("plan stores grounded evidence",
            out["plan"][0]["evidence"]["output"]["table"] == "filter_001" and
            out["plan"][0]["status"] == "done",
            str(out))
    t.check("plan snapshot hides internal bookkeeping",
            set(state.snapshot()["plan"][0]) == {"id", "goal", "status", "evidence"},
            str(state.snapshot()))

    upsert = state.apply_plan_ops([
        {"op": "update", "id": "p2", "goal": "check whether a follow-up join is needed", "status": "pending"},
        {"op": "update", "id": "p2", "evidence": "none", "status": "done"},
    ], "step_1c", history)
    t.check("plan update on unknown item upserts",
            any(item["id"] == "p2" and item["status"] == "done" for item in upsert["plan"]),
            str(upsert))
    t.check("plan evidence none clears evidence",
            all(item.get("evidence") is None for item in upsert["plan"] if item["id"] == "p2"),
            str(upsert))
    empty_evidence = state.apply_plan_ops([
        {"op": "update", "id": "p2", "evidence": "", "status": "pending"},
    ], "step_1c_empty", history)
    p2_empty = [item for item in empty_evidence["plan"] if item["id"] == "p2"][0]
    t.check("plan empty-string evidence is treated as absent",
            "evidence" not in p2_empty,
            str(empty_evidence))

    unresolved = state.apply_plan_ops([
        {"op": "update", "id": "p2", "evidence": "step_future"},
    ], "step_1d", history)
    p2 = [item for item in unresolved["plan"] if item["id"] == "p2"][0]
    t.check("plan unknown evidence is nonfatal",
            p2["evidence"]["step_id"] == "step_future" and p2["evidence"]["unresolved"],
            str(unresolved))

    desc = h.describe_table(["employees"])
    state.apply_tool_result("describe_table", {"tables": ["employees"]}, desc, "step_2")
    snap = state.snapshot()
    t.check("schema stored under table",
            snap["tables"]["employees"]["schema"]["from_step"] == "step_2", str(snap))

    ins = h.inspect_column("employees", "dept")
    state.apply_tool_result("inspect_column", {"table": "employees", "column": "dept"}, ins, "step_3")
    snap = state.snapshot()
    t.check("column domain stored under table",
            snap["tables"]["employees"]["inspected_columns"]["dept"]["from_step"] == "step_3",
            str(snap))

    f = h.condition_filter("employees", {"column": "dept", "op": "=", "value": "eng"})
    output = {"table": f["table_name"], "kind": f["kind"],
              "columns": f["columns"], "row_count": f["row_count"],
              "derivation": {
                  "schema": "relation-derivation-v1",
                  "operator": "condition_filter",
                  "inputs": [{"kind": "table", "role": "input", "ref": "employees"}],
                  "semantics": {
                      "row_operation": "filter",
                      "predicate": {"column": "dept", "op": "=", "value": "eng"},
                      "predicate_columns": ["dept"],
                      "column_operation": "preserve",
                      "projected_columns": f["columns"],
                  },
              }}
    state.apply_tool_result("condition_filter", {"table": "employees", "conditions": {}}, output, "step_4")
    snap = state.snapshot()
    t.check("relation derivation is bound to its output table",
            snap["tables"][f["table_name"]]["derivation"]["operator"] == "condition_filter" and
            snap["tables"][f["table_name"]]["derivation"]["semantics"]["row_operation"] == "filter",
            str(snap))
    rows = h.read_subtable(f["table_name"], limit=2)
    state.apply_tool_result(
        "read_subtable",
        {"table": f["table_name"], "limit": 2},
        {"rows": [list(r) for r in rows], "row_count": f["row_count"]},
        "step_5",
    )
    snap = state.snapshot()
    t.check("derived handle stored",
            snap["tables"][f["table_name"]]["created_by"] == "step_4", str(snap))
    t.check("read rows grouped under handle",
            snap["tables"][f["table_name"]]["reads"][0]["from_step"] == "step_5", str(snap))
    t.check("perception preserves table-bound derivation",
            snap["tables"][f["table_name"]]["derivation"]["operator"] == "condition_filter",
            str(snap))
    state.apply_tool_result(
        "read_subtable",
        {"table": f["table_name"], "limit": 2},
        {"rows": [["replacement"]], "row_count": f["row_count"]},
        "step_5b",
    )
    snap = state.snapshot()
    reads = snap["tables"][f["table_name"]]["reads"]
    t.check("duplicate read_subtable replaces previous read",
            len(reads) == 1 and reads[0]["from_step"] == "step_5b", str(snap))

    state.apply_tool_result(
        "project",
        {"table": f["table_name"], "expressions": ["dept"]},
        {
            "table": "project_999",
            "kind": "project",
            "columns": ["dept"],
            "row_count": 2,
            "derivation": {
                "schema": "relation-derivation-v1",
                "operator": "project",
                "inputs": [{"kind": "table", "role": "input", "ref": f["table_name"]}],
                "semantics": {
                    "row_operation": "preserve",
                    "column_operation": "project",
                    "column_lineage": [
                        {
                            "output": "dept",
                            "sources": ["dept"],
                            "kind": "column",
                            "expression": "dept",
                        },
                    ],
                },
            },
        },
        "step_5_clear",
    )
    t.check("new derivation does not erase prior table metadata",
            state.snapshot()["tables"][f["table_name"]]["derivation"]["operator"]
            == "condition_filter" and
            state.snapshot()["tables"]["project_999"]["derivation"]["operator"] == "project" and
            "latest_structural_feedback" not in state.snapshot(),
            str(state.snapshot()))

    try:
        state.apply_tool_result(
            "project",
            {"table": f["table_name"], "expressions": ["dept"]},
            {
                "table": "project_without_derivation",
                "kind": "project",
                "columns": ["dept"],
                "row_count": 2,
            },
            "step_missing_derivation",
        )
    except EnvironmentStateError:
        missing_derivation_rejected = True
    else:
        missing_derivation_rejected = False
    t.check(
        "table operators cannot bypass relation derivation",
        missing_derivation_rejected
        and "project_without_derivation" not in state.snapshot()["tables"],
        str(state.snapshot()),
    )

    state.apply_tool_result(
        "aggregate",
        {"table": f["table_name"], "column": "*", "op": "count"},
        {"result_sample": [[2]], "row_count": 1},
        "step_5c",
    )
    snap = state.snapshot()
    t.check("scalar result stored in values",
            snap["values"]["step_5c"]["result_sample"] == [[2]], str(snap))

    state.apply_plan_ops([
        {"op": "delete", "id": "p1", "reason": "complete"},
        {"op": "delete", "id": "p2", "reason": "complete"},
    ], "step_6")
    t.check("deleted plan hidden from snapshot", state.snapshot()["plan"] == [], str(state.snapshot()))
    return t.result()
