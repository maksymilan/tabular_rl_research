"""Tests for resident plan + table-context state."""
from common import T, employees_db
from environment_state import EnvironmentState


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
        {"op": "update", "id": "p1", "status": "in_progress", "notes": "schema next"},
    ], "step_1")
    t.check("plan creates one item", len(out["plan"]) == 1, str(out))
    t.check("plan updates status", out["plan"][0]["status"] == "in_progress", str(out))

    desc = h.describe_table(["employees"])
    state.apply_tool_result("describe_table", {"tables": ["employees"]}, desc, "step_2")
    snap = state.snapshot()
    t.check("schema stored under table",
            snap["tables"]["employees"]["schema"]["from_step"] == "step_2", str(snap))

    ins = h.inspect_column("employees", "dept", value="eng")
    state.apply_tool_result("inspect_column", {"table": "employees", "column": "dept"}, ins, "step_3")
    snap = state.snapshot()
    t.check("column domain stored under table",
            snap["tables"]["employees"]["inspected_columns"]["dept"]["from_step"] == "step_3",
            str(snap))

    f = h.condition_filter("employees", {"column": "dept", "op": "=", "value": "eng"})
    output = {"table": f["table_name"], "kind": f["kind"],
              "columns": f["columns"], "row_count": f["row_count"]}
    state.apply_tool_result("condition_filter", {"table": "employees", "conditions": {}}, output, "step_4")
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

    state.apply_plan_ops([{"op": "delete", "id": "p1", "reason": "complete"}], "step_6")
    t.check("deleted plan hidden from snapshot", state.snapshot()["plan"] == [], str(state.snapshot()))
    return t.result()
