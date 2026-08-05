#!/usr/bin/env python3
"""Build the frozen 23-task BIRD counterfactual candidate suite.

The builder copies each immutable source database before applying one task-specific mutation.
Every generated database must preserve the complete SQLite table/column/FK schema and change the
hidden BIRD gold denotation.  The emitted manifest deliberately remains ``quality_gate=pending``;
only a separate replay audit may promote it to a trainer-loadable v2 manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Callable

from trajectory_replay import bird_set_fingerprint, sqlite_schema_fingerprint


Mutation = Callable[[sqlite3.Connection], dict[str, Any]]


def quoted(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_examples(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    examples = payload.get("examples") if isinstance(payload, dict) else payload
    if not isinstance(examples, list):
        raise ValueError("examples JSON must be a list or an object with an examples list")
    return examples


def fetch_rows(
    connection: sqlite3.Connection,
    sql: str,
    parameters: tuple[Any, ...] = (),
) -> list[list[Any]]:
    return [list(row) for row in connection.execute(sql, parameters).fetchall()]


def execute_required(
    connection: sqlite3.Connection,
    sql: str,
    parameters: tuple[Any, ...] = (),
) -> int:
    cursor = connection.execute(sql, parameters)
    changed = cursor.rowcount
    if changed == 0:
        raise RuntimeError(f"mutation changed no rows: {sql}")
    return changed


def scalar(
    connection: sqlite3.Connection,
    sql: str,
    parameters: tuple[Any, ...] = (),
) -> Any:
    row = connection.execute(sql, parameters).fetchone()
    if row is None:
        raise RuntimeError(f"query returned no row: {sql}")
    return row[0]


def mutate_5647_a(connection: sqlite3.Connection) -> dict[str, Any]:
    changed = execute_required(
        connection,
        """
        UPDATE PlayerInfo
        SET PlayerName = PlayerName || ' [CF-A]'
        WHERE rowid IN (
          SELECT P.rowid
          FROM PlayerInfo P JOIN weight_info W ON P.weight = W.weight_id
          WHERE W.weight_in_kg > 90
            AND P.sum_7yr_GP = (
              SELECT MAX(P2.sum_7yr_GP)
              FROM PlayerInfo P2 JOIN weight_info W2 ON P2.weight = W2.weight_id
              WHERE W2.weight_in_kg > 90
            )
        )
        """,
    )
    return {"kind": "answer_cell_shift", "changed_rows": changed}


def mutate_5647_b(connection: sqlite3.Connection) -> dict[str, Any]:
    maximum = scalar(
        connection,
        """
        SELECT MAX(P.sum_7yr_GP)
        FROM PlayerInfo P JOIN weight_info W ON P.weight = W.weight_id
        WHERE W.weight_in_kg > 90
        """,
    )
    rowid = scalar(
        connection,
        """
        SELECT P.rowid
        FROM PlayerInfo P JOIN weight_info W ON P.weight = W.weight_id
        WHERE W.weight_in_kg > 90 AND P.sum_7yr_GP < ?
        ORDER BY P.sum_7yr_GP DESC, P.rowid
        LIMIT 1
        """,
        (maximum,),
    )
    changed = execute_required(
        connection,
        """
        UPDATE PlayerInfo
        SET sum_7yr_GP = ?, PlayerName = PlayerName || ' [CF-B]'
        WHERE rowid = ?
        """,
        (maximum + 100, rowid),
    )
    return {"kind": "argmax_competitor_flip", "changed_rows": changed}


def mutate_1622_a(connection: sqlite3.Connection) -> dict[str, Any]:
    rowid = scalar(
        connection,
        """
        SELECT V.rowid
        FROM Episode E JOIN Vote V ON V.episode_id = E.episode_id
        WHERE E.title = 'Lost Verizon'
        ORDER BY V.votes DESC, V.rowid
        LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE Vote SET stars = COALESCE(stars, 0) + 17 WHERE rowid = ?",
        (rowid,),
    )
    return {"kind": "selected_value_shift", "changed_rows": changed}


def mutate_1622_b(connection: sqlite3.Connection) -> dict[str, Any]:
    maximum = scalar(
        connection,
        """
        SELECT MAX(V.votes)
        FROM Episode E JOIN Vote V ON V.episode_id = E.episode_id
        WHERE E.title = 'Lost Verizon'
        """,
    )
    rowid = scalar(
        connection,
        """
        SELECT V.rowid
        FROM Episode E JOIN Vote V ON V.episode_id = E.episode_id
        WHERE E.title = 'Lost Verizon'
        ORDER BY V.votes DESC, V.rowid
        LIMIT 1 OFFSET 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE Vote SET votes = ?, stars = COALESCE(stars, 0) + 29 WHERE rowid = ?",
        (maximum + 1000, rowid),
    )
    return {"kind": "order_by_competitor_flip", "changed_rows": changed}


def mutate_3256_a(connection: sqlite3.Connection) -> dict[str, Any]:
    station = scalar(
        connection,
        """
        SELECT station_nbr FROM relation
        GROUP BY station_nbr ORDER BY COUNT(store_nbr) DESC, station_nbr LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        """
        UPDATE weather SET tmax = tmax + 1
        WHERE station_nbr = ? AND SUBSTR(date, 1, 7) = '2012-02'
        """,
        (station,),
    )
    return {"kind": "aggregate_value_shift", "changed_rows": changed, "station": station}


def mutate_3256_b(connection: sqlite3.Connection) -> dict[str, Any]:
    stations = fetch_rows(
        connection,
        """
        SELECT station_nbr, COUNT(store_nbr) AS n
        FROM relation GROUP BY station_nbr
        ORDER BY n DESC, station_nbr LIMIT 2
        """,
    )
    if len(stations) < 2:
        raise RuntimeError("sales_in_weather has no runner-up station")
    winner, runner = stations[0][0], stations[1][0]
    changed = execute_required(
        connection,
        "UPDATE relation SET station_nbr = ? WHERE station_nbr = ?",
        (runner, winner),
    )
    return {
        "kind": "group_argmax_flip",
        "changed_rows": changed,
        "winner_station": winner,
        "runner_station": runner,
    }


def mutate_5366_a(connection: sqlite3.Connection) -> dict[str, Any]:
    target_date = scalar(
        connection,
        """
        SELECT S."Order Date"
        FROM people P JOIN central_superstore S ON P."Customer ID" = S."Customer ID"
        WHERE P."Customer Name" = 'Alan Barnes'
          AND STRFTIME('%Y', S."Order Date") = '2015'
        LIMIT 1
        """,
    )
    rowid = scalar(
        connection,
        """
        SELECT S.rowid
        FROM people P JOIN central_superstore S ON P."Customer ID" = S."Customer ID"
        WHERE P."Customer Name" = 'Alan Barnes'
          AND STRFTIME('%Y', S."Order Date") != '2015'
        LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        'UPDATE central_superstore SET "Order Date" = ? WHERE rowid = ?',
        (target_date, rowid),
    )
    return {"kind": "population_membership_add_one", "changed_rows": changed}


def mutate_5366_b(connection: sqlite3.Connection) -> dict[str, Any]:
    target_date = scalar(
        connection,
        """
        SELECT S."Order Date"
        FROM people P JOIN central_superstore S ON P."Customer ID" = S."Customer ID"
        WHERE P."Customer Name" = 'Alan Barnes'
          AND STRFTIME('%Y', S."Order Date") = '2015'
        LIMIT 1
        """,
    )
    rowids = [
        row[0]
        for row in connection.execute(
            """
            SELECT S.rowid
            FROM people P JOIN central_superstore S ON P."Customer ID" = S."Customer ID"
            WHERE P."Customer Name" = 'Alan Barnes'
              AND STRFTIME('%Y', S."Order Date") != '2015'
            LIMIT 2
            """
        )
    ]
    if not rowids:
        raise RuntimeError("no non-2015 Alan Barnes order is available")
    placeholders = ",".join("?" for _ in rowids)
    changed = execute_required(
        connection,
        f'UPDATE central_superstore SET "Order Date" = ? WHERE rowid IN ({placeholders})',
        (target_date, *rowids),
    )
    return {"kind": "population_membership_add_many", "changed_rows": changed}


def _move_ra_rows(connection: sqlite3.Connection, limit: int) -> dict[str, Any]:
    professor = scalar(
        connection,
        "SELECT prof_id FROM prof WHERE teachingability = '1' AND gender = 'Female' LIMIT 1",
    )
    rowids = [
        row[0]
        for row in connection.execute(
            "SELECT rowid FROM RA WHERE prof_id != ? LIMIT ?",
            (professor, limit),
        )
    ]
    if not rowids:
        raise RuntimeError("no RA control row is available")
    placeholders = ",".join("?" for _ in rowids)
    changed = execute_required(
        connection,
        f"UPDATE RA SET prof_id = ? WHERE rowid IN ({placeholders})",
        (professor, *rowids),
    )
    return {"kind": "join_population_shift", "changed_rows": changed}


def mutate_3925_a(connection: sqlite3.Connection) -> dict[str, Any]:
    return _move_ra_rows(connection, 1)


def mutate_3925_b(connection: sqlite3.Connection) -> dict[str, Any]:
    return _move_ra_rows(connection, 2)


def mutate_5761_a(connection: sqlite3.Connection) -> dict[str, Any]:
    commander = scalar(
        connection,
        """
        SELECT D.commander
        FROM IUCR I JOIN Crime C ON C.iucr_no = I.iucr_no
        JOIN District D ON D.district_no = C.district_no
        WHERE I.secondary_description = 'CRIMINAL SEXUAL ABUSE'
        GROUP BY D.commander ORDER BY COUNT(*) DESC, D.commander LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE District SET commander = commander || ' [CF-A]' WHERE commander = ?",
        (commander,),
    )
    return {"kind": "answer_cell_shift", "changed_rows": changed}


def mutate_5761_b(connection: sqlite3.Connection) -> dict[str, Any]:
    winner = scalar(
        connection,
        """
        SELECT D.commander
        FROM IUCR I JOIN Crime C ON C.iucr_no = I.iucr_no
        JOIN District D ON D.district_no = C.district_no
        WHERE I.secondary_description = 'CRIMINAL SEXUAL ABUSE'
        GROUP BY D.commander ORDER BY COUNT(*) DESC, D.commander LIMIT 1
        """,
    )
    runner_district = scalar(
        connection,
        "SELECT district_no FROM District WHERE commander != ? ORDER BY district_no LIMIT 1",
        (winner,),
    )
    changed = execute_required(
        connection,
        """
        UPDATE Crime SET district_no = ?
        WHERE iucr_no IN (
          SELECT iucr_no FROM IUCR
          WHERE secondary_description = 'CRIMINAL SEXUAL ABUSE'
        )
        """,
        (runner_district,),
    )
    return {"kind": "group_argmax_flip", "changed_rows": changed}


def mutate_362_a(connection: sqlite3.Connection) -> dict[str, Any]:
    professor = scalar(
        connection,
        """
        SELECT P.prof_id
        FROM RA R JOIN prof P ON R.prof_id = P.prof_id
        GROUP BY R.prof_id HAVING COUNT(R.student_id) > 2
        ORDER BY P.teachingability DESC, P.prof_id LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        """
        UPDATE prof
        SET first_name = first_name || ' [CF-A]',
            last_name = last_name || ' [CF-A]'
        WHERE prof_id = ?
        """,
        (professor,),
    )
    return {"kind": "answer_cell_shift", "changed_rows": changed}


def mutate_362_b(connection: sqlite3.Connection) -> dict[str, Any]:
    professors = fetch_rows(
        connection,
        """
        SELECT P.prof_id
        FROM RA R JOIN prof P ON R.prof_id = P.prof_id
        GROUP BY R.prof_id HAVING COUNT(R.student_id) > 2
        ORDER BY P.teachingability DESC, P.prof_id LIMIT 2
        """,
    )
    if len(professors) < 2:
        raise RuntimeError("no qualifying runner-up professor")
    professor = professors[1][0]
    changed = execute_required(
        connection,
        """
        UPDATE prof
        SET teachingability = '999',
            first_name = first_name || ' [CF-B]',
            last_name = last_name || ' [CF-B]'
        WHERE prof_id = ?
        """,
        (professor,),
    )
    return {"kind": "argmax_competitor_flip", "changed_rows": changed}


def mutate_6501_a(connection: sqlite3.Connection) -> dict[str, Any]:
    rowid = scalar(connection, "SELECT rowid FROM Country ORDER BY LifeExpectancy, rowid LIMIT 1")
    changed = execute_required(
        connection,
        "UPDATE Country SET Name = Name || ' [CF-A]' WHERE rowid = ?",
        (rowid,),
    )
    return {"kind": "answer_cell_shift", "changed_rows": changed}


def mutate_6501_b(connection: sqlite3.Connection) -> dict[str, Any]:
    winner = scalar(connection, "SELECT rowid FROM Country ORDER BY LifeExpectancy, rowid LIMIT 1")
    runner = scalar(
        connection,
        "SELECT rowid FROM Country WHERE rowid != ? ORDER BY LifeExpectancy, rowid LIMIT 1",
        (winner,),
    )
    execute_required(
        connection,
        "UPDATE Country SET LifeExpectancy = 999 WHERE rowid = ?",
        (winner,),
    )
    changed = execute_required(
        connection,
        "UPDATE Country SET LifeExpectancy = -1, Name = Name || ' [CF-B]' WHERE rowid = ?",
        (runner,),
    )
    return {"kind": "argmin_competitor_flip", "changed_rows": changed + 1}


def mutate_3443_a(connection: sqlite3.Connection) -> dict[str, Any]:
    changed = execute_required(
        connection,
        """
        UPDATE Customers
        SET age = age + 1
        WHERE SEX = 'Female' AND MARITAL_STATUS = 'Widowed'
          AND GEOID IN (SELECT GEOID FROM Demog WHERE INHABITANTS_K = 33.658)
        """,
    )
    return {"kind": "answer_cell_shift", "changed_rows": changed}


def mutate_3443_b(connection: sqlite3.Connection) -> dict[str, Any]:
    target_geoid = scalar(
        connection,
        "SELECT GEOID FROM Demog WHERE INHABITANTS_K = 33.658 LIMIT 1",
    )
    rowid = scalar(
        connection,
        """
        SELECT rowid FROM Customers
        WHERE SEX = 'Female' AND MARITAL_STATUS = 'Widowed' AND GEOID != ?
        ORDER BY rowid LIMIT 1
        """,
        (target_geoid,),
    )
    changed = execute_required(
        connection,
        "UPDATE Customers SET GEOID = ?, age = age + 7 WHERE rowid = ?",
        (target_geoid, rowid),
    )
    return {"kind": "join_population_add", "changed_rows": changed}


def mutate_1296_a(connection: sqlite3.Connection) -> dict[str, Any]:
    publisher_id = scalar(
        connection,
        """
        SELECT P.id
        FROM game G JOIN game_publisher GP ON G.id = GP.game_id
        JOIN publisher P ON GP.publisher_id = P.id
        JOIN genre R ON G.genre_id = R.id
        WHERE R.genre_name = 'Puzzle'
        ORDER BY P.id LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE publisher SET publisher_name = publisher_name || ' [CF-A]' WHERE id = ?",
        (publisher_id,),
    )
    return {"kind": "set_member_value_shift", "changed_rows": changed}


def mutate_1296_b(connection: sqlite3.Connection) -> dict[str, Any]:
    puzzle_genre = scalar(connection, "SELECT id FROM genre WHERE genre_name = 'Puzzle' LIMIT 1")
    game_id = scalar(
        connection,
        """
        SELECT G.id
        FROM game G JOIN game_publisher GP ON G.id = GP.game_id
        WHERE GP.publisher_id NOT IN (
          SELECT DISTINCT GP2.publisher_id
          FROM game G2 JOIN game_publisher GP2 ON G2.id = GP2.game_id
          JOIN genre R2 ON G2.genre_id = R2.id
          WHERE R2.genre_name = 'Puzzle'
        )
        ORDER BY G.id LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE game SET genre_id = ? WHERE id = ?",
        (puzzle_genre, game_id),
    )
    return {"kind": "set_membership_add", "changed_rows": changed}


def mutate_3409_a(connection: sqlite3.Connection) -> dict[str, Any]:
    changed = execute_required(
        connection,
        """
        UPDATE politics SET Independence = '1900-01-01'
        WHERE Country IN (SELECT Code FROM country WHERE Name = 'United States')
        """,
    )
    return {"kind": "answer_cell_shift", "changed_rows": changed}


def mutate_3409_b(connection: sqlite3.Connection) -> dict[str, Any]:
    old_code = scalar(connection, "SELECT Code FROM country WHERE Name = 'United States' LIMIT 1")
    new_code = "ZCF"
    execute_required(
        connection,
        "UPDATE politics SET Country = ?, Independence = '1901-02-03' WHERE Country = ?",
        (new_code, old_code),
    )
    changed = execute_required(
        connection,
        "UPDATE country SET Code = ? WHERE Code = ?",
        (new_code, old_code),
    )
    return {"kind": "join_key_rebinding", "changed_rows": changed + 1}


def mutate_5172_a(connection: sqlite3.Connection) -> dict[str, Any]:
    rowid = scalar(
        connection,
        """
        SELECT E.rowid FROM employee E JOIN position P ON P.positionID = E.positionID
        WHERE P.positiontitle = 'Manager' AND E.performance = 'Poor'
        ORDER BY E.rowid LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE employee SET firstname = firstname || ' [CF-A]' WHERE rowid = ?",
        (rowid,),
    )
    return {"kind": "set_member_value_shift", "changed_rows": changed}


def mutate_5172_b(connection: sqlite3.Connection) -> dict[str, Any]:
    rowid = scalar(
        connection,
        """
        SELECT E.rowid FROM employee E JOIN position P ON P.positionID = E.positionID
        WHERE P.positiontitle = 'Manager' AND E.performance != 'Poor'
        ORDER BY E.rowid LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE employee SET performance = 'Poor', firstname = firstname || ' [CF-B]' WHERE rowid = ?",
        (rowid,),
    )
    return {"kind": "population_membership_add", "changed_rows": changed}


def mutate_1802_a(connection: sqlite3.Connection) -> dict[str, Any]:
    rowid = scalar(
        connection,
        """
        SELECT A.rowid
        FROM Answer A
        WHERE A.QuestionID = 1 AND A.UserID IN (
          SELECT UserID FROM Answer WHERE QuestionID = 3 AND AnswerText = 'United States'
        )
        ORDER BY A.rowid LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE Answer SET AnswerText = CAST(AnswerText AS INTEGER) + 11 WHERE rowid = ?",
        (rowid,),
    )
    return {"kind": "aggregate_value_shift", "changed_rows": changed}


def mutate_1802_b(connection: sqlite3.Connection) -> dict[str, Any]:
    rowid = scalar(
        connection,
        """
        SELECT A.rowid
        FROM Answer A
        WHERE A.QuestionID = 3 AND A.AnswerText != 'United States'
          AND EXISTS (
            SELECT 1 FROM Answer Age
            WHERE Age.UserID = A.UserID AND Age.QuestionID = 1
          )
        ORDER BY A.rowid LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE Answer SET AnswerText = 'United States' WHERE rowid = ?",
        (rowid,),
    )
    return {"kind": "population_membership_add", "changed_rows": changed}


def mutate_5260_a(connection: sqlite3.Connection) -> dict[str, Any]:
    rowid = scalar(
        connection,
        """
        SELECT T.rowid
        FROM twitter T JOIN user U ON T.UserID = U.UserID
        WHERE U.Gender = 'Female' AND T.IsReshare = 'TRUE'
        ORDER BY T.rowid LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE twitter SET IsReshare = 'FALSE' WHERE rowid = ?",
        (rowid,),
    )
    return {"kind": "count_membership_remove", "changed_rows": changed}


def mutate_5260_b(connection: sqlite3.Connection) -> dict[str, Any]:
    user_id = scalar(
        connection,
        """
        SELECT U.UserID
        FROM user U JOIN twitter T ON T.UserID = U.UserID
        WHERE U.Gender != 'Female' AND T.IsReshare = 'TRUE'
        ORDER BY U.UserID LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE user SET Gender = 'Female' WHERE UserID = ?",
        (user_id,),
    )
    return {"kind": "join_population_add", "changed_rows": changed}


def mutate_278_a(connection: sqlite3.Connection) -> dict[str, Any]:
    maximum = scalar(connection, "SELECT MAX(Watchers) FROM Repo")
    changed = execute_required(
        connection,
        """
        UPDATE Solution SET Id = Id + 1000000000
        WHERE RepoId IN (SELECT Id FROM Repo WHERE Watchers = ?)
        """,
        (maximum,),
    )
    return {"kind": "set_member_value_shift", "changed_rows": changed}


def mutate_278_b(connection: sqlite3.Connection) -> dict[str, Any]:
    maximum = scalar(connection, "SELECT MAX(Watchers) FROM Repo")
    repo_id = scalar(
        connection,
        """
        SELECT R.Id FROM Repo R
        WHERE R.Watchers < ? AND EXISTS (SELECT 1 FROM Solution S WHERE S.RepoId = R.Id)
        ORDER BY R.Watchers DESC, R.Id LIMIT 1
        """,
        (maximum,),
    )
    changed = execute_required(
        connection,
        "UPDATE Repo SET Watchers = ? WHERE Id = ?",
        (maximum + 1, repo_id),
    )
    return {"kind": "argmax_competitor_flip", "changed_rows": changed}


def mutate_3837_a(connection: sqlite3.Connection) -> dict[str, Any]:
    changed = execute_required(
        connection,
        """
        UPDATE events SET Product = Product || ' [CF-A]'
        WHERE "Complaint ID" IN (
          SELECT E."Complaint ID"
          FROM callcenterlogs C JOIN events E ON C."Complaint ID" = E."Complaint ID"
          WHERE C.server = 'TOVA' AND E."Date received" LIKE '2017-03%'
        )
        """,
    )
    return {"kind": "set_member_value_shift", "changed_rows": changed}


def mutate_3837_b(connection: sqlite3.Connection) -> dict[str, Any]:
    complaint_id = scalar(
        connection,
        """
        SELECT E."Complaint ID"
        FROM events E JOIN callcenterlogs C ON C."Complaint ID" = E."Complaint ID"
        WHERE E."Date received" LIKE '2017-03%' AND C.server != 'TOVA'
          AND E.Product NOT IN (
            SELECT E2.Product
            FROM events E2 JOIN callcenterlogs C2 ON C2."Complaint ID" = E2."Complaint ID"
            WHERE E2."Date received" LIKE '2017-03%' AND C2.server = 'TOVA'
          )
        ORDER BY E."Complaint ID" LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        'UPDATE callcenterlogs SET server = \'TOVA\' WHERE "Complaint ID" = ?',
        (complaint_id,),
    )
    return {"kind": "join_population_add", "changed_rows": changed}


def mutate_1305_a(connection: sqlite3.Connection) -> dict[str, Any]:
    rowid = scalar(
        connection,
        """
        SELECT GP.rowid
        FROM game_platform GP JOIN game_publisher GPub ON GP.game_publisher_id = GPub.id
        JOIN game G ON GPub.game_id = G.id
        WHERE G.game_name = 'Pro Evolution Soccer 2016'
        ORDER BY GP.rowid LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "DELETE FROM game_platform WHERE rowid = ?",
        (rowid,),
    )
    return {"kind": "count_membership_remove", "changed_rows": changed}


def mutate_1305_b(connection: sqlite3.Connection) -> dict[str, Any]:
    game_publisher_id = scalar(
        connection,
        """
        SELECT GPub.id
        FROM game_publisher GPub JOIN game G ON GPub.game_id = G.id
        WHERE G.game_name = 'Pro Evolution Soccer 2016'
        ORDER BY GPub.id LIMIT 1
        """,
    )
    platform_id = scalar(
        connection,
        """
        SELECT P.id FROM platform P
        WHERE P.id NOT IN (
          SELECT GP.platform_id FROM game_platform GP
          JOIN game_publisher GPub ON GP.game_publisher_id = GPub.id
          JOIN game G ON GPub.game_id = G.id
          WHERE G.game_name = 'Pro Evolution Soccer 2016'
        )
        ORDER BY P.id LIMIT 1
        """,
    )
    rowid = scalar(
        connection,
        """
        SELECT rowid FROM game_platform
        WHERE game_publisher_id != ?
          AND NOT EXISTS (
            SELECT 1 FROM game_platform X
            WHERE X.game_publisher_id = ? AND X.platform_id = ?
          )
        ORDER BY rowid LIMIT 1
        """,
        (game_publisher_id, game_publisher_id, platform_id),
    )
    changed = execute_required(
        connection,
        """
        UPDATE game_platform
        SET game_publisher_id = ?, platform_id = ?
        WHERE rowid = ?
        """,
        (game_publisher_id, platform_id, rowid),
    )
    return {"kind": "count_membership_add", "changed_rows": changed}


def mutate_1655_a(connection: sqlite3.Connection) -> dict[str, Any]:
    episode_id = scalar(
        connection,
        """
        SELECT E.episode_id
        FROM Award A JOIN Episode E ON A.episode_id = E.episode_id
        GROUP BY A.episode_id ORDER BY COUNT(*) DESC, A.episode_id LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE Episode SET title = title || ' [CF-A]' WHERE episode_id = ?",
        (episode_id,),
    )
    return {"kind": "answer_cell_shift", "changed_rows": changed}


def mutate_1655_b(connection: sqlite3.Connection) -> dict[str, Any]:
    episodes = fetch_rows(
        connection,
        """
        SELECT A.episode_id, COUNT(*) AS n
        FROM Award A JOIN Episode E ON A.episode_id = E.episode_id
        GROUP BY A.episode_id
        ORDER BY n DESC, A.episode_id LIMIT 2
        """,
    )
    if len(episodes) < 2:
        raise RuntimeError("no award runner-up episode")
    winner, runner = episodes[0][0], episodes[1][0]
    changed = execute_required(
        connection,
        "UPDATE Award SET episode_id = ? WHERE episode_id = ?",
        (runner, winner),
    )
    return {"kind": "group_argmax_flip", "changed_rows": changed}


def mutate_5429_a(connection: sqlite3.Connection) -> dict[str, Any]:
    publisher_id = scalar(
        connection,
        """
        SELECT P.id
        FROM game_platform GP JOIN game_publisher GPub ON GP.game_publisher_id = GPub.id
        JOIN publisher P ON GPub.publisher_id = P.id
        WHERE GP.release_year = 2007
        GROUP BY P.id ORDER BY COUNT(DISTINCT GPub.game_id) DESC, P.id LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE publisher SET publisher_name = publisher_name || ' [CF-A]' WHERE id = ?",
        (publisher_id,),
    )
    return {"kind": "answer_cell_shift", "changed_rows": changed}


def mutate_5429_b(connection: sqlite3.Connection) -> dict[str, Any]:
    publishers = fetch_rows(
        connection,
        """
        SELECT P.id, COUNT(DISTINCT GPub.game_id) AS n
        FROM game_platform GP JOIN game_publisher GPub ON GP.game_publisher_id = GPub.id
        JOIN publisher P ON GPub.publisher_id = P.id
        WHERE GP.release_year = 2007
        GROUP BY P.id ORDER BY n DESC, P.id LIMIT 2
        """,
    )
    if len(publishers) < 2:
        raise RuntimeError("no 2007 publisher runner-up")
    winner, runner = publishers[0][0], publishers[1][0]
    changed = execute_required(
        connection,
        """
        UPDATE game_publisher SET publisher_id = ?
        WHERE publisher_id = ? AND id IN (
          SELECT game_publisher_id FROM game_platform WHERE release_year = 2007
        )
        """,
        (runner, winner),
    )
    return {"kind": "group_argmax_flip", "changed_rows": changed}


def mutate_808_a(connection: sqlite3.Connection) -> dict[str, Any]:
    role_id = scalar(
        connection,
        """
        SELECT PM.Role_Id
        FROM "Match" M JOIN Player_Match PM ON M.Match_Id = PM.Match_Id
        JOIN Player P ON PM.Player_Id = P.Player_Id
        ORDER BY P.DOB DESC, PM.rowid LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE Rolee SET Role_Desc = Role_Desc || ' [CF-A]' WHERE Role_Id = ?",
        (role_id,),
    )
    return {"kind": "answer_cell_shift", "changed_rows": changed}


def mutate_808_b(connection: sqlite3.Connection) -> dict[str, Any]:
    player_match = connection.execute(
        """
        SELECT PM.Player_Id, MIN(PM.Match_Id)
        FROM Player_Match PM
        WHERE PM.Player_Id != (
          SELECT PM2.Player_Id
          FROM Player_Match PM2 JOIN Player P2 ON PM2.Player_Id = P2.Player_Id
          ORDER BY P2.DOB DESC, PM2.rowid LIMIT 1
        )
        GROUP BY PM.Player_Id
        HAVING COUNT(*) = 1
        ORDER BY PM.Player_Id LIMIT 1
        """
    ).fetchone()
    if player_match is None:
        raise RuntimeError("no alternate player with exactly one match")
    player_id, match_id = player_match
    execute_required(
        connection,
        "UPDATE Player SET DOB = '2099-01-01' WHERE Player_Id = ?",
        (player_id,),
    )
    changed = execute_required(
        connection,
        'UPDATE "Match" SET Match_Date = \'2099-12-31\' WHERE Match_Id = ?',
        (match_id,),
    )
    return {"kind": "order_by_competitor_flip", "changed_rows": changed + 1}


def mutate_898_a(connection: sqlite3.Connection) -> dict[str, Any]:
    country = scalar(
        connection,
        """
        SELECT M.country
        FROM movies2directors MD JOIN movies M ON MD.movieid = M.movieid
        WHERE MD.genre = 'Action'
        GROUP BY M.country ORDER BY COUNT(M.country) DESC, M.country LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        """
        UPDATE movies SET country = country || ' [CF-A]'
        WHERE country = ? AND movieid IN (
          SELECT movieid FROM movies2directors WHERE genre = 'Action'
        )
        """,
        (country,),
    )
    return {"kind": "answer_cell_shift", "changed_rows": changed}


def mutate_898_b(connection: sqlite3.Connection) -> dict[str, Any]:
    countries = fetch_rows(
        connection,
        """
        SELECT M.country, COUNT(M.country) AS n
        FROM movies2directors MD JOIN movies M ON MD.movieid = M.movieid
        WHERE MD.genre = 'Action'
        GROUP BY M.country ORDER BY n DESC, M.country LIMIT 2
        """,
    )
    if len(countries) < 2:
        raise RuntimeError("no Action-country runner-up")
    winner, runner = countries[0][0], countries[1][0]
    changed = execute_required(
        connection,
        """
        UPDATE movies SET country = ?
        WHERE country = ? AND movieid IN (
          SELECT movieid FROM movies2directors WHERE genre = 'Action'
        )
        """,
        (runner, winner),
    )
    return {"kind": "group_argmax_flip", "changed_rows": changed}


def mutate_1320_a(connection: sqlite3.Connection) -> dict[str, Any]:
    publisher_id = scalar(
        connection,
        """
        SELECT P.id
        FROM game_publisher GP JOIN publisher P ON GP.publisher_id = P.id
        GROUP BY P.id HAVING COUNT(DISTINCT GP.game_id) = 1
        ORDER BY P.id LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE publisher SET publisher_name = publisher_name || ' [CF-A]' WHERE id = ?",
        (publisher_id,),
    )
    return {"kind": "set_member_value_shift", "changed_rows": changed}


def mutate_1320_b(connection: sqlite3.Connection) -> dict[str, Any]:
    publisher_id = scalar(
        connection,
        """
        SELECT P.id
        FROM game_publisher GP JOIN publisher P ON GP.publisher_id = P.id
        GROUP BY P.id HAVING COUNT(DISTINCT GP.game_id) = 1
        ORDER BY P.id LIMIT 1
        """,
    )
    current_game = scalar(
        connection,
        "SELECT game_id FROM game_publisher WHERE publisher_id = ? LIMIT 1",
        (publisher_id,),
    )
    rowid = scalar(
        connection,
        """
        SELECT GP.rowid
        FROM game_publisher GP
        WHERE GP.game_id != ?
          AND NOT EXISTS (
            SELECT 1 FROM game_publisher X
            WHERE X.publisher_id = ? AND X.game_id = GP.game_id
          )
        ORDER BY GP.rowid LIMIT 1
        """,
        (current_game, publisher_id),
    )
    changed = execute_required(
        connection,
        "UPDATE game_publisher SET publisher_id = ? WHERE rowid = ?",
        (publisher_id, rowid),
    )
    return {"kind": "set_membership_remove_by_cardinality", "changed_rows": changed}


def mutate_4576_a(connection: sqlite3.Connection) -> dict[str, Any]:
    region_id = scalar(
        connection,
        """
        SELECT R.id
        FROM medal M JOIN competitor_event CE ON M.id = CE.medal_id
        JOIN games_competitor GC ON CE.competitor_id = GC.id
        JOIN person_region PR ON GC.person_id = PR.person_id
        JOIN noc_region R ON PR.region_id = R.id
        WHERE M.id != 4
        GROUP BY R.id ORDER BY COUNT(CE.competitor_id) DESC, R.id LIMIT 1
        """,
    )
    changed = execute_required(
        connection,
        "UPDATE noc_region SET region_name = region_name || ' [CF-A]' WHERE id = ?",
        (region_id,),
    )
    return {"kind": "answer_cell_shift", "changed_rows": changed}


def mutate_4576_b(connection: sqlite3.Connection) -> dict[str, Any]:
    regions = fetch_rows(
        connection,
        """
        SELECT R.id, COUNT(CE.competitor_id) AS n
        FROM medal M JOIN competitor_event CE ON M.id = CE.medal_id
        JOIN games_competitor GC ON CE.competitor_id = GC.id
        JOIN person_region PR ON GC.person_id = PR.person_id
        JOIN noc_region R ON PR.region_id = R.id
        WHERE M.id != 4
        GROUP BY R.id ORDER BY n DESC, R.id LIMIT 2
        """,
    )
    if len(regions) < 2:
        raise RuntimeError("no medal-count runner-up region")
    winner, runner = regions[0][0], regions[1][0]
    changed = execute_required(
        connection,
        "UPDATE person_region SET region_id = ? WHERE region_id = ?",
        (runner, winner),
    )
    return {"kind": "group_argmax_flip", "changed_rows": changed}


MUTATIONS: dict[int, tuple[Mutation, Mutation]] = {
    278: (mutate_278_a, mutate_278_b),
    362: (mutate_362_a, mutate_362_b),
    808: (mutate_808_a, mutate_808_b),
    898: (mutate_898_a, mutate_898_b),
    1296: (mutate_1296_a, mutate_1296_b),
    1305: (mutate_1305_a, mutate_1305_b),
    1320: (mutate_1320_a, mutate_1320_b),
    1622: (mutate_1622_a, mutate_1622_b),
    1655: (mutate_1655_a, mutate_1655_b),
    1802: (mutate_1802_a, mutate_1802_b),
    3256: (mutate_3256_a, mutate_3256_b),
    3409: (mutate_3409_a, mutate_3409_b),
    3443: (mutate_3443_a, mutate_3443_b),
    3837: (mutate_3837_a, mutate_3837_b),
    3925: (mutate_3925_a, mutate_3925_b),
    4576: (mutate_4576_a, mutate_4576_b),
    5172: (mutate_5172_a, mutate_5172_b),
    5260: (mutate_5260_a, mutate_5260_b),
    5366: (mutate_5366_a, mutate_5366_b),
    5429: (mutate_5429_a, mutate_5429_b),
    5647: (mutate_5647_a, mutate_5647_b),
    5761: (mutate_5761_a, mutate_5761_b),
    6501: (mutate_6501_a, mutate_6501_b),
}


def build_variant(
    *,
    source_path: Path,
    target_path: Path,
    gold_sql: str,
    mutation: Mutation,
    source_schema: str,
    source_gold_fingerprint: str,
    resume_existing: bool,
) -> dict[str, Any]:
    if target_path.exists():
        if not resume_existing:
            raise FileExistsError(f"refusing to overwrite existing candidate DB: {target_path}")
        connection = sqlite3.connect(str(target_path))
        try:
            gold_rows = fetch_rows(connection, gold_sql)
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        finally:
            connection.close()
        if integrity != "ok":
            raise RuntimeError(
                f"resumed SQLite integrity check failed for {target_path}: {integrity}"
            )
        actual_schema = sqlite_schema_fingerprint(target_path)
        if actual_schema != source_schema:
            raise RuntimeError(
                f"resumed schema changed for {target_path}: "
                f"{actual_schema} != {source_schema}"
            )
        gold_fingerprint = bird_set_fingerprint(gold_rows)
        if gold_fingerprint == source_gold_fingerprint:
            raise RuntimeError(
                f"resumed candidate is not informative and must be replaced: {target_path}"
            )
        return {
            "path": str(target_path.resolve()),
            "sha256": sha256_file(target_path),
            "bytes": target_path.stat().st_size,
            "schema_fingerprint": actual_schema,
            "gold_fingerprint": gold_fingerprint,
            "gold_row_count": len(gold_rows),
            "gold_sample": gold_rows[:5],
            "mutation": {
                "kind": "resumed_existing_candidate",
                "function": mutation.__name__,
            },
        }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)
    connection = sqlite3.connect(str(target_path))
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("BEGIN IMMEDIATE")
        mutation_record = mutation(connection)
        connection.commit()
        gold_rows = fetch_rows(connection, gold_sql)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
    if integrity != "ok":
        raise RuntimeError(f"SQLite integrity check failed for {target_path}: {integrity}")
    actual_schema = sqlite_schema_fingerprint(target_path)
    if actual_schema != source_schema:
        raise RuntimeError(
            f"schema changed for {target_path}: {actual_schema} != {source_schema}"
        )
    gold_fingerprint = bird_set_fingerprint(gold_rows)
    if gold_fingerprint == source_gold_fingerprint:
        raise RuntimeError(f"mutation did not change gold denotation: {target_path}")
    return {
        "path": str(target_path.resolve()),
        "sha256": sha256_file(target_path),
        "bytes": target_path.stat().st_size,
        "schema_fingerprint": actual_schema,
        "gold_fingerprint": gold_fingerprint,
        "gold_row_count": len(gold_rows),
        "gold_sample": gold_rows[:5],
        "mutation": mutation_record,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()

    examples = read_examples(args.examples_json)
    indices = {int(example["example_index"]) for example in examples}
    if indices != set(MUTATIONS):
        raise ValueError(
            f"example set differs from frozen mutation set: "
            f"missing={sorted(set(MUTATIONS) - indices)} extra={sorted(indices - set(MUTATIONS))}"
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = output_dir / "candidate_generation_audit.json"
    manifest_path = output_dir / "counterfactual_suite_v2.pending.json"
    if audit_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate artifacts in {output_dir}")

    tasks: dict[str, Any] = {}
    audits: list[dict[str, Any]] = []
    source_hash_cache: dict[Path, str] = {}
    for example in examples:
        index = int(example["example_index"])
        source_path = Path(example["db_path"]).resolve()
        gold_sql = str(example.get("gold_sql") or example["query"])
        source_schema = sqlite_schema_fingerprint(source_path)
        source_connection = sqlite3.connect(str(source_path))
        try:
            source_gold_rows = fetch_rows(source_connection, gold_sql)
        finally:
            source_connection.close()
        source_gold_fingerprint = bird_set_fingerprint(source_gold_rows)
        source_sha = source_hash_cache.get(source_path)
        if source_sha is None:
            source_sha = sha256_file(source_path)
            source_hash_cache[source_path] = source_sha

        variant_records = []
        for label, mutation in zip(("a", "b"), MUTATIONS[index], strict=True):
            target_path = (
                output_dir
                / f"spider_train_{index:05d}"
                / f"{example['db_id']}.cf_{label}.sqlite"
            )
            variant_records.append(
                build_variant(
                    source_path=source_path,
                    target_path=target_path,
                    gold_sql=gold_sql,
                    mutation=mutation,
                    source_schema=source_schema,
                    source_gold_fingerprint=source_gold_fingerprint,
                    resume_existing=args.resume_existing,
                )
            )

        trainer_task_id = f"spider_train_{index:05d}"
        trajectory_task_id = f"bird_train_{index:05d}"
        tasks[trainer_task_id] = {
            "example_index": index,
            "trajectory_task_id": trajectory_task_id,
            "db_id": example["db_id"],
            "source_db_sha256": source_sha,
            "gold_sql_sha256": sha256_text(gold_sql),
            "min_informative_databases": 2,
            "databases": [
                {"path": record["path"], "sha256": record["sha256"]}
                for record in variant_records
            ],
        }
        audits.append(
            {
                "trainer_task_id": trainer_task_id,
                "trajectory_task_id": trajectory_task_id,
                "example_index": index,
                "db_id": example["db_id"],
                "source_db": str(source_path),
                "source_db_sha256": source_sha,
                "source_schema_fingerprint": source_schema,
                "gold_sql_sha256": sha256_text(gold_sql),
                "source_gold_fingerprint": source_gold_fingerprint,
                "source_gold_row_count": len(source_gold_rows),
                "source_gold_sample": source_gold_rows[:5],
                "variants": variant_records,
            }
        )
        print(
            json.dumps(
                {
                    "event": "task_complete",
                    "trainer_task_id": trainer_task_id,
                    "variants": len(variant_records),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    audit_payload = {
        "schema_version": "process-counterfactual-generation-audit-v1",
        "examples_json": str(args.examples_json.resolve()),
        "examples_json_sha256": sha256_file(args.examples_json),
        "builder_script_sha256": sha256_file(Path(__file__)),
        "tasks": audits,
    }
    audit_path.write_text(
        json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_payload = {
        "schema_version": "process-counterfactual-suite-v2",
        "denotation_comparison": "bird-set",
        "tasks": tasks,
        "generator": {
            "name": "build_bird_counterfactual_suite_v2.py",
            "strategy": "full-source-copy-plus-task-specific-causal-mutation",
            "generation_audit_path": str(audit_path),
            "generation_audit_sha256": sha256_file(audit_path),
        },
        "quality_gate": {
            "status": "pending",
            "reason": "candidate databases require known-correct replay and shortcut regression audit",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "tasks": len(tasks),
                "databases": sum(len(task["databases"]) for task in tasks.values()),
                "audit": str(audit_path),
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
