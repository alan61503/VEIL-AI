import sqlite3
from typing import List, Tuple

DB_NAME = "vehicles.db"


def get_conn():
    return sqlite3.connect(DB_NAME)


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("PRAGMA table_info(vehicles)")
    columns = [row[1] for row in cur.fetchall()]

    if not columns:
        _create_schema(cur)
    elif "id" not in columns:
        _migrate_schema(cur)
    else:
        _ensure_indexes(cur)

    conn.commit()
    conn.close()


def _create_schema(cur) -> None:
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate TEXT NOT NULL,
            type TEXT,
            entry_time TEXT,
            exit_time TEXT,
            synced INTEGER DEFAULT 0
        )
        """
    )
    _ensure_indexes(cur)


def _ensure_indexes(cur) -> None:
    cur.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_vehicles_plate ON vehicles (plate)
        """
    )


def _migrate_schema(cur) -> None:
    cur.execute("ALTER TABLE vehicles RENAME TO vehicles_legacy")
    _create_schema(cur)
    cur.execute(
        """
        INSERT INTO vehicles (plate, type, entry_time, exit_time, synced)
        SELECT plate, type, entry_time, exit_time, synced FROM vehicles_legacy
        """
    )
    cur.execute("DROP TABLE vehicles_legacy")


def add_entry(plate: str, vehicle_type: str, entry_time: str) -> int:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO vehicles (plate, type, entry_time)
        VALUES (?, ?, ?)
        """,
        (plate, vehicle_type, entry_time),
    )

    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def add_exit(row_id: int, exit_time: str):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE vehicles
        SET exit_time = ?
        WHERE id = ? AND exit_time IS NULL
        """,
        (exit_time, row_id),
    )

    conn.commit()
    conn.close()


def get_unsynced() -> List[Tuple]:
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT id, plate, type, entry_time, exit_time
        FROM vehicles
        WHERE synced = 0 AND exit_time IS NOT NULL
        ORDER BY entry_time ASC
        """
    )

    rows = cur.fetchall()
    conn.close()
    return rows


def mark_synced(row_id: int):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        UPDATE vehicles SET synced = 1 WHERE id = ?
        """,
        (row_id,),
    )

    conn.commit()
    conn.close()
