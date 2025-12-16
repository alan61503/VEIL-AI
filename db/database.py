import sqlite3
from datetime import datetime
from typing import List, Tuple

DB_NAME = "vehicles.db"


def get_connection():
    return sqlite3.connect(DB_NAME)


def init_db():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS vehicles (
        plate TEXT PRIMARY KEY,
        type TEXT,
        entry_time TEXT,
        exit_time TEXT,
        synced INTEGER DEFAULT 0
    )
    """)

    conn.commit()
    conn.close()


def add_vehicle(plate: str, vehicle_type: str, entry_time: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO vehicles (plate, type, entry_time)
    VALUES (?, ?, ?)
    ON CONFLICT(plate) DO NOTHING
    """, (plate, vehicle_type, entry_time))

    conn.commit()
    conn.close()


def mark_exit(plate: str, exit_time: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    UPDATE vehicles
    SET exit_time = ?
    WHERE plate = ? AND exit_time IS NULL
    """, (exit_time, plate))

    conn.commit()
    conn.close()


def get_unsynced() -> List[Tuple]:
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    SELECT plate, type, entry_time, exit_time
    FROM vehicles
    WHERE synced = 0 AND exit_time IS NOT NULL
    """)

    rows = cur.fetchall()
    conn.close()
    return rows


def mark_synced(plate: str):
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
    UPDATE vehicles SET synced = 1 WHERE plate = ?
    """, (plate,))

    conn.commit()
    conn.close()
