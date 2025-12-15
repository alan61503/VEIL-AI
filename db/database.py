import sqlite3

conn = sqlite3.connect("vehicles.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS vehicles (
    plate TEXT PRIMARY KEY,
    type TEXT,
    entry_time TEXT,
    exit_time TEXT,
    synced INTEGER DEFAULT 0
)
""")
conn.commit()

def add_vehicle(plate, vehicle_type, entry_time, exit_time=None):
    cursor.execute("""
    INSERT OR REPLACE INTO vehicles (plate, type, entry_time, exit_time)
    VALUES (?, ?, ?, ?)
    """, (plate, vehicle_type, entry_time, exit_time))
    conn.commit()

def mark_exit(plate, exit_time):
    cursor.execute("""
    UPDATE vehicles SET exit_time = ? WHERE plate = ?
    """, (exit_time, plate))
    conn.commit()

def get_unsynced():
    cursor.execute("SELECT * FROM vehicles WHERE synced = 0")
    return cursor.fetchall()

def mark_synced(plate):
    cursor.execute("UPDATE vehicles SET synced = 1 WHERE plate = ?", (plate,))
    conn.commit()
