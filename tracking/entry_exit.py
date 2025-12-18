from collections import deque
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any, Deque, Dict, Optional, Tuple

from db.database import add_entry, add_exit
from config import ENTRY_DEDUP_SIMILARITY, ENTRY_DEDUP_WINDOW_SECONDS

vehicle_log: Dict[str, Dict[str, Any]] = {}
recent_entries: Deque[Tuple[str, datetime]] = deque()


def _prune_recent(now: datetime) -> None:
    if ENTRY_DEDUP_WINDOW_SECONDS <= 0:
        recent_entries.clear()
        return

    cutoff = now - timedelta(seconds=ENTRY_DEDUP_WINDOW_SECONDS)
    while recent_entries and recent_entries[0][1] < cutoff:
        recent_entries.popleft()


def _plate_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _is_duplicate_plate(plate: str, now: datetime) -> bool:
    if ENTRY_DEDUP_WINDOW_SECONDS <= 0:
        return False

    _prune_recent(now)
    for recent_plate, _ in recent_entries:
        if _plate_similarity(plate, recent_plate) >= ENTRY_DEDUP_SIMILARITY:
            return True
    return False


def _remember_plate(plate: str, timestamp: datetime) -> None:
    recent_entries.append((plate, timestamp))


def vehicle_entry(plate: str, vehicle_type: str) -> Dict[str, Any]:
    now = datetime.now()
    entry_time = now.strftime("%Y-%m-%d %H:%M:%S")

    if _is_duplicate_plate(plate, now):
        print(
            f"[ENTRY:DEDUP] {plate} ignored (similar plate seen within {ENTRY_DEDUP_WINDOW_SECONDS}s)"
        )
        return vehicle_log.get(
            plate,
            {
                "plate": plate,
                "type": vehicle_type,
                "entry_time": entry_time,
                "exit_time": None,
                "db_id": None,
            },
        )

    if plate not in vehicle_log:
        db_id = add_entry(plate, vehicle_type, entry_time)

        vehicle_log[plate] = {
            "plate": plate,
            "type": vehicle_type,
            "entry_time": entry_time,
            "exit_time": None,
            "db_id": db_id,
        }

        _remember_plate(plate, now)
        print(f"[ENTRY] {plate}")

    return vehicle_log[plate]


def vehicle_exit(plate: str) -> Optional[Dict[str, Any]]:
    record = vehicle_log.get(plate)
    if not record:
        return None

    exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    record["exit_time"] = exit_time

    add_exit(record["db_id"], exit_time)

    vehicle_log.pop(plate, None)
    print(f"[EXIT] {plate}")
    return record
