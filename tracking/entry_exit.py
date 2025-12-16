from datetime import datetime
from typing import Any, Dict, Optional

from db.database import add_entry, add_exit

vehicle_log: Dict[str, Dict[str, Any]] = {}


def vehicle_entry(plate: str, vehicle_type: str) -> Dict[str, Any]:
    if plate not in vehicle_log:
        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        db_id = add_entry(plate, vehicle_type, entry_time)

        vehicle_log[plate] = {
            "plate": plate,
            "type": vehicle_type,
            "entry_time": entry_time,
            "exit_time": None,
            "db_id": db_id,
        }

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
