from datetime import datetime
from typing import Dict, Optional

from db.database import add_vehicle, mark_exit

vehicle_log: Dict[str, Dict[str, Optional[str]]] = {}


def vehicle_entry(plate: str, vehicle_type: str):
    if plate not in vehicle_log:
        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        record = {
            "plate": plate,
            "type": vehicle_type,
            "entry_time": entry_time,
            "exit_time": None,
        }

        vehicle_log[plate] = record
        add_vehicle(plate, vehicle_type, entry_time)

        print(f"[ENTRY] {plate} ({vehicle_type})")
        return record

    return vehicle_log[plate]


def vehicle_exit(plate: str):
    if plate in vehicle_log:
        exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        vehicle_log[plate]["exit_time"] = exit_time
        mark_exit(plate, exit_time)

        record = vehicle_log[plate]
        del vehicle_log[plate]

        print(f"[EXIT] {plate}")
        return record

    return None
