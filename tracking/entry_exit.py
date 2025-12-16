from datetime import datetime
from typing import Dict, Optional

from db.database import add_vehicle, mark_exit

# Keeps an in-memory view of vehicles currently tracked on the edge device.
vehicle_log: Dict[str, Dict[str, Optional[str]]] = {}

def vehicle_entry(number_plate: str, vehicle_type: str) -> Dict[str, Optional[str]]:
    """Record an entry event locally and in the persistent store."""
    if number_plate not in vehicle_log:
        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        record = {
            "plate": number_plate,
            "entry_time": entry_time,
            "exit_time": None,
            "type": vehicle_type,
        }
        vehicle_log[number_plate] = record
        add_vehicle(number_plate, vehicle_type, entry_time)
        print(f"Vehicle Entered: {number_plate}, Type: {vehicle_type}")
        return record
    return vehicle_log[number_plate]

def vehicle_exit(number_plate: str) -> Optional[Dict[str, Optional[str]]]:
    """Update exit details for a tracked vehicle and return the record."""
    if number_plate in vehicle_log:
        exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vehicle_log[number_plate]["exit_time"] = exit_time
        mark_exit(number_plate, exit_time)
        print(f"Vehicle Exited: {number_plate}")
        return vehicle_log[number_plate]
    return None

def clear_vehicle(number_plate: str) -> Optional[Dict[str, Optional[str]]]:
    """Remove a vehicle from the in-memory log once processing is complete."""
    return vehicle_log.pop(number_plate, None)
