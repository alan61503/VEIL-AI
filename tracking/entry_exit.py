from datetime import datetime
from db.database import add_vehicle, mark_exit

vehicle_log = {}

def vehicle_entry(number_plate, vehicle_type):
    if number_plate not in vehicle_log:
        entry_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vehicle_log[number_plate] = {
            "entry_time": entry_time,
            "exit_time": None,
            "type": vehicle_type
        }
        add_vehicle(number_plate, vehicle_type, entry_time)
        print(f"Vehicle Entered: {number_plate}, Type: {vehicle_type}")

def vehicle_exit(number_plate):
    if number_plate in vehicle_log:
        exit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vehicle_log[number_plate]["exit_time"] = exit_time
        mark_exit(number_plate, exit_time)
        print(f"Vehicle Exited: {number_plate}")
