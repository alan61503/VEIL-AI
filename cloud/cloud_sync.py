from typing import Mapping, Optional

from config import DEVICE_ID
from db.database import mark_synced

def sync_to_cloud(vehicle_data: Optional[Mapping[str, str]]) -> bool:
    """Push a completed vehicle record to the cloud backend (simulated)."""
    if not vehicle_data:
        return False

    plate = vehicle_data.get("plate")
    exit_time = vehicle_data.get("exit_time")

    if not plate or not exit_time:
        return False

    print(f"[{DEVICE_ID}] Syncing to cloud: {plate} @ {exit_time}")
    # TODO: Add Firebase REST API or Supabase logic here
    mark_synced(plate)
    return True
