from db.database import get_unsynced, mark_synced
from cloud.cloud_sync import sync_to_cloud


def sync_pending() -> None:
    rows = get_unsynced()

    for row_id, plate, vtype, entry, exit_time in rows:
        record = {
            "plate": plate,
            "type": vtype,
            "entry_time": entry,
            "exit_time": exit_time,
            "db_id": row_id,
        }

        if sync_to_cloud(record):
            mark_synced(row_id)
            print(f"[SYNCED] {plate}")
