from typing import Dict

from config import DEVICE_ID, FIREBASE_COLLECTION
from cloud.firebase_client import init_firebase


def sync_to_firebase(record: Dict) -> bool:
    try:
        db = init_firebase()

        payload = {
            "plate": record["plate"],
            "type": record["type"],
            "entry_time": record["entry_time"],
            "exit_time": record["exit_time"],
            "device_id": DEVICE_ID,
            "db_id": record.get("db_id"),
        }

        doc_id = str(record.get("db_id", record["plate"]))
        db.collection(FIREBASE_COLLECTION).document(doc_id).set(payload)

        return True

    except Exception as exc:  # pragma: no cover - logging only
        print("[FIREBASE ERROR]", exc)
        return False
