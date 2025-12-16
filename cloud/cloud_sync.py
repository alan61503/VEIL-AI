from typing import Dict

import requests

from config import CLOUD_API_KEY, CLOUD_ENDPOINT, CLOUD_PROVIDER
from cloud.firebase_sync import sync_to_firebase


def sync_to_cloud(record: Dict) -> bool:
    if CLOUD_PROVIDER == "firebase":
        return sync_to_firebase(record)
    if CLOUD_PROVIDER == "rest":
        return _sync_via_rest(record)
    return False


def _sync_via_rest(record: Dict) -> bool:
    payload = {
        "plate": record["plate"],
        "type": record["type"],
        "entry_time": record["entry_time"],
        "exit_time": record["exit_time"],
    }
    headers = {
        "Authorization": f"Bearer {CLOUD_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            CLOUD_ENDPOINT,
            json=payload,
            headers=headers,
            timeout=10,
        )
        return response.status_code in (200, 201)
    except Exception as exc:  # pragma: no cover - logging only
        print("[CLOUD ERROR]", exc)
        return False
