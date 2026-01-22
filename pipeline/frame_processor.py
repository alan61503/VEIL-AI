"""Shared frame processing logic for camera and video pipelines."""
from typing import Any, Dict

from config import CLOUD_ENABLED, MIN_PLATE_HITS
from classification.plate_color import classify_plate_color
from cloud.cloud_sync import sync_to_cloud
from db.database import mark_synced
from detection.detector import detect_plate
from ocr.plate_reader import read_plate
from tracking.entry_exit import vehicle_entry, vehicle_exit, vehicle_log
from tracking.plate_confirmer import clear_plate_vote, register_plate_vote


def process_frame(
    frame: Any,
    cloud_enabled: bool = CLOUD_ENABLED,
    min_plate_hits: int = MIN_PLATE_HITS,
    *,
    dedupe_enabled: bool = True,
    track_exits: bool = True,
) -> Dict[str, Any]:
    """Detect plates in a frame, persist entries, and return a summary."""
    plates = detect_plate(frame)
    required_hits = max(1, min_plate_hits)
    summary: Dict[str, Any] = {
        "detections": len(plates),
        "recognized": [],
        "entries": [],
        "exits": [],
    }

    for plate_img in plates:
        plate_read = read_plate(plate_img)
        if not plate_read:
            continue

        number, confidence = plate_read
        summary["recognized"].append(number)
        vehicle_type = classify_plate_color(plate_img)

        if number not in vehicle_log or not track_exits:
            if required_hits > 1:
                if not register_plate_vote(number, confidence, required_hits=required_hits):
                    continue
            vehicle_entry(number, vehicle_type, dedupe_enabled=dedupe_enabled)
            summary["entries"].append(number)
            clear_plate_vote(number)
            continue

        if track_exits:
            record = vehicle_exit(number)
            if record:
                summary["exits"].append(number)
                if cloud_enabled and sync_to_cloud(record):
                    mark_synced(record["db_id"])
                    print(f"{number} synced to cloud.")
                clear_plate_vote(number)

    return summary
