"""Shared frame processing logic for camera and video pipelines."""
from typing import Any

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
) -> None:
    """Detect plates in a frame, persist entries, and sync exits when available."""
    plates = detect_plate(frame)
    required_hits = max(1, min_plate_hits)

    for plate_img in plates:
        plate_read = read_plate(plate_img)
        if not plate_read:
            continue

        number, confidence = plate_read
        vehicle_type = classify_plate_color(plate_img)

        if number not in vehicle_log:
            if required_hits > 1:
                if not register_plate_vote(number, confidence, required_hits=required_hits):
                    continue
            vehicle_entry(number, vehicle_type)
            clear_plate_vote(number)
            continue

        record = vehicle_exit(number)
        if record and cloud_enabled and sync_to_cloud(record):
            mark_synced(record["db_id"])
            print(f"{number} synced to cloud.")
            clear_plate_vote(number)
