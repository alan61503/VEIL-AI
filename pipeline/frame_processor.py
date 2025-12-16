"""Shared frame processing logic for camera and video pipelines."""
from typing import Any

from config import CLOUD_ENABLED
from classification.plate_color import classify_plate_color
from cloud.cloud_sync import sync_to_cloud
from db.database import mark_synced
from detection.detector import detect_plate
from ocr.plate_reader import read_plate
from tracking.entry_exit import vehicle_entry, vehicle_exit, vehicle_log


def process_frame(frame: Any, cloud_enabled: bool = CLOUD_ENABLED) -> None:
    """Detect plates in a frame, persist entries, and sync exits when available."""
    plates = detect_plate(frame)

    for plate_img in plates:
        number = read_plate(plate_img)
        if not number:
            continue

        number = number.upper()
        vehicle_type = classify_plate_color(plate_img)

        if number not in vehicle_log:
            vehicle_entry(number, vehicle_type)
            continue

        record = vehicle_exit(number)
        if record and cloud_enabled and sync_to_cloud(record):
            mark_synced(number)
            print(f"{number} synced to cloud.")
