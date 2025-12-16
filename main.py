import cv2

from config import CAMERA_SOURCE, CLOUD_ENABLED
from detection.detector import detect_plate
from ocr.plate_reader import read_plate
from classification.plate_color import classify_plate_color
from tracking.entry_exit import (
    vehicle_entry,
    vehicle_exit,
    vehicle_log,
    clear_vehicle,
)
from cloud.cloud_sync import sync_to_cloud


def main() -> None:
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        raise RuntimeError("Unable to access camera source. Check CAMERA_SOURCE in config.py")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

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
                if not record:
                    continue

                if CLOUD_ENABLED and sync_to_cloud(record):
                    print(f"{number} synced to cloud.")
                    clear_vehicle(number)

            cv2.imshow("VEIL", frame)
            if cv2.waitKey(1) == 27:  # ESC to quit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
