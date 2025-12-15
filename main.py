import cv2
from config import CAMERA_SOURCE
from detection.detector import detect_plate
from ocr.plate_reader import read_plate
from classification.plate_color import classify_plate_color
from tracking.entry_exit import vehicle_entry, vehicle_exit, vehicle_log
from cloud.cloud_sync import sync_to_cloud

cap = cv2.VideoCapture(CAMERA_SOURCE)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    plates = detect_plate(frame)

    for plate_img in plates:
        number = read_plate(plate_img)
        if number:
            vehicle_type = classify_plate_color(plate_img)

            # For demo, toggle entry/exit manually
            if number not in vehicle_log:
                vehicle_entry(number, vehicle_type)
            else:
                vehicle_exit(number)
                if sync_to_cloud(vehicle_log[number]):
                    print(f"{number} synced to cloud.")

    cv2.imshow("VEIL", frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
