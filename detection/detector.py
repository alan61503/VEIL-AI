from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  # pretrained nano model

def detect_plate(frame):
    results = model(frame)[0]
    plate_boxes = []
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        # Assuming YOLO trained with class 0=car, 1=plate
        if int(cls) == 1:  # 1 = license plate
            x1, y1, x2, y2 = map(int, box)
            plate_boxes.append(frame[y1:y2, x1:x2])
    return plate_boxes
