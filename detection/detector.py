from pathlib import Path
from typing import List

import requests
from ultralytics import YOLO

from config import PLATE_CLASS_IDS, PLATE_CONFIDENCE, PLATE_MODEL_PATH, PLATE_MODEL_URL
from detection.fallback import contour_detect_plates

_model_path = Path(PLATE_MODEL_PATH)
if not _model_path.exists():
    if not PLATE_MODEL_URL:
        raise FileNotFoundError(
            f"Plate model not found at {_model_path}. Set PLATE_MODEL_PATH to a valid .pt file."
        )

    _model_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading plate model to {_model_path} ...")
    response = requests.get(PLATE_MODEL_URL, timeout=60)
    if not response.ok:
        raise RuntimeError(
            "Unable to download plate model automatically. "
            "Please download it manually and update PLATE_MODEL_PATH."
        )
    _model_path.write_bytes(response.content)
    print("Plate model download complete.")

model = YOLO(str(_model_path))


def detect_plate(frame) -> List:
    """Return cropped plate regions detected in the provided frame."""
    plate_boxes = _detect_with_yolo(frame)
    if plate_boxes:
        return plate_boxes
    return contour_detect_plates(frame)


def _detect_with_yolo(frame) -> List:
    try:
        results = model(frame, conf=PLATE_CONFIDENCE, verbose=False)[0]
    except Exception as exc:  # pragma: no cover - logging only
        print("[YOLO ERROR]", exc)
        return []

    boxes = getattr(results, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    plate_boxes = []
    height, width = frame.shape[:2]

    for box in boxes:
        cls_id = int(box.cls[0])
        if PLATE_CLASS_IDS and cls_id not in PLATE_CLASS_IDS:
            continue

        crop = _crop(frame, box.xyxy[0].tolist(), width, height)
        if crop is not None:
            plate_boxes.append(crop)

    return plate_boxes


def _crop(frame, xyxy, width: int, height: int):
    x1, y1, x2, y2 = xyxy
    x1 = max(0, min(width, int(x1)))
    y1 = max(0, min(height, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop