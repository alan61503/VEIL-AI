"""YOLO-based plate detection with dynamic cropping heuristics."""

from pathlib import Path
from typing import List

import requests
from ultralytics import YOLO

from config import (
    PLATE_CLASS_IDS,
    PLATE_CONFIDENCE,
    PLATE_FORCE_TALL,
    PLATE_MARGIN,
    PLATE_MAX_RATIO,
    PLATE_MAX_RESULTS,
    PLATE_MIN_RATIO,
    PLATE_MODEL_PATH,
    PLATE_MODEL_URL,
    PLATE_TALL_MULTIPLIER,
    PLATE_TALL_PAD,
    PLATE_TALL_RATIO,
    PLATE_TALL_TARGET_RATIO,
    PLATE_TALL_UP_BIAS,
    PLATE_TALL_WIDTH_PAD,
    PLATE_TOP_EXTRA,
)
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

    height, width = frame.shape[:2]

    try:
        xyxy_list = boxes.xyxy.tolist()
        cls_list = [int(c) for c in boxes.cls.tolist()]
        conf_list = boxes.conf.tolist()
    except Exception:  # pragma: no cover - tensor conversion fallback
        xyxy_list, cls_list, conf_list = [], [], []
        for box in boxes:
            xyxy_list.append(box.xyxy[0].tolist())
            cls_list.append(int(box.cls[0]))
            conf_list.append(float(box.conf[0]))

    detections = sorted(
        zip(xyxy_list, cls_list, conf_list),
        key=lambda item: item[2],
        reverse=True,
    )

    plate_boxes = []
    for xyxy, cls_id, conf in detections[:PLATE_MAX_RESULTS]:
        if PLATE_CLASS_IDS and cls_id not in PLATE_CLASS_IDS:
            continue

        crop = _crop(frame, xyxy, width, height, PLATE_MARGIN)
        if crop is not None:
            plate_boxes.append(crop)

    return plate_boxes


def _crop(frame, xyxy, width: int, height: int, margin: float = 0.0):
    x1, y1, x2, y2 = xyxy
    pad_x = int((x2 - x1) * margin)
    pad_y = int((y2 - y1) * margin)

    x1 = max(0, min(width, int(x1) - pad_x))
    y1 = max(0, min(height, int(y1) - pad_y))
    x2 = max(0, min(width, int(x2) + pad_x))
    y2 = max(0, min(height, int(y2) + pad_y))

    if x2 <= x1 or y2 <= y1:
        return None

    x1, y1, x2, y2 = _expand_for_ratio(x1, y1, x2, y2, width, height)

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _expand_for_ratio(x1, y1, x2, y2, width, height):
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    ratio = box_w / float(box_h)

    if PLATE_FORCE_TALL or ratio >= PLATE_TALL_RATIO:
        desired_height = int(box_w / max(0.5, PLATE_TALL_TARGET_RATIO))
        target_height = max(box_h * PLATE_TALL_MULTIPLIER, desired_height)
        extra_needed = max(0, target_height - box_h)
        derived_from_width = int(box_w * PLATE_TALL_WIDTH_PAD)
        y_pad = max(int(box_h * PLATE_TALL_PAD), derived_from_width, extra_needed // 2)
        if y_pad > 0:
            upper_pad = max(1, int(y_pad * PLATE_TALL_UP_BIAS))
            lower_pad = max(1, y_pad - upper_pad)
            y1 = max(0, y1 - upper_pad)
            y2 = min(height, y2 + lower_pad)
            box_h = max(1, y2 - y1)
            ratio = box_w / float(box_h)

        extra_top = int((y2 - y1) * PLATE_TOP_EXTRA)
        if extra_top > 0:
            y1 = max(0, y1 - extra_top)
            box_h = max(1, y2 - y1)

    if ratio < PLATE_MIN_RATIO:
        needed = int(((PLATE_MIN_RATIO * box_h) - box_w) / 2)
        if needed > 0:
            x1 = max(0, x1 - needed)
            x2 = min(width, x2 + needed)

    if ratio > PLATE_MAX_RATIO:
        needed = int(((box_w / PLATE_MAX_RATIO) - box_h) / 2)
        if needed > 0:
            y1 = max(0, y1 - needed)
            y2 = min(height, y2 + needed)

    return x1, y1, x2, y2