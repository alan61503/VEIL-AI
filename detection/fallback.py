"""Classical computer-vision fallback for license plate detection."""
from typing import List

import cv2
import numpy as np


def contour_detect_plates(frame, max_results: int = 3) -> List:
    """Detect plate-shaped contours when the ML model finds nothing."""
    if frame is None or frame.size == 0:
        return []

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    plates = []
    height, width = frame.shape[:2]
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 1500:
            continue

        ratio = w / float(h)
        if ratio < 2.0 or ratio > 6.0:
            continue

        pad_w = int(w * 0.1)
        pad_h = int(h * 0.2)

        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(width, x + w + pad_w)
        y2 = min(height, y + h + pad_h)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        plates.append(crop)
        if len(plates) >= max_results:
            break

    return plates
