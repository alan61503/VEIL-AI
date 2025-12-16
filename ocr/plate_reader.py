import re

import cv2
import easyocr
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)


def _preprocess(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )
    upscaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return upscaled


def _clean_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def read_plate(plate_img):
    if plate_img is None or plate_img.size == 0:
        return None

    processed = _preprocess(plate_img)
    results = reader.readtext(
        processed,
        detail=0,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )

    for text in results:
        cleaned = _clean_text(text)
        if len(cleaned) >= 4:
            return cleaned

    return None
