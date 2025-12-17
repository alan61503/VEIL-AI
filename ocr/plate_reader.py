import re

import cv2
import easyocr
import numpy as np

reader = easyocr.Reader(['en'], gpu=False)
<<<<<<< HEAD
PLATE_PATTERN = re.compile(PLATE_REGEX) if PLATE_REGEX else None
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
FAST_VARIANT_COUNT = 2
MIN_VARIANT_DIM = 96
MAX_VARIANT_DIM = 320
LINE_GAP_FRACTION = 0.2
=======
>>>>>>> parent of 1b12b7d (.)


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
<<<<<<< HEAD

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    inverted = cv2.bitwise_not(adaptive)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, morph_kernel)

    return [gray, adaptive, inverted, clahe, closed]
=======
    upscaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return upscaled
>>>>>>> parent of 1b12b7d (.)


def _clean_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


<<<<<<< HEAD
def _valid_candidate(text: str) -> bool:
    if len(text) < MIN_PLATE_LENGTH:
        return False
    if PLATE_PATTERN and not PLATE_PATTERN.fullmatch(text):
        return False
    return True


def _read_variant(img) -> List[tuple[float, str]]:
    hits: List[tuple[float, str]] = []
    line_entries: List[dict] = []
    results = reader.readtext(img, detail=1, allowlist=ALLOWLIST)
    height = img.shape[0] if len(img.shape) > 1 else 0
    line_gap = max(12.0, height * LINE_GAP_FRACTION)

    for bbox, text, conf in results:
        cleaned = _clean_text(text)
        if not cleaned or conf < MIN_OCR_CONFIDENCE:
            continue
        hits.append((conf, cleaned))

        if not bbox:
            continue

        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        line_entries.append({"x": center_x, "y": center_y, "conf": conf, "text": cleaned})

    hits.extend(_merge_multiline(line_entries, line_gap))

    paragraph_text = reader.readtext(img, detail=0, paragraph=True)
    for text in paragraph_text:
        cleaned = _clean_text(text)
        if not cleaned:
            continue
        hits.append((0.45, cleaned))

    return sorted(hits, key=lambda item: item[0], reverse=True)


def _merge_multiline(entries: List[dict], line_gap: float) -> List[tuple[float, str]]:
    if len(entries) < 2:
        return []

    sorted_entries = sorted(entries, key=lambda item: item["y"])
    mid_index = len(sorted_entries) // 2
    median_y = sorted_entries[mid_index]["y"]

    top = [entry for entry in sorted_entries if entry["y"] <= median_y]
    bottom = [entry for entry in sorted_entries if entry["y"] > median_y]

    if not top or not bottom:
        return []

    top_max = max(entry["y"] for entry in top)
    bottom_min = min(entry["y"] for entry in bottom)

    if bottom_min - top_max < max(10.0, line_gap):
        return []

    def build_line(items: List[dict]) -> tuple[str, float]:
        ordered = sorted(items, key=lambda item: item["x"])
        text = "".join(item["text"] for item in ordered)
        conf = min(item["conf"] for item in ordered)
        return text, conf

    top_text, top_conf = build_line(top)
    bottom_text, bottom_conf = build_line(bottom)
    if not top_text or not bottom_text:
        return []

    merged_conf = min(top_conf, bottom_conf)
    return [(merged_conf, top_text + bottom_text)]


def _evaluate_variants(variants: List) -> tuple[List[tuple[float, str]], List[tuple[float, str]]]:
    valid: List[tuple[float, str]] = []
    collected: List[tuple[float, str]] = []

    for img in variants:
        hits = _read_variant(img)
        if not hits:
            continue

        collected.extend(hits)
        variant_valid = [hit for hit in hits if _valid_candidate(hit[1])]
        if variant_valid:
            valid.extend(variant_valid)
            break

    return valid, collected


def _read_candidates(images: Iterable) -> List:
    variants = list(images)
    fast_variants = variants[:FAST_VARIANT_COUNT]
    slow_variants = variants[FAST_VARIANT_COUNT:]

    valid, collected = _evaluate_variants(fast_variants)
    if valid:
        return valid

    slow_valid, slow_collected = _evaluate_variants(slow_variants)
    if slow_valid:
        return slow_valid

    collected.extend(slow_collected)
    return collected


def _select_best(candidates: List[tuple[float, str]]):
    if not candidates:
        return None

    return max(candidates, key=lambda c: (len(c[1]), c[0]))


=======
>>>>>>> parent of 1b12b7d (.)
def read_plate(plate_img):
    if plate_img is None or plate_img.size == 0:
        return None

    processed = _preprocess(plate_img)
    results = reader.readtext(
        processed,
        detail=0,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    )

<<<<<<< HEAD
    filtered = [c for c in candidates if _valid_candidate(c[1])]
    choice = _select_best(filtered) or _select_best(candidates)
    if choice:
        return choice[1], choice[0]
=======
    for text in results:
        cleaned = _clean_text(text)
        if len(cleaned) >= 4:
            return cleaned

>>>>>>> parent of 1b12b7d (.)
    return None
