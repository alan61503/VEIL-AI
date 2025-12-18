"""OCR helpers for extracting reliable plate strings from cropped regions."""

import re
from typing import Iterable, List

import cv2
import easyocr

from config import (
    MIN_OCR_CONFIDENCE,
    MIN_PLATE_LENGTH,
    PLATE_MAX_LENGTH,
    PLATE_MIN_DIGITS,
    PLATE_REGEX,
    PLATE_REQUIRE_REGEX,
)

reader = easyocr.Reader(['en'], gpu=False)
PLATE_PATTERN = re.compile(PLATE_REGEX) if PLATE_REGEX else None
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
FAST_VARIANT_COUNT = 2
MIN_VARIANT_DIM = 96
MAX_VARIANT_DIM = 320
LINE_GAP_FRACTION = 0.2
PARAGRAPH_FALLBACK_CONF = 0.45
ROTATION_ANGLES = (-15, -10, -5, 5, 10, 15)


def _scale_variant(img):
    height, width = img.shape[:2]
    if height == 0 or width == 0:
        return img

    largest = max(height, width)
    smallest = min(height, width)
    scale = 1.0

    if largest > MAX_VARIANT_DIM:
        scale = MAX_VARIANT_DIM / float(largest)
    elif smallest < MIN_VARIANT_DIM and smallest > 0:
        scale = min(1.5, MIN_VARIANT_DIM / float(smallest))

    if scale == 1.0:
        return img

    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(img, new_size, interpolation=interpolation)


def _rotate(img, angle: float):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return img
    center = (w // 2, h // 2)
    mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _sharpen(gray):
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)


def _preprocess_variants(plate_img) -> List:
    scaled = _scale_variant(plate_img)

    variants: List = []
    bases = [scaled] + [_rotate(scaled, ang) for ang in ROTATION_ANGLES]

    for base in bases:
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 17, 17)
        sharp = _sharpen(gray)

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        inverted = cv2.bitwise_not(adaptive)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, morph_kernel)

        variants.extend([gray, sharp, adaptive, inverted, clahe, closed])

    return variants


def _clean_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _valid_candidate(text: str) -> bool:
    length = len(text)
    if length < MIN_PLATE_LENGTH or length > PLATE_MAX_LENGTH:
        return False

    digit_count = sum(ch.isdigit() for ch in text)
    if digit_count < PLATE_MIN_DIGITS:
        return False

    regex_ok = PLATE_PATTERN.fullmatch(text) if PLATE_PATTERN else True
    heuristics_ok = _basic_plate_heuristics(text, digit_count)

    if PLATE_PATTERN:
        if regex_ok:
            return True
        if PLATE_REQUIRE_REGEX:
            return False

    return heuristics_ok


def _basic_plate_heuristics(text: str, digit_count: int) -> bool:
    letter_count = sum(ch.isalpha() for ch in text)
    if letter_count == 0 or digit_count == 0:
        return False

    ratio = digit_count / max(1, letter_count)
    if ratio < 0.25 or ratio > 4.0:
        return False

    prefix_letters = len(text) >= 2 and text[:2].isalpha()
    suffix_digits = len(text) >= 2 and text[-2:].isdigit()
    return prefix_letters or suffix_digits


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

    # Skip paragraph-mode fallback when strict regex is required to avoid loose text
    if not PLATE_REQUIRE_REGEX:
        for text in reader.readtext(img, detail=0, paragraph=True):
            cleaned = _clean_text(text)
            if cleaned:
                hits.append((PARAGRAPH_FALLBACK_CONF, cleaned))

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


def read_plate(plate_img):
    if plate_img is None or plate_img.size == 0:
        return None

    variants = _preprocess_variants(plate_img)
    candidates = _read_candidates(variants)

    filtered = [c for c in candidates if _valid_candidate(c[1])]
    choice = _select_best(filtered)
    if choice:
        return choice[1], choice[0]
    return None
