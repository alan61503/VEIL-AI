"""OCR helpers for extracting reliable plate strings from cropped regions."""

from collections import deque
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
PLATE_GROUP_PATTERN = re.compile(r"^([A-Z]{2})([0-9]{1,2})([A-Z]{1,3})([0-9]{3,4})$") if PLATE_REGEX else None
ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
FAST_VARIANT_COUNT = 2
MAX_VALID_CANDIDATES = 8
MIN_VARIANT_DIM = 96
MAX_VARIANT_DIM = 320
LINE_GAP_FRACTION = 0.2
PARAGRAPH_FALLBACK_CONF = 0.45
ROTATION_ANGLES = (-15, -10, -5, 5, 10, 15)
LOW_CONFIDENCE_FLOOR = max(0.2, MIN_OCR_CONFIDENCE - 0.2)
NOISE_TOKENS = (
    "INDIAN",
    "INDIAH",
    "INDIA",
    "IND1A",
    "IND",
    "BHARATH",
    "BHARAT",
)
STATE_PREFIXES = {
    "AN",
    "AP",
    "AR",
    "AS",
    "BR",
    "CG",
    "CH",
    "DD",
    "DL",
    "GA",
    "GJ",
    "HP",
    "HR",
    "JH",
    "JK",
    "KA",
    "KL",
    "LA",
    "LD",
    "MH",
    "ML",
    "MN",
    "MP",
    "MZ",
    "NL",
    "OD",
    "PB",
    "PY",
    "RJ",
    "SK",
    "TN",
    "TS",
    "TR",
    "UK",
    "UP",
    "WB",
}
STATE_PREFIX_OVERRIDES = {
    "HB": "WB",
    "IH": "HR",
    "NH": "MH",
}
SUBSTITUTIONS = {
    "H": ["W", "R"],
    "W": ["H"],
    "O": ["0"],
    "0": ["O", "Q", "D"],
    "Q": ["O"],
    "I": ["H", "1", "L"],
    "1": ["I", "L"],
    "L": ["4", "1"],
    "4": ["L", "A"],
    "B": ["8"],
    "8": ["B"],
    "G": ["6"],
    "6": ["G"],
    "S": ["5"],
    "5": ["S"],
    "J": ["3"],
    "3": ["J"],
    "Z": ["2"],
    "2": ["Z"],
    "7": ["T", "Y"],
    "9": ["G"],
    "D": ["0"],
}
SERIES_OPTION_LIMIT = 16
SERIES_CHAR_OPTIONS = {
    "0": ["D", "Q"],
    "O": ["D", "O", "Q"],
    "1": ["I"],
    "2": ["Z"],
    "3": ["B"],
    "4": ["A"],
    "5": ["S"],
    "6": ["G"],
    "7": ["Y", "T"],
    "8": ["B"],
    "9": ["G"],
}
NUMBER_CHAR_FIX = {
    "O": "0",
    "D": "0",
    "Q": "0",
    "B": "8",
    "S": "5",
    "G": "6",
    "Z": "2",
    "I": "1",
    "L": "1",
}


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


def _post_correct(text: str) -> str:
    if not PLATE_PATTERN:
        return text

    seeds = _generate_correction_seeds(text)
    match = _match_with_substitutions(seeds)
    if match:
        enforced = _enforce_state_prefix(match)
        return _normalize_plate_segments(enforced)

    if PLATE_PATTERN.fullmatch(text):
        enforced = _enforce_state_prefix(text)
        return _normalize_plate_segments(enforced)
    return text


def _strip_noise(text: str) -> str:
    cleaned = text
    for noise in NOISE_TOKENS:
        cleaned = cleaned.replace(noise, "")
    return cleaned


def _generate_correction_seeds(text: str) -> List[str]:
    cleaned = _strip_noise(text)
    seeds = []
    seen = set()

    def add_candidate(candidate: str):
        if candidate and candidate not in seen:
            seen.add(candidate)
            seeds.append(candidate)

    for candidate in filter(None, [text, cleaned]):
        add_candidate(candidate)
        length = len(candidate)
        for idx in range(1, length):
            rotated = candidate[idx:] + candidate[:idx]
            add_candidate(rotated)

    seeds.sort(key=_seed_priority, reverse=True)
    return seeds


def _seed_priority(text: str) -> tuple:
    has_prefix_letters = int(len(text) >= 2 and text[:2].isalpha())
    has_district_digits = int(len(text) >= 4 and text[2:4].isdigit())
    return (has_prefix_letters, has_district_digits, -len(text))


def _match_with_substitutions(seeds: List[str]) -> str | None:
    queue = deque()
    seen = set()

    for seed in seeds:
        if seed in seen:
            continue
        seen.add(seed)
        queue.append((seed, 0))

    best_match = None
    best_score = None
    max_depth = 4

    while queue:
        current, depth = queue.popleft()

        matched = bool(PLATE_PATTERN.fullmatch(current))
        if matched:
            score = (_candidate_score(current, 1.0), -depth)
            if best_score is None or score > best_score:
                best_match = current
                best_score = score

        if depth >= max_depth:
            continue

        for idx, ch in enumerate(current):
            replacements = SUBSTITUTIONS.get(ch, [])
            if not replacements:
                continue
            for replacement in replacements:
                candidate = current[:idx] + replacement + current[idx + 1 :]
                if candidate in seen:
                    continue
                seen.add(candidate)
                queue.append((candidate, depth + 1))

    return best_match


def _enforce_state_prefix(text: str) -> str:
    if len(text) < 2 or not STATE_PREFIXES:
        return text

    prefix = text[:2]
    override = STATE_PREFIX_OVERRIDES.get(prefix)
    if override:
        candidate = override + text[2:]
        if not PLATE_PATTERN or PLATE_PATTERN.fullmatch(candidate):
            return candidate

    if prefix in STATE_PREFIXES:
        return text

    best_code = prefix
    best_diff = 3
    for code in STATE_PREFIXES:
        diff = sum(1 for a, b in zip(prefix, code) if a != b)
        if diff < best_diff:
            best_diff = diff
            best_code = code

    if best_code != prefix and best_diff <= 2:
        candidate = best_code + text[2:]
        if not PLATE_PATTERN or PLATE_PATTERN.fullmatch(candidate):
            return candidate
    return text


def _normalize_plate_segments(text: str) -> str:
    if not PLATE_GROUP_PATTERN:
        return text

    match = PLATE_GROUP_PATTERN.fullmatch(text)
    if not match:
        return text

    state, district, series, number = match.groups()

    if len(district) == 1 and series and len(series) > 1:
        leading = series[0]
        if leading in {"O", "Q"}:
            district = district + "0"
            series = series[1:]

    series_options = _expand_series_options(series)
    normalized_number = _fix_number_block(number)

    best_text = state + district + series_options[0] + normalized_number
    best_score = _candidate_score(best_text, 1.0)

    for option in series_options[1:SERIES_OPTION_LIMIT]:
        candidate = state + district + option + normalized_number
        score = _candidate_score(candidate, 1.0)
        if score > best_score:
            best_text = candidate
            best_score = score

    return best_text


def _expand_series_options(series: str) -> List[str]:
    if not series:
        return [series]

    options = [""]
    for ch in series:
        replacements = SERIES_CHAR_OPTIONS.get(ch, [ch])
        new_options = []
        for base in options:
            for repl in replacements:
                new_options.append(base + repl)
        options = new_options[:SERIES_OPTION_LIMIT]
    augmented = options[:]
    for option in options:
        if len(option) > 1:
            augmented.append(option[:-1])
    deduped = []
    seen = set()
    for option in augmented:
        if option and option not in seen:
            seen.add(option)
            deduped.append(option)
    return deduped or [series]


def _fix_number_block(number: str) -> str:
    chars = []
    for ch in number:
        if ch in NUMBER_CHAR_FIX:
            chars.append(NUMBER_CHAR_FIX[ch])
        else:
            chars.append(ch)
    return "".join(chars)


def _candidate_score(text: str, confidence: float) -> tuple:
    regex_match = 1 if PLATE_PATTERN and PLATE_PATTERN.fullmatch(text) else 0
    prefix_letters = 1 if len(text) >= 2 and text[:2].isalpha() else 0
    suffix_digits = 1 if len(text) >= 2 and text[-2:].isdigit() else 0
    state_bonus = 1 if len(text) >= 2 and text[:2] in STATE_PREFIXES else 0
    length_penalty = -abs(len(text) - 10)
    district_bonus = 0
    series_bonus = 0
    if PLATE_GROUP_PATTERN:
        match = PLATE_GROUP_PATTERN.fullmatch(text)
        if match:
            district_len = len(match.group(2))
            series_len = len(match.group(3))
            district_bonus = 1 if district_len == 2 else 0
            series_bonus = 1 if 1 <= series_len <= 2 else 0

    return (
        state_bonus,
        district_bonus,
        series_bonus,
        regex_match,
        prefix_letters + suffix_digits,
        confidence,
        length_penalty,
    )


def _valid_candidate(text: str) -> bool:
    length = len(text)
    if length < MIN_PLATE_LENGTH or length > PLATE_MAX_LENGTH:
        return False

    digit_count = sum(ch.isdigit() for ch in text)
    if digit_count < PLATE_MIN_DIGITS:
        return False

    regex_ok = PLATE_PATTERN.fullmatch(text) if PLATE_PATTERN else True
    heuristics_ok = _basic_plate_heuristics(text, digit_count)

    if PLATE_PATTERN and PLATE_REQUIRE_REGEX:
        return bool(regex_ok)

    return heuristics_ok or bool(regex_ok)


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
        if not cleaned:
            continue
        adjusted_conf = conf
        if conf < MIN_OCR_CONFIDENCE:
            if conf < LOW_CONFIDENCE_FLOOR or len(cleaned) < 4:
                continue
            adjusted_conf = conf * 0.85
        hits.append((adjusted_conf, cleaned))

        if not bbox:
            continue

        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        center_x = sum(xs) / len(xs)
        center_y = sum(ys) / len(ys)
        line_entries.append({"x": center_x, "y": center_y, "conf": adjusted_conf, "text": cleaned})

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
            if len(valid) >= MAX_VALID_CANDIDATES:
                break

    return valid, collected


def _read_candidates(images: Iterable) -> List:
    variants = list(images)
    fast_variants = variants[:FAST_VARIANT_COUNT]
    slow_variants = variants[FAST_VARIANT_COUNT:]

    valid, collected = _evaluate_variants(fast_variants)
    slow_valid: List[tuple[float, str]] = []
    slow_collected: List[tuple[float, str]] = []

    if len(valid) < MAX_VALID_CANDIDATES and slow_variants:
        slow_valid, slow_collected = _evaluate_variants(slow_variants)
        collected.extend(slow_collected)
        valid.extend(slow_valid)

    if valid:
        return valid[:MAX_VALID_CANDIDATES]

    return collected


def _select_best(candidates: List[tuple[float, str]]):
    if not candidates:
        return None

    return max(candidates, key=lambda c: _candidate_score(c[1], c[0]))


def _combine_candidates(candidates: List[tuple[float, str]]):
    if len(candidates) < 2:
        return None

    sorted_cands = sorted(candidates, key=lambda c: _candidate_score(c[1], c[0]), reverse=True)[:5]
    best = None
    best_score = None

    for i in range(len(sorted_cands)):
        for j in range(len(sorted_cands)):
            if i == j:
                continue
            combined_text = sorted_cands[i][1] + sorted_cands[j][1]
            if len(combined_text) > PLATE_MAX_LENGTH:
                continue
            if not _valid_candidate(combined_text):
                continue
            conf = min(sorted_cands[i][0], sorted_cands[j][0])
            corrected = _post_correct(combined_text)
            score = _candidate_score(corrected, conf)
            if not best_score or score > best_score:
                best = (corrected, conf)
                best_score = score

    return best



def read_plate(plate_img):
    if plate_img is None or plate_img.size == 0:
        return None

    variants = _preprocess_variants(plate_img)
    candidates = _read_candidates(variants)

    filtered = [c for c in candidates if _valid_candidate(c[1])]
    choice = _select_best(filtered) if filtered else None
    combined = None

    if choice:
        corrected = _post_correct(choice[1])
        if not (PLATE_PATTERN and PLATE_PATTERN.fullmatch(corrected)):
            combined = _combine_candidates(candidates)
            if combined:
                combined_corrected = _post_correct(combined[0])
                if _candidate_score(combined_corrected, combined[1]) > _candidate_score(corrected, choice[0]):
                    return combined_corrected, combined[1]
        return corrected, choice[0]

    combined = _combine_candidates(candidates)
    if combined:
        corrected = _post_correct(combined[0])
        return corrected, combined[1]
    return None
