"""Batch evaluation of detection + OCR on a labeled plate dataset.

Run from the repo root, pointing to a folder with plate images and a CSV/JSON
file that maps each image to its ground-truth plate string. Example:

    python scripts/eval_plate_dataset.py \
        --images datasets/plates/images \
        --labels datasets/plates/labels.csv \
        --output reports/plate_eval.csv

The script will run the standard VEIL detector + OCR stack, compare predictions
to the ground truth, and print consolidated accuracy metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import cv2

from detection.detector import detect_plate
from ocr.plate_reader import read_plate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="Folder containing plate images to evaluate.")
    parser.add_argument(
        "--labels",
        type=Path,
        help="Optional CSV/JSON mapping image filenames to ground-truth text.",
    )
    parser.add_argument(
        "--image-field",
        default="image",
        help="CSV/JSON field that stores the image filename (default: image).",
    )
    parser.add_argument(
        "--label-field",
        default="plate",
        help="CSV/JSON field that stores the plate text (default: plate).",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        default=["*.jpg", "*.jpeg", "*.png"],
        help="Glob patterns to collect images (defaults to common formats).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file to save per-image predictions and scores.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images for a quick smoke test.")
    parser.add_argument(
        "--fallback-stem",
        action="store_true",
        help="Use the image filename stem as the label when --labels is omitted or missing entries.",
    )
    return parser.parse_args()


def load_labels(path: Optional[Path], image_field: str, label_field: str) -> Dict[str, str]:
    if not path:
        return {}

    if not path.exists():
        raise FileNotFoundError(f"Label file not found at {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_labels_csv(path, image_field, label_field)
    if suffix == ".json":
        return _load_labels_json(path, image_field, label_field)
    raise ValueError(f"Unsupported label format: {path.suffix}")


def _load_labels_csv(path: Path, image_field: str, label_field: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if image_field not in reader.fieldnames or label_field not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain '{image_field}' and '{label_field}' columns. Found: {reader.fieldnames}"
            )
        for row in reader:
            image = row[image_field].strip()
            plate = row[label_field].strip()
            if image and plate:
                mapping[_normalize_key(image)] = _clean_text(plate)
    return mapping


def _load_labels_json(path: Path, image_field: str, label_field: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        iterable = data.items()
        for key, value in iterable:
            mapping[_normalize_key(str(key))] = _clean_text(str(value))
        return mapping
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            if image_field in entry and label_field in entry:
                mapping[_normalize_key(str(entry[image_field]))] = _clean_text(str(entry[label_field]))
        return mapping
    raise ValueError("JSON labels must be a dict or a list of objects")


def _normalize_key(value: str) -> str:
    return value.replace("\\", "/").lower()


def _clean_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def collect_images(folder: Path, patterns: Iterable[str]) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder not found: {folder}")

    files: List[Path] = []
    for pattern in patterns:
        files.extend(folder.rglob(pattern))
    return sorted(set(files))


def evaluate_image(image_path: Path, ground_truth: Optional[str]) -> dict:
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    plate_crops = detect_plate(frame)
    best_prediction: Optional[str] = None
    best_conf = 0.0

    for crop in plate_crops:
        result = read_plate(crop)
        if not result:
            continue
        text, conf = result
        if conf > best_conf:
            best_conf = conf
            best_prediction = _clean_text(text)

    gt_clean = _clean_text(ground_truth) if ground_truth else None
    detection_hit = len(plate_crops) > 0
    exact_match = bool(best_prediction and gt_clean and best_prediction == gt_clean)
    similarity = SequenceMatcher(None, best_prediction or "", gt_clean or "").ratio()

    return {
        "image": image_path,
        "ground_truth": gt_clean,
        "prediction": best_prediction,
        "confidence": best_conf,
        "detected": detection_hit,
        "exact_match": exact_match,
        "similarity": similarity,
    }


def save_report(rows: List[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["image", "ground_truth", "prediction", "confidence", "detected", "exact_match", "similarity"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "image": row["image"].as_posix()})


def infer_label_from_stem(path: Path) -> Optional[str]:
    return _clean_text(path.stem)


def main() -> None:
    args = parse_args()
    label_map = load_labels(args.labels, args.image_field, args.label_field)
    files = collect_images(args.images, args.patterns)
    if args.limit > 0:
        files = files[: args.limit]

    if not files:
        raise RuntimeError("No images found for the provided patterns.")

    rows: List[dict] = []
    for idx, image_path in enumerate(files, start=1):
        lookup_key = _normalize_key(image_path.name)
        ground_truth = label_map.get(lookup_key)
        if ground_truth is None and args.fallback_stem:
            ground_truth = infer_label_from_stem(image_path)

        result = evaluate_image(image_path, ground_truth)
        rows.append(result)

        if idx % 25 == 0 or idx == len(files):
            print(f"Processed {idx}/{len(files)} images...")

    detection_rate = sum(1 for row in rows if row["detected"]) / len(rows)
    exact_rate = sum(1 for row in rows if row["exact_match"]) / len(rows)
    similarities = [row["similarity"] for row in rows]
    avg_similarity = mean(similarities) if similarities else 0.0

    print("\n===== Evaluation Summary =====")
    print(f"Images evaluated    : {len(rows)}")
    print(f"Detection hit rate  : {detection_rate:.2%}")
    print(f"Exact OCR match rate: {exact_rate:.2%}")
    print(f"Avg. similarity     : {avg_similarity:.3f}")

    if args.output:
        save_report(rows, args.output)
        print(f"Detailed report written to {args.output}")


if __name__ == "__main__":
    main()