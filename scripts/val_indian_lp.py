"""Quick Ultralytics YOLO validation helper for the Indian LP dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the detector checkpoint (best.pt).",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/indian_lp/data.yaml"),
        help="YOLO data.yaml file produced by prepare_indian_lp_dataset.py.",
    )
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate (test/val/train).")
    parser.add_argument("--imgsz", type=int, default=640, help="Evaluation image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for validation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(f"Missing model at {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Missing data config at {args.data}")

    model = YOLO(str(args.model))
    results = model.val(data=str(args.data), split=args.split, imgsz=args.imgsz, batch=args.batch)
    print("\nValidation metrics:")
    for key, value in sorted(results.results_dict.items()):
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()