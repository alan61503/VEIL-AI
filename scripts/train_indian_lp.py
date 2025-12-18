"""Convenience wrapper around Ultralytics YOLO training for the Indian LP dataset.

Example usage:

```
C:/btech/development/VEIL-AI/veilenv/Scripts/python.exe scripts/train_indian_lp.py \
    --model yolov8n.pt \
    --epochs 80 \
    --imgsz 640
```

By default it looks for the dataset prepared by
`scripts/prepare_indian_lp_dataset.py` (i.e. `data/indian_lp/data.yaml`).
The script simply forwards arguments to Ultralytics' `YOLO.train()` API
while keeping sensible defaults for a lightweight-yet-accurate setup.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_DATA = Path("data/indian_lp/data.yaml")
DEFAULT_MODEL = "yolov8n.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to the YOLO data.yaml file.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Base model checkpoint (pt or yaml).")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--name",
        type=str,
        default="indian-lp",
        help="Custom run name under runs/detect/",
    )
    parser.add_argument("--device", type=str, default="", help="Optional CUDA device (e.g., '0' or '0,1').")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume the last run for the provided model/data pair.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.data.exists():
        raise FileNotFoundError(
            f"Missing data config at {args.data}. Run scripts/prepare_indian_lp_dataset.py first."
        )

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device or None,
        resume=args.resume,
        optimizer="AdamW",
        cos_lr=True,
        patience=20,
    )


if __name__ == "__main__":
    main()
