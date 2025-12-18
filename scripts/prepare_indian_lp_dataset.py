"""Utility to download and split the Kaggle Indian license plate dataset.

This script uses kagglehub to fetch the dataset
`kedarsai/indian-license-plates-with-labels`, performs a deterministic
train/val/test split, and creates a YOLO-friendly folder structure under
`data/indian_lp/`:

```
data/indian_lp/
  raw/                # Optional cache of the original download
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
  data.yaml           # Ready to use with Ultralytics YOLO
  summary.json        # Helpful counts for reference
```

Example usage (from the repo root):

```
C:/btech/development/VEIL-AI/veilenv/Scripts/python.exe scripts/prepare_indian_lp_dataset.py
```
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import kagglehub

DATASET_ID = "kedarsai/indian-license-plates-with-labels"
DEFAULT_SPLITS = {
    "train": 0.8,
    "val": 0.1,
    "test": 0.1,
}
RNG_SEED = 1337
CLASS_NAMES = ["plate"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/indian_lp"),
        help="Destination folder that will store the split dataset.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help=(
            "Optional path to an existing download that already contains "
            "'images' and 'labels' folders. When omitted the dataset is "
            "downloaded with kagglehub."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RNG_SEED,
        help="Seed for the deterministic shuffle.",
    )
    return parser.parse_args()


def download_source() -> Path:
    path = Path(kagglehub.dataset_download(DATASET_ID))
    print(f"Downloaded dataset to {path}")
    return path


def collect_pairs(base: Path) -> Tuple[List[Tuple[Path, Path]], List[Path]]:
    img_dir = base / "images"
    label_dir = base / "labels"
    if not img_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            f"Expected 'images' and 'labels' folders under {base}, "
            "but at least one of them is missing."
        )

    pairs: List[Tuple[Path, Path]] = []
    unlabeled: List[Path] = []

    for img_path in sorted(img_dir.glob("*")):
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            pairs.append((img_path, label_path))
        else:
            unlabeled.append(img_path)

    if not pairs:
        raise RuntimeError(f"No labeled images found inside {base}")

    print(
        f"Found {len(pairs)} labeled images and {len(unlabeled)} unlabeled images under {base}."
    )
    return pairs, unlabeled


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def distribute_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    output: Path,
    splits: dict,
) -> dict:
    split_names = list(splits.keys())
    total = len(pairs)
    counts = {key: 0 for key in split_names}

    allocated = []
    running_total = 0
    for name in split_names[:-1]:
        size = int(round(splits[name] * total))
        allocated.append((name, size))
        running_total += size
    last_name = split_names[-1]
    allocated.append((last_name, total - running_total))

    offset = 0
    for split_name, split_count in allocated:
        split_pairs = pairs[offset : offset + split_count]
        offset += split_count

        img_dest = output / split_name / "images"
        label_dest = output / split_name / "labels"
        ensure_clean_dir(img_dest)
        ensure_clean_dir(label_dest)

        for img_path, label_path in split_pairs:
            shutil.copy2(img_path, img_dest / img_path.name)
            shutil.copy2(label_path, label_dest / label_path.name)

        counts[split_name] = len(split_pairs)
        print(f"Copied {len(split_pairs)} samples into {split_name}.")

    return counts


def copy_unlabeled(unlabeled: Iterable[Path], output: Path) -> int:
    unlabeled_list = list(unlabeled)
    if not unlabeled_list:
        return 0

    img_dest = output / "unlabeled" / "images"
    ensure_clean_dir(img_dest)
    for img_path in unlabeled_list:
        shutil.copy2(img_path, img_dest / img_path.name)
    print(f"Copied {len(unlabeled_list)} unlabeled images for future use.")
    return len(unlabeled_list)


def write_yaml(output: Path) -> None:
    yaml_path = output / "data.yaml"
    content = """
    path: {path}
    train: train/images
    val: val/images
    test: test/images
    names:
      - {class_name}
    """.strip().format(path=output.resolve().as_posix(), class_name=CLASS_NAMES[0])
    yaml_path.write_text(content)
    print(f"Wrote YOLO data config to {yaml_path}")


def write_summary(output: Path, counts: dict, unlabeled_count: int) -> None:
    summary_path = output / "summary.json"
    payload = {
        "dataset_id": DATASET_ID,
        "splits": counts,
        "unlabeled": unlabeled_count,
        "class_names": CLASS_NAMES,
    }
    summary_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote dataset summary to {summary_path}")


def main() -> None:
    args = parse_args()
    output_root = args.output
    output_root.mkdir(parents=True, exist_ok=True)

    base = args.source if args.source else download_source()
    if args.source:
        print(f"Using existing dataset at {base}")

    pairs, unlabeled = collect_pairs(base)

    random.seed(args.seed)
    shuffled = list(pairs)
    random.shuffle(shuffled)

    counts = distribute_pairs(shuffled, output_root, DEFAULT_SPLITS)
    unlabeled_count = copy_unlabeled(unlabeled, output_root)

    raw_dir = output_root / "raw"
    ensure_clean_dir(raw_dir)
    shutil.copytree(base, raw_dir, dirs_exist_ok=True)

    write_yaml(output_root)
    write_summary(output_root, counts, unlabeled_count)


if __name__ == "__main__":
    main()
