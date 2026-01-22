from pathlib import Path
import warnings

import cv2
warnings.filterwarnings(
    "ignore",
    message=".*'pin_memory' argument is set as true but no accelerator is found.*",
    category=UserWarning,
    module="torch.utils.data.dataloader",
)

from config import CLOUD_ENABLED
from cloud.sync_worker import sync_pending
from db.database import init_db
from pipeline.frame_processor import process_frame

IMAGE_DIR = Path("data/images")


def process_images(image_dir: Path = IMAGE_DIR) -> None:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = sorted(p for p in image_dir.iterdir() if p.is_file())
    if not image_paths:
        print(f"No images found in {image_dir}.")
        return

    total = len(image_paths)
    recognized_hits = 0
    saved_entries = 0

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{total}] {image_path.name}")
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        summary = process_frame(
            frame,
            min_plate_hits=1,
            dedupe_enabled=False,
            track_exits=False,
        )

        recognized = summary.get("recognized", [])
        entries = summary.get("entries", [])
        detections = summary.get("detections", 0)

        if recognized:
            recognized_hits += 1
            saved_entries += len(entries)
            entry_note = f"saved: {', '.join(entries)}" if entries else "not persisted (duplicate/regex)"
            print(f"    Plates read: {', '.join(recognized)} ({entry_note})")
        else:
            if detections > 0:
                print("    Plate detected but OCR rejected the text.")
            else:
                print("    No plate detected in this frame.")

    print("\n===== Image Batch Summary =====")
    print(f"Images processed : {total}")
    print(f"OCR hits         : {recognized_hits}")
    print(f"Unique entries   : {saved_entries}")
    print(f"Misses           : {total - recognized_hits}")


def main() -> None:
    init_db()
    process_images()
    if CLOUD_ENABLED:
        sync_pending()


if __name__ == "__main__":
    main()
