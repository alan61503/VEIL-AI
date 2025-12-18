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

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        process_frame(frame, min_plate_hits=1)

    print("Image processing finished.")


def main() -> None:
    init_db()
    process_images()
    if CLOUD_ENABLED:
        sync_pending()


if __name__ == "__main__":
    main()
