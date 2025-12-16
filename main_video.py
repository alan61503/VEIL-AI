import cv2
from pathlib import Path

from config import CLOUD_ENABLED
from cloud.sync_worker import sync_pending
from db.database import init_db
from pipeline.frame_processor import process_frame

VIDEO_PATH = Path("sample_car_video.mp4")
FRAME_INTERVAL = 5  # process every Nth frame to reduce load


def process_video(video_path: Path = VIDEO_PATH, frame_interval: int = FRAME_INTERVAL) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Unable to read video file: {video_path}")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_interval > 1 and frame_count % frame_interval != 0:
                continue

            process_frame(frame)
    finally:
        cap.release()

    print("Video processing finished.")


def main() -> None:
    init_db()
    process_video()
    if CLOUD_ENABLED:
        sync_pending()


if __name__ == "__main__":
    main()
