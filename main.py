import cv2

from config import CAMERA_SOURCE, CLOUD_ENABLED
from cloud.sync_worker import sync_pending
from db.database import init_db
from pipeline.frame_processor import process_frame


def run_camera() -> None:
    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        raise RuntimeError("Unable to access camera source. Check CAMERA_SOURCE in config.py")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            process_frame(frame)

            cv2.imshow("VEIL", frame)
            if cv2.waitKey(1) == 27:  # ESC to quit
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> None:
    init_db()
    run_camera()
    if CLOUD_ENABLED:
        sync_pending()


if __name__ == "__main__":
    main()
