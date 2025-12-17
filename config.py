# config.py

import os
from pathlib import Path


CAMERA_SOURCE = int(os.getenv("CAMERA_SOURCE", 0))  # 0 = laptop webcam, later Pi camera
DEVICE_ID = os.getenv("DEVICE_ID", "V.E.I.L_01")  # unique device identifier


CLOUD_ENABLED = os.getenv("CLOUD_ENABLED", "true").lower() == "true"
CLOUD_PROVIDER = os.getenv("CLOUD_PROVIDER", "firebase")
CLOUD_ENDPOINT = os.getenv("CLOUD_ENDPOINT", "https://example.com/api/vehicles")
CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "CHANGE_ME")
FIREBASE_CREDENTIALS = Path(os.getenv("FIREBASE_CREDENTIALS", "serviceAccount.json"))
FIREBASE_COLLECTION = os.getenv("FIREBASE_COLLECTION", "vehicles")
FIREBASE_CREDENTIALS_JSON = os.getenv("FIREBASE_CREDENTIALS_JSON")


MODELS_DIR = Path(os.getenv("MODELS_DIR", "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_default_model_path = MODELS_DIR / "yolov8n-license-plate.pt"
PLATE_MODEL_PATH = Path(os.getenv("PLATE_MODEL_PATH", str(_default_model_path)))
PLATE_MODEL_URL = os.getenv(
	"PLATE_MODEL_URL",
	"https://huggingface.co/keremberke/yolov8n-license-plate/resolve/main/yolov8n-license-plate.pt",
)

_plate_classes = os.getenv("PLATE_CLASS_IDS", "0")
PLATE_CLASS_IDS = [int(cls.strip()) for cls in _plate_classes.split(",") if cls.strip()]
PLATE_CONFIDENCE = float(os.getenv("PLATE_CONFIDENCE", "0.25"))
<<<<<<< HEAD
PLATE_MAX_RESULTS = int(os.getenv("PLATE_MAX_RESULTS", "3"))
PLATE_MARGIN = float(os.getenv("PLATE_MARGIN", "0.12"))
PLATE_MIN_RATIO = float(os.getenv("PLATE_MIN_RATIO", "2.4"))
PLATE_TALL_RATIO = float(os.getenv("PLATE_TALL_RATIO", "1.8"))
PLATE_TALL_PAD = float(os.getenv("PLATE_TALL_PAD", "0.75"))
PLATE_TALL_MULTIPLIER = float(os.getenv("PLATE_TALL_MULTIPLIER", "2.5"))
PLATE_TALL_TARGET_RATIO = float(os.getenv("PLATE_TALL_TARGET_RATIO", "1.5"))
PLATE_TALL_WIDTH_PAD = float(os.getenv("PLATE_TALL_WIDTH_PAD", "0.18"))
PLATE_TALL_UP_BIAS = float(os.getenv("PLATE_TALL_UP_BIAS", "0.7"))
PLATE_TOP_EXTRA = float(os.getenv("PLATE_TOP_EXTRA", "0.35"))
PLATE_FORCE_TALL = os.getenv("PLATE_FORCE_TALL", "true").lower() == "true"
PLATE_MAX_RATIO = float(os.getenv("PLATE_MAX_RATIO", "6.5"))

PLATE_REGEX = os.getenv("PLATE_REGEX", r"[A-Z0-9]{4,10}")
MIN_PLATE_LENGTH = int(os.getenv("MIN_PLATE_LENGTH", "5"))
MIN_OCR_CONFIDENCE = float(os.getenv("MIN_OCR_CONFIDENCE", "0.4"))
MIN_PLATE_HITS = int(os.getenv("MIN_PLATE_HITS", "2"))
=======
>>>>>>> parent of 1b12b7d (.)

