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

