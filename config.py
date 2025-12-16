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

