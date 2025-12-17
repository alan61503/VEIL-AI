# VEIL-AI
Vehicle Edge Intelligence Logger - advanced AI for vehicular security and monitoring.

## Plate detection model

Download a license-plate YOLO checkpoint (for example [keremberke/yolov8n-license-plate](https://huggingface.co/keremberke/yolov8n-license-plate)) and place it at `models/yolov8n-license-plate.pt`, or set the `PLATE_MODEL_PATH` environment variable to your custom `.pt` file before running `main.py` or `main_video.py`.

## Cloud sync configuration

By default the pipeline attempts to sync completed entries to Firebase Cloud Firestore. Provide credentials in one of two ways:

1. **File path** – download your Firebase service-account JSON and set `FIREBASE_CREDENTIALS` to its absolute path (defaults to `serviceAccount.json` in the project root).
2. **Inline JSON** – set `FIREBASE_CREDENTIALS_JSON` to the raw JSON string (useful for CI or secrets managers). The app writes it to `.cache/firebase_credentials.json` automatically.

If you prefer a REST endpoint instead of Firebase, set `CLOUD_PROVIDER=rest`, point `CLOUD_ENDPOINT` to your API, and set `CLOUD_API_KEY` to the corresponding bearer token. Use `--no-cloud` on `main.py` / `main_video.py` for fully offline runs.
