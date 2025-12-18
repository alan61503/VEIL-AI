# VEIL-AI
Vehicle Edge Intelligence Logger - advanced AI for vehicular security and monitoring.

## Plate detection model

Download a license-plate YOLO checkpoint (for example [keremberke/yolov8n-license-plate](https://huggingface.co/keremberke/yolov8n-license-plate)) and place it at `models/yolov8n-license-plate.pt`, or set the `PLATE_MODEL_PATH` environment variable to your custom `.pt` file before running `main.py` or `main_video.py`.

### Training with the Indian Kaggle dataset

You can fine-tune a YOLO detector on the [Indian license plates with labels](https://www.kaggle.com/datasets/kedarsai/indian-license-plates-with-labels) dataset directly from this repo:

1. **Prepare the splits**

	```bash
	C:/btech/development/VEIL-AI/veilenv/Scripts/python.exe scripts/prepare_indian_lp_dataset.py
	```

	The script downloads the dataset via `kagglehub`, creates deterministic `train/val/test` splits under `data/indian_lp/`, and writes a YOLO `data.yaml` file.

2. **Train with Ultralytics YOLO**

	```bash
	yolo detect train model=yolov8n.pt data=data/indian_lp/data.yaml epochs=50 imgsz=640
	```

	Replace `yolov8n.pt` with any starting checkpoint. Final weights will be emitted under `runs/detect/.../weights/best.pt`.

3. **Update the runtime model**

	Point `PLATE_MODEL_PATH` to the newly trained checkpoint (or copy it to `models/yolov8n-license-plate.pt`) and rerun `main_video.py` to evaluate accuracy improvements in the VEIL pipeline.

## Cloud sync configuration

By default the pipeline attempts to sync completed entries to Firebase Cloud Firestore. Provide credentials in one of two ways:

1. **File path** – download your Firebase service-account JSON and set `FIREBASE_CREDENTIALS` to its absolute path (defaults to `serviceAccount.json` in the project root).
2. **Inline JSON** – set `FIREBASE_CREDENTIALS_JSON` to the raw JSON string (useful for CI or secrets managers). The app writes it to `.cache/firebase_credentials.json` automatically.

If you prefer a REST endpoint instead of Firebase, set `CLOUD_PROVIDER=rest`, point `CLOUD_ENDPOINT` to your API, and set `CLOUD_API_KEY` to the corresponding bearer token. Use `--no-cloud` on `main.py` / `main_video.py` for fully offline runs.
