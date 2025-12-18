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

	The script downloads the dataset via `kagglehub`, creates deterministic `train/val/test` splits under `data/indian_lp/`, mirrors the original download to `data/indian_lp/raw/`, and writes both a YOLO `data.yaml` and a `summary.json` with useful counts.

2. **Train with Ultralytics YOLO**

	```bash
	# Option A: call Ultralytics directly
	yolo detect train model=yolov8n.pt data=data/indian_lp/data.yaml epochs=80 imgsz=640

	# Option B: use the helper wrapper (adds AdamW + cosine LR + patience)
	C:/btech/development/VEIL-AI/veilenv/Scripts/python.exe scripts/train_indian_lp.py \
		 --model yolov8n.pt \
		 --epochs 80 \
		 --imgsz 640
	```

	Replace `yolov8n.pt` with any starting checkpoint. Final weights will be emitted under `runs/detect/.../weights/best.pt`.

3. **Validate + wire into VEIL**

	```bash
	# Quick validation report on the held-out test split
	yolo detect val model=runs/detect/<run-name>/weights/best.pt data=data/indian_lp/data.yaml split=test
	# or use the helper if PowerShell quoting is painful
	C:/btech/development/VEIL-AI/veilenv/Scripts/python.exe scripts/val_indian_lp.py \
		--model runs/detect/<run-name>/weights/best.pt \
		--split test --imgsz 640 --batch 16

	# Update VEIL to use the refined detector
	copy runs/detect/<run-name>/weights/best.pt models/yolov8n-license-plate.pt
	# or set PLATE_MODEL_PATH directly in config.py / env vars
	```

	The latest YOLOv8n fine-tune (test split with 202 images) yields precision 0.995, recall 0.990, mAP50 0.995, and mAP50-95 0.865, confirming the new checkpoint surpasses the base detector.

	Afterwards rerun `python main_video.py` (or the live camera pipeline) and compare plate detection accuracy + speed.

## Cloud sync configuration

By default the pipeline attempts to sync completed entries to Firebase Cloud Firestore. Provide credentials in one of two ways:

1. **File path** – download your Firebase service-account JSON and set `FIREBASE_CREDENTIALS` to its absolute path (defaults to `serviceAccount.json` in the project root).
2. **Inline JSON** – set `FIREBASE_CREDENTIALS_JSON` to the raw JSON string (useful for CI or secrets managers). The app writes it to `.cache/firebase_credentials.json` automatically.

If you prefer a REST endpoint instead of Firebase, set `CLOUD_PROVIDER=rest`, point `CLOUD_ENDPOINT` to your API, and set `CLOUD_API_KEY` to the corresponding bearer token. Use `--no-cloud` on `main.py` / `main_video.py` for fully offline runs.
