# VEIL-AI
Vehicle Edge Intelligence Logger - advanced AI for vehicular security and monitoring.

## Plate detection model

Download a license-plate YOLO checkpoint (for example [keremberke/yolov8n-license-plate](https://huggingface.co/keremberke/yolov8n-license-plate)) and place it at `models/yolov8n-license-plate.pt`, or set the `PLATE_MODEL_PATH` environment variable to your custom `.pt` file before running `main.py` or `main_video.py`.
