import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ALLOWED_NODES = [n.strip() for n in os.getenv("ALLOWED_NODES", "").split(",") if n.strip()]

STORAGE_BASE = Path(__file__).parent.parent / "storage"

BODY_MODEL_PATH = "models/yolo11n.pt"
HEAD_MODEL_PATH = "models/yolov8n-head.pt"
HEAD_MODEL_URL = (
    "https://huggingface.co/AmineSam/irail-crowd-counting-yolov8n/resolve/main/best.pt"
)
