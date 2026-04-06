import os
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(ENV_PATH)

ALLOWED_NODES = [n.strip() for n in os.getenv("ALLOWED_NODES", "").split(",") if n.strip()]
ZENOH_LISTEN_ENDPOINT = os.getenv("ZENOH_LISTEN_ENDPOINT", "tcp/0.0.0.0:7447")

STORAGE_BASE = Path(__file__).parent.parent / "storage"

BODY_MODEL_PATH = "models/yolo11n.pt"
HEAD_MODEL_PATH = "models/yolov8n-head.pt"
HEAD_MODEL_URL = (
    "https://huggingface.co/AmineSam/irail-crowd-counting-yolov8n/resolve/main/best.pt"
)
