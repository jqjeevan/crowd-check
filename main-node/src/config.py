import os
from pathlib import Path
from dotenv import load_dotenv

ENV_PATH = Path(__file__).with_name(".env")
load_dotenv(ENV_PATH)


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: str) -> Path:
    raw_path = Path(os.getenv(name, default))
    if raw_path.is_absolute():
        return raw_path
    return Path(__file__).parent.parent / raw_path


ALLOWED_NODES = [n.strip() for n in os.getenv("ALLOWED_NODES", "").split(",") if n.strip()]
ZENOH_LISTEN_ENDPOINT = os.getenv("ZENOH_LISTEN_ENDPOINT", "tcp/0.0.0.0:7447")

STORAGE_BASE = _env_path("STORAGE_BASE_PATH", "storage")

ENABLE_CSV_SINK = _env_flag("ENABLE_CSV_SINK", True)
ENABLE_CLICKHOUSE_SINK = _env_flag("ENABLE_CLICKHOUSE_SINK", False)
STATS_QUEUE_MAX_SIZE = int(os.getenv("STATS_QUEUE_MAX_SIZE", "5000"))
STATS_BATCH_SIZE = int(os.getenv("STATS_BATCH_SIZE", "250"))
STATS_FLUSH_INTERVAL_S = float(os.getenv("STATS_FLUSH_INTERVAL_S", "5"))

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "127.0.0.1")
CLICKHOUSE_PORT = int(os.getenv("CLICKHOUSE_PORT", "8123"))
CLICKHOUSE_DATABASE = os.getenv("CLICKHOUSE_DATABASE", "crowd_check")
CLICKHOUSE_USERNAME = os.getenv("CLICKHOUSE_USERNAME", "crowdcheck")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "crowdcheck")
CLICKHOUSE_SECURE = _env_flag("CLICKHOUSE_SECURE", False)

BODY_MODEL_PATH = "models/yolo11n.pt"
HEAD_MODEL_PATH = "models/yolov8n-head.pt"
HEAD_MODEL_URL = (
    "https://huggingface.co/AmineSam/irail-crowd-counting-yolov8n/resolve/main/best.pt"
)
