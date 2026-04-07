import queue
import json

import cv2
import numpy as np
import zenoh

from camera_registry import CameraRegistry
from clickhouse_config import ClickHouseConfig
from config import (
    ALLOWED_NODES,
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_SECURE,
    CLICKHOUSE_USERNAME,
    ENABLE_CLICKHOUSE_SINK,
    ZENOH_LISTEN_ENDPOINT,
)

frame_queues: dict[str, queue.Queue] = {node: queue.Queue() for node in ALLOWED_NODES}
camera_registry: CameraRegistry | None = None


def _frame_handler(sample):
    topic = str(sample.key_expr)
    node_id = topic.split("/")[-1]

    if node_id not in ALLOWED_NODES:
        return

    data = np.frombuffer(sample.payload.to_bytes(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if frame is not None:
        frame_queues[node_id].put(frame)


def _metadata_handler(sample):
    topic = str(sample.key_expr)
    node_id = topic.split("/")[-1]

    if node_id not in ALLOWED_NODES:
        return

    try:
        payload = json.loads(sample.payload.to_bytes().decode("utf-8"))
        latitude = float(payload["latitude"])
        longitude = float(payload["longitude"])
        label = str(payload.get("label", node_id))
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"[{node_id}] Ignoring invalid camera metadata payload: {exc}")
        return

    if camera_registry is not None:
        camera_registry.register(
            node_id=node_id,
            latitude=latitude,
            longitude=longitude,
            label=label,
        )
        print(
            f"[{node_id}] Registered camera location ({latitude:.6f}, {longitude:.6f})."
        )


def start_subscriber():
    global camera_registry
    conf = zenoh.Config()
    conf.insert_json5("listen/endpoints", f'["{ZENOH_LISTEN_ENDPOINT}"]')
    session = zenoh.open(conf)

    clickhouse_config = None
    if ENABLE_CLICKHOUSE_SINK:
        clickhouse_config = ClickHouseConfig(
            host=CLICKHOUSE_HOST,
            port=CLICKHOUSE_PORT,
            database=CLICKHOUSE_DATABASE,
            username=CLICKHOUSE_USERNAME,
            password=CLICKHOUSE_PASSWORD,
            secure=CLICKHOUSE_SECURE,
        )

    camera_registry = CameraRegistry(ALLOWED_NODES, clickhouse_config=clickhouse_config)
    frame_sub = session.declare_subscriber("cme466/camera/*", _frame_handler)
    metadata_sub = session.declare_subscriber("cme466/camera-meta/*", _metadata_handler)
    print(f"Receiver Active on {ZENOH_LISTEN_ENDPOINT}. Waiting for frames...")
    return session, [frame_sub, metadata_sub], camera_registry
