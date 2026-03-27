import queue

import cv2
import numpy as np
import zenoh

from config import ALLOWED_NODES

frame_queues: dict[str, queue.Queue] = {node: queue.Queue() for node in ALLOWED_NODES}


def _frame_handler(sample):
    topic = str(sample.key_expr)
    node_id = topic.split("/")[-1]

    if node_id not in ALLOWED_NODES:
        return

    data = np.frombuffer(sample.payload.to_bytes(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if frame is not None:
        frame_queues[node_id].put(frame)


def start_subscriber():
    conf = zenoh.Config()
    session = zenoh.open(conf)
    sub = session.declare_subscriber("cme466/camera/*", _frame_handler)
    print("Receiver Active. Waiting for frames...")
    return session, sub
