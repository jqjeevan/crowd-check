import os
import argparse
import json
import cv2
import zenoh
import time
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

load_dotenv()

FILTER_MODEL_NAME = "models/yolo26n.pt"
PERSON_CLASS_ID = 0  # COCO class 0 = person
CAMERA_CATALOG_PATH = Path(__file__).resolve().parents[2] / "camera_nodes.json"


def load_camera_catalog() -> dict[str, dict[str, float | str]]:
    if not CAMERA_CATALOG_PATH.exists():
        return {}

    with open(CAMERA_CATALOG_PATH, encoding="utf-8") as f:
        raw_catalog = json.load(f)

    return {str(node_id): dict(spec) for node_id, spec in raw_catalog.items()}


CAMERA_CATALOG = load_camera_catalog()


def load_filter_model() -> YOLO:
    """Load the YOLO26n person-detection model (auto-downloads on first run)."""
    print(f"Loading YOLO26n person-detection filter ({FILTER_MODEL_NAME})...")
    model = YOLO(FILTER_MODEL_NAME)
    print("Filter model loaded.\n")
    return model


def person_detected(model: YOLO, frame, conf_threshold: float) -> bool:
    """Return True if at least one person survives area-ratio filtering.

    Uses the same detection strategy as the main node's detection.py:
    - Low initial confidence to catch dense/small crowd detections
    - Area-ratio post-filtering to discard large low-conf false positives
    - imgsz=640 for better small-object recall on CPU (vs. 1280 on GPU)
    """
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_height * frame_width

    results = model.predict(
        frame,
        classes=[PERSON_CLASS_ID],
        imgsz=640,
        conf=conf_threshold,
        iou=0.45,
        verbose=False,
    )

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf_val = box.conf.cpu().item()

        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / frame_area

        # Mirror the main node's area-ratio filtering rules
        if area_ratio > 0.08 and conf_val < 0.60:
            continue
        elif area_ratio > 0.02 and conf_val < 0.35:
            continue
        elif conf_val < 0.15:
            continue

        # At least one valid person detection — frame is worth sending
        return True

    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Edge node camera stream sender")
    parser.add_argument(
        "--node-id",
        default=os.getenv("NODE_ID"),
        help="Node ID (default: from .env)",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=None,
        help="Latitude for this camera node (default: from .env or camera catalog)",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=None,
        help="Longitude for this camera node (default: from .env or camera catalog)",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=["test_058", "test_059"],
        help="Target folders to stream (default: test_058 test_059)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable person-detection filter and send all frames (useful for debugging)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="Initial confidence threshold for person detection filter (default: 0.10)",
    )
    return parser.parse_args()


def resolve_camera_metadata(
    node_id: str,
    latitude: float | None,
    longitude: float | None,
) -> tuple[str, float, float]:
    catalog_entry = CAMERA_CATALOG.get(node_id, {})
    label = str(catalog_entry.get("label", node_id))

    if latitude is None:
        raw_latitude = os.getenv("NODE_LATITUDE")
        if raw_latitude:
            latitude = float(raw_latitude)
        elif "latitude" in catalog_entry:
            latitude = float(catalog_entry["latitude"])

    if longitude is None:
        raw_longitude = os.getenv("NODE_LONGITUDE")
        if raw_longitude:
            longitude = float(raw_longitude)
        elif "longitude" in catalog_entry:
            longitude = float(catalog_entry["longitude"])

    if latitude is None or longitude is None:
        raise ValueError(
            f"Camera coordinates are required for {node_id}. Set NODE_LATITUDE/NODE_LONGITUDE "
            "or use one of the catalog camera names."
        )

    return label, latitude, longitude


def publish_camera_metadata(meta_pub, node_id: str, label: str, latitude: float, longitude: float):
    payload = json.dumps(
        {
            "node_id": node_id,
            "label": label,
            "latitude": latitude,
            "longitude": longitude,
            "announced_at": datetime.now().isoformat(),
        }
    )
    meta_pub.put(payload.encode("utf-8"))
    print(f"Published camera metadata for {node_id} ({latitude:.6f}, {longitude:.6f})")


def main():
    args = parse_args()
    NODE_ID = args.node_id
    MAIN_NODE_IP = os.getenv("MAIN_NODE_IP")
    DATASET_PATH = Path(os.getenv("DATASET_PATH")).resolve()
    TARGET_FOLDERS = args.folders
    label, latitude, longitude = resolve_camera_metadata(
        NODE_ID,
        args.latitude,
        args.longitude,
    )

    # Load the YOLO26n filter model unless disabled
    filter_model = None
    if not args.no_filter:
        filter_model = load_filter_model()
    else:
        print("Person-detection filter DISABLED — all frames will be sent.\n")

    conf = zenoh.Config()

    # Apply the main node IP to ensure direct connection across subnets
    if MAIN_NODE_IP:
        conf.insert_json5("connect/endpoints", f'["tcp/{MAIN_NODE_IP}:7447"]')

    session = zenoh.open(conf)
    pub = session.declare_publisher(f"cme466/camera/{NODE_ID}")
    meta_pub = session.declare_publisher(f"cme466/camera-meta/{NODE_ID}")
    publish_camera_metadata(meta_pub, NODE_ID, label, latitude, longitude)

    print(f"Node {NODE_ID} streaming at 1 frame per second. Press Ctrl+C to stop.")

    sent_count = 0
    skipped_count = 0
    last_metadata_publish = time.monotonic()

    try:
        for folder in TARGET_FOLDERS:
            folder_path = DATASET_PATH / folder
            if not folder_path.exists():
                print(f"Skipping missing directory: {folder_path}")
                continue

            images = sorted(list(folder_path.glob("*.jpg")))

            for img_path in images:
                frame = cv2.imread(str(img_path))
                if frame is None:
                    continue

                if time.monotonic() - last_metadata_publish >= 30:
                    publish_camera_metadata(meta_pub, NODE_ID, label, latitude, longitude)
                    last_metadata_publish = time.monotonic()

                # Apply person-detection filter before sending
                if filter_model is not None:
                    if not person_detected(filter_model, frame, args.conf):
                        skipped_count += 1
                        print(f"Skipped (no person): {img_path.name}")
                        time.sleep(1.0)
                        continue

                _, buffer = cv2.imencode('.jpg', frame)
                pub.put(buffer.tobytes())
                sent_count += 1
                print(f"Sent: {img_path.name}")

                time.sleep(1.0)

    except KeyboardInterrupt:
        print("\nStreaming interrupted by user.")
    finally:
        print(f"\nSession summary — Sent: {sent_count} | Skipped: {skipped_count}")
        print("Closing Zenoh session safely...")
        session.close()
        print("Sender shut down complete.")


if __name__ == "__main__":
    main()
