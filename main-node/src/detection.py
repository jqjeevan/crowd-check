import cv2
import numpy as np
from datetime import datetime

from statistics import FrameStats

# IoU threshold — any overlap above this makes a body box Tier 2
OVERLAP_IOU_THRESHOLD = 0.1

# Heatmap colours (BGR) and blend alpha
TIER_COLOURS = {
    1: (0, 200, 0),      # green  — low congestion
    2: (0, 220, 220),    # yellow — moderate congestion
    3: (0, 0, 220),      # red    — high congestion
}
HEATMAP_ALPHA = 0.30


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def process_frame(
    frame: np.ndarray, body_model, head_model, node_id: str
) -> tuple[np.ndarray, FrameStats]:
    """Run body + head detection, build congestion heatmap, and collect stats.

    Returns (annotated_frame, frame_stats).
    """
    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_height * frame_width

    # --- body detection -------------------------------------------------
    body_results = body_model.predict(
        frame, device="cuda:0", classes=[0], imgsz=1280,
        conf=0.10, iou=0.45, verbose=False
    )

    body_boxes: list[list[float]] = []
    for box in body_results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf_val = box.conf.cpu().item()

        box_area = (x2 - x1) * (y2 - y1)
        area_ratio = box_area / frame_area

        if area_ratio > 0.08 and conf_val < 0.60:
            continue
        elif area_ratio > 0.02 and conf_val < 0.35:
            continue
        elif conf_val < 0.15:
            continue

        body_boxes.append([float(x1), float(y1), float(x2), float(y2)])

    # --- head detection -------------------------------------------------
    head_results = head_model.predict(
        frame, device="cuda:0", conf=0.15, iou=0.60,
        max_det=1500, verbose=False
    )

    all_head_boxes = head_results[0].boxes.xyxy.cpu().numpy()

    orphan_head_boxes: list[list[float]] = []
    for hbox in all_head_boxes:
        hx1, hy1, hx2, hy2 = map(int, hbox)
        hcx = (hx1 + hx2) // 2
        hcy = (hy1 + hy2) // 2
        is_orphan = True

        for bbox in body_boxes:
            bx1, by1, bx2, by2 = map(int, bbox)
            if bx1 <= hcx <= bx2 and by1 <= hcy <= by2:
                is_orphan = False
                break

        if is_orphan:
            orphan_head_boxes.append([float(hbox[0]), float(hbox[1]),
                                      float(hbox[2]), float(hbox[3])])

    # --- congestion tiers -----------------------------------------------
    congestion_tiers: dict[int, int] = {}

    # Bodies: Tier 1 (no overlap) or Tier 2 (overlapping)
    for i, box_a in enumerate(body_boxes):
        has_overlap = False
        for j, box_b in enumerate(body_boxes):
            if i == j:
                continue
            if _iou(box_a, box_b) > OVERLAP_IOU_THRESHOLD:
                has_overlap = True
                break
        congestion_tiers[i] = 2 if has_overlap else 1

    # Orphan heads: always Tier 3
    body_count = len(body_boxes)
    for k in range(len(orphan_head_boxes)):
        congestion_tiers[body_count + k] = 3

    # --- build annotated frame -----------------------------------------
    annotated = frame.copy()

    # Draw body rectangles
    for idx, bbox in enumerate(body_boxes):
        bx1, by1, bx2, by2 = map(int, bbox)
        tier = congestion_tiers.get(idx, 1)
        colour = TIER_COLOURS[tier]
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), colour, 2)

    # Draw orphan head rectangles
    for k, hbox in enumerate(orphan_head_boxes):
        hx1, hy1, hx2, hy2 = map(int, hbox)
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), TIER_COLOURS[3], 2)

    total_headcount = body_count + len(orphan_head_boxes)

    cv2.putText(annotated, f"Total: {total_headcount}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(annotated, f"Bodies: {body_count} | Heads: {len(orphan_head_boxes)}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- congestion heatmap overlay -------------------------------------
    annotated = _overlay_heatmap(
        annotated, body_boxes, orphan_head_boxes, congestion_tiers
    )

    # --- collect stats --------------------------------------------------
    stats = FrameStats(
        timestamp=datetime.now().isoformat(),
        node_id=node_id,
        body_count=body_count,
        head_count=len(orphan_head_boxes),
        total_headcount=total_headcount,
        body_boxes=body_boxes,
        head_boxes=orphan_head_boxes,
        congestion_tiers=congestion_tiers,
    )

    return annotated, stats


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def _overlay_heatmap(
    frame: np.ndarray,
    body_boxes: list[list[float]],
    orphan_head_boxes: list[list[float]],
    congestion_tiers: dict[int, int],
) -> np.ndarray:
    """Render a semi-transparent congestion heatmap over the frame.

    Each pixel takes the colour of the highest congestion tier covering it.
    """
    h, w = frame.shape[:2]
    tier_map = np.zeros((h, w), dtype=np.uint8)    # 0 = no detection

    body_count = len(body_boxes)

    # Paint body-box regions
    for idx, bbox in enumerate(body_boxes):
        bx1, by1, bx2, by2 = (
            max(0, int(bbox[0])), max(0, int(bbox[1])),
            min(w, int(bbox[2])), min(h, int(bbox[3])),
        )
        tier = congestion_tiers.get(idx, 1)
        # Keep pixel-wise max tier
        roi = tier_map[by1:by2, bx1:bx2]
        tier_map[by1:by2, bx1:bx2] = np.maximum(roi, tier)

    # Paint orphan-head regions
    for k, hbox in enumerate(orphan_head_boxes):
        hx1, hy1, hx2, hy2 = (
            max(0, int(hbox[0])), max(0, int(hbox[1])),
            min(w, int(hbox[2])), min(h, int(hbox[3])),
        )
        tier = congestion_tiers.get(body_count + k, 3)
        roi = tier_map[hy1:hy2, hx1:hx2]
        tier_map[hy1:hy2, hx1:hx2] = np.maximum(roi, tier)

    # Build colour overlay
    overlay = np.zeros_like(frame)
    for tier_val, colour in TIER_COLOURS.items():
        mask = tier_map == tier_val
        overlay[mask] = colour

    # Blend only where tier_map > 0
    has_detection = tier_map > 0
    blended = frame.copy()
    blended[has_detection] = cv2.addWeighted(
        frame, 1.0 - HEATMAP_ALPHA, overlay, HEATMAP_ALPHA, 0
    )[has_detection]

    return blended

