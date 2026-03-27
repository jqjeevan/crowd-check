import cv2
import numpy as np


def process_frame(frame: np.ndarray, body_model, head_model) -> np.ndarray:
    """Run body + head detection on a frame and return the annotated result."""

    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_height * frame_width

    body_results = body_model.predict(
        frame, device="cuda:0", classes=[0], imgsz=1280,
        conf=0.10, iou=0.45, verbose=False
    )

    body_boxes = []
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

        body_boxes.append([x1, y1, x2, y2])

    head_results = head_model.predict(
        frame, device="cuda:0", conf=0.15, iou=0.60,
        max_det=1500, verbose=False
    )

    head_boxes = head_results[0].boxes.xyxy.cpu().numpy()

    annotated = frame.copy()
    for bbox in body_boxes:
        bx1, by1, bx2, by2 = map(int, bbox)
        cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 255, 0), 2)

    orphan_heads_count = 0

    for hbox in head_boxes:
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
            orphan_heads_count += 1
            cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)

    total_headcount = len(body_boxes) + orphan_heads_count

    cv2.putText(annotated, f"Total: {total_headcount}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.putText(annotated, f"Bodies: {len(body_boxes)} | Heads: {orphan_heads_count}",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return annotated
