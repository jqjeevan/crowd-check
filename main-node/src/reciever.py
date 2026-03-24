import os
import sys
import time
import queue
import urllib.request
import shutil
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import torch
import zenoh
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()
ALLOWED_NODES = [n.strip() for n in os.getenv("ALLOWED_NODES", "").split(",") if n.strip()]
STORAGE_BASE = Path(__file__).parent.parent / "storage"

BODY_MODEL_PATH = "models/yolo11n.pt"
HEAD_MODEL_PATH = "models/yolov8n-head.pt" 
HEAD_MODEL_URL = "https://huggingface.co/AmineSam/irail-crowd-counting-yolov8n/resolve/main/best.pt"

frame_queues = {node: queue.Queue() for node in ALLOWED_NODES}

def verify_hardware():
    print("Hardware Verification Started")
    if not torch.cuda.is_available():
        print("CRITICAL ERROR: CUDA is not available.")
        sys.exit(1)
        
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    
    print(f"GPU Detected: {device_name}")
    print(f"Compute Capability: sm_{capability}{capability}")
    
    try:
        test_tensor = torch.ones(1).cuda() + 1
        print(f"Logic Check Passed: {test_tensor.item()}\n")
    except Exception as e:
        print(f"CRITICAL ERROR: GPU tensor operation failed: {e}")
        sys.exit(1)

def ensure_models_exist():
    os.makedirs("models", exist_ok=True)
    
    if not os.path.exists(BODY_MODEL_PATH):
        print(f"'{BODY_MODEL_PATH}' not found. Downloading standard YOLO11n weights...")
        YOLO(BODY_MODEL_PATH)

    if not os.path.exists(HEAD_MODEL_PATH):
        print(f"'{HEAD_MODEL_PATH}' not found locally.")
        print("Downloading Railway Crowd Head model from Hugging Face...")
        try:
            req = urllib.request.Request(HEAD_MODEL_URL, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(HEAD_MODEL_PATH, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
            print("Download complete!\n")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to download head model. {e}")
            sys.exit(1)

def frame_handler(sample):
    topic = str(sample.key_expr)
    node_id = topic.split("/")[-1]

    if node_id not in ALLOWED_NODES:
        return

    data = np.frombuffer(sample.payload.to_bytes(), dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if frame is not None:
        frame_queues[node_id].put(frame)

def main():
    verify_hardware()
    ensure_models_exist()
    
    print("Loading YOLO Body and Head models into VRAM...")
    body_model = YOLO(BODY_MODEL_PATH) 
    head_model = YOLO(HEAD_MODEL_PATH)

    conf = zenoh.Config()
    session = zenoh.open(conf)
    sub = session.declare_subscriber("cme466/camera/*", frame_handler)

    print("Receiver Active. Opening displays and waiting for frames...")
    
    display_frames = {node: np.zeros((480, 640, 3), dtype=np.uint8) for node in ALLOWED_NODES}

    try:
        while True:
            for node_name in ALLOWED_NODES:
                try:
                    while True:
                        frame = frame_queues[node_name].get_nowait()
                        
                        save_dir = STORAGE_BASE / node_name
                        save_dir.mkdir(parents=True, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        filename = f"{timestamp}.jpg"
                        cv2.imwrite(str(save_dir / filename), frame)
                        print(f"[{node_name}] Processed & Archived: {filename}")

                        frame_height, frame_width = frame.shape[:2]
                        frame_area = frame_height * frame_width
                        
                        body_results = body_model.predict(
                            frame, device="cuda:0", classes=[0], imgsz=1280, 
                            conf=0.10, iou=0.45, verbose=False
                        )
                        
                        body_boxes = []
                        for box in body_results[0].boxes:
                            # The index extracts the coordinates correctly
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
                        
                        annotated_frame = frame.copy()
                        for bbox in body_boxes:
                            bx1, by1, bx2, by2 = map(int, bbox)
                            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                            
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
                                cv2.rectangle(annotated_frame, (hx1, hy1), (hx2, hy2), (0, 0, 255), 2)
                        
                        total_headcount = len(body_boxes) + orphan_heads_count
                        
                        cv2.putText(annotated_frame, f"Node: {node_name} | Total: {total_headcount}", (20, 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                        cv2.putText(annotated_frame, f"Bodies: {len(body_boxes)} | Heads: {orphan_heads_count}", 
                                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        display_frames[node_name] = annotated_frame

                except queue.Empty:
                    pass

                cv2.imshow(f"Stream: {node_name}", display_frames[node_name])

            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                print("\nShutting down receiver.")
                break

    except KeyboardInterrupt:
        print("\nStopping receiver via interrupt.")
    finally:
        session.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()