import os
import sys
import shutil
import urllib.request

import torch
from ultralytics import YOLO

from config import BODY_MODEL_PATH, HEAD_MODEL_PATH, HEAD_MODEL_URL


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
            req = urllib.request.Request(HEAD_MODEL_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as response, open(HEAD_MODEL_PATH, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
            print("Download complete!\n")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to download head model. {e}")
            sys.exit(1)


def load_models():
    print("Loading YOLO Body and Head models into VRAM...")
    body_model = YOLO(BODY_MODEL_PATH)
    head_model = YOLO(HEAD_MODEL_PATH)
    return body_model, head_model
