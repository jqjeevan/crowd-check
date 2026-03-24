import os
import sys
import argparse
import cv2
import zenoh
import time
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Edge node camera stream sender")
    parser.add_argument("--node-id", default=os.getenv("NODE_ID"), help="Node ID (default: from .env)")
    parser.add_argument("--folders", nargs="+", default=["test_058", "test_059"], help="Target folders to stream (default: test_058 test_059)")
    return parser.parse_args()

def main():
    args = parse_args()
    NODE_ID = args.node_id
    MAIN_NODE_IP = os.getenv("MAIN_NODE_IP")
    DATASET_PATH = Path(os.getenv("DATASET_PATH")).resolve()
    TARGET_FOLDERS = args.folders
    conf = zenoh.Config()
    
    # Apply the main node IP to ensure direct connection across subnets
    if MAIN_NODE_IP:
        conf.insert_json5("connect/endpoints", f'["tcp/{MAIN_NODE_IP}:7447"]')
        
    session = zenoh.open(conf)
    pub = session.declare_publisher(f"cme466/camera/{NODE_ID}")

    print(f"Node {NODE_ID} streaming at 1 frame per second. Press Ctrl+C to stop.")

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

                _, buffer = cv2.imencode('.jpg', frame)
                pub.put(buffer.tobytes())
                print(f"Sent: {img_path.name}")
                
                time.sleep(1.0) 
                
    except KeyboardInterrupt:
        print("\nStreaming interrupted by user.")
    finally:
        print("Closing Zenoh session safely...")
        session.close()
        print("Sender shut down complete.")

if __name__ == "__main__":
    main()