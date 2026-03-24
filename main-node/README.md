cme466-iot-system/
├── .gitignore
├── README.md
├── edge-node/               # Raspberry Pi (ARM64)
│   ├── pyproject.toml       # Pi-specific environment
│   ├── uv.lock
│   ├── src/                 # All edge logic (Zenoh/Camera)
│   │   ├── camera_stream.py
│   │   └── protocols.py     # Shared logic copied/linked here
│   └── storage/             # Dataset location
│       └── videos_vscrowd/  # Source files for the Pi to stream
└── main-node/               # Windows 11 PC (sm_120)
    ├── pyproject.toml       # Nightly CUDA 12.8 config
    ├── uv.lock
    ├── src/                 # All AI and server logic
    │   ├── main.py          # YOLO Crowd Analysis
    │   └── server_node.py   # Zenoh subscriber
    ├── models/              # weights downloaded by script
    │   ├── yolo11n.pt
    │   └── yolov8n-head.pt
    └── storage/             # Received image archive
        └── raw_frames/      # Security camera frame storage