## Prerequisites

* **Python:** Version 3.12
* **Package Manager:** [uv](https://github.com/astral-sh/uv) installed on all machines.
* **Hardware (Main Node):** An NVIDIA GPU with CUDA support.
* **Network:** Both nodes must be on the same local network or reachable by IP.
* **Dataset:** Download Dataset From (https://huggingface.co/datasets/HopLeeTop/VSCrowd/tree/main) and insert into storage folder as outlined in Configuration.

## Project Structure

The project is split into two distinct applications:

* `edge-node/`: Runs on the camera devices (e.g., Raspberry Pi) to read and send images.
* `main-node/`: Runs on the central computer to receive, process, and display images.

## 1. Edge Node Setup

The edge node reads images from a local dataset and streams them at a fixed interval to the main node.

### Configuration
Navigate to the `edge-node` directory and create a `src/.env` file with the following variables:

```env
# Unique ID for this specific Edge Node
NODE_ID=Raspi-1

# Target IP address of the Main Node
MAIN_NODE_IP=###.##.#.###

# Path to the dataset relative to the src folder
DATASET_PATH=dataset/videos_vscrowd
```

### Running the Edge Node
Use `uv` to automatically handle dependencies and run the script:

```bash
cd edge-node
uv run src/sender.py
```

## 2. Main Node Setup

The main node listens for incoming frames, runs them through YOLO11n (body) and YOLOv8n-head (head) models, saves the original frames, and displays the real-time analysis.

### Configuration
Navigate to the `main-node` directory and create a `src/.env` file with the following variables:

```env
# Comma-separated list of Node IDs permitted to send data
ALLOWED_NODES=Raspi-1,Raspi-2

# Storage path relative to the src folder
STORAGE_BASE_PATH=storage
```

### Model Management
On the first run, the system will automatically download the standard YOLO11n weights and the specific Hugging Face head detection model into a `models/` directory parallel to `src/` folder.

### Running the Main Node
The `pyproject.toml` is configured to pull PyTorch Nightly to ensure it works with the latest NVIDIA GPU. Run the receiver using `uv`:

```bash
cd main-node
uv run src/receiver.py
```

## Execution Flow

1. Start the **Main Node** first so it is ready to receive incoming network traffic.
2. Start the **Edge Node(s)**.
3. The Main Node will automatically open a display window for each allowed node. If an edge node is connected but hasn't sent a frame yet, the window will appear black.
4. Press `q` while focused on the Main Node display windows to safely shut down the receiver.
5. Press `Ctrl+C` on the Edge Node terminal to stop the camera stream safely.