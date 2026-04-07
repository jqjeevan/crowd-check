import queue
from datetime import datetime

import cv2
from PySide6.QtWidgets import QMainWindow, QWidget, QGridLayout
from PySide6.QtCore import QTimer

from config import (
    CLICKHOUSE_DATABASE,
    CLICKHOUSE_HOST,
    CLICKHOUSE_PASSWORD,
    CLICKHOUSE_PORT,
    CLICKHOUSE_SECURE,
    CLICKHOUSE_USERNAME,
    ENABLE_CLICKHOUSE_SINK,
    ENABLE_CSV_SINK,
    STATS_BATCH_SIZE,
    STATS_FLUSH_INTERVAL_S,
    STATS_QUEUE_MAX_SIZE,
    STORAGE_BASE,
)
from network import frame_queues
from detection import process_frame
from clickhouse_config import ClickHouseConfig
from statistics import StatsBuffer
from gui.camera_widget import CameraWidget
from gui.theme import WINDOW_BG, GRID_MARGIN, GRID_SPACING


class ReceiverWindow(QMainWindow):

    def __init__(
        self,
        node_names,
        body_model,
        head_model,
        zenoh_session,
        zenoh_subscribers,
        camera_registry,
    ):
        super().__init__()
        self.setWindowTitle("Crowd Check")
        self.setStyleSheet(f"QMainWindow {{ background-color: {WINDOW_BG}; }}")

        self._body_model = body_model
        self._head_model = head_model
        self._zenoh_session = zenoh_session
        self._zenoh_subscribers = zenoh_subscribers
        self._camera_registry = camera_registry
        self._node_names = node_names

        central = QWidget()
        central.setStyleSheet(f"background-color: {WINDOW_BG};")
        self.setCentralWidget(central)

        self._grid = QGridLayout(central)
        self._grid.setContentsMargins(GRID_MARGIN, GRID_MARGIN, GRID_MARGIN, GRID_MARGIN)
        self._grid.setSpacing(GRID_SPACING)

        self._cameras: dict[str, CameraWidget] = {}
        self._stats_buffers: dict[str, StatsBuffer] = {}
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

        for node in node_names:
            self._cameras[node] = CameraWidget()
            self._stats_buffers[node] = StatsBuffer(
                node,
                STORAGE_BASE,
                csv_enabled=ENABLE_CSV_SINK,
                clickhouse_enabled=ENABLE_CLICKHOUSE_SINK,
                clickhouse_config=clickhouse_config,
                batch_size=STATS_BATCH_SIZE,
                flush_interval_s=STATS_FLUSH_INTERVAL_S,
                max_queue_size=STATS_QUEUE_MAX_SIZE,
            )

        self._arrange_grid()

        self._display_frames: dict[str, any] = {node: None for node in node_names}

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_frames)
        self._timer.start(30)

        self.resize(960, 600)

    def _arrange_grid(self):
        n = len(self._cameras)
        nodes = self._node_names

        if n == 1:
            self._grid.addWidget(self._cameras[nodes[0]], 0, 0)
        elif n == 2:
            self._grid.addWidget(self._cameras[nodes[0]], 0, 0)
            self._grid.addWidget(self._cameras[nodes[1]], 0, 1)
            self._grid.setColumnStretch(0, 1)
            self._grid.setColumnStretch(1, 1)
        elif n == 3:
            self._grid.addWidget(self._cameras[nodes[0]], 0, 0)
            self._grid.addWidget(self._cameras[nodes[1]], 1, 0)
            self._grid.addWidget(self._cameras[nodes[2]], 0, 1, 2, 1)
            self._grid.setColumnStretch(0, 1)
            self._grid.setColumnStretch(1, 1)
            self._grid.setRowStretch(0, 1)
            self._grid.setRowStretch(1, 1)
        else:
            cols = 2 if n <= 4 else 3
            for i, node in enumerate(nodes):
                r, c = divmod(i, cols)
                self._grid.addWidget(self._cameras[node], r, c)
            for c in range(cols):
                self._grid.setColumnStretch(c, 1)
            rows = (n + cols - 1) // cols
            for r in range(rows):
                self._grid.setRowStretch(r, 1)

    def _poll_frames(self):
        for node_name in self._node_names:
            try:
                while True:
                    frame = frame_queues[node_name].get_nowait()

                    save_dir = STORAGE_BASE / node_name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    filename = f"{timestamp}.jpg"
                    cv2.imwrite(str(save_dir / filename), frame)
                    print(f"[{node_name}] Processed & Archived: {filename}")

                    annotated, stats = process_frame(
                        frame, self._body_model, self._head_model, node_name
                    )
                    self._display_frames[node_name] = annotated
                    self._stats_buffers[node_name].push(stats)

            except queue.Empty:
                pass

            if self._display_frames[node_name] is not None:
                self._cameras[node_name].update_frame(self._display_frames[node_name])

    def closeEvent(self, event):
        self._timer.stop()

        for node_name, buf in self._stats_buffers.items():
            print(f"[{node_name}] Draining statistics queue before shutdown.")
            buf.close()

        if self._camera_registry is not None:
            self._camera_registry.close()

        self._zenoh_session.close()
        print("\nShutting down receiver.")
        event.accept()
