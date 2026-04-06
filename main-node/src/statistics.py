"""Frame statistics data model, message buffer, and CSV export."""

import csv
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class FrameStats:
    """All statistics collected from a single processed frame."""

    timestamp: str                                      # ISO 8601
    node_id: str
    body_count: int
    head_count: int                                     # orphan heads only
    total_headcount: int
    body_boxes: list[list[float]] = field(default_factory=list)   # [[x1,y1,x2,y2], ...]
    head_boxes: list[list[float]] = field(default_factory=list)   # orphan heads
    congestion_tiers: dict[int, int] = field(default_factory=dict)
    # key = detection index (bodies first, then heads), value = tier (1/2/3)


class StatsBuffer:
    """Thread-safe per-node message buffer backed by a deque."""

    def __init__(self, maxlen: int = 5000):
        self._buf: deque[FrameStats] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def push(self, stats: FrameStats) -> None:
        with self._lock:
            self._buf.append(stats)

    def flush(self) -> list[FrameStats]:
        """Drain the buffer and return all items."""
        with self._lock:
            items = list(self._buf)
            self._buf.clear()
            return items

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def export_csv(self, node_id: str, storage_base: Path) -> None:
        """Flush buffer and write two CSV files into storage/<node_id>/statistics/."""

        items = self.flush()
        if not items:
            print(f"[{node_id}] No statistics to export.")
            return

        stats_dir = storage_base / node_id / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)

        batch_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self._write_frame_summary(items, stats_dir, batch_ts, node_id)
        self._write_detections(items, stats_dir, batch_ts, node_id)

    # -- internal helpers ------------------------------------------------

    @staticmethod
    def _write_frame_summary(
        items: list[FrameStats], out_dir: Path, batch_ts: str, node_id: str
    ) -> None:
        path = out_dir / f"frame_summary_{batch_ts}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "node_id",
                "body_count", "head_count", "total_headcount",
            ])
            for s in items:
                writer.writerow([
                    s.timestamp, s.node_id,
                    s.body_count, s.head_count, s.total_headcount,
                ])
        print(f"[{node_id}] Exported frame summary → {path}")

    @staticmethod
    def _write_detections(
        items: list[FrameStats], out_dir: Path, batch_ts: str, node_id: str
    ) -> None:
        path = out_dir / f"detections_{batch_ts}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "node_id", "box_type",
                "x1", "y1", "x2", "y2", "congestion_tier",
            ])
            for s in items:
                idx = 0
                for box in s.body_boxes:
                    tier = s.congestion_tiers.get(idx, 0)
                    writer.writerow([
                        s.timestamp, s.node_id, "body",
                        *[f"{v:.1f}" for v in box], tier,
                    ])
                    idx += 1
                for box in s.head_boxes:
                    tier = s.congestion_tiers.get(idx, 3)
                    writer.writerow([
                        s.timestamp, s.node_id, "head",
                        *[f"{v:.1f}" for v in box], tier,
                    ])
                    idx += 1
        print(f"[{node_id}] Exported detections   → {path}")
