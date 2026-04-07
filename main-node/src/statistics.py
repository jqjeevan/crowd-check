"""Frame statistics data model, fan-out queue, and sink workers."""

import csv
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

try:
    import clickhouse_connect
except ImportError:  # pragma: no cover - import is validated by config/dependency setup
    clickhouse_connect = None

from clickhouse_config import ClickHouseConfig


@dataclass
class FrameImage:
    """Image metadata for a single saved frame variant."""

    frame_variant: str
    image_path: str
    image_url: str


@dataclass
class FrameStats:
    """All statistics collected from a single processed frame."""

    timestamp: str                                      # ISO 8601
    frame_id: str
    node_id: str
    body_count: int
    head_count: int                                     # orphan heads only
    total_headcount: int
    body_boxes: list[list[float]] = field(default_factory=list)   # [[x1,y1,x2,y2], ...]
    head_boxes: list[list[float]] = field(default_factory=list)   # orphan heads
    congestion_tiers: dict[int, int] = field(default_factory=dict)
    frame_images: list[FrameImage] = field(default_factory=list)
    # key = detection index (bodies first, then heads), value = tier (1/2/3)


@dataclass
class _QueuedStats:
    stats: FrameStats
    pending_sinks: set[str]


class StatsBuffer:
    """Per-node fan-out buffer retained until every sink acknowledges a record."""

    def __init__(
        self,
        node_id: str,
        storage_base: Path,
        *,
        csv_enabled: bool = True,
        clickhouse_enabled: bool = False,
        clickhouse_config: ClickHouseConfig | None = None,
        batch_size: int = 250,
        flush_interval_s: float = 5.0,
        max_queue_size: int = 5000,
    ):
        self._node_id = node_id
        self._storage_base = storage_base
        self._batch_size = max(1, batch_size)
        self._flush_interval_s = max(0.1, flush_interval_s)
        self._max_queue_size = max(1, max_queue_size)
        self._records: deque[_QueuedStats] = deque()
        self._condition = threading.Condition()
        self._closing = False
        self._workers: list[threading.Thread] = []
        self._clickhouse_config = clickhouse_config
        self._clickhouse_client = None
        self._sink_names: list[str] = []

        if csv_enabled:
            self._sink_names.append("csv")
            self._start_worker("csv", self._flush_csv_batch)

        if clickhouse_enabled:
            if clickhouse_config is None:
                raise ValueError("ClickHouse is enabled but no ClickHouseConfig was provided.")
            if clickhouse_connect is None:
                raise RuntimeError(
                    "ClickHouse sink is enabled but clickhouse-connect is not installed."
                )
            self._sink_names.append("clickhouse")
            self._start_worker("clickhouse", self._flush_clickhouse_batch)

        if not self._sink_names:
            print(f"[{self._node_id}] No statistics sinks are enabled.")

    def push(self, stats: FrameStats) -> None:
        """Add a processed frame to the fan-out queue."""
        if not self._sink_names:
            return

        with self._condition:
            while len(self._records) >= self._max_queue_size and not self._closing:
                print(f"[{self._node_id}] Stats queue full; waiting for sinks to catch up.")
                self._condition.wait(timeout=0.5)

            if self._closing:
                return

            self._records.append(
                _QueuedStats(stats=stats, pending_sinks=set(self._sink_names))
            )
            self._condition.notify_all()

    def close(self) -> None:
        """Drain the queue and stop the sink workers."""
        with self._condition:
            self._closing = True
            self._condition.notify_all()

        for worker in self._workers:
            worker.join()

        if self._clickhouse_client is not None:
            self._clickhouse_client.close()
            self._clickhouse_client = None

    def _start_worker(
        self,
        sink_name: str,
        handler: Callable[[list[FrameStats]], None],
    ) -> None:
        worker = threading.Thread(
            target=self._run_worker,
            args=(sink_name, handler),
            daemon=True,
            name=f"{self._node_id}-{sink_name}-sink",
        )
        worker.start()
        self._workers.append(worker)

    def _run_worker(
        self,
        sink_name: str,
        handler: Callable[[list[FrameStats]], None],
    ) -> None:
        while True:
            batch = self._wait_for_batch(sink_name)
            if batch is None:
                return

            try:
                handler([record.stats for record in batch])
            except Exception as exc:
                print(f"[{self._node_id}] {sink_name} sink failed: {exc}")
                time.sleep(min(self._flush_interval_s, 5.0))
                continue

            self._acknowledge_batch(sink_name, batch)

    def _wait_for_batch(self, sink_name: str) -> list[_QueuedStats] | None:
        deadline = None

        with self._condition:
            while True:
                available = [
                    record for record in self._records if sink_name in record.pending_sinks
                ]

                if available:
                    if len(available) >= self._batch_size or self._closing:
                        return available[:self._batch_size]

                    if deadline is None:
                        deadline = time.monotonic() + self._flush_interval_s

                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return available[:self._batch_size]

                    self._condition.wait(timeout=remaining)
                    continue

                if self._closing:
                    return None

                self._condition.wait()

    def _acknowledge_batch(self, sink_name: str, batch: list[_QueuedStats]) -> None:
        with self._condition:
            for record in batch:
                record.pending_sinks.discard(sink_name)

            while self._records and not self._records[0].pending_sinks:
                self._records.popleft()

            self._condition.notify_all()

    def _flush_csv_batch(self, items: list[FrameStats]) -> None:
        if not items:
            return

        stats_dir = self._storage_base / self._node_id / "statistics"
        stats_dir.mkdir(parents=True, exist_ok=True)

        batch_ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._write_frame_summary(items, stats_dir, batch_ts)
        self._write_detections(items, stats_dir, batch_ts)
        self._write_frame_images(items, stats_dir, batch_ts)

    def _flush_clickhouse_batch(self, items: list[FrameStats]) -> None:
        if not items:
            return

        client = self._get_clickhouse_client()
        frame_rows = []
        detection_rows = []
        image_rows = []

        for stats in items:
            ts = datetime.fromisoformat(stats.timestamp)
            frame_rows.append([
                ts,
                stats.frame_id,
                stats.node_id,
                stats.body_count,
                stats.head_count,
                stats.total_headcount,
            ])

            idx = 0
            for box in stats.body_boxes:
                detection_rows.append([
                    ts,
                    stats.frame_id,
                    stats.node_id,
                    "body",
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    stats.congestion_tiers.get(idx, 0),
                ])
                idx += 1

            for box in stats.head_boxes:
                detection_rows.append([
                    ts,
                    stats.frame_id,
                    stats.node_id,
                    "head",
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    stats.congestion_tiers.get(idx, 3),
                ])
                idx += 1

            for image in stats.frame_images:
                image_rows.append([
                    ts,
                    stats.frame_id,
                    stats.node_id,
                    image.frame_variant,
                    image.image_path,
                    image.image_url,
                    stats.body_count,
                    stats.head_count,
                    stats.total_headcount,
                ])

        client.insert(
            "frame_summary",
            frame_rows,
            database=self._clickhouse_config.database,
            column_names=[
                "timestamp",
                "frame_id",
                "node_id",
                "body_count",
                "head_count",
                "total_headcount",
            ],
        )

        if detection_rows:
            client.insert(
                "detections",
                detection_rows,
                database=self._clickhouse_config.database,
                column_names=[
                    "timestamp",
                    "frame_id",
                    "node_id",
                    "box_type",
                    "x1",
                    "y1",
                    "x2",
                    "y2",
                    "congestion_tier",
                ],
            )

        if image_rows:
            client.insert(
                "frame_images",
                image_rows,
                database=self._clickhouse_config.database,
                column_names=[
                    "timestamp",
                    "frame_id",
                    "node_id",
                    "frame_variant",
                    "image_path",
                    "image_url",
                    "body_count",
                    "head_count",
                    "total_headcount",
                ],
            )

    def _get_clickhouse_client(self):
        if self._clickhouse_client is None:
            self._clickhouse_client = clickhouse_connect.get_client(
                host=self._clickhouse_config.host,
                port=self._clickhouse_config.port,
                username=self._clickhouse_config.username,
                password=self._clickhouse_config.password,
                database=self._clickhouse_config.database,
                secure=self._clickhouse_config.secure,
            )
        return self._clickhouse_client

    def _write_frame_summary(
        self,
        items: list[FrameStats],
        out_dir: Path,
        batch_ts: str,
    ) -> None:
        path = out_dir / f"frame_summary_{batch_ts}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "frame_id", "node_id",
                "body_count", "head_count", "total_headcount",
            ])
            for stats in items:
                writer.writerow([
                    stats.timestamp,
                    stats.frame_id,
                    stats.node_id,
                    stats.body_count,
                    stats.head_count,
                    stats.total_headcount,
                ])
        print(f"[{self._node_id}] Exported frame summary -> {path}")

    def _write_detections(
        self,
        items: list[FrameStats],
        out_dir: Path,
        batch_ts: str,
    ) -> None:
        path = out_dir / f"detections_{batch_ts}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "frame_id", "node_id", "box_type",
                "x1", "y1", "x2", "y2", "congestion_tier",
            ])
            for stats in items:
                idx = 0
                for box in stats.body_boxes:
                    tier = stats.congestion_tiers.get(idx, 0)
                    writer.writerow([
                        stats.timestamp,
                        stats.frame_id,
                        stats.node_id,
                        "body",
                        *[f"{value:.1f}" for value in box],
                        tier,
                    ])
                    idx += 1
                for box in stats.head_boxes:
                    tier = stats.congestion_tiers.get(idx, 3)
                    writer.writerow([
                        stats.timestamp,
                        stats.frame_id,
                        stats.node_id,
                        "head",
                        *[f"{value:.1f}" for value in box],
                        tier,
                    ])
                    idx += 1
        print(f"[{self._node_id}] Exported detections   -> {path}")

    def _write_frame_images(
        self,
        items: list[FrameStats],
        out_dir: Path,
        batch_ts: str,
    ) -> None:
        path = out_dir / f"frame_images_{batch_ts}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "frame_id",
                "node_id",
                "frame_variant",
                "image_path",
                "image_url",
                "body_count",
                "head_count",
                "total_headcount",
            ])
            for stats in items:
                for image in stats.frame_images:
                    writer.writerow([
                        stats.timestamp,
                        stats.frame_id,
                        stats.node_id,
                        image.frame_variant,
                        image.image_path,
                        image.image_url,
                        stats.body_count,
                        stats.head_count,
                        stats.total_headcount,
                    ])
        print(f"[{self._node_id}] Exported frame images -> {path}")
