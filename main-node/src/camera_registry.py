import threading
from dataclasses import dataclass
from datetime import datetime

try:
    import clickhouse_connect
except ImportError:  # pragma: no cover - import is validated by dependency setup
    clickhouse_connect = None

from clickhouse_config import ClickHouseConfig
from config import CAMERA_CATALOG


@dataclass
class CameraMetadata:
    node_id: str
    label: str
    latitude: float
    longitude: float
    updated_at: datetime


class CameraRegistry:
    """Tracks bus stop metadata and persists it for Grafana geomaps."""

    def __init__(
        self,
        allowed_nodes: list[str],
        clickhouse_config: ClickHouseConfig | None = None,
    ):
        self._allowed_nodes = set(allowed_nodes)
        self._clickhouse_config = clickhouse_config
        self._lock = threading.Lock()
        self._client = None
        self._metadata: dict[str, CameraMetadata] = {}
        self._persistence_disabled_reason: str | None = None

        for node_id in allowed_nodes:
            catalog_entry = CAMERA_CATALOG.get(node_id)
            if catalog_entry is None:
                continue

            self.register(
                node_id=node_id,
                label=str(catalog_entry.get("label", node_id)),
                latitude=float(catalog_entry["latitude"]),
                longitude=float(catalog_entry["longitude"]),
            )

    def register(
        self,
        node_id: str,
        latitude: float,
        longitude: float,
        label: str | None = None,
    ) -> None:
        if node_id not in self._allowed_nodes:
            return

        metadata = CameraMetadata(
            node_id=node_id,
            label=label or node_id,
            latitude=latitude,
            longitude=longitude,
            updated_at=datetime.now(),
        )

        with self._lock:
            self._metadata[node_id] = metadata

        if self._clickhouse_config is not None and self._persistence_disabled_reason is None:
            try:
                self._persist(metadata)
            except Exception as exc:
                message = str(exc)
                if "UNKNOWN_TABLE" in message or "does not exist" in message:
                    self._persistence_disabled_reason = (
                        "camera_nodes table is missing in ClickHouse. Run the schema migration "
                        "or recreate the ClickHouse volume so the init SQL is applied."
                    )
                elif "ACCESS_DENIED" in message or "Not enough privileges" in message:
                    self._persistence_disabled_reason = (
                        "ClickHouse user lacks schema migration privileges. Grant CREATE TABLE "
                        "on crowd_check.* to crowdcheck, then restart the main node."
                    )
                else:
                    self._persistence_disabled_reason = f"unexpected ClickHouse error: {exc}"

                print(
                    f"[{node_id}] Failed to persist camera metadata; disabling metadata writes: "
                    f"{self._persistence_disabled_reason}"
                )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def _persist(self, metadata: CameraMetadata) -> None:
        client = self._get_client()
        client.insert(
            "camera_nodes",
            [[
                metadata.node_id,
                metadata.label,
                metadata.latitude,
                metadata.longitude,
                metadata.updated_at,
            ]],
            database=self._clickhouse_config.database,
            column_names=[
                "node_id",
                "label",
                "latitude",
                "longitude",
                "updated_at",
            ],
        )

    def _get_client(self):
        if clickhouse_connect is None:
            raise RuntimeError(
                "Camera metadata persistence requires clickhouse-connect to be installed."
            )

        if self._client is None:
            self._client = clickhouse_connect.get_client(
                host=self._clickhouse_config.host,
                port=self._clickhouse_config.port,
                username=self._clickhouse_config.username,
                password=self._clickhouse_config.password,
                database=self._clickhouse_config.database,
                secure=self._clickhouse_config.secure,
            )

        return self._client
