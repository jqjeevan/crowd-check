from dataclasses import dataclass


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str
    port: int
    database: str
    username: str
    password: str
    secure: bool = False
