CREATE DATABASE IF NOT EXISTS crowd_check;

CREATE USER IF NOT EXISTS crowdcheck
IDENTIFIED WITH plaintext_password BY 'crowdcheck';

GRANT SELECT, INSERT ON crowd_check.* TO crowdcheck;

CREATE TABLE IF NOT EXISTS crowd_check.frame_summary
(
    `timestamp` DateTime64(3),
    `node_id` LowCardinality(String),
    `body_count` UInt32,
    `head_count` UInt32,
    `total_headcount` UInt32
)
ENGINE = MergeTree
ORDER BY (`node_id`, `timestamp`);

CREATE TABLE IF NOT EXISTS crowd_check.detections
(
    `timestamp` DateTime64(3),
    `node_id` LowCardinality(String),
    `box_type` LowCardinality(String),
    `x1` Float32,
    `y1` Float32,
    `x2` Float32,
    `y2` Float32,
    `congestion_tier` UInt8
)
ENGINE = MergeTree
ORDER BY (`node_id`, `timestamp`, `box_type`, `congestion_tier`);
