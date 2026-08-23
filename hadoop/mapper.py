#!/usr/bin/env python3
"""
mapper.py — Hadoop Streaming mapper

Reads JSONL traffic events from stdin (as staged into HDFS from the video
pipeline's output) and emits key/value pairs for aggregation:

    <hour>|<vehicle_class>|line_cross \t 1
    <hour>|congestion|<level>          \t 1
    <hour>|<vehicle_class>|speed       \t <speed_px_s>

The reducer sums/aggregates these per key. Bucketing by hour lets us
compute hourly traffic volume and congestion trends at scale, exactly
the kind of batch analytics Hadoop is suited for (as opposed to the
real-time per-frame processing done by the YOLO pipeline).
"""

import sys
import json
from datetime import datetime, timezone


def hour_bucket(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H")


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue

        ts = ev.get("timestamp")
        if ts is None:
            continue
        hour = hour_bucket(ts)
        etype = ev.get("event_type")

        if etype == "line_cross":
            vclass = ev.get("vehicle_class", "unknown")
            lane = ev.get("lane", "lane_1")
            print(f"{hour}|{vclass}|count\t1")
            print(f"{hour}|lane_{lane}|count\t1")
        elif etype == "detected":
            vclass = ev.get("vehicle_class", "unknown")
            speed = ev.get("speed_px_s", 0)
            print(f"{hour}|{vclass}|speed\t{speed}")
        elif etype == "summary":
            level = ev.get("congestion_level", "unknown")
            print(f"{hour}|congestion|{level}\t1")
        elif etype == "signal_plan":
            for lane, secs in ev.get("green_seconds", {}).items():
                print(f"{hour}|signal_{lane}|speed\t{secs}")  # reuse avg aggregation for avg green seconds


if __name__ == "__main__":
    main()
