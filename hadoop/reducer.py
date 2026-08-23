#!/usr/bin/env python3
"""
reducer.py — Hadoop Streaming reducer

Consumes the sorted key/value stream from mapper.py and produces one
aggregated result per key:
  - "*|count"     -> total count of vehicles crossing the line that hour
  - "*|speed"     -> average speed (px/s) that hour
  - "*|congestion|<level>" -> total sampled frames at that congestion level

Hadoop guarantees keys arrive sorted and grouped, so a simple running
total per key change is sufficient (standard streaming-reducer pattern).
"""

import sys


def emit(key, kind, values):
    if kind == "speed":
        avg = sum(values) / len(values) if values else 0.0
        print(f"{key}\tavg_speed_px_s={avg:.2f}\tsamples={len(values)}")
    else:
        print(f"{key}\ttotal={sum(values)}")


def main():
    current_key = None
    current_kind = None
    values = []

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line or "\t" not in line:
            continue
        key, val = line.split("\t", 1)
        parts = key.split("|")
        kind = parts[-1] if parts[-1] in ("count", "speed") else "congestion"

        try:
            val = float(val)
        except ValueError:
            continue

        if key != current_key:
            if current_key is not None:
                emit(current_key, current_kind, values)
            current_key = key
            current_kind = kind
            values = []
        values.append(val)

    if current_key is not None:
        emit(current_key, current_kind, values)


if __name__ == "__main__":
    main()
