"""
traffic_analyzer.py
====================
Lightweight local analytics over JSONL event logs produced by
video_pipeline.py — useful for quick checks without spinning up Hadoop.
For large-scale/batch analysis across many days of logs, use the
MapReduce jobs in ../hadoop/ instead.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load_events(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def analyze(path: str):
    class_counts = Counter()
    congestion_counts = Counter()
    speeds = defaultdict(list)
    total_line_crossings = 0

    for ev in load_events(path):
        etype = ev.get("event_type")
        if etype == "line_cross":
            class_counts[ev["vehicle_class"]] += 1
            total_line_crossings += 1
        elif etype == "detected":
            speeds[ev["vehicle_class"]].append(ev["speed_px_s"])
        elif etype == "summary":
            congestion_counts[ev["congestion_level"]] += 1

    print("=== Traffic Summary ===")
    print(f"Total vehicles counted (line crossings): {total_line_crossings}")
    print("By class:")
    for cls, n in class_counts.most_common():
        print(f"  {cls:12s} {n}")

    print("\nCongestion level distribution (by sampled frames):")
    for level, n in congestion_counts.most_common():
        print(f"  {level:10s} {n}")

    print("\nAverage speed (px/s) by class:")
    for cls, vals in speeds.items():
        avg = sum(vals) / len(vals) if vals else 0
        print(f"  {cls:12s} {avg:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze traffic event JSONL logs")
    parser.add_argument("--input", required=True, help="Path to events.jsonl")
    args = parser.parse_args()
    analyze(args.input)
