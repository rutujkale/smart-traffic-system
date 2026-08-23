"""
app.py — Smart Traffic Management dashboard

Flask web app that reads the JSONL event log (produced live by
video_pipeline.py, or the Hadoop-aggregated output) and serves:
  - GET /                 dashboard page
  - GET /api/summary       current totals, per-class counts, congestion
  - GET /api/timeseries    per-minute vehicle counts for charting

Run:
    pip install flask
    python app.py --log ../data/sample_logs/sample_events.jsonl
Then open http://localhost:5000
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template

app = Flask(__name__)
LOG_PATH = Path("../data/sample_logs/sample_events.jsonl")


def load_events():
    if not LOG_PATH.exists():
        return []
    events = []
    with LOG_PATH.open() as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return events


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/summary")
def api_summary():
    events = load_events()
    class_counts = Counter()
    congestion = Counter()
    for ev in events:
        if ev.get("event_type") == "line_cross":
            class_counts[ev.get("vehicle_class", "unknown")] += 1
        elif ev.get("event_type") == "summary":
            congestion[ev.get("congestion_level", "unknown")] += 1

    latest_level = None
    for ev in reversed(events):
        if ev.get("event_type") == "summary":
            latest_level = ev.get("congestion_level")
            break

    return jsonify({
        "total_vehicles": sum(class_counts.values()),
        "by_class": dict(class_counts),
        "congestion_distribution": dict(congestion),
        "current_congestion": latest_level or "unknown",
    })


@app.route("/api/signal-plan")
def api_signal_plan():
    events = load_events()
    latest_plan = None
    for ev in reversed(events):
        if ev.get("event_type") == "signal_plan":
            latest_plan = ev
            break
    if latest_plan is None:
        return jsonify({"available": False})
    return jsonify({
        "available": True,
        "green_seconds": latest_plan.get("green_seconds", {}),
        "cycle_time": latest_plan.get("cycle_time"),
        "lane_counts": latest_plan.get("lane_counts", {}),
        "solver": latest_plan.get("solver"),
    })


@app.route("/api/timeseries")
def api_timeseries():
    events = load_events()
    buckets = defaultdict(int)
    for ev in events:
        if ev.get("event_type") == "line_cross":
            ts = ev.get("timestamp")
            if ts is None:
                continue
            minute = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
            buckets[minute] += 1
    series = [{"time": k, "count": v} for k, v in sorted(buckets.items())]
    return jsonify(series)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default=None, help="Path to JSONL event log")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    # Resolve relative paths against CWD *now* and store as an absolute
    # path — also exported via env var so Flask's reloader subprocess
    # (if enabled) picks up the exact same file.
    import os
    if os.environ.get("TRAFFIC_LOG"):
        LOG_PATH = Path(os.environ["TRAFFIC_LOG"])
    elif args.log:
        LOG_PATH = Path(args.log).resolve()
    print(f"[dashboard] reading event log: {LOG_PATH.resolve()}")
    app.run(debug=False, use_reloader=False, port=args.port)
