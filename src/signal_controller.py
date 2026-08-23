"""
signal_controller.py — ties live per-lane vehicle counts to the QUBO
signal optimizer, on a repeating cycle.

In a real intersection you'd run one video_pipeline.py per camera
(--lane north / south / east / west), each appending lane-tagged events
to a shared log. This controller tails that log, sums each lane's
vehicle arrivals over the current cycle, calls signal_optimizer, and
writes the resulting green-time plan back to the log as a "signal_plan"
event — which both the Hadoop batch job and the dashboard can read.

Usage:
    python signal_controller.py --log ../data/sample_logs/events.jsonl --cycle-time 90 --interval 30
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

from signal_optimizer import optimize_signal_timings


def count_recent_arrivals(log_path: Path, since_ts: float) -> dict:
    """Sum line_cross events per lane since `since_ts`."""
    counts = defaultdict(int)
    if not log_path.exists():
        return counts
    with log_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event_type") == "line_cross" and ev.get("timestamp", 0) >= since_ts:
                counts[ev.get("lane", "lane_1")] += 1
    return counts


def run_controller(log_path: str, cycle_time: int = 90, interval: int = 30,
                    min_green: int = 10, once: bool = False):
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    while True:
        window_start = time.time() - interval
        counts = count_recent_arrivals(path, window_start)

        if not counts:
            print("No lane traffic observed yet — waiting for detection events...")
        else:
            # Every observed lane must be represented, even at 0, so the
            # optimizer still gives it the minimum green floor.
            plan = optimize_signal_timings(dict(counts), cycle_time=cycle_time, min_green=min_green)
            record = {
                "timestamp": time.time(),
                "event_type": "signal_plan",
                "lane_counts": dict(counts),
                **plan,
            }
            with path.open("a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"[signal_controller] counts={dict(counts)} -> "
                  f"green={plan['green_seconds']} (solver={plan['solver']})")

        if once:
            break
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live signal-timing controller")
    parser.add_argument("--log", default="../data/sample_logs/events.jsonl")
    parser.add_argument("--cycle-time", type=int, default=90, help="Signal cycle length in seconds")
    parser.add_argument("--interval", type=int, default=30, help="How often to recompute, in seconds")
    parser.add_argument("--min-green", type=int, default=10)
    parser.add_argument("--once", action="store_true", help="Run a single cycle and exit (for testing/demo)")
    args = parser.parse_args()

    run_controller(args.log, args.cycle_time, args.interval, args.min_green, args.once)
