"""
Smart Traffic Management System — Video Processing Pipeline
=============================================================

Uses YOLOv8 (Ultralytics) to detect and track vehicles in a video stream
(file or live camera/RTSP), counts vehicles crossing a virtual line per
lane, estimates traffic density/congestion, and logs structured events
(JSON Lines) that downstream Hadoop jobs can batch-process.

Usage:
    python video_pipeline.py --source traffic.mp4 --output ../data/sample_logs/events.jsonl
    python video_pipeline.py --source 0                     # webcam
    python video_pipeline.py --source rtsp://camera-ip/stream

Design notes:
- Detection: YOLOv8n (nano) for real-time speed; swap `--model` for larger
  weights (yolov8s/m/l) if more accuracy is needed and hardware allows.
- Tracking: Ultralytics' built-in ByteTrack (via `model.track`) gives each
  vehicle a persistent ID so we don't double-count across frames.
- Counting: a configurable virtual counting line. A track is counted once
  when its centroid crosses the line.
- Congestion estimate: vehicles-in-frame / lane_capacity, bucketed into
  free / moderate / heavy.
- Output: one JSON object per line (JSONL) — easy to ingest into HDFS and
  process with the MapReduce jobs in ../hadoop/.
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
from ultralytics import YOLO

# COCO class ids we care about for "vehicle" traffic
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class TrafficEvent:
    timestamp: float
    frame_id: int
    track_id: int
    vehicle_class: str
    x: float
    y: float
    speed_px_s: float
    direction: str
    event_type: str  # "detected" | "line_cross"
    lane: str = "lane_1"


class LineCounter:
    """Counts track centroids crossing a horizontal or vertical virtual line."""

    def __init__(self, line_y: int = None, line_x: int = None):
        if line_y is None and line_x is None:
            raise ValueError("Provide line_y (horizontal) or line_x (vertical).")
        self.line_y = line_y
        self.line_x = line_x
        self.prev_positions = {}
        self.counted_ids = set()
        self.counts = defaultdict(int)  # per vehicle class

    def update(self, track_id: int, cx: float, cy: float, vclass: str):
        crossed = False
        prev = self.prev_positions.get(track_id)
        if prev is not None:
            if self.line_y is not None:
                if (prev[1] < self.line_y <= cy) or (prev[1] > self.line_y >= cy):
                    crossed = True
            else:
                if (prev[0] < self.line_x <= cx) or (prev[0] > self.line_x >= cx):
                    crossed = True
        self.prev_positions[track_id] = (cx, cy)

        if crossed and track_id not in self.counted_ids:
            self.counted_ids.add(track_id)
            self.counts[vclass] += 1
            return True
        return False

    def total(self):
        return sum(self.counts.values())


def congestion_level(vehicles_in_frame: int, lane_capacity: int = 15) -> str:
    ratio = vehicles_in_frame / max(lane_capacity, 1)
    if ratio < 0.4:
        return "free"
    elif ratio < 0.8:
        return "moderate"
    return "heavy"


def run_pipeline(source: str, output_path: str, model_name: str = "yolov8n.pt",
                  line_y_ratio: float = 0.6, lane_capacity: int = 15,
                  conf: float = 0.35, show: bool = False, lane: str = "lane_1"):
    model = YOLO(model_name)

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    line_y = int(frame_h * line_y_ratio)
    counter = LineCounter(line_y=line_y)

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    prev_centroids = {}
    frame_id = 0
    t_start = time.time()

    with out_file.open("w") as f:
        for result in model.track(source=source, stream=True, persist=True,
                                   conf=conf, classes=list(VEHICLE_CLASSES.keys()),
                                   verbose=False):
            frame_id += 1
            now = time.time()
            frame = result.orig_img
            vehicles_in_frame = 0

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy().astype(int)
                clss = result.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls_id in zip(boxes, ids, clss):
                    if cls_id not in VEHICLE_CLASSES:
                        continue
                    vehicles_in_frame += 1
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    vclass = VEHICLE_CLASSES[cls_id]

                    # speed estimate (pixels/sec) from previous centroid
                    speed = 0.0
                    direction = "unknown"
                    if tid in prev_centroids:
                        px, py, pt = prev_centroids[tid]
                        dt = max(now - pt, 1e-3)
                        # cast out of numpy float32 so json.dumps can serialize it
                        speed = float(((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / dt)
                        direction = "down" if cy > py else "up"
                    prev_centroids[tid] = (cx, cy, now)

                    event = TrafficEvent(
                        timestamp=now, frame_id=frame_id, track_id=int(tid),
                        vehicle_class=vclass, x=float(cx), y=float(cy),
                        speed_px_s=round(speed, 2), direction=direction,
                        event_type="detected", lane=lane,
                    )
                    f.write(json.dumps(asdict(event)) + "\n")

                    if counter.update(tid, cx, cy, vclass):
                        cross_event = TrafficEvent(
                            timestamp=now, frame_id=frame_id, track_id=int(tid),
                            vehicle_class=vclass, x=float(cx), y=float(cy),
                            speed_px_s=round(speed, 2), direction=direction,
                            event_type="line_cross", lane=lane,
                        )
                        f.write(json.dumps(asdict(cross_event)) + "\n")

                    if show:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv2.putText(frame, f"{vclass}#{tid}", (int(x1), int(y1) - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            level = congestion_level(vehicles_in_frame, lane_capacity)

            if show:
                cv2.line(frame, (0, line_y), (frame.shape[1], line_y), (0, 0, 255), 2)
                cv2.putText(frame, f"Count: {counter.total()}  Congestion: {level}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("Smart Traffic Management", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_id % 30 == 0:  # periodic summary line every ~1s @30fps
                summary = {
                    "timestamp": now, "frame_id": frame_id, "event_type": "summary",
                    "lane": lane,
                    "vehicles_in_frame": vehicles_in_frame,
                    "congestion_level": level,
                    "cumulative_counts": dict(counter.counts),
                    "fps": round(frame_id / max(now - t_start, 1e-3), 2),
                }
                f.write(json.dumps(summary) + "\n")

    cap.release()
    if show:
        cv2.destroyAllWindows()

    print(f"Processed {frame_id} frames. Total vehicle count: {counter.total()}")
    print(f"Per-class counts: {dict(counter.counts)}")
    print(f"Events written to: {out_file.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Traffic Management — video pipeline")
    parser.add_argument("--source", required=True, help="Video file path, webcam index (0), or RTSP URL")
    parser.add_argument("--output", default="../data/sample_logs/events.jsonl", help="Path to JSONL event log")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLO weights (yolov8n/s/m/l.pt)")
    parser.add_argument("--line-y-ratio", type=float, default=0.6, help="Counting line as fraction of frame height")
    parser.add_argument("--lane-capacity", type=int, default=15, help="Vehicles considered 'full' lane capacity")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--show", action="store_true", help="Display annotated video window")
    parser.add_argument("--lane", default="lane_1",
                         help="Name of the approach/lane this camera covers (e.g. north, south, east, west). "
                              "Run one pipeline instance per camera at an intersection, each with its own --lane, "
                              "so signal_controller.py can compare arrivals across approaches.")
    args = parser.parse_args()

    run_pipeline(args.source, args.output, args.model, args.line_y_ratio,
                 args.lane_capacity, args.conf, args.show, args.lane)
