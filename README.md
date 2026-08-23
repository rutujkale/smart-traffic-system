# Intelligent Traffic Management Using AI, Big Data, and Quantum Computing

[![Python](https://img.shields.io/badge/python-3.10+-blue?style=flat-square)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-lightgrey?style=flat-square)](#)

Real-time vehicle detection with YOLOv8, scalable log processing with
Hadoop, and quantum-ready signal-timing optimization via QUBO — plus a
live web dashboard.

**Stack:** Python · YOLOv8 (Ultralytics) · OpenCV · Hadoop Streaming ·
dimod + neal (D-Wave Ocean SDK) · Flask

## What it does

1. **`src/video_pipeline.py`** — reads a video (file, webcam, or RTSP
   stream) per camera/approach, detects vehicles with YOLOv8, tracks
   them frame-to-frame, counts vehicles crossing a virtual line per
   lane, estimates congestion level, and logs everything as JSONL
   events.
2. **`src/signal_optimizer.py` + `src/signal_controller.py`** —
   **quantum-ready signal timing**: formulates "how much green time
   should each approach get this cycle" as a QUBO (the native problem
   format for quantum annealers), built with D-Wave's own `dimod`
   library and solved with `neal`'s simulated-annealing sampler. Same
   sampler interface as real D-Wave quantum hardware — swapping to
   actual hardware is a one-line change (`solve_on_quantum_hardware`).
3. **`hadoop/`** — a Hadoop Streaming MapReduce job (`mapper.py` +
   `reducer.py`) that batch-processes accumulated JSONL logs into hourly
   vehicle counts per lane, average speeds per vehicle class, average
   green-time allocation, and congestion distribution — the Big Data
   layer for processing logs from many cameras/intersections/days at once.
4. **`web/app.py`** — a Flask dashboard visualizing current congestion,
   vehicle counts by class, volume over time, and the live signal-timing
   plan per lane.

See `docs/architecture.md` for the full data-flow diagram.

## For your resume / portfolio

**Suggested bullet points:**

> **Intelligent Traffic Management System** — Python, YOLOv8, Hadoop, QUBO/Quantum Computing, Flask
> - Designed and implemented a real-time video processing pipeline using Python and YOLOv8 to detect, track, and count vehicles per intersection approach, with automated congestion-level estimation.
> - Integrated Hadoop for scalable data processing: built a Hadoop Streaming MapReduce job to aggregate high-volume, multi-camera traffic logs into hourly per-lane volume, speed, and congestion analytics, optimizing backend performance and ensuring reliable data ingestion.
> - Formulated traffic signal timing as a QUBO (Quadratic Unconstrained Binary Optimization) problem using D-Wave's Ocean SDK, solved via simulated annealing with a direct upgrade path to real quantum annealing hardware.
> - Built and deployed a live monitoring dashboard (Flask) visualizing real-time congestion, vehicle composition, and signal-timing decisions.

Swap in real metrics once you've run it against real video (e.g. "processed N hours of footage across 4 lanes," "aggregated X events/hour via Hadoop," "reduced simulated average wait time by X%").

**For a portfolio site or resume link:** since this runs a live video pipeline and (optionally) a Hadoop cluster, it isn't a fit for static hosting like Vercel. Better options: link the GitHub repo directly, or record a short screen capture (GIF or video) of the dashboard and YOLO detection window running, and embed that at the top of this README — recruiters skim, they don't clone.



## Quick start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the video pipeline (one instance per camera/approach)
```bash
cd src
# on a video file, tagging which approach this camera covers
python video_pipeline.py --source /path/to/north_cam.mp4 --lane north --show
python video_pipeline.py --source /path/to/south_cam.mp4 --lane south --show
# on a webcam
python video_pipeline.py --source 0 --lane north --show
```
All instances append lane-tagged events to `data/sample_logs/events.jsonl`
by default. First run downloads YOLOv8n weights automatically (~6MB).

### 3. Run the signal-timing optimizer
```bash
# one-off: try the QUBO optimizer directly with made-up counts
python signal_optimizer.py --counts '{"north":18,"south":15,"east":4,"west":3}' --cycle-time 120

# continuous: pulls live lane counts from the shared event log every cycle
python signal_controller.py --log ../data/sample_logs/events.jsonl --cycle-time 90 --interval 30
```
Each cycle appends a `signal_plan` event (green seconds per lane) back
into the same log, so the dashboard and Hadoop job both pick it up.

### 4. Inspect results locally (no cluster needed)
```bash
python traffic_analyzer.py --input ../data/sample_logs/events.jsonl
```

### 5. Run the Hadoop batch job (requires a Hadoop cluster)
```bash
cd hadoop
./run_hadoop_job.sh ../data/sample_logs/events.jsonl
```
You can also test the map/reduce logic locally without a cluster,
exactly the way Hadoop Streaming would run it:
```bash
cat ../data/sample_logs/sample_events.jsonl | python3 mapper.py | sort | python3 reducer.py
```

### 6. Launch the dashboard
```bash
cd web
python app.py --log ../data/sample_logs/sample_events.jsonl
# open http://localhost:5000
```

## Project layout
```
smart-traffic-system/
├── src/
│   ├── video_pipeline.py      # YOLO detection + tracking + per-lane counting
│   ├── signal_optimizer.py    # QUBO signal-timing optimizer (dimod + neal)
│   ├── signal_controller.py   # live loop: log -> lane counts -> optimizer -> plan
│   └── traffic_analyzer.py    # local ad-hoc analytics over JSONL logs
├── hadoop/
│   ├── mapper.py               # Hadoop Streaming mapper
│   ├── reducer.py              # Hadoop Streaming reducer
│   └── run_hadoop_job.sh       # stages data to HDFS and runs the job
├── web/
│   ├── app.py                  # Flask dashboard API
│   ├── templates/index.html
│   └── static/style.css
├── data/sample_logs/           # sample JSONL for testing without a camera/cluster
├── docs/architecture.md
└── requirements.txt
```

## Notes for the pitch (Technical Approach slide)

- **Artificial Intelligence** → `src/video_pipeline.py`: YOLOv8 detects
  and tracks vehicles in real time, per approach, feeding both the
  congestion estimate and the signal optimizer.
- **Quantum Computing** → `src/signal_optimizer.py`: signal timing is
  formulated as a QUBO — the native problem format for quantum
  annealers — and solved with D-Wave's own `dimod`/`neal` libraries.
  It's genuinely quantum-ready: `solve_on_quantum_hardware()` runs the
  identical QUBO on real D-Wave hardware with a one-line sampler swap.
- **Big Data / Cloud** → `hadoop/`: MapReduce job aggregating high-volume
  multi-camera JSONL logs into hourly per-lane traffic volume, speeds,
  and average signal-timing decisions at scale.
- **IoT** → out of scope for this build (no physical signal controller
  hardware); `signal_plan` events are the interface a real IoT
  controller would subscribe to.
