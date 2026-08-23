# Architecture

```
 ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │ Camera: north│    │ Camera: south│    │ Camera: east │    │ Camera: west │
 └──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
        │  video_pipeline.py --lane <name>  (YOLOv8 + ByteTrack, per approach)
        └────────────────────┴──────────────────┴──────────────────┘
                                     │ lane-tagged JSONL events
                                     ▼
                      data/sample_logs/events.jsonl  (shared log)
                                     │
                 ┌───────────────────┼────────────────────────┐
                 ▼                   ▼                        ▼
     signal_controller.py    web/app.py (dashboard)    hadoop/ (batch analytics)
     rolls up per-lane        live congestion,          mapper.py -> shuffle/sort
     arrivals every cycle     counts, signal plan        -> reducer.py: hourly
        │                                                 volume/lane, avg speed,
        ▼                                                 avg green-time/lane
     signal_optimizer.py
     QUBO formulation, solved
     via dimod + neal (D-Wave's
     own simulated-annealing
     sampler — quantum-ready,
     swaps to real hardware
     with one line)
        │
        ▼
     "signal_plan" event
     (green seconds per lane)
     written back to the log
```

## Component responsibilities

**Video pipeline (`src/video_pipeline.py`)** — real-time layer.
Runs YOLOv8 object detection with built-in ByteTrack tracking on each
frame, filters to vehicle classes (car/truck/bus/motorcycle), tracks a
virtual counting line to avoid double counting, and estimates local
congestion from vehicle density. Every detection and line-crossing event
is appended to a JSONL log — a natural, append-friendly format for
streaming into HDFS.

**Batch layer (`hadoop/`)** — scale layer.
`mapper.py` and `reducer.py` implement a classic Hadoop Streaming job.
The mapper bucket events by hour and emits partial keys; Hadoop's
shuffle/sort groups identical keys across however many mapper tasks ran;
the reducer aggregates each group into hourly vehicle counts, average
speeds, and congestion distributions. This is where Hadoop's value shows:
processing weeks/months of accumulated logs from many intersections in
parallel, rather than one process reading one file.

**Signal optimization (`src/signal_optimizer.py`, `src/signal_controller.py`)** — quantum-ready layer.
Every cycle, `signal_controller.py` sums recent line-crossing events per
lane and hands the counts to `signal_optimizer.py`, which formulates
"how many seconds of green light should each approach get" as a QUBO
(Quadratic Unconstrained Binary Optimization) — the exact problem format
quantum annealers solve. It's built with `dimod` (D-Wave's own modeling
library) and solved with `neal.SimulatedAnnealingSampler`, a classical
algorithm that implements the identical sampler interface as D-Wave's
real quantum hardware. `solve_on_quantum_hardware()` is the same code
path pointed at `DWaveSampler()` instead — a one-line swap, not a
rewrite, once real hardware access is available. The result (green
seconds per lane) is written back to the log as a `signal_plan` event.

**Dashboard (`web/`)** — presentation layer.
A small Flask app exposes the current state (`/api/summary`) and a
time-series (`/api/timeseries`) consumed by a vanilla-JS dashboard. In
production this would read from the Hadoop output / a database instead
of the raw JSONL, but pointing it at the same event log keeps local
development simple.

## Scaling this into "real-world"

- Swap the flat JSONL file for a Kafka topic between the video pipeline
  and HDFS for true streaming ingestion.
- Run the Hadoop job on a schedule (Oozie/Airflow) against the day's
  accumulated logs in HDFS.
- Replace the dashboard's file read with queries against the reducer's
  output (e.g. loaded into Hive or a small Postgres table).
- Run one video_pipeline.py process per camera/intersection, each
  writing to its own HDFS path partitioned by intersection ID and date.
- Point `signal_optimizer.solve_on_quantum_hardware` at a provisioned
  D-Wave Leap account to run the identical QUBO on real quantum
  annealing hardware instead of the classical sampler.
