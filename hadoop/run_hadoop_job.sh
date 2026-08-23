#!/usr/bin/env bash
# run_hadoop_job.sh
# ------------------
# Stages event logs into HDFS and runs a Hadoop Streaming MapReduce job
# using mapper.py / reducer.py to compute hourly traffic volume, average
# speed per vehicle class, and congestion-level distribution.
#
# Prerequisites: a running Hadoop cluster (HADOOP_HOME set, HDFS + YARN up)
# and the hadoop-streaming jar available.
#
# Usage:
#   ./run_hadoop_job.sh /local/path/to/events.jsonl
#
set -euo pipefail

LOCAL_INPUT="${1:?Usage: $0 <local-events.jsonl>}"
HDFS_INPUT_DIR="/traffic/input"
HDFS_OUTPUT_DIR="/traffic/output/$(date +%Y%m%d_%H%M%S)"
STREAMING_JAR="${HADOOP_STREAMING_JAR:-$HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar}"

echo "== Staging input into HDFS =="
hdfs dfs -mkdir -p "$HDFS_INPUT_DIR"
hdfs dfs -put -f "$LOCAL_INPUT" "$HDFS_INPUT_DIR/"

echo "== Running Hadoop Streaming job =="
hadoop jar $STREAMING_JAR \
  -files mapper.py,reducer.py \
  -mapper "python3 mapper.py" \
  -reducer "python3 reducer.py" \
  -input "$HDFS_INPUT_DIR/*" \
  -output "$HDFS_OUTPUT_DIR"

echo "== Job complete. Results: =="
hdfs dfs -cat "$HDFS_OUTPUT_DIR/part-*" | sort
