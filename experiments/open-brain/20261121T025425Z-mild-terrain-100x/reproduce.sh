#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash build_load.sh
bash collect.sh &
collect_pid=$!
nice -n 10 bash env.sh practice.py > train.log 2>&1 &
fit_pid=$!
wait "$collect_pid"
wait "$fit_pid"
bash env.sh audit.py
bash env.sh report.py
