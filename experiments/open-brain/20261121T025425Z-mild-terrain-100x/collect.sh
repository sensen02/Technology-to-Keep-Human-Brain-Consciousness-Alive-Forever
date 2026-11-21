#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
worker_count=9
pids=()
for ((lane=0; lane<worker_count; lane++)); do
    mapfile -t held < <(seq "$((3000 + lane))" "$worker_count" 3019)
    mapfile -t train < <(seq "$((20 + lane))" "$worker_count" 2791)
    nice -n 12 bash env.sh generate.py "${held[@]}" "${train[@]}" >> "logs/terrain-$lane.log" 2>&1 &
    pids+=("$!")
done
code=0
for pid in "${pids[@]}"; do
    wait "$pid" || code=1
done
printf "%s\n" "$code" > collection.exit
exit "$code"
