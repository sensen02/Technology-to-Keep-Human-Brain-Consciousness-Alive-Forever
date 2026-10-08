#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
cd "$ROOT"
for i in 0 1 2 3 4 5; do
  nohup "$ROOT/venv/bin/python" -u run_p1c_phase.py $i 6 > "$ROOT/outputs/p1c_$i.log" 2>&1 &
  echo "  shard $i -> pid $!"
done
