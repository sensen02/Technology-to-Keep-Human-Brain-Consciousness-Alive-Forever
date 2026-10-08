#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
PY=$ROOT/venv/bin/python
cd "$ROOT"
nohup "$PY" -u run_p3_wound.py > "$ROOT/outputs/p3.log" 2>&1 &
echo "p3 -> pid $!"
