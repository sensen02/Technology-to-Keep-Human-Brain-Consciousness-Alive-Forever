#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
cd "$ROOT"
nohup "$ROOT/venv/bin/python" -u run_flywire_brain.py > "$ROOT/outputs/flywire_brain.log" 2>&1 &
echo "flywire brain -> pid $!"
