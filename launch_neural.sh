#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
cd "$ROOT"
nohup "$ROOT/venv/bin/python" -u run_neural_selftest.py > "$ROOT/outputs/neural_selftest.log" 2>&1 &
echo "neural selftest -> pid $!"
