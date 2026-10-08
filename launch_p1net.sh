#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
cd "$ROOT"
nohup "$ROOT/venv/bin/python" -u run_p1_network.py > "$ROOT/outputs/p1net.log" 2>&1 &
echo "p1net -> pid $!"
