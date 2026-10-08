#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
cd "$ROOT"
nohup "$ROOT/venv/bin/python" -u run_p1b_merks.py > "$ROOT/outputs/p1b.log" 2>&1 &
echo "p1b -> pid $!"
