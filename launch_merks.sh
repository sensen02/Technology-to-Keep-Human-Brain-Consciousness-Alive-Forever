#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
cd "$ROOT"
nohup "$ROOT/venv/bin/python" -u try_merks.py > "$ROOT/outputs/merks.log" 2>&1 &
echo "merks -> pid $!"
