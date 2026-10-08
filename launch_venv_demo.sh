#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
cd "$ROOT"
nohup "$ROOT/venv/bin/python" -u run_virtual_env.py > "$ROOT/outputs/virtual_env.log" 2>&1 &
echo "virtual env -> pid $!"
