#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
cd "$ROOT"
nohup "$ROOT/venv/bin/python" -u build_flywire_connectome.py > "$ROOT/outputs/flywire_build.log" 2>&1 &
echo "build -> pid $!"
