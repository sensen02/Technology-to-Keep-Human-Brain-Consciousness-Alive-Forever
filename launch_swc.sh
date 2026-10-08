#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
cd "$ROOT"
nohup "$ROOT/venv/bin/python" -u download_swc.py > "$ROOT/outputs/swc_download.log" 2>&1 &
echo "swc download -> pid $!"
