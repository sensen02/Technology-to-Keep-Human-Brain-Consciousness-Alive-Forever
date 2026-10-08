#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
cd "$ROOT"
nohup "$ROOT/venv/bin/python" -u flywire_download.py banc > "$ROOT/outputs/banc_download.log" 2>&1 &
echo "banc download -> pid $!"
