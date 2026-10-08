#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
cd "$ROOT"
nohup "$ROOT/venv/bin/python" -u render_merks_fine.py > "$ROOT/outputs/p1d_render.log" 2>&1 &
echo "render -> pid $!"
