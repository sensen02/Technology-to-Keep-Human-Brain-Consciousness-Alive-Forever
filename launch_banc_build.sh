#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
cd "$ROOT"
nohup "$ROOT/venv/bin/python" -u build_banc_cns.py > "$ROOT/outputs/banc_build.log" 2>&1 &
echo "banc build -> pid $!"
