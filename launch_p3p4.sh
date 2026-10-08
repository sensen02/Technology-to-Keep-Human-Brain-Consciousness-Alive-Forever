#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
PY=$ROOT/venv/bin/python
cd "$ROOT"
# stop the number-only duplicate wound scans (run_p3_wound.py supersedes them)
for pid in $(pgrep -f "^$PY -u run_wound_scan.py" || true); do kill "$pid" 2>/dev/null; done
sleep 2
bg(){ local tag="$1"; shift; nohup "$@" > "$ROOT/outputs/$tag.log" 2>&1 & echo "  $tag -> pid $!"; }
bg p3 "$PY" -u run_p3_wound.py
bg p4 "$PY" -u run_p4_epidermis.py
