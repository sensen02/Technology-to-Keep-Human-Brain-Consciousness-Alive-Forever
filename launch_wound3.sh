#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
PY=$ROOT/venv/bin/python
cd "$ROOT"
w(){ nohup "$PY" -u run_wound_scan.py "$1" > "$ROOT/outputs/$2.log" 2>&1 & echo "  $2 pid $!"; }
w '[{"prolif":true,"lv":6,"lf":8,"mcs":5000,"r_um":20}]'                 f20
w '[{"prolif":false,"lv":6,"lf":8,"mcs":5000,"r_um":20}]'                f20noP
w '[{"prolif":true,"lv":6,"lf":8,"mcs":5000,"r_um":15}]'                 f15
