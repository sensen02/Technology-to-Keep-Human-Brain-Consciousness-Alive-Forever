#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
PY=$ROOT/venv/bin/python
cd "$ROOT"
w(){ nohup "$PY" -u run_wound_scan.py "$1" > "$ROOT/outputs/$2.log" 2>&1 & echo "  $2 pid $!"; }
w '[{"prolif":true,"lv":8,"lf":4,"mcs":4000,"r_um":20,"cycle_mcs":100}]'    x_c100
w '[{"prolif":false,"lv":8,"lf":4,"mcs":4000,"r_um":20,"cycle_mcs":100}]'   x_c100noP
