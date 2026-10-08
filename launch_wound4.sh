#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
PY=$ROOT/venv/bin/python
cd "$ROOT"
w(){ nohup "$PY" -u run_wound_scan.py "$1" > "$ROOT/outputs/$2.log" 2>&1 & echo "  $2 pid $!"; }
# slower mechanical closure -> margin cells keep a free surface long enough to divide
w '[{"prolif":true,"lv":8,"lf":2.5,"mcs":6000,"r_um":20}]'                 v_lf2.5
w '[{"prolif":false,"lv":8,"lf":2.5,"mcs":6000,"r_um":20}]'                v_lf2.5noP
w '[{"prolif":true,"lv":12,"lf":3.0,"mcs":6000,"r_um":20}]'                v_lf3lv12
