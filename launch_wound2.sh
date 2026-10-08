#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
PY=$ROOT/venv/bin/python
cd "$ROOT"
wrun(){ nohup "$PY" -u run_wound_scan.py "$1" > "$ROOT/outputs/scx_$2.log" 2>&1 & echo "  $2 pid $!"; }
# stretch-induced proliferation (default), varying wound size and mechanics
wrun '[{"prolif":true,"lv":6,"lf":8,"mcs":4000,"r_um":20}]'                        s20
wrun '[{"prolif":true,"lv":6,"lf":8,"mcs":4000,"r_um":10}]'                        s10
wrun '[{"prolif":true,"lv":6,"lf":8,"mcs":6000,"r_um":30}]'                        s30
wrun '[{"prolif":true,"lv":8,"lf":10,"mcs":4000,"r_um":20}]'                       sstrong
wrun '[{"prolif":false,"lv":6,"lf":8,"mcs":4000,"r_um":20}]'                       snoP
wrun '[{"prolif":true,"lv":6,"lf":8,"mcs":4000,"r_um":20,"seed":7}]'               sseed7
