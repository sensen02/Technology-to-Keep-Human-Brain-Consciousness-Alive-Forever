#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
PY=$ROOT/venv/bin/python
cd "$ROOT"
wrun(){ nohup "$PY" -u run_wound_scan.py "$1" > "$ROOT/outputs/scanw_$2.log" 2>&1 & echo "  launched $2 (pid $!)"; }
wrun '[{"prolif":false,"lv":6,"lf":8,"mcs":3000}]'                             a
wrun '[{"prolif":true,"lv":6,"lf":8,"mcs":3000}]'                              b
wrun '[{"prolif":true,"lv":3,"lf":8,"mcs":3000}]'                              c
wrun '[{"prolif":true,"lv":6,"lf":8,"mcs":3000,"r_um":10}]'                    d
wrun '[{"prolif":true,"lv":6,"lf":8,"mcs":5000,"r_um":30}]'                    e
wrun '[{"prolif":true,"lv":6,"lf":8,"mcs":3000,"seed":7}]'                     f
