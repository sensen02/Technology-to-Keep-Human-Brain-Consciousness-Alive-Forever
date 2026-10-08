#!/bin/bash
# Relaunch all phases on the corrected engine (division now inherits per-cell
# parameters).  Each entry is a single-threaded worker process.
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
PY=$ROOT/venv/bin/python
cd "$ROOT"
mkdir -p outputs

bg(){ # $1 = tag, rest = command
  local tag="$1"; shift
  nohup "$@" > "$ROOT/outputs/$tag.log" 2>&1 &
  echo "  $tag -> pid $!"
}

# P1 + P2 + P2b: sprouting angiogenesis, with and without contact inhibition
bg p1        "$PY" -u run_p1_vasculogenesis.py

# P3: wound closure, with and without proliferation (the two-phase test)
bg w20       "$PY" -u run_wound_scan.py '[{"prolif":true,"lv":6,"lf":8,"mcs":4000,"r_um":20}]'
bg wNoP      "$PY" -u run_wound_scan.py '[{"prolif":false,"lv":6,"lf":8,"mcs":4000,"r_um":20}]'
bg w10       "$PY" -u run_wound_scan.py '[{"prolif":true,"lv":6,"lf":8,"mcs":4000,"r_um":10}]'

# P4: epidermis (quick check first that the tissue now grows upward)
bg epcheck   "$PY" -u check_epidermis.py
