#!/bin/bash
set -u
ROOT=/run/media/sensen/Data2/cell_wound_prototype
PY=$ROOT/venv/bin/python
cd "$ROOT"
run(){ nohup "$PY" -u run_vasco_grid.py "$1" > "$ROOT/outputs/gh_$2.log" 2>&1 & echo "  $2 pid $!"; }
# vary: how the attractant is sourced, its range, chemotaxis strength, cohesion
run '{"tag":"ecsrc_short","L":150,"n_ec":120,"mcs":2500,"lambda_length":1.5,"vegf_secretion_ec":0.02,"vegf_secretion_fib":0.0,"vegf_decay":0.006}' h1
run '{"tag":"fibsrc","L":150,"n_ec":120,"mcs":2500,"lambda_length":1.5,"vegf_secretion_ec":0.002,"vegf_secretion_fib":0.02,"vegf_decay":0.006}' h2
run '{"tag":"fibsrc_uptake","L":150,"n_ec":120,"mcs":2500,"lambda_length":1.5,"vegf_secretion_ec":0.002,"vegf_secretion_fib":0.02,"vegf_decay":0.006,"vegf_ec_uptake":0.05}' h3
run '{"tag":"long_range","L":150,"n_ec":120,"mcs":2500,"lambda_length":1.5,"vegf_secretion_ec":0.02,"vegf_secretion_fib":0.0,"vegf_decay":0.0008}' h4
run '{"tag":"coh0.5","L":150,"n_ec":120,"mcs":2500,"lambda_length":1.5,"vegf_secretion_ec":0.02,"vegf_secretion_fib":0.0,"vegf_decay":0.006,"J_ec_ec":0.5,"J_ec_medium":6.0}' h5
run '{"tag":"noCI_short","L":150,"n_ec":120,"mcs":2500,"lambda_length":1.5,"vegf_secretion_ec":0.02,"vegf_secretion_fib":0.0,"vegf_decay":0.006,"ci":false}' h6
