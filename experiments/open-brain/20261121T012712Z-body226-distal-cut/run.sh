#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
p=/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python
"$p" cut.py
"$p" record.py
"$p" fit_body.py
"$p" verify.py
"$p" plot.py
