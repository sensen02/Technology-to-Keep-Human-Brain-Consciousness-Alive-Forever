#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
archive=/run/media/sensen/Data2/cell_wound_prototype
flight="$archive/experiments/open-brain/20261117T051249Z-bidirectional/flight_env/bin/python"
neural="$archive/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=egl TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=-1 LP_NUM_THREADS=2
target="${1:?Use a new absolute output directory}"
[[ "$target" = /* ]]
mkdir -- "$target"
"$flight" -uB live_body.py "$target/HIGH" --raised
"$flight" -uB live_body.py "$target/CUT" --raised --cut --duration .80
"$flight" -uB live_body.py "$target/LOW" --duration 4.44
"$flight" -uB live_body.py "$target/NO_ADAPTER" --raised --no-adapter --duration 4.44
"$neural" -uB verify_live.py --root "$target"
