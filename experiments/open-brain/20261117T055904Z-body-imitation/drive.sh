#!/usr/bin/env bash
set -euo pipefail
work_dir=$(cd -- "$(dirname -- "$0")" && pwd)
project=$(cd -- "$work_dir/../../.." && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
exec "$project/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python" -uB "$work_dir/$1" "${@:2}"
