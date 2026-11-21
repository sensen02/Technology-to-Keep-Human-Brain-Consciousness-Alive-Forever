#!/usr/bin/env bash
set -euo pipefail
task_dir=$(cd -- "$(dirname -- "$0")" && pwd)
archive_root=${P1_ARCHIVE_ROOT:-/run/media/sensen/Data2/cell_wound_prototype}
export P1_ARCHIVE_ROOT="$archive_root" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
exec "$archive_root/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python" -uB "$task_dir/$1" "${@:2}"
