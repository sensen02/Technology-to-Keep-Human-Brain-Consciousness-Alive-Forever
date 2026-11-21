#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
archive="${P1_ARCHIVE_ROOT:-/run/media/sensen/Data2/cell_wound_prototype}"
py="$archive/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python"
out="${1:?supply a new output directory}"
mkdir -- "$out"
export P1_ARCHIVE_ROOT="$archive"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$py" -uB "$here/preflight.py" "$out/FORMAL"
"$py" -uB "$here/teacher.py" open "$out/FORMAL" "$out/OPEN_R" 137
"$py" -uB "$here/teacher.py" wall "$out/FORMAL" "$out/WALL_L" 463
"$py" -uB "$here/teacher.py" wall "$out/FORMAL" "$out/WALL_R" 463 --mirror
"$py" -uB "$here/probe_eon_odor.py" "$out/FORMAL/RESULT.json" "$out/EON_ODOR"
"$py" -uB "$here/transport_probe.py" "$out/EON_ODOR" "$out/ODOR_BODY"
"$py" -uB "$here/make_figure.py" "$out"
