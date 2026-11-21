#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
archive="${P1_ARCHIVE_ROOT:-/run/media/sensen/Data2/cell_wound_prototype}"
py="$archive/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python"
out="${1:?supply a new output directory}"
mkdir -- "$out"
export P1_ARCHIVE_ROOT="$archive"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ln -s -- "$here/SENSOR_FIT" "$out/SENSOR_FIT"
for entry in 'left intact FORMAL_L' 'right intact FORMAL_R' 'left cut_dn CUT_L' \
             'left vision_only VISION_L' 'left odor_only ODOR_L' 'left blank BLANK_L'; do
  read -r scene mode label <<< "$entry"
  "$py" -uB "$here/live.py" "$scene" "$mode" "$out/$label" --duration 3.5 > "$out/$label.log" 2>&1
done
"$py" -uB "$here/finish.py" "$out"
"$py" -uB "$here/wall_baseline.py" "$out/WALL_PRIMARY" > "$out/WALL_PRIMARY.log" 2>&1
"$py" -uB "$here/wall_baseline.py" "$out/WALL_ALT" --variant alternate > "$out/WALL_ALT.log" 2>&1
"$py" -uB "$here/wall_baseline.py" "$out/WALL_VIS" --mode vision_only > "$out/WALL_VIS.log" 2>&1
"$py" -uB "$here/wall_baseline.py" "$out/WALL_ODOR" --mode odor_only > "$out/WALL_ODOR.log" 2>&1
"$py" -uB "$here/wall_figure.py" "$out"
