#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
archive="${P1_ARCHIVE_ROOT:-/run/media/sensen/Data2/cell_wound_prototype}"
py="$archive/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python"
out="${1:?supply a new output directory}"
mkdir -- "$out"
export P1_ARCHIVE_ROOT="$archive"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

for seed in 4339 7213; do
  "$py" -uB "$here/meal.py" "$out/TAGGED$seed" "$seed" --fit "$here/FIT_TAGGED.json"
done
"$py" -uB "$here/assess.py" "$out/ACCEPTANCE.json" "$out/TAGGED4339" "$out/TAGGED7213"
"$py" -uB "$here/figure.py" "$out" TAGGED4339 TAGGED7213
