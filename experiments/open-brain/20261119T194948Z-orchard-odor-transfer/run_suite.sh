#!/usr/bin/env bash
set -euo pipefail
run_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner="$run_dir/../20261119T012530Z-survival-orchard/run_env.sh"
python_cpu=/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python
results="${1:-$run_dir}"
mkdir -p "$results"
results="$(cd -- "$results" && pwd)"

"$python_cpu" -B - "$run_dir/PREFLIGHT.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as fh:
    report=json.load(fh)
assert {r['site'] for r in report} == {'CLEAR','DEBRIS'}
assert all(r['support_pass'] for r in report)
PY

for site in CLEAR DEBRIS; do
  "$runner" "$run_dir/transfer.py" "$site" "$results/LIVE_${site}_ON" --duration 6.4 2>&1 | tee "$results/live-${site}-on.log"
  "$runner" "$run_dir/transfer.py" "$site" "$results/LIVE_${site}_OFF" --no-odor --duration 3.2 2>&1 | tee "$results/live-${site}-off.log"
done
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 "$python_cpu" -uB "$run_dir/audit.py" "$results"
