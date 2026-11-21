#!/usr/bin/env bash
set -euo pipefail

base=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
archive=/run/media/sensen/Data2/cell_wound_prototype
neural=$archive/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python
runner=$base/../20261119T012530Z-survival-orchard/run_env.sh
budget=$base/BUDGET_1800
previous=$base/../20261119T152434Z-reward-repair
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1

"$neural" "$base/check_checkpoint.py" "$budget" > "$base/reload.log" 2>&1
"$neural" "$base/proxy_transfer.py" "$previous/POLICY.npz" "$budget" > "$base/proxy-transfer.log" 2>&1
python3 "$base/compare_results.py" --training > "$base/plot-training.log" 2>&1
"$neural" "$base/physical_suite.py" "$base/LIVE_SELECTED" --policy "$budget/POLICY.npz" > "$base/selected.log" 2>&1

out=$base/VISIBLE_SELECTED
mkdir -p "$out"
for case_name in SLOPE_L COMPOUND_L; do
  if [[ ! -f "$out/$case_name/AUDIT.json" ]]; then
    "$runner" "$base/visible_food_probe.py" "$out/$case_name" --policy "$budget/POLICY.npz" --case "$case_name" > "$out/$case_name.log" 2>&1
  fi
done
"$runner" "$base/risk_transfer.py" BASELINE SELECTED > "$base/risk-comparison.log" 2>&1
python3 "$base/compare_results.py" --physical --visible-cue --hazard > "$base/comparison.log" 2>&1
"$runner" "$base/render_live.py" "$out/SLOPE_L" "$base/LIVE_VISIBLE_SLOPE.png" --visible-cue > "$base/render-selected.log" 2>&1
