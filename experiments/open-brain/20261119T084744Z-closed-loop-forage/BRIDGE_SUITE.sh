#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runner="$root/../20261119T012530Z-survival-orchard/run_env.sh"
run() {
    local label="$1" stage="$2" seed="$3" side="$4"
    "$runner" "$root/run_course.py" "$root/VISION/VISION.npz" "$root/$label" \
      --policy "$root/STRONG/POLICY.npz" --profile strong --challenge "$stage" \
      --seed "$seed" --side "$side" --duration 5.2 --early > "$root/$label.log" 2>&1
    printf '%s exit 0\n' "$label"
}
run WIDE_L wide_bridge 15121 1
run WIDE_R wide_bridge 15121 -1
run BRIDGE_L bridge 15131 1
run BRIDGE_R bridge 15131 -1
run LONG_L long_bridge 15131 1
run LONG_R long_bridge 15131 -1
"$runner" "$root/run_course.py" "$root/VISION/VISION.npz" "$root/NO_FOOD" \
  --policy "$root/STRONG/POLICY.npz" --profile strong --challenge wide_bridge \
  --seed 15121 --side 1 --duration 1.6 --no-food --early > "$root/NO_FOOD.log" 2>&1
