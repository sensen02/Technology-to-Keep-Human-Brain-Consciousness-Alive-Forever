#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runner="$root/../20261119T012530Z-survival-orchard/run_env.sh"
run() {
    local label="$1" stage="$2" seed="$3" side="$4" duration="$5"
    shift 5
    "$runner" "$root/run_course.py" "$root/VISION/VISION.npz" "$root/$label" \
      --policy "$root/STRONG/POLICY.npz" --profile strong --challenge "$stage" \
      --seed "$seed" --side "$side" --duration "$duration" --early "$@" > "$root/$label.log" 2>&1
    printf '%s exit 0\n' "$label"
}
run FINAL_FLAT_A flat 15103 1 4.8
run FINAL_FLAT_B flat 15107 -1 4.8
run WIDE_L wide_bridge 15121 1 5.2
run WIDE_R wide_bridge 15121 -1 5.2
run BRIDGE_L bridge 15131 1 5.2
run BRIDGE_R bridge 15131 -1 5.2
run LONG_L long_bridge 15131 1 5.2
run LONG_R long_bridge 15131 -1 5.2
run NO_FOOD wide_bridge 15121 1 1.6 --no-food
