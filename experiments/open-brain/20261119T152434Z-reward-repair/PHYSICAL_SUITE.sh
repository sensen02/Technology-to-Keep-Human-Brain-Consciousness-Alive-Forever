#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
parent="$root/../20261119T084744Z-closed-loop-forage"
runner="$root/../20261119T012530Z-survival-orchard/run_env.sh"
trial() {
    local name="$1" stage="$2" seed="$3" side="$4" secs="$5"
    "$runner" "$parent/run_course_docking.py" "$parent/VISION/VISION.npz" "$root/$name" \
      --policy "$root/POLICY.npz" --profile strong --challenge "$stage" \
      --seed "$seed" --side "$side" --duration "$secs" --early > "$root/$name.log" 2>&1
    printf '%s exit 0\n' "$name"
}
trial REPEAT_L wide_bridge 15121 1 5.2
trial REPEAT_R wide_bridge 15121 -1 5.2
trial HOLD_L wide_bridge 15671 1 5.2
trial HOLD_R wide_bridge 15671 -1 5.2
trial FLAT_CONTROL flat 15107 1 5.2
