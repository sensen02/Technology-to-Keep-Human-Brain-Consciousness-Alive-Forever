#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runner="$root/../20261119T012530Z-survival-orchard/run_env.sh"
for seed in 15103 15107; do
    name="DOCK_${seed}"
    "$runner" "$root/run_course_docking.py" "$root/VISION/VISION.npz" "$root/$name" \
      --policy "$root/STRONG/POLICY.npz" --profile strong --challenge flat \
      --seed "$seed" --duration 5.2 --early > "$root/$name.log" 2>&1
    printf '%s exit 0\n' "$name"
done
