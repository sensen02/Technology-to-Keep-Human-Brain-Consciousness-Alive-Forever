#!/usr/bin/env bash
set -euo pipefail

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runner="$root/../20261119T012530Z-survival-orchard/run_env.sh"
output=$(realpath -m -- "${1:?provide an unused output directory}")
mkdir -- "$output"
"$runner" "$root/compile.py"
"$runner" "$root/body_righting_stow.py" "$output/stances" \
  left_side right_side back orchard_fall test_shallow_left test_shallow_right test_dorsal --handoff
"$runner" "$root/body_righting_stow.py" "$output/held_out" \
  test_left test_right test_back test_diagonal
"$runner" "$root/check_and_render.py" "$output/stances" "$output/held_out" --output "$output"
"$runner" "$root/orchard_recovery.py" FADE "$output/continuous" --allow-first-fall
"$runner" "$root/orchard_recovery.py" FADE "$output/guarded"
"$runner" "$root/assess_live.py" --continuous "$output/continuous" --guarded "$output/guarded" \
  --output "$output/LIVE_AUDIT.json"
