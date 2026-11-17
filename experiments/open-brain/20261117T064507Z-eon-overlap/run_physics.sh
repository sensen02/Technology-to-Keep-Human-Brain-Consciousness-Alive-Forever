#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
out=$1
calibration=${2:-CALIBRATION.json}
mkdir "$out"
{
  printf '181_off open\n'
  for seed in 271 362; do
    for name in cruise fast a01_left a01_right a02_left a02_right; do
      printf '%s_%s open\n' "$seed" "$name"
    done
    for name in a01_left a01_right a02_left a02_right; do
      printf '%s_%s steer\n' "$seed" "$name"
    done
  done
  printf '271_cruise speed\n271_a02_left all\n'
} | xargs -P "${OVERLAP_JOBS:-10}" -n 2 bash -c '
  run="$1/${3}_$4"
  ./drive.sh overlap.py "$4" "BRAIN/$3" "$run" --calibration "$2" > "$run.log" 2>&1
  code=$?
  echo "$code" > "$run.exit"
  exit "$code"
' _ "$out" "$calibration"
