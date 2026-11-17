#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir BRAIN
{
  for seed in 181 271 362; do
    for name in off cruise fast a01_left a01_right a02_left a02_right; do
      if [ "$name" != off ] || [ "$seed" -eq 181 ]; then
        printf '%s %s\n' "$seed" "$name"
      fi
    done
  done
  printf '271 pfl_left\n271 pfl_right\n'
} | xargs -P 6 -n 2 bash -c '
  run="BRAIN/${1}_${2}"
  ./drive.sh brain_events.py "$2" "$1" "$run" > "$run.log" 2>&1
  code=$?
  echo "$code" > "$run.exit"
  exit "$code"
' _
