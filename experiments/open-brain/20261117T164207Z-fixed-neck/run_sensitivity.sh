#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir SENSITIVITY
for mode in slow fast kinetic08 kinetic12 leak08 leak12; do
 for task in 'body dev_cruise' 'body dev_back' 'body dev_brake' 'ground stride_left__held0__open' 'ground bluebell__dev__open'; do
  printf '%s %s\n' "$mode" "$task"
 done
done | xargs -P 6 -n 3 bash -c '
 out="SENSITIVITY/${1}_${2}_$3"
 ./drive.sh sensitivity.py "$1" "$2" "$3" "$out" > "$out.log" 2>&1
 code=$?
 echo "$code" > "$out.exit"
 exit "$code"
' _
