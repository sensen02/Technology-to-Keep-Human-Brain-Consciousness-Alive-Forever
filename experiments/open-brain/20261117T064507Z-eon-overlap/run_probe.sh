#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir PROBE
printf '%s\n' off cruise fast a01_left a01_right a02_left a02_right | xargs -P 7 -I '{}' bash -c '
  ./drive.sh overlap.py probe "BRAIN/181_$1" "PROBE/$1" > "PROBE/$1.log" 2>&1
  code=$?
  echo "$code" > "PROBE/$1.exit"
  exit "$code"
' _ '{}'
./drive.sh overlap.py calibrate PROBE CALIBRATION.json > calibration.log 2>&1
