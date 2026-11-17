#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
out=$1
mkdir "$out"
xargs -P "${FIXED_NECK_JOBS:-8}" -n 2 bash -c '
 out="$1/${2}_$3"
 ./drive.sh regress.py "$2" "$3" "$out" --mapped > "$out.log" 2>&1
 code=$?
 echo "$code" > "$out.exit"
 exit "$code"
' _ "$out" < FORMAL_JOBS.txt
