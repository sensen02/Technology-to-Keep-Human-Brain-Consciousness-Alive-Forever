#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
out=$1
weight=$2
mkdir "$out"
xargs -P "${BODY_ADAPT_JOBS:-8}" -n 2 bash -c '
 out="$1/${3}_$4"
 ./drive.sh evaluate.py "$3" "$4" "$2" "$out" > "$out.log" 2>&1
 code=$?
 echo "$code" > "$out.exit"
 exit "$code"
' _ "$out" "$weight" < ../20261117T164207Z-fixed-neck/FORMAL_JOBS.txt
