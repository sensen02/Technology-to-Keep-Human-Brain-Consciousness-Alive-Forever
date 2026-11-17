#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
out=$1
split=$2
calibration=${3:-}
mkdir "$out"
for target in stride_left stride_right bluebell web_b; do
    if [ "$split" = dev ]; then variants=(dev); else variants=(held0 held1); fi
    for variant in "${variants[@]}"; do
        for mode in open cut; do
            printf '%s %s %s\n' "$target" "$variant" "$mode"
        done
    done
done | xargs -P "${GROUND_JOBS:-8}" -n 3 bash -c '
    set -u
    name="$1/${3}_${4}_$5"
    args=()
    if [ -n "$2" ]; then args+=(--calibration "$2"); fi
    ./drive.sh ground_response.py "$3" "$4" "$5" "$name" "${args[@]}" > "$name.log" 2>&1
    code=$?
    echo "$code" > "$name.exit"
    exit "$code"
' _ "$out" "$calibration"
