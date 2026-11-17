#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
mkdir "$1"
for target in stride_left stride_right; do
    for variant in held0 held1; do
        for mode in open cut; do
            printf '%s %s %s\n' "$target" "$variant" "$mode"
        done
    done
done | xargs -P 8 -n 3 bash -c '
    out="$1/${2}_${3}_$4"
    ../drive.sh REFINE/refine.py run "$2" "$3" "$4" "$out" > "$out.log" 2>&1
    code=$?
    echo "$code" > "$out.exit"
    exit "$code"
' _ "$1"
