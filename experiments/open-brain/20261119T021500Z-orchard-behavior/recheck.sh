#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
scene="$PWD/../20261119T012530Z-survival-orchard/run_env.sh"
out="${1:?Pass a new absolute result directory}"
[[ "$out" = /* ]]
mkdir -- "$out"
for case in FRESH EMPTY_GROUND; do
    "$scene" "$PWD/steering_recheck.py" "$case" "$out/$case" > "$out/$case.log" 2>&1
done
