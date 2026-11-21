#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
scene="$PWD/../20261119T012530Z-survival-orchard/run_env.sh"
out="${1:?Pass a new absolute result directory}"
[[ "$out" = /* ]]
mkdir -p "$out"
for case in GUIDED FRESH EMPTY_GROUND; do
    "$scene" "$PWD/run_case.py" "$case" "$out/$case" > "$out/$case.log" 2>&1
done
"$scene" "$PWD/support_probe.py" "$out/SUPPORT" > "$out/SUPPORT.log" 2>&1
