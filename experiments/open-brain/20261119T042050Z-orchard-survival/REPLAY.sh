#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
runner="$PWD/../20261119T012530Z-survival-orchard/run_env.sh"
out="${1:?Provide a new absolute output directory}"
[[ "$out" = /* ]]
mkdir -- "$out"
for case in FRESH EMPTY_GROUND GUIDED; do
    "$runner" "$PWD/survival.py" "$case" "$out/$case" > "$out/$case.log" 2>&1
done
"$runner" "$PWD/assess.py" "$out"
