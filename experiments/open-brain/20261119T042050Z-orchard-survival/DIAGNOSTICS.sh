#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
runner="$PWD/../20261119T012530Z-survival-orchard/run_env.sh"
out="${1:?Provide a new absolute output directory}"
[[ "$out" = /* ]]
mkdir -- "$out"
"$runner" "$PWD/survival.py" GUIDED "$out/GUIDED" --duration 4 --curved-guide > "$out/GUIDED.log" 2>&1
"$runner" "$PWD/survival.py" FRESH "$out/FRESH" --duration 5 --approach-brake > "$out/FRESH.log" 2>&1
"$runner" "$PWD/assess.py" "$out" --cases FRESH GUIDED
