#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
runner="$PWD/../20261119T012530Z-survival-orchard/run_env.sh"
out="${1:?Provide a new absolute evidence folder}"
[[ "$out" = /* ]]
mkdir -- "$out"
for name in HUNGRY FULL FADE; do
    "$runner" "$PWD/orchard_trial.py" "$name" "$out/$name" > "$out/$name.log" 2>&1
done
"$runner" "$PWD/verify.py" "$out"
