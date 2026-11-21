#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
runner="$PWD/../20261119T012530Z-survival-orchard/run_env.sh"
target="${1:?Absolute output folder}"
[[ "$target" = /* ]]
mkdir -- "$target"
status_all=0
for variant in FADE HUNGRY FADE_YAW FADE_BRAKE FADE_LATE_YAW FADE_LATE_BRAKE; do
    status=0
    "$runner" "$PWD/replay.py" "$variant" "$target/$variant" > "$target/$variant.log" 2>&1 || status=$?
    printf '%s %s\n' "$variant" "$status"
    if (( status )); then
        status_all=$status
        [[ -s "$target/$variant/DIAGNOSTIC.json" ]] || exit "$status"
    fi
done
exit "$status_all"
