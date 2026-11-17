#!/usr/bin/env bash
set -eu
cd "$(dirname -- "$0")"
root="$PWD"
run="$root/../20261117T022700Z-sac-startup-replay/drive.sh"
mkdir "$root/HELD"
jobs=()
for arm in warm original route; do
    mkdir "$root/HELD/$arm"
    for case in held_flat held_box; do
        out="$root/HELD/$arm/$case"
        (set +e
         timeout 2700 "$run" "$root/evaluate_pair.py" "$arm" "$case" "$out" >"$out.log" 2>&1
         rc=$?
         printf '%d\n' "$rc" >"$out.exit"
         exit "$rc") &
        jobs+=("$!")
    done
done
status=0
for job in "${jobs[@]}"; do
    wait "$job" || status=1
done
exit "$status"
