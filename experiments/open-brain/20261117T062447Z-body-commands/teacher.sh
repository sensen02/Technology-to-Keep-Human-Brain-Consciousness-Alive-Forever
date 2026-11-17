#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
for family in cruise fast arc_left arc_right pivot_left pivot_right back brake; do
    for variant in 0 1 2; do
        printf '%s\n' "train_${family}_${variant}"
    done
    printf '%s\n' "dev_${family}"
done | xargs -P 8 -I '{}' bash -c './drive.sh expansion.py teacher "$1" "TEACHER/$1" > "TEACHER/$1.log" 2>&1; result=$?; echo "$result" > "TEACHER/$1.exit"; exit "$result"' _ '{}'
