#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")"
for seed in "$@"; do
 /home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T012530Z-survival-orchard/run_env.sh "$PWD/collect.py" "$seed" > "$PWD/teacher-$seed.log" 2>&1
done
