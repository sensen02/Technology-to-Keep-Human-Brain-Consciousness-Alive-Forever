#!/usr/bin/env bash
set -euo pipefail
entry=/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain
policy="$entry/20261119T190005Z-odor-sac"
out=${1:?Specify a fresh output directory}
"$entry/20261119T012530Z-survival-orchard/run_env.sh" "$policy/live.py" "$out" \
    --policy "$policy/TRAIN_DETOUR/POLICY.npz" --stage open --sense odor \
    --duration 6.4 --seed 88207 --assisted
