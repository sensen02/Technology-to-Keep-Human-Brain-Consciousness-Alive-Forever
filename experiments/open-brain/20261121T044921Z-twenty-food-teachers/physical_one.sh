#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec systemd-run --user --scope -q -p MemoryHigh=3550M -p MemoryMax=4200M -p MemorySwapMax=384M   nice -n 13 /home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261119T012530Z-survival-orchard/run_env.sh "$PWD/collect.py" "$1"
