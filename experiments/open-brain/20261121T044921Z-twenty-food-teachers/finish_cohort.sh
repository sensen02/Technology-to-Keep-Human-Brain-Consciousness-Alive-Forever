#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
root=../20261121T040639Z-feeding-body-practice
until [[ -s cohort.json ]]; do sleep 8; done
if [[ ! -s twenty/cohort-identity.json ]]; then
    nice -n 13 bash env.sh train_cohort.py > logs/train.log 2>&1
fi
for seed in 88207 88224 88225; do
    if [[ ! -e live-$seed-on-previous ]]; then
        ln -s "$root/live-$seed-on-long-rollout" "live-$seed-on-previous"
    fi
    if [[ ! -s live-$seed-on-twenty/result.json ]]; then
        bash run_live_capped.sh "$seed" on 6.4 twenty > "logs/live-$seed.log" 2>&1
    fi
done
bash env.sh assess.py twenty 88207 88211 88224 88225 > logs/replay.log 2>&1
bash env.sh summarize.py twenty 88207 88224 88225
bash env.sh report.py
