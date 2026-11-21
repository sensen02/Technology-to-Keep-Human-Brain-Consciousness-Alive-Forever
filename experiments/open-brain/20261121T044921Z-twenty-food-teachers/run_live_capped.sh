#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec systemd-run --user --scope -q -p MemoryHigh=2380M -p MemoryMax=2950M -p MemorySwapMax=256M     nice -n 13 bash env.sh live_student.py "$@"
