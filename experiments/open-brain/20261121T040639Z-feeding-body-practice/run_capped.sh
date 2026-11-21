#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec systemd-run --user --scope -q -p MemoryHigh=3550M -p MemoryMax=4200M -p MemorySwapMax=384M     nice -n 12 bash teachers.sh "$@"
