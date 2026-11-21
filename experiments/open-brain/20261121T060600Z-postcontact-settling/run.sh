#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
here=$PWD
src="$here/../20261121T044921Z-twenty-food-teachers"
exec systemd-run --user --scope -q -p MemoryHigh=2380M -p MemoryMax=2950M -p MemorySwapMax=256M nice -n 13 bash "$src/env.sh" "$here/live.py" "$@"
