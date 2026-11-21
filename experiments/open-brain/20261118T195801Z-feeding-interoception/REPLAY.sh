#!/usr/bin/env bash
set -euo pipefail
trial="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root="${P1_ARCHIVE_ROOT:-/run/media/sensen/Data2/cell_wound_prototype}"
python_bin="$root/experiments/banc-walking-recovery/20261110T125712Z/env/bin/python"
body="$root/experiments/open-brain/20261117T184709Z-body-terrain-repair"
output="${1:?supply a new output directory}"
mkdir -- "$output"
export P1_ARCHIVE_ROOT="$root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

(
  cd -- "$trial/sources"
  author='https://raw.githubusercontent.com/ZepengYao-Lab/Giakoumas-et-al/9a82bcc1877e1285ac3a6db3c2417b8bb507d161/input/PhN'
  for number in 1 2 3 4 5 6; do
    csv="set_$number.csv"
    if [[ ! -f "$csv" ]]; then
      curl --fail --silent --show-error --location "$author/$csv" --output "$csv.tmp"
      mv -- "$csv.tmp" "$csv"
    fi
  done
  sha256sum -c SHA256SUMS
)

"$python_bin" -uB "$trial/bench.py" "$output/BODY_FINAL"
"$python_bin" -uB "$trial/mouth_bench.py" "$body" "$output/MOUTH"
"$python_bin" -uB "$trial/probe.py" "$output/EON_FINAL127" 127
"$python_bin" -uB "$trial/probe.py" "$output/EON_FINAL509" 509
"$python_bin" -uB "$trial/audit_banc.py" "$root/data/banc888/banc_888_meta.feather" \
  "$output/EON_FINAL127/BINDING.json" "$output/BANC_STN_AUDIT.json"
"$python_bin" -uB "$trial/figure.py" "$output"
