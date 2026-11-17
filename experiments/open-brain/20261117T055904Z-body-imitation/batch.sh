#!/usr/bin/env bash
set -euo pipefail
run=$(cd -- "$(dirname -- "$0")" && pwd)
mode=$1
folder=$2
checkpoint=$3
mix=$4
shift 4
mkdir "$folder" "$folder/log"
case_run() {
  local k=$1
  local rc=0
  if [[ -n "$checkpoint" ]]; then
    "$run/drive.sh" body_run.py "$mode" "$k" "$folder/$k" --checkpoint "$checkpoint" --mix "$mix" > "$folder/log/$k.out" 2>&1 || rc=$?
  else
    "$run/drive.sh" body_run.py "$mode" "$k" "$folder/$k" > "$folder/log/$k.out" 2>&1 || rc=$?
  fi
  printf '%s\n' "$rc" > "$folder/log/$k.exit"
  cat "$folder/log/$k.out"
  return "$rc"
}
pids=()
for k in "$@"; do
  case_run "$k" & pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
