#!/bin/bash
# Stop only the simulation worker processes.
# Matches processes whose argv[0] is exactly the project python, which never
# matches the calling shell (the shell's argv[0] is "bash").
set -u
PY=/run/media/sensen/Data2/cell_wound_prototype/venv/bin/python
n=0
for pid in $(pgrep -f "^$PY" || true); do
  kill "$pid" 2>/dev/null && n=$((n+1))
done
sleep 2
echo "killed $n worker(s); still alive:"
pgrep -af "^$PY" | wc -l
