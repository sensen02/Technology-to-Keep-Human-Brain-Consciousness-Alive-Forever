#!/bin/sh
# Run only with the frozen graph, trained endpoints and completed development checks.
set -eu
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
trial_python=../../banc-walking-recovery/20261110T125712Z/env/bin/python
export trial_python
"$trial_python" -B - <<'PY'
from pathlib import Path
import hashlib, json

def signature(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()

expected = Path('dev_names.txt').read_text().split()
for name in expected:
    result = json.loads((Path('DEV_CAL')/name/'RESULT.json').read_text())
    assert result['passed'], (name, result)
    assert result['checkpoint_sha256'] == signature('VISION_CAL.pt')
    assert result['codec_sha256'] == signature('CODEC.pt')
files = ('OFFICIAL_GRAPH.json', 'DNa02_BINDING.json', 'CODEC.pt', 'VISION_CAL.pt',
         'TRAIN_CASES.json', 'EVAL_CASES.json', 'trained_link.py', 'codec.py', 'brain.py',
         'bench_runtime.py', 'sequence.py', 'optics.py', 'evaluate.sh')
with Path('EVAL_INPUTS.json').open('x') as file:
    json.dump({'dev_checks_passed':expected,'source_sha256':{name:signature(name) for name in files}},file,indent=2)
PY
mkdir EVAL EVAL_LOGS NOISY NOISY_LOGS
"$trial_python" -B -c 'import json; print(*json.load(open("EVAL_CASES.json")),sep="\n")' > eval_names.txt
printf 'L04 17012\nR04 38517\nL_target 50911\nR_target 82390\nL_restore 16982\nR_restore 66841\noff 49013\n' > noisy_names.txt
# Predeclared matching directional cases have raw native contact instrumentation.
xargs -n1 -P8 sh -c '
  raw=
  case "$0" in L_target|R_target) raw=--native;; esac
  "$trial_python" -B -u sequence.py EVAL_CASES.json "$0" "EVAL/$0" --checkpoint VISION_CAL.pt --codec CODEC.pt $raw > "EVAL_LOGS/$0.log" 2>&1
' < eval_names.txt
xargs -n2 -P7 sh -c '
  raw=
  case "$0" in L_target|R_target) raw=--native;; esac
  "$trial_python" -B -u sequence.py EVAL_CASES.json "$0" "NOISY/$0" --checkpoint VISION_CAL.pt --codec CODEC.pt --profile combined --seed "$1" $raw > "NOISY_LOGS/$0.log" 2>&1
' < noisy_names.txt
