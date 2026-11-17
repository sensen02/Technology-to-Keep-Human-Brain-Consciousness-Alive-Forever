#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname -- "$0")"
FIRST="../20261117T020300Z-sac-single-box"

python3 - <<'PY'
import time
from pathlib import Path
needed=[Path(x+".exit") for x in ("seed_left","seed_right","offset_left","offset_right")]
limit=time.monotonic()+1500.
while not all(p.exists() for p in needed):
    assert time.monotonic()<limit, "Physical seed collection exceeded bounded wall budget"
    time.sleep(5)
for p in needed:
    assert p.read_text().strip()=="0",p
PY
./drive.sh - "$FIRST" <<'PY'
import sys,numpy as np
from pathlib import Path
for name in ("flat","contact"):
    src=Path(sys.argv[1])/"SEED"/name/"EPISODE.npz"
    with np.load(src,allow_pickle=False) as z:
        moving=np.any(z["requested"]!=0.,axis=-1)
        expected=np.where(moving,4.,-4.)[:,None]
        np.testing.assert_array_equal(z["x"][1:,:,0],np.broadcast_to(expected,z["x"][1:,:,0].shape))
    print(name,"v1 archived request-drive identical to v2",flush=True)
PY
SEEDS=("$FIRST/SEED/flat" "$FIRST/SEED/contact" SEED/cal_left SEED/cal_right SEED/offset_left SEED/offset_right)
DEMOS=("$FIRST/SEED/flat" SEED/cal_left SEED/cal_right)
for name in offset_left offset_right; do
  if python3 - "$name" <<'PY'
import json,sys
from pathlib import Path
p=Path("SEED")/sys.argv[1]/"RESULT.json"
r=json.loads(p.read_text())
sys.exit(0 if r["reason"]=="success" else 1)
PY
  then
    DEMOS+=("SEED/$name")
  fi
done
sha256sum course.py model.py sac_replay.py train_sac.py seed_imitation.py evaluate_policy.py     SCENES.json FLAT_DEMOS.json ../20261117T014000Z-distance-reward/{reward.py,stall.py}     ../20261117T010100Z-unified-neck-final/{BRAIN.pt,UNIFIED_BINDING.json} > RUN_INPUTS.sha256
./drive.sh seed_imitation.py "$FIRST/WARM.pt" "${DEMOS[@]}" > warm_box.log
./drive.sh train_sac.py WARM_BOX.pt PHASE_A --updates 384 "${SEEDS[@]}" > phase_a.log
sha256sum -c RUN_INPUTS.sha256

mkdir LIVE
job() {
  local tag="$1" model="$2" task="$3"
  shift 3
  set +e
  ./drive.sh evaluate_policy.py "$model" "$task" "LIVE/$tag" "$@" > "LIVE/$tag.log" 2>&1
  local status="$?"
  printf '%s\n' "$status" > "LIVE/$tag.exit"
  return "$status"
}
job online_left PHASE_A/0384.pt online_left --sample 4009 & a=$!
job online_right PHASE_A/0384.pt online_right --sample 4109 & b=$!
job phase_a_held_flat PHASE_A/0384.pt held_flat & c=$!
job warm_held_box WARM_BOX.pt held_box & d=$!
job phase_a_held_box PHASE_A/0384.pt held_box & e=$!
control=0
for pid in "$a" "$b" "$c" "$d" "$e"; do
  if ! wait "$pid"; then control=1; fi
done
if [ "$control" -ne 0 ]; then exit "$control"; fi

./drive.sh train_sac.py PHASE_A/0384.pt PHASE_B --updates 192     "${SEEDS[@]}" LIVE/online_left LIVE/online_right > phase_b.log
sha256sum -c RUN_INPUTS.sha256
job phase_b_held_flat PHASE_B/0576.pt held_flat & f=$!
job phase_b_held_box PHASE_B/0576.pt held_box & g=$!
for pid in "$f" "$g"; do
  if ! wait "$pid"; then control=1; fi
done
exit "$control"
