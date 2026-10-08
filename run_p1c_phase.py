"""run_p1c_phase.py -- search for the published direction of the control.

WHY a 2D search
---------------
`run_p1b_merks.py` reproduced the de novo network but NOT the direction of the
paper's cluster control: at lambda_chem = 400 and lambda_volume = 4 the cluster
became COMPACT with contact inhibition (chi=0) and DENDRITIC without it
(chi=1), whereas Merks et al. 2008 report the opposite.

The paper gives a reason to look at the volume constraint.  Its buckling
mechanism rests on the cells being nearly incompressible:

    "because each cell's volume is nearly conserved (apart from small
     fluctuations around its target volume), the core cells can only release
     the pressure the ingressing cells exert on them by moving outwards as
     sprouts"

With a SOFT volume constraint a cluster can simply compress into a ball instead
of sprouting, which would mask the instability.  So the relevant plane is
(lambda_chem, lambda_volume), not lambda_chem alone.

This script scans that plane at chi=0 and chi=1, reports where the direction
agrees with the paper (compactness LOWER with contact inhibition), then runs a
fine scan over the chi ratio at the best point found, looking for the
transition near 0.5 that the paper reports.

Usage:  venv/bin/python run_p1c_phase.py <shard_index> <n_shards>
Writes: outputs/p1c_shard_<i>.json      (per-shard raw rows)
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from merks import MerksVasculogenesis

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
L = 200
N_CELLS = 128
MCS = 3000

LAM_CHEM = (100.0, 400.0, 1200.0)
LAM_VOL = (4.0, 20.0, 60.0)
CHIS = (0.0, 1.0)
MCS_FINE = 6000


def grid_configs():
    out = []
    for lv in LAM_VOL:
        for lc in LAM_CHEM:
            for chi in CHIS:
                out.append((lv, lc, chi))
    return out


def run_one(lv, lc, chi, mcs, seed=1):
    v = MerksVasculogenesis(
        L=L, seed=seed, mode="cluster", n_cells=N_CELLS, chem_cc_ratio=chi,
        params={"lambda_chem": lc, "lambda_volume": lv})
    v.run(mcs, record_every=max(200, mcs // 6), verbose=False)
    h = v.history[-1]
    h.update({"lambda_chem": lc, "lambda_volume": lv, "chem_cc_ratio": chi,
              "mcs": mcs, "seed": seed})
    return h


def main():
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    cfgs = grid_configs()
    mine = [c for i, c in enumerate(cfgs) if i % nsh == shard]
    rows = []
    for lv, lc, chi in mine:
        t0 = time.time()
        h = run_one(lv, lc, chi, MCS)
        rows.append(h)
        print(f"  [shard {shard}] lamV={lv:5.1f} lamChem={lc:7.1f} chi={chi:.2f} "
              f"-> C={h['compactness']} comp={h['n_components']} "
              f"lacunae={h['n_lacunae']} ({time.time()-t0:.0f}s)", flush=True)
    path = os.path.join(OUT, f"p1c_shard_{shard}.json")
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=2)
    print("wrote", path, len(rows), "rows")


if __name__ == "__main__":
    main()
