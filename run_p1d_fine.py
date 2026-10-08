"""run_p1d_fine.py -- fine scan over the contact-inhibition ratio at the one
point of the (lambda_chem, lambda_volume) plane that showed the paper's
direction, with several seeds to test whether it is robust or noise.

Point found by run_p1c_phase.py: lambda_volume = 20, lambda_chem = 100, where
compactness was LOWER with contact inhibition (0.868) than without (0.949),
which is the direction Merks et al. 2008 report.

Usage: venv/bin/python run_p1d_fine.py <shard> <n_shards>
Writes: outputs/p1d_shard_<i>.json
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from merks import MerksVasculogenesis

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
L = 200
N_CELLS = 128
MCS = 6000
LAM_VOL = 20.0
LAM_CHEM = 100.0
CHIS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0)
SEEDS = (1, 2, 3)


def configs():
    return [(chi, s) for chi in CHIS for s in SEEDS]


def main():
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nsh = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    mine = [c for i, c in enumerate(configs()) if i % nsh == shard]
    rows = []
    for chi, seed in mine:
        t0 = time.time()
        v = MerksVasculogenesis(L=L, seed=seed, mode="cluster", n_cells=N_CELLS,
                                chem_cc_ratio=chi,
                                params={"lambda_chem": LAM_CHEM,
                                        "lambda_volume": LAM_VOL})
        v.run(MCS, record_every=max(200, MCS // 6), verbose=False)
        h = v.history[-1]
        h.update({"chi": chi, "seed": seed, "lambda_chem": LAM_CHEM,
                  "lambda_volume": LAM_VOL, "mcs": MCS})
        rows.append(h)
        print(f"  [shard {shard}] chi={chi:.3f} seed={seed} -> C={h['compactness']:.4f} "
              f"comp={h['n_components']} lacunae={h['n_lacunae']} "
              f"({time.time()-t0:.0f}s)", flush=True)
    path = os.path.join(OUT, f"p1d_shard_{shard}.json")
    with open(path, "w") as fh:
        json.dump(rows, fh, indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
