"""Short tuning probe: does a network actually form?  Not a result."""
import sys, time, json
sys.path.insert(0, '/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np
from vasculogenesis import Vasculogenesis, p

def probe(L, n_ec, mcs, lam_tip, contact_inhibited=True, seed=0, lc=None):
    v = Vasculogenesis(L=L, seed=seed, n_ec=n_ec, n_fibroblast=20,
                       contact_inhibited=contact_inhibited,
                       params={"lambda_chem_tip": lam_tip, "lambda_length": lc} if lc else {"lambda_chem_tip": lam_tip})
    v.run(mcs, record_every=150, verbose=True)
    h = v.history[-1]
    return h

if __name__ == "__main__":
    for lam in (3.0, 7.0, 15.0):
        print(f"=== lambda_chem_tip={lam}  contact_inhibited=True ===")
        t0=time.time()
        h = probe(L=160, n_ec=140, mcs=600, lam_tip=lam)
        print(f"   wall={time.time()-t0:.1f}s  -> comp={h['ec_components']} loops={h['ec_loops']} "
              f"area={h['ec_area_fraction']:.4f} tip={h['n_tip']} elong={h['ec_mean_elongation']:.2f} "
              f"vegf_mean={h['vegf_mean']:.3f}")
