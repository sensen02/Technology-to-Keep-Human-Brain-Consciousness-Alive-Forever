import sys, time
sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np
from wound_healing import Epithelium

for prolif in (False, True):
    for lf in (1.0, 2.2, 4.0):
        t0=time.time()
        e = Epithelium(L=120, seed=1, proliferative=prolif, params={"lambda_fill": lf})
        e.m.step(150, field_every=0)
        info = e.wound(radius_um=20.0)
        e.run(1500, record_every=100, verbose=False)
        h = e.history[-1]
        fit = e.two_phase_fit()
        print(f"prolif={prolif} lam_fill={lf}: R={info['radius_sites']:.1f}sites "
              f"({info['radius_um']:.1f}um) A0={info['area_um2']:.0f}um2 -> closed={h['closed_fraction']:.3f} "
              f"t98={e.closure_mcs_98} cells={h['n_cells']} wall={time.time()-t0:.0f}s "
              f"slopes={fit['slope1_um2_per_mcs']} / {fit['slope2_um2_per_mcs']} two_phase={fit['two_phase_preferred']}")
