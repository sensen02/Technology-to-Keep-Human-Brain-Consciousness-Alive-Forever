"""P3 parameter scan: which regimes actually close a 40 um wound?"""
import sys, json, time
sys.path.insert(0,'/run/media/sensen/Data2/cell_wound_prototype')
import numpy as np
from wound_healing import Epithelium

def one(L=120, seed=1, prolif=True, lv=6.0, lf=4.0, le=0.28, clamp=2, mcs=2500,
        r_um=20.0, cycle_mcs=None, division_stretch_threshold=None):
    params = {"lambda_volume": lv, "lambda_fill": lf, "lambda_edge": le}
    if cycle_mcs is not None:
        params["cycle_mcs"] = cycle_mcs
    if division_stretch_threshold is not None:
        params["division_stretch_threshold"] = division_stretch_threshold
    e = Epithelium(L=L, seed=seed, proliferative=prolif, params=params)
    if clamp: e.clamp_outer_ring(clamp)
    e.m.step(150, field_every=0)
    info = e.wound(radius_um=r_um)
    e.run(mcs, record_every=100, verbose=False)
    h = e.history[-1]
    fit = e.two_phase_fit()
    return dict(prolif=prolif, lv=lv, lf=lf, le=le, clamp=clamp, cycle_mcs=cycle_mcs,
                A0_um2=info["area_um2"], R_um=info["radius_um"],
                closed=h["closed_fraction"], t98=e.closure_mcs_98,
                cells=h["n_cells"], perim=h["front_perimeter_um"],
                div_total=h["n_divisions_total"], accept=round(h["acceptance"],4),
                slope1=fit["slope1_um2_per_mcs"], slope2=fit["slope2_um2_per_mcs"],
                two_phase=fit["two_phase_preferred"],
                ratio=(fit["rate_ratio_slow_over_fast"]))

if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1])
    for c in cfgs:
        t0=time.time()
        r = one(**c)
        r["wall_s"]=round(time.time()-t0,1)
        print("RESULT " + json.dumps(r), flush=True)
