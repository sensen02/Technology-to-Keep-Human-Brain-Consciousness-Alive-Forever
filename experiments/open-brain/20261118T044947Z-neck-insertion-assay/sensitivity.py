"""Test whether one quantization crossing is stable to numerical and proxy choices."""
import json
from itertools import product
from pathlib import Path

import numpy as np
from cable import FIXED, ROOTS, NEURAL_STEP, propagate, ticks, sha

HERE = Path(__file__).resolve().parent


def main():
    stored=json.loads((HERE/"CABLE.json").read_text())
    reference=json.loads(FIXED.read_text())["ports"][ROOTS[1]]
    series={}
    with np.load(HERE/"MECHANICS.npz") as m:
        for name in ("release","retained","stick_release","refined"):
            v=m[name]
            series[name]=(v[:,0],v[:,7]/v[0,7])
        length=m["release"][0,7]
    min_stretch=min(x[1].min() for x in series.values())
    grid=np.unique(np.r_[np.linspace(min_stretch,1.,13),1.])
    results={}
    for freeze_radius, fine_step in product((False,True),(False,True)):
        dt=NEURAL_STEP/2 if fine_step else NEURAL_STEP
        conduction=np.array([propagate(length,x,not freeze_radius,dt,reverse=True) for x in grid])
        normalized=conduction/conduction[-1]
        by_trajectory={}
        for name,(t,stretch) in series.items():
            local=reference["propagation_ms"]*np.interp(stretch,grid,normalized)
            route=np.interp(np.arange(20000)*.0001,t,local)
            sample=ticks(route)
            changed=np.flatnonzero(sample!=reference["delay_ticks"])
            by_trajectory[name]=dict(min_ms=float(route.min()),max_ms=float(route.max()),
                changed_ticks=len(changed), first_changed_s=float(changed[0]*.0001) if changed.size else None,
                last_changed_s=float(changed[-1]*.0001) if changed.size else None)
        results[f"{'constant_diameter' if freeze_radius else 'constant_volume'}_{'finer_step' if fine_step else 'main_step'}"]=by_trajectory
    assert results["constant_volume_main_step"]["release"]["changed_ticks"]==stored["changed_departure_ticks"][2][1]
    output=dict(sampled_root=ROOTS[1],snapshots_per_grid=len(grid),alternatives=results,
                input_sha256={str(f):sha(f) for f in (FIXED,HERE/"MECHANICS.npz",HERE/"CABLE.json",HERE/"cable.py",Path(__file__))})
    (HERE/"SENSITIVITY.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    for key,data in results.items():
        print(key,[(mode,row["changed_ticks"],round(row["min_ms"],7)) for mode,row in data.items()],flush=True)


if __name__=="__main__":
    main()
