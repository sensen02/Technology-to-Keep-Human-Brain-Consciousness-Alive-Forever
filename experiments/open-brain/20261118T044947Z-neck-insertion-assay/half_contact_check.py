"""Supplemental sensitivity to the unmeasured contact displacement."""
from pathlib import Path
import json

import numpy as np

import mechanics
from cable import ticks

HERE=Path(__file__).resolve().parent


def main():
    source=json.loads((HERE/"MECHANICS.json").read_text())
    chart=source["chart"]
    with np.load(HERE/"MECHANICS.npz") as data:
        paths=[data[f"axon{i}"] for i in range(3)]
    mechanics.PENETRATION=5.0
    raw,_,kkt=mechanics.run_case(72,28,.01,paths,chart["length_um"],chart["width_um"],True,False)
    fixed=json.loads(mechanics.FIXED.read_text())["ports"]
    authored=json.loads((HERE/"CABLE.json").read_text())["measured_cables"]
    result=dict(explicit_supplemental=True, assumed_contact_um=mechanics.PENETRATION,
                max_deformation_um=float(raw[:,3].max()), peak_strain=float(raw[:,4].max()),
                min_delay_ms={}, changed_clock_opportunities={}, iteration_steps=len(raw))
    for index,root in enumerate(mechanics.ROOTS):
        fit=authored[root]
        stretch=raw[:,5+index*2]/raw[0,5+index*2]
        xs=np.array(fit["stretch_grid"])
        assert stretch.min()>=xs.min()-1e-8 and stretch.max()<=xs.max()+1e-8
        latency=np.interp(stretch,xs,fit["time_ms"]["True"])/fit["reference_proxy_ms"]["True"]
        continuous=np.interp(np.arange(20000)*.0001,raw[:,0],latency*fixed[root]["propagation_ms"])
        result["min_delay_ms"][root]=float(continuous.min())
        result["changed_clock_opportunities"][root]=int((ticks(continuous)!=fixed[root]["delay_ticks"]).sum())
    result["kkt_free_residual"],result["kkt_wall_gap"]=kkt
    (HERE/"HALF_CONTACT.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result,flush=True)


if __name__=="__main__":
    main()
