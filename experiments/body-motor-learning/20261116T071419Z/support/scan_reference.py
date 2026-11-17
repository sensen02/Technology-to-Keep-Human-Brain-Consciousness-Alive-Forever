"""Check nominal pad clearance relative to the clipping-plane seam positions."""
from pathlib import Path
import argparse

import mujoco as mj
import numpy as np

from support_experiment import HERE, W, ARCHIVE, bench, read, stan


def scan(cases, stop, output):
    model = stan.physics.act_mod.load_full_model(stan.physics.PAD_XML)
    data = mj.MjData(model)
    ids = np.array([[i for i in range(model.ngeom) if model.geom(i).name.startswith(leg+'Tarsus5_pad')]
                    for leg in bench.LEGS])
    radius = model.geom_size[ids,0]
    with np.load(ARCHIVE/"TRACE.npz",allow_pickle=False) as z:
        root = z["qpos"][:stop+1]
        w = z["loads"][:stop+1]/W
    positions = []
    for values in root:
        data.qpos[:] = values
        mj.mj_fwdPosition(model,data)
        positions.append(data.geom_xpos[ids].copy())
    xyz = np.asarray(positions)
    mask = (xyz[...,2]-radius < .0005) & (w[:,:,None] > .001)
    result = []
    for case in cases:
        layout = read(case)
        x0,x1,y0,y1 = layout["opening"]
        x,y = xyz[...,0],xyz[...,1]
        close_x = (abs(x-x0)<radius) | (abs(x-x1)<radius)
        close_y = (abs(y-y0)<radius) | (abs(y-y1)<radius)
        if layout.get("axis_first") == "x":
            hit = close_x | (close_y & (x>=x0) & (x<=x1))
        else:
            hit = close_y | (close_x & (y>=y0) & (y<=y1))
        counts = {}
        for i,leg in enumerate(bench.LEGS):
            frames = np.flatnonzero((mask[:,i] & hit[:,i]).any(1))
            counts[leg] = dict(samples=len(frames),first=frames[:3].tolist(),last=frames[-3:].tolist())
        result.append(dict(case=str(case),possible_loaded_pad_overlapping_seam=counts))
    stan.put(output,dict(scope="preflight from already recorded nominal configs; no integration, not physical seam equivalence",
                        through_ms=stop,trace_sha256=stan.sha(ARCHIVE/"TRACE.npz"),cases=result))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("stop",type=int)
    p.add_argument("cases",type=Path,nargs="+")
    a=p.parse_args()
    scan([p.resolve() for p in a.cases],a.stop,a.output.resolve())
