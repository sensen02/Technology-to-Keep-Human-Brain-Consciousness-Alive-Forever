"""Complete families whose restricted placement scan had no separated option."""
import argparse
from pathlib import Path

import mujoco as mj
import numpy as np
import torch

from sensory_corpus import Apparatus, RADIUS, case_for, sha, turn


def separated_center(bench, track, case):
    side = case["side"]
    body = bench.body
    pads = bench.pad_ids[3*side:3*side+3]
    radii = body.plant.geom_size[pads,0]
    first = track[:case["start"]+100:20,3*side:3*side+3]
    angles = np.linspace(0.,2*np.pi,32,endpoint=False)
    directions = np.column_stack([np.cos(angles),np.sin(angles),np.zeros(len(angles))])
    start = case["start"]+180
    options = []
    for when in range(start,len(track)-40,12):
        positions = track[when,3*side:3*side+3]
        centers = positions[:,None,:] + directions[None,:,:]*(radii[:,None,None]+RADIUS-.004)
        gaps = np.linalg.norm(centers[:,:,None,None,:]-first[None,None,:,:,:],axis=-1)-radii-RADIUS
        clearance = gaps.min((2,3))
        speed = np.linalg.norm(positions-track[when-25,3*side:3*side+3],axis=1)
        lateral = directions[:,1]*(1-2*side)
        for point, direction in zip(*np.where(clearance>.008),strict=True):
            merit = (clearance[point,direction]+3.*speed[point])/(1+abs(when-case["target_ms"])/420)
            merit *= .85+.15*lateral[direction]
            options.append((merit, centers[point,direction]))
    if not options:
        raise RuntimeError(f"No fixed placement from full measured path: {case}")
    bench.reset()
    probe = body.plant.geom("touch_tip").id
    for _, center in sorted(options,key=lambda item:item[0],reverse=True):
        body.d.mocap_pos[0] = center
        mj.mj_forward(body.plant,body.d)
        if not any(probe in contact.geom for contact in body.d.contact[:body.d.ncon]):
            return center
    raise RuntimeError(f"No separated placement from the complete initial geometry: {case}")


def recover(root, groups):
    torch.set_num_threads(1)
    folder = root/f"recovery_{min(groups):02d}"
    folder.mkdir()
    bench = Apparatus(folder)
    scripts = [Path(__file__), Path(__file__).with_name("sensory_corpus.py")]
    before = {str(file):sha(file) for file in scripts}
    turn.stand.put(folder/"RECOVERY_IDENTITY.json",before)
    for group in groups:
        case = case_for(group)
        family = root / f"group_{group:02d}"
        family.mkdir(exist_ok=True)
        if (family/"clear/RESULT.json").exists():
            stored = family/"clear/SAMPLES.npz"
            assert sha(stored)==turn.read(family/"clear/RESULT.json")["sha256"]
            with np.load(stored,allow_pickle=False) as data:
                reference = data["pads"]
        else:
            reference,_ = bench.run(family/"clear",case,"clear")
        for mode in ("fixed","external","sham"):
            if not (family/mode).exists():
                position = separated_center(bench,reference,case) if mode=="fixed" else None
                bench.run(family/mode,case,mode,position)
    assert before == {str(file):sha(file) for file in scripts}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("groups",nargs="+",type=int)
    args = parser.parse_args()
    recover(args.root,args.groups)
