"""Replay the prior physical walking observations across the four fixed source axons."""
import argparse
from pathlib import Path

import numpy as np

from bind import BINDING, GRAPH
from separation import GROUPS, SCALE, SeparatedLink
from touch_runtime import W, sha, turn

WALKS = Path(__file__).resolve().parent.parent/"20261116T211600Z-touch-prediction/DATA"
STEP_MS = 5
WALK_GROUPS = 20


def replay_archive(input_activity, commands, cut=None, isolate=None):
    link = SeparatedLink()
    received = []
    for step, physical in enumerate(input_activity):
        if cut and step in cut:
            link.cut_an(step==cut[0])
        if step%STEP_MS==0:
            link.send(int(commands[step]))
        row,valid = link.observe_packet()
        received.append(np.r_[row,valid])
        if not link.an_cut:
            activities = physical.copy()
            if isolate is not None:
                activities[isolate] = 0.
            link.sense(activities.tolist())
        link.advance()
    return np.asarray(received), link.transport.verify_final()


def walking(root, ids):
    for number in ids:
        source = WALKS/f"group_{number:02d}"
        target = root/source.name
        target.mkdir()
        for variant in ("clear","fixed","external","sham"):
            meta = turn.read(source/variant/"RESULT.json")
            path = source/variant/"SAMPLES.npz"
            assert sha(path)==meta["sha256"]
            with np.load(path,allow_pickle=False) as z:
                raw, loads, lateral, requested, original = (z[key] for key in (
                    "native","state","touch","commands","activity"))
            side_loads = loads[:,[12,15]]
            millisecond_load = np.repeat(side_loads,STEP_MS,axis=0)
            mv = np.repeat(requested[:,1],STEP_MS)
            sensor = np.clip(raw[:,:2].reshape(len(mv),80,2).mean(1)/(.02*W),0.,1.)
            preceding = np.vstack([np.zeros(2),sensor[:-1]])
            activity = np.column_stack([preceding,np.clip(millisecond_load/SCALE,0.,1.)])
            arrivals, identity = replay_archive(activity,mv)
            np.testing.assert_allclose(arrivals[::STEP_MS,:2], original,rtol=0.,atol=2e-15)
            archived = target/f"{variant}.npz"
            np.savez_compressed(archived,received=arrivals[::STEP_MS].copy(),normal_load=side_loads,
                                truth_tactile=lateral,commands=requested[:,1])
            turn.stand.put(target/f"{variant}.json",dict(native_origin=str(path),native_archive_sha256=meta["sha256"],
                replay_sha256=sha(archived),case=meta["case"],variant=variant,physical_steps_this_run=0,
                timing="5 ms load sample held causally until following sample; 1 ms lateral sensor from native solve",
                routes=identity))
        print(source.name,flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("ids",nargs="+",type=int)
    args=parser.parse_args()
    walking(args.root,args.ids)
