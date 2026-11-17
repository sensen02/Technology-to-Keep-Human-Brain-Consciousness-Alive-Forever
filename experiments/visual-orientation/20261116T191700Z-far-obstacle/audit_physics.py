"""Report each continuous propulsion segment independently of detour acceptance."""
import argparse
from pathlib import Path

import numpy as np

from approach_runtime import turn, sha
from walking_bench import cycles


def report(folder):
    record = turn.read(folder / "RESULT.json")
    with np.load(folder / "OBS.npz") as z:
        commands = z["decisions"]
    forward = commands[commands[:,3] > .06,0].astype(int)
    with np.load(folder / "TRACE.npz") as z:
        packed = {key:z[key] for key in ("time","loads","padz","foot")}
        com = z["com"]
    with np.load(folder / "NATIVE.npz") as z:
        forces = z["forces"]
    with np.load(folder / "BLOCK_FORCE.npz") as z:
        impact = z["force"]
    assert len(forces) == len(impact) == record["native_steps"]
    assert np.isfinite(forces).all() and np.isfinite(impact).all()
    movement = []
    for segment in np.split(forward, np.flatnonzero(np.diff(forward)>5)+1):
        if not len(segment):
            continue
        start, end = int(segment[0]), int(segment[-1]+5)
        if end >= len(com):
            end = len(com)-1
        raw = forces[start*80:end*80]
        ground = raw[:,:6]
        slide = raw[:,6:12]
        elapsed = (end-start)*.001
        support = float(ground.sum(1).mean()/turn.W)
        displacement = com[end,:2]-com[start,:2]
        speed = float(np.linalg.norm(displacement)/elapsed)
        slip = (slide.sum(0)/ground.sum(0))/speed if speed > 0 and np.all(ground.sum(0)>0) else None
        gait,_ = cycles(packed,start,end,arm_before=20)
        movement.append(dict(
            interval_ms=[start,end], speed_native_s=speed, support_W=support, per_leg_slip=None if slip is None else slip.tolist(),
            maximum_slip=None if slip is None else float(slip.max()), cycles=gait, COM_min=float(com[start:end+1,2].min()),
            nonfoot_native=float(raw[:,12].max()),
            original_walk_engineering_gate=bool(support>=.9 and speed>=.05 and slip is not None and slip.max()<=.5
                and all(gait.values()) and raw[:,12].max()==0 and com[start:end+1,2].min()>=1.2)))
    return dict(trial_result_sha256=sha(folder/"RESULT.json"), obstacle_native_max=float(impact.max()),
                native_samples_with_any_obstacle_load=int(np.count_nonzero(impact)),
                native_samples=len(impact), continuous_propulsion=movement)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("trial",type=Path)
    args = parser.parse_args()
    outcome = report(args.trial)
    turn.stand.put(args.out,outcome)
    print(outcome,flush=True)
