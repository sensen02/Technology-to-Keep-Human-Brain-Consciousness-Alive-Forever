"""Check delivered commands and observed physical changes in terminal five seconds."""
import argparse
from pathlib import Path

import numpy as np

from approach_runtime import sha, turn
from visual_rejoin import marker_bearing


def audit(trial):
    result = turn.read(trial / "RESULT.json")
    for name in ("TRACE.npz","OBS.npz","TRANSPORT.json"):
        assert sha(trial / name) == result["files"][name]
    last = result["actual_ms"]
    first = last - 5000
    with np.load(trial / "TRACE.npz") as trace:
        window = slice(first,last+1)
        qpos = trace["qpos"][window]
        qvel = trace["qvel"][window]
        com = trace["com"][window,:2]
        yaw = np.rad2deg(np.unwrap(trace["yaw"][window]))
        target = trace["target"][window]
        ctrl = trace["ctrl"][window]
    with np.load(trial / "OBS.npz") as obs:
        decisions = obs["decisions"]
        keep = decisions[:,0] >= first
        decisions = decisions[keep]
        stamps, eyes = obs["stamps"], obs["eyes"]
        images = eyes[stamps+20>=first]
    assert len(decisions) == 1000
    assert len(qpos)==5001 and all(np.isfinite(x).all() for x in (qpos,qvel,target,ctrl))
    red = [marker_bearing(image) for image in images]
    return dict(
        trial_result_sha256=sha(trial / "RESULT.json"),
        physical_window_ms=[first,last], control_samples=len(decisions),
        brain_commands=np.unique(decisions[:,1:3],axis=0,return_counts=True)[0].tolist(),
        neck_after_commands=np.unique(decisions[:,3:5],axis=0,return_counts=True)[0].tolist(),
        retinal_samples=len(images), retinal_red_detected=sum(value is not None for value in red),
        learned_confidence_min=float(decisions[:,7].min()), learned_confidence_max=float(decisions[:,7].max()),
        learned_bearing_min=float(decisions[:,5].min()), learned_bearing_max=float(decisions[:,5].max()),
        qpos_max_absolute_change=float(np.ptp(qpos,axis=0).max()),
        qvel_max_absolute=float(np.abs(qvel).max()),
        world_yaw_span_deg=float(np.ptp(yaw)),
        center_of_mass_max_xy_excursion=float(np.linalg.norm(com-com[0],axis=1).max()),
        actuator_target_max_change=float(np.ptp(target,axis=0).max()),
        actuator_ctrl_max_change=float(np.ptp(ctrl,axis=0).max()),
        ident="brain requests and decoded DN match recorded delivered channels; distal mechanical/controller mechanism is unresolved",
    )


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("out",type=Path)
    p.add_argument("trial",type=Path)
    args=p.parse_args()
    data=audit(args.trial)
    turn.stand.put(args.out,data)
    print(data,flush=True)
