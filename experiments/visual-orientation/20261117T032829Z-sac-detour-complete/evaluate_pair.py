"""Evaluate one saved reward arm on the shared untouched flat or single-box layout."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/"20261117T022700Z-sac-startup-replay"))
import course
from evaluate_policy import Policy
from model import SACBrain
from route_reward import RouteConfig
from verify_runs import deterministic_commands, verify
from validate import Kinematics, load, score_rollout


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("arm",choices=("warm","original","route"))
    cli.add_argument("case",choices=("held_flat","held_box"))
    cli.add_argument("out",type=Path)
    args = cli.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(3373)
    settings = load(HERE/"SCORING_PLAN.json")
    paired = load(HERE/"PAIRED/PAIRED.json")
    pin = HERE/"WARM_COMPLETE.pt" if args.arm=="warm" else Path(paired["checkpoint"][args.arm])
    state = torch.load(pin,map_location="cpu",weights_only=True)
    agent = SACBrain().eval()
    agent.load_state_dict(state["model"])
    out = args.out.resolve()
    course.SCENES = HERE/"HELD_SCENES.json"
    identity = dict(path=str(pin),sha256=course.sha(pin),sac_updates=state.get("sac_updates",0),sample_seed=None)
    course.rollout(out,args.case,Policy(agent,True),agent.brain,"sac" if identity["sac_updates"] else "warm",pin=identity)
    assert course.sha(pin)==identity["sha256"]
    r,z = verify(out)
    deterministic_commands(out,pin)
    kin = Kinematics()
    with np.load(out/"EPISODE.npz",allow_pickle=False) as saved:
        xy, goal = saved["com"],saved["goal"]
    boxes = r["geometry"]["boxes"]
    tail = None
    confirmed_at = None
    if boxes:
        edge = boxes[0][0]+boxes[0][3]
        tail = float(kin.footprint.bounds(z["qpos"][-1])[0] - edge)
        if tail >= .03:
            tail_all = np.array([kin.footprint.bounds(q)[0]-edge for q in z["qpos"]])
            sustained = np.logical_and.accumulate((tail_all>=.03)[::-1])[::-1]
            confirmed_at = float(np.flatnonzero(sustained)[0] * course.ACTOR_S)
    score = score_rollout(out,kin,RouteConfig(**settings["route_reward"]),out)
    metrics = dict(arm=args.arm,case=args.case,success=(r["reason"]=="success"),
                   ending=r["reason"],seconds=r["actor_ticks"]*course.ACTOR_S,
                   final_range=r["final_range"],initial_range=r["initial_range"],
                   final_tail_margin=tail,sustained_full_clearance_from_s=confirmed_at,
                   lateral_extent=float(np.max(np.abs(xy[:,1]-xy[0,1]))),
                   target_progress_com=float(np.linalg.norm(xy[0,:2]-goal[:2])-np.linalg.norm(xy[-1,:2]-goal[:2])),
                   native_steps=r["native_steps"],released=r["delivered_packets"],
                   box_contact_peak_W=r["contact_peak_W"],deterministic_actions_verified=True,
                   recorded_motion_s=float(.005*np.count_nonzero(z["delivered"].any(axis=-1))),
                   recorded_forward_s=float(.005*np.count_nonzero(z["delivered"][...,0]>0.)),
                   recorded_turn_s=float(.005*np.count_nonzero(z["delivered"][...,1])),
                   policy_sha256=identity["sha256"],rescore=score)
    (out/"EVALUATION.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics,indent=2),flush=True)


if __name__ == "__main__":
    main()
