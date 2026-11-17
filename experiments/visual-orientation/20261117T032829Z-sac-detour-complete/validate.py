"""Qualify native demonstrations and rescore the identical logged transitions."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import mujoco as mj
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "20261117T022700Z-sac-startup-replay"
sys.path.insert(0, str(SOURCE))
from course import ACTOR_S, BODY, Footprint, GAMMA, sha
from verify_runs import verify

from route_reward import BoxRoute, RouteConfig, rescore


def load(path):
    return json.loads(path.read_text())


class Kinematics:
    def __init__(self):
        self.model = BODY.act_mod.load_full_model(BODY.PAD_XML)
        self.data = mj.MjData(self.model)
        self.thorax = self.model.body("Thorax").id
        self.head = self.model.body("Head").id
        pads = {leg:[self.model.geom(f"{leg}Tarsus5_pad{i}").id for i in range(3)]
                for leg in BODY.LEG_ORDER}
        self.footprint = Footprint(SimpleNamespace(plant=self.model,pads=pads))

    def state(self, qpos):
        self.data.qpos[:] = qpos
        mj.mj_kinematics(self.model,self.data)
        mj.mj_comPos(self.model,self.data)
        return self.data.subtree_com[self.thorax].copy()

    def read(self, qpos, edge):
        xyz = self.state(qpos)
        head = self.data.xpos[self.head,:2].copy()
        yaw = float(np.arctan2(self.data.xmat[self.thorax,3],self.data.xmat[self.thorax,0]))
        tail, _ = self.footprint.bounds(qpos)
        return np.r_[xyz,yaw,head,tail-edge]


def qualify(root, kin, gate):
    meta, _ = verify(root)
    receipt = load(root/"REFERENCE.json")
    assert receipt["reference_sha256"] == sha(root/"REFERENCE.npz")
    assert all(sha(Path(path)) == expected for path,expected in receipt["sources"].items())
    with np.load(root/"REFERENCE.npz",allow_pickle=False) as z:
        raw = z["state"]
        phase = z["phase"]
    with np.load(root/"EPISODE.npz",allow_pickle=False) as z:
        qpos, target, contact, delivered, com, mode = (
            z[key] for key in ("qpos","goal","contact_W","delivered","com","mode"))
        d = z["distance"]
    x, _, _, half_x, *_ = meta["geometry"]["boxes"][0]
    positions = np.asarray([kin.read(q,x+half_x) for q in qpos])
    np.testing.assert_allclose(positions[:,6],raw[:,6],rtol=0.,atol=1e-9)
    np.testing.assert_allclose(positions[:,:3],raw[:,:3],rtol=0.,atol=6e-5)
    np.testing.assert_allclose(positions[1:,:3],com,rtol=0.,atol=6e-5)
    np.testing.assert_allclose(np.linalg.norm(positions[1:,4:6]-target[:2],axis=1),d,rtol=0.,atol=5e-5)

    edge_margin = positions[:,6]
    safely_beyond = np.logical_and.accumulate((edge_margin>=gate["tail_margin"])[::-1])[::-1]
    sustained = np.flatnonzero(safely_beyond)
    clear = int(sustained[0]) if len(sustained) else len(mode)
    rejoin = np.flatnonzero(phase == "rejoin")
    after_clear = max(clear,int(rejoin[0])) if len(rejoin) else len(mode)
    radius = np.linalg.norm(positions[:,:2]-target[:2],axis=1)
    forward_s = float(np.count_nonzero(delivered[after_clear:,:,0] > 0.) * .005)
    advance = float(radius[after_clear]-radius[-1])
    checks = dict(no_native_box_contact=receipt["box_contact_native_counts"]==[0] and not contact.any(),
                  genuine_success=receipt["ending"]==meta["reason"]=="success" and meta["terminated"]
                        and not meta["truncated"],
                  full_body_stays_beyond=bool(len(sustained)),
                  resumed_forward=forward_s>=gate["min_forward_seconds_after_clear"],
                  approached_target_after_clear=advance>=gate["net_com_advance_after_clear"],
                  neutral_at_goal=not bool(delivered[-30:].any()) and not bool(mode[-30:].any()))
    result = dict(case=meta["case"],checks=checks,passed=all(checks.values()),reason=meta["reason"],
                  actor_seconds=float(len(mode)*ACTOR_S),native_steps=meta["native_steps"],
                  native_steps_ref=receipt["native_steps_including_warm"],
                  neck_released_packets=meta["delivered_packets"],
                  clear_at_s=float(clear*ACTOR_S) if len(sustained) else None,
                  rejoin_at_s=float(rejoin[0]*ACTOR_S) if len(rejoin) else None,
                  forward_seconds_after_clear=forward_s,com_target_approach_after_clear=advance,
                  final_range=float(d[-1]),final_tail_margin=float(edge_margin[-1]),
                  max_floor_nonfoot_20ms_W=float(raw[:,7].max()),
                  final_yaw_200ms_deg=float(abs(np.degrees(np.unwrap(positions[-11:,3])[-1]
                                                           - np.unwrap(positions[-11:,3])[0]))),
                  com_drift_last_600ms=float(np.linalg.norm(positions[-1,:2]-positions[-31,:2])),
                  episode_sha256=meta["files"]["EPISODE.npz"])
    assert result["native_steps_ref"] == result["native_steps"] + 40000
    (root/"QUALIFIED.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)
    return result, phase


def score_rollout(root, kin, config, output, phase=None):
    meta = load(root/"RESULT.json")
    path = root/"EPISODE.npz"
    assert sha(path) == meta["files"]["EPISODE.npz"]
    with np.load(path,allow_pickle=False) as z:
        terms,columns,com,qpos,goal,reasons,old,delivered = (z[key] for key in (
            "terms","term_names","com","qpos","goal","reasons","reward","delivered"))
    columns = columns.tolist()
    xy = np.vstack((kin.state(qpos[0])[:2],com[:,:2]))
    boxes = meta["geometry"]["boxes"]
    assert len(boxes)<=1
    new, remaining = rescore(terms,columns,xy,goal[:2],boxes[0] if boxes else None,reasons,config)
    locked = [i for i,k in enumerate(columns) if k not in ("progress","potential")]
    np.testing.assert_array_equal(new[:,locked],terms[:,locked])
    np.testing.assert_allclose(old,terms.sum(axis=1),rtol=0.,atol=2e-14)
    if boxes:
        summed = new.sum(axis=1)
    else:
        np.testing.assert_array_equal(new,terms)
        summed = old.copy()
    extra = {}
    if phase is not None:
        masks = {label:phase[1:]==label for label in dict.fromkeys(phase[1:])}
        if "rejoin" in masks:
            forward = np.any(delivered[...,0]>0.,axis=1)
            masks["rejoin_forward"] = masks["rejoin"] & forward
            masks["rejoin_no_forward"] = masks["rejoin"] & ~forward
        for label,mask in masks.items():
            if not mask.any():
                continue
            first,last = np.flatnonzero(mask)[[0,-1]]
            extra[label] = dict(steps=int(mask.sum()),old=float(old[mask].sum()),new=float(summed[mask].sum()),
                                route_remaining=[float(remaining[first]),float(remaining[last+1])])
    np.savez_compressed(output/f"{meta['case']}.npz",reward=summed,terms=new,route_remaining=remaining,
                        episode_sha256=meta["files"]["EPISODE.npz"])
    discount = GAMMA**np.arange(len(old))
    return dict(case=meta["case"],reason=meta["reason"],source=str(root),sha256=sha(path),
                duration_s=len(old)*ACTOR_S, original=float(old.sum()),route=float(summed.sum()),
                original_discounted=float(old @ discount),route_discounted=float(summed @ discount),
                unchanged_components=[columns[i] for i in locked], phases=extra,
                score_sha256=sha(output/f"{meta['case']}.npz"))


def geometry_controls(config):
    box = (1.5,0.,0.5,.045,.22,.50)
    target = np.array([5.8,0.])
    route = BoxRoute(box,target,config)
    a = np.array([-1.,0.])
    assert abs(route.distance(a+np.array([0.,.04]))-route.distance(a-np.array([0.,.04]))) < 1.e-12
    assert route.distance(a) > np.linalg.norm(target-a)
    side = np.array([1.6,route.hi[1]+.1])
    assert route.distance(side+np.array([.4,0.])) < route.distance(side)
    inside = np.mean((route.lo,route.hi),axis=0)
    assert route.distance(inside) > route.distance(side)
    columns = ["event","time","potential","toe","stall","progress"]
    terms = np.tile([0.,-.005,0.,0.,0.,0.],(250,1))
    controls = {}
    for name, trajectory in (
        ("stationary_or_head_rotates",np.repeat([side],251,axis=0)),
        ("walk_alongside",side+np.c_[np.linspace(0.,.4,251),np.zeros(251)]),
        ("reverse_then_recover",side+np.c_[np.r_[np.linspace(0.,.2,126),np.linspace(.2,0.,126)[1:]],
                                          np.zeros(251)]),
    ):
        result, _ = rescore(terms,columns,trajectory,target,box,[""]*250,config)
        controls[name] = dict(total=float(result.sum()),route_component=float(result[:,-1].sum()),
                              discounted_total=float(result.sum(1) @ GAMMA**np.arange(250)))
    assert controls["stationary_or_head_rotates"]["route_component"] == 0.
    assert controls["walk_alongside"]["total"] > 0.
    assert controls["reverse_then_recover"]["route_component"] <= 1.e-12
    assert controls["reverse_then_recover"]["discounted_total"] < 0.
    return controls


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("left",type=Path)
    cli.add_argument("right",type=Path)
    cli.add_argument("--extra",type=Path,nargs="*",default=[])
    args = cli.parse_args()
    torch.set_num_threads(1)
    config = RouteConfig(**load(HERE/"SCORING_PLAN.json")["route_reward"])
    controls = geometry_controls(config)
    kin = Kinematics()
    gate = load(HERE/"PLAN.json")["acceptance"]
    left,right = (p.resolve() for p in (args.left,args.right))
    demos = {path:qualify(path,kin,gate) for path in (left,right)}
    out=HERE/"SCORES"
    out.mkdir(exist_ok=True)
    scores=[]
    for path in [left,right,*map(Path.resolve,args.extra)]:
        scores.append(score_rollout(path,kin,config,out,demos[path][1] if path in demos else None))
    report=dict(both_demos_qualified=all(meta["passed"] for meta,_ in demos.values()),
                demonstrations=[meta for meta,_ in demos.values()], score_controls=controls, comparisons=scores,
                source_sha256={p.name:sha(p) for p in (HERE/"SCORING_PLAN.json",HERE/"route_reward.py",Path(__file__))})
    (HERE/"COMPARISON.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2),flush=True)
    raise SystemExit(0 if report["both_demos_qualified"] else 1)


if __name__ == "__main__":
    main()
