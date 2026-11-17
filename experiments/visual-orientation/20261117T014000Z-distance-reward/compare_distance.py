"""Recompute nonterminal navigation rewards from saved physical walk trajectories."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/"20261117T011500Z-navigation-stall-reward"
PHYSICAL=HERE.parent/"20261117T010100Z-unified-neck-final"
BAND=(1.75,2.34)
SCENES=("straight","pace","left","right")
STRIDE_MS=20


def load_module(path, name):
    sys.path.insert(0,str(path.parent))
    try:
        spec=importlib.util.spec_from_file_location(name,path)
        mod=importlib.util.module_from_spec(spec)
        sys.modules[name]=mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def summarize(total, active, closer, selected):
    mask=selected & active
    steps=total[mask]
    advances=total[mask & closer]
    assert len(steps) and len(advances)
    return dict(walking_steps=len(steps),closer_steps=len(advances),walking_mean=float(steps.mean()),
                walking_positive=float(np.mean(steps>0.)),closer_mean=float(advances.mean()),
                closer_positive=float(np.mean(advances>0.)))


def calculate(version, compare=None):
    metadata=json.loads((PHYSICAL/"ACCEPTANCE.json").read_text())["cases"]
    results={}
    for case in SCENES:
        path=PHYSICAL/"RUNS"/case/"TRACE.npz"
        assert sha(path)==metadata[case]["hashes"]["TRACE.npz"]
        with np.load(path,allow_pickle=False) as z:
            rows=z["decisions"]
            head=z["head"]
            goal=z["goal"]
        distance=np.linalg.norm(head[:,:2]-goal[:2],axis=1)
        start=metadata[case]["scene"]["on"]
        times=np.arange(start,len(distance)-STRIDE_MS,STRIDE_MS,dtype=int)
        before,after=distance[times],distance[times+STRIDE_MS]
        active=np.min(rows[(times[:,None]+np.arange(0,STRIDE_MS,5))//5,3],axis=1)>0.
        outside=(before>BAND[1]) & (after>BAND[1])
        closer=(before-after)>1e-9
        fixture=metadata[case]["scene"]
        modes={"walking_outside_goal":outside}
        if case=="straight":
            modes.update(mid=(times>=12_000)&(times<20_000)&outside,late=(times>=22_000)&(times<30_000)&outside)
        if case=="pace":
            speed=rows[times//5,3]
            modes.update(fast=outside&(speed>.080),slow=outside&(speed>.069)&(speed<.076))

        def score(engine):
            # Snapshot case explicitly excludes safety channels, allows a
            # common geometric band, and tests only the distance/time terms.
            task=engine.Task(float(distance[start]),BAND,False)
            reward=engine.Reward(task)
            terms=[]
            for i,b,a in zip(times,before,after,strict=True):
                # Prevent an artificial terminal from the audit's isolated
                # measurements. Full event/reward replay has its own suite.
                obs=engine.Measurement(float(a),0.,.0,1.,1.275,1.,4,navigation_active=True)
                step=reward.step(float(b),obs)
                assert not step["terminated"]
                terms.append(step["reward"])
            return [summarize(np.asarray(terms),active,closer,mask) for mask in modes.values()]

        scores=[score(version)]
        if compare is not None:
            scores.append(score(compare))
        results[case]=dict(source=str(path),sha256=sha(path),fixture=fixture,
                           windows={name:{("old" if i==0 else "candidate"):s[j] for i,s in enumerate(scores)}
                                    for j,name in enumerate(modes)})
    return results


def illustrative(old,new):
    out=[]
    for d in (5.,3.,1.,.1):
        for speed in (.0,.072,.084):
            a=d+.02*speed
            values={}
            for key,engine in (("old",old),("candidate",new)):
                if engine is None:
                    continue
                reward=engine.Reward(engine.Task(BAND[1]+a,BAND,False))
                step=reward.step(BAND[1]+a,engine.Measurement(BAND[1]+d,0.,speed,1.,1.275,1.,4))
                values[key]=step["terms"]|{"total":step["reward"]}
            out.append(dict(excess_after=d,radial_speed=speed,**values))
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate",action="store_true")
    args=p.parse_args()
    old=load_module(OLD/"reward.py","old_distance_reward")
    candidate=load_module(HERE/"reward.py","candidate_distance_reward") if args.candidate else None
    result=dict(goal_band=BAND,step_ms=STRIDE_MS,illustration=illustrative(old,candidate),
                physical=calculate(old,candidate))
    if candidate:
        for record in result["physical"].values():
            for case in record["windows"].values():
                assert case["old"]["walking_mean"]<0.<case["candidate"]["walking_mean"], case
                assert case["old"]["closer_positive"]<=.01 and case["candidate"]["closer_positive"]>.70
        assert all(item["candidate"]["total"]<0. for item in result["illustration"]
                   if item["radial_speed"]==0.)
        assert all(item["candidate"]["total"]>0. for item in result["illustration"]
                   if item["radial_speed"]>0.)
    target=HERE/("DISTANCE_COMPARISON.json" if candidate else "ORIGINAL_DISTANCE_AUDIT.json")
    target.write_text(json.dumps(result,indent=2)+"\n")
    for case,row in result["physical"].items():
        for window,report in row["windows"].items():
            print(case,window,json.dumps(report))
