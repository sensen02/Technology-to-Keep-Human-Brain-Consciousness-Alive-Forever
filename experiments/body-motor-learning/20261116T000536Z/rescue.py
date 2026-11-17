"""Compare recovery from identical physical states visited by the student."""
import argparse
from pathlib import Path

import numpy as np
import torch

from fit_walk import load_policy
from walking_bench import HERE, STARTS, W, cycles, features, load_initial, read, stand
from snapshots import capture

CASES = {
    "a": ("nominal",345,0),
    "b": ("lean_left",770,11),
    "c": ("moving_forward",1250,0),
    "d": ("uneven_feet",1725,31),
    "e": ("lean_left",2300,0),
    "f": ("moving_forward",2800,17),
}
STEPS_MS = 1600
CHECKPOINT = HERE/"FIT_INITIAL/step_1200.pt"


def sample(c,m):
    return dict(time=float(c.d.time), qpos=c.d.qpos.copy(), qvel=c.d.qvel.copy(),
                com=c.d.subtree_com[c.th].copy(),loads=m["loads"].copy(),nonfoot=float(m["nonfoot"]),
                padz=m["padz"].copy(),speed=m["speed"].copy(),foot=m["foot"].copy(),
                target=c.actuator.target_rad.copy())


def rollout(c,ms,policy):
    m=c.measure()
    trajectory=[sample(c,m)]
    labels=[]
    unsafe=0
    for tick in range(ms):
        if not tick%5:
            x=features(c,m)
            a=policy(x,tick,c.actuator.target_rad)
            labels.append((x.copy(),a.copy()))
        m=c.tick(a)
        if np.any(c.d.xfrc_applied) or np.any(c.d.qfrc_applied):
            raise RuntimeError("Unexpected wrench")
        trajectory.append(sample(c,m))
        unsafe=unsafe+1 if m["com"]<.8 or m["nonfoot"]>.5*W else 0
        if unsafe>=100:
            break
    return ({k:np.asarray([row[k] for row in trajectory]) for k in trajectory[0]},
            {k:np.asarray(item) for k,item in zip(("x","action"),zip(*labels),strict=True)})


def stats(z, ref):
    num=len(z["time"])-1
    gate=[]
    gates=dict(duration=num==STEPS_MS,com=bool(z["com"][:,2].min()>=1.2),
               support=bool(z["loads"][1:].sum(-1).mean()>=.9*W),
               nonfoot_zero=bool(z["nonfoot"].max()==0.),
               advance=bool(z["com"][-1,0]-z["com"][0,0]>=.03))
    counts,_=cycles(z,0,num)
    gates["one_cycle"]=bool(sum(counts.values())>=1)
    return dict(gates=gates, passed=all(gates.values()), duration_ms=num,
                target_error_end=float(np.sqrt(np.square(z["target"][-1]-ref[num]).mean())),
                cycles=counts,min_com=float(z["com"][:,2].min()),
                forward_delta=float(z["com"][-1,0]-z["com"][0,0]))


def trial(name):
    initial,cut,seed=CASES[name]
    output=HERE/f"RESCUE_{name}"
    output.mkdir()
    bench=stand.Bench(output)
    c=bench.body
    c.physics_order=stand.physics.rc.SENS_IDX
    source=STARTS/f"{initial}.npz"
    bank=read(STARTS/"BANK.json")
    load_initial(bench,source,bank[initial]["sha256"])
    policy=load_policy(CHECKPOINT)
    data=HERE/f"TEACH_{initial}"
    if not read(data/"RESULT.json")["first_step"]:
        raise ValueError("No usable reference from this initial")
    ident=stand.identity.capture(c,dict(initial=initial,cutoff_ms=cut,pulse_seed=seed),
        extra_paths=[CHECKPOINT,HERE/"RESCUE_PREREG.zh-CN.md",source,data/"TRACE.npz",data/"DECISIONS.npz"])
    stand.put(output/"RUN_IDENTITY.json",ident)
    motor=np.random.default_rng(seed).normal(0.,.04,42) if seed else np.zeros(42)

    def prefix(x,t,target):
        a=policy(x,int(t>=500))
        return np.clip(a+motor,-1,1) if 500+cut-35<=t<500+cut-15 else a

    before,_=rollout(c,500+cut,prefix)
    np.savez_compressed(output/"PREFIX.npz",**before)
    if len(before["time"]) != 501+cut or before["nonfoot"][-1]>.01*W or before["com"][-1,2]<1.15:
        stand.put(output/"RESULT.json",dict(admitted=False,reason="prefix already collapsed",
            prefix_ms=len(before["time"])-1,native_steps=(len(before["time"])-1)*80,
            checkpoint=stand.sha(CHECKPOINT)))
        return
    saved=capture(bench)
    destination=output/"HANDOFF.npz"
    np.savez_compressed(destination,**saved)
    saved_sha=stand.sha(destination)
    with np.load(data/"DECISIONS.npz",allow_pickle=False) as d,np.load(data/"TRACE.npz",allow_pickle=False) as t:
        from_ms=max(500,500+cut-225)
        to_ms=min(500+cut+225,5300-STEPS_MS-5)
        indexes=np.arange(from_ms//5,to_ms//5+1)
        current=features(c,c.measure())
        columns=np.r_[0:42,126:174,183:207]
        x=d["x"][indexes][:,columns]
        error=np.square((x-current[columns])/np.maximum(x.std(0),.015)).mean(1)
        ref_ms=int(indexes[error.argmin()])*5
        reference=t["target"][ref_ms:ref_ms+STEPS_MS+1].copy()
        actions=d["action"][ref_ms//5:(ref_ms+STEPS_MS)//5].copy()
    if len(actions)!=STEPS_MS//5:
        raise ValueError("Reference endpoint out of range")
    report={}
    for arm in ("network","expert"):
        load_initial(bench,destination,saved_sha)

        def act(x,t,target):
            if arm=="network":
                return policy(x,1)
            return np.clip(actions[t//5]+7.*(reference[t]-target)/2.,-1.,1.)

        trace,labels=rollout(c,STEPS_MS,act)
        np.testing.assert_array_equal(trace["qpos"][0],saved["qpos"])
        np.testing.assert_array_equal(trace["qvel"][0],saved["qvel"])
        np.testing.assert_array_equal(trace["target"][0],saved["act_target_rad"])
        np.savez_compressed(output/f"{arm.upper()}_TRACE.npz",**trace)
        report[arm]=stats(trace,reference)
        if arm=="expert":
            path=output/"EXPERT.npz"
            np.savez_compressed(path,**labels,intent=np.ones(len(labels["x"])))
    ref=report["network"]
    expert=report["expert"]
    improved=not ref["passed"] or sum(expert["cycles"].values())>sum(ref["cycles"].values()) or (
        expert["target_error_end"] < .9*ref["target_error_end"])
    result=dict(admitted=bool(expert["passed"] and improved), arms=report,ref_ms=ref_ms,parent_path=str(data),
                checkpoint=stand.sha(CHECKPOINT),data_sha256=stand.sha(path),switch_sha256=saved_sha,
                native_steps=80*(len(before["time"])-1+sum(r["duration_ms"] for r in report.values())))
    stand.identity.verify_runtime(ident,c)
    stand.put(output/"RESULT.json",result)
    print(name,result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("case",choices=CASES)
    a=p.parse_args()
    torch.set_num_threads(1)
    trial(a.case)
