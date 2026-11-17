"""Closed-loop gait and stop trials with a declared thorax impulse."""
import argparse
from pathlib import Path

import numpy as np
import torch

from prototype import policy
from walking_bench import HERE, STARTS, W, features, load_initial, metrics, read, stand

CASES = {
    "walking": ("lean_right",3700),
    "stopping": ("moving_backward",10110),
}
WALK_MS=9600
TOTAL=11300


def sample(c,m):
    return (float(c.d.time),c.d.qpos.copy(),c.d.qvel.copy(),c.d.subtree_com[c.th].copy(),
            m["loads"].copy(),float(m["nonfoot"]),m["padz"].copy(),m["speed"].copy(),m["foot"].copy(),
            float(c.d.xmat[c.th,8]),c.actuator.target_rad.copy(),c.d.ctrl.copy())


def run(which):
    initial,onset=CASES[which]
    root=HERE/f"PUSH_{which}"
    root.mkdir()
    player=policy()
    bench=stand.Bench(root)
    c=bench.body
    c.physics_order=stand.physics.rc.SENS_IDX
    source=STARTS/f"{initial}.npz"
    initial_sha=read(STARTS/"BANK.json")[initial]["sha256"]
    load_initial(bench,source,initial_sha)
    ident=stand.identity.capture(c,dict(initial=initial,push_onset_ms=onset,push_duration_ms=20,
        push_w=[0.,.2,0.]),extra_paths=[source,HERE/"PUSH_PREREG.zh-CN.md",HERE/"PROTOTYPES.npz"])
    stand.put(root/"RUN_IDENTITY.json",ident)
    expected=np.zeros((TOTAL*2,6))
    expected[2*onset:2*(onset+20),1]=.2*W
    original=c.apply_adhesion
    observed=[]
    m=c.measure()
    trace=[sample(c,m)]
    decisions=[]

    def force(loads):
        original(loads)
        c.d.xfrc_applied[c.th]=expected[len(observed)]
        observed.append(c.d.xfrc_applied[c.th].copy())
        if np.any(c.d.qfrc_applied) or np.any(c.d.xfrc_applied[:c.th]) or np.any(c.d.xfrc_applied[c.th+1:]):
            raise RuntimeError("Unexpected force applied")

    c.apply_adhesion=force
    try:
        for tick in range(TOTAL):
            if tick%5==0:
                x=features(c,m)
                intent=int(500<=tick<10100)
                a=player(x,intent)
                decisions.append((x.copy(),intent,a.copy()))
            m=c.tick(a)
            trace.append(sample(c,m))
    finally:
        c.apply_adhesion=original
        c.d.xfrc_applied[:]=0.
    np.testing.assert_array_equal(observed,expected)
    ms={key:np.asarray(value) for key,value in zip(("time","qpos","qvel","com","loads","nonfoot",
        "padz","speed","foot","upright","target","ctrl"),zip(*trace),strict=True)}
    base=HERE/f"PROTO_{initial}"
    old_result=read(base/"RESULT.json")
    assert stand.sha(base/"TRACE.npz")==old_result["trace_sha256"]
    with np.load(base/"TRACE.npz",allow_pickle=False) as matrix:
        for key,z in ms.items():
            np.testing.assert_array_equal(z[:onset+1],matrix[key][:onset+1])
    path=root/"TRACE.npz"
    saved_decisions=root/"DECISIONS.npz"
    np.savez_compressed(path,**ms,actual_wrench=observed)
    np.savez_compressed(saved_decisions,**{k:np.asarray(v) for k,v in zip(("x","intent","action"),zip(*decisions),strict=True)})
    result=metrics(ms,500,WALK_MS,TOTAL)
    result.update(force_w=.20,duration_ms=20,onset_ms=onset,which=which,initial=initial,
        native_steps=TOTAL*80,settle_steps=80000,trace_sha256=stand.sha(path),
        decision_sha256=stand.sha(saved_decisions),physical_prefix_exact=True,control=str(base))
    stand.identity.verify_runtime(ident,c)
    stand.put(root/"RESULT.json",result)
    print(result["joint_sampled_pass"],result["walk"]["forward_signed_native_per_s"],
          result["walk"]["slip_ratio_max"],result["terminal_stand"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("which",choices=CASES)
    torch.set_num_threads(1)
    run(parser.parse_args().which)
