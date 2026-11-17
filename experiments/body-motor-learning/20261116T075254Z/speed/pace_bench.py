"""Native replay of a physically paced frozen fruit fly, with common teacher labels."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

from speed_teacher import HERE, NOMINAL_V, REFERENCE, SOURCE, PhaseFollower, WARM, stock_policy

sys.path.insert(0,str(HERE.parents[1]/"20261116T004235Z"))
from verify_native import Record
import walking_bench as bench

STOP = 1200
PER_MS = 80
DT = .0000125
W = bench.W
stan = bench.stand


def profile_value(spec, t):
    value = 0.
    knots = spec["pace"]
    if 0<=t<spec["walk_ms"]:
        before = np.asarray(knots,dtype=float)
        segment = int(np.searchsorted(before[:,0],t,side="right"))-1
        if segment>=len(before)-1:
            value = before[-1,1]
        else:
            a,b = before[segment:segment+2]
            u = (t-a[0])/(b[0]-a[0])
            blend = u*u*(3.-2.*u)
            value = a[1]+(b[1]-a[1])*blend
    return NOMINAL_V*value


def raw_window(f,p,z,a,b):
    f,p = f[a*PER_MS:b*PER_MS],p[a*PER_MS:b*PER_MS]
    support = f[:,:6].sum(1)/W
    totals = f[:,:6].sum(0)
    walk_dist = z["com"][b,:2]-z["com"][a,:2]
    seconds =(b-a)*.001
    speed = np.linalg.norm(walk_dist)/seconds
    foot_slip = f[:,6:12].sum(0)/np.maximum(totals,1e-10)
    return dict(signed_forward_native_s=float(walk_dist[0]/seconds),
                lateral_native_s=float(walk_dist[1]/seconds),
                ground_foot_support_mean_W=float(support.mean()),support_min_W=float(support.min()),
                minimum_com=float(min(p[:,2].min(),z["com"][b,2])),
                ground_nonfoot_peak_W=float(f[:,12].max()/W),
                max_ground_normalized_slip=float(foot_slip.max()/speed) if speed>0 else None,
                valid_loaded_legs=[n for i,n in enumerate(bench.LEGS) if totals[i]>1e-10],
                absolute_slide_native_s=dict(zip(bench.LEGS,foot_slip.tolist(),strict=True)))


def standing_window(f,p,z,a,b):
    active = f[a*PER_MS:b*PER_MS]
    pose = p[a*PER_MS:b*PER_MS]
    support = active[:,:6]/W
    sampled=bench.standing(z,a,b)
    gates = dict(sampled_stand=sampled["passed"],native_support=float(support.sum(1).mean())>=.90,
                 native_three_leg=float(((support>=.02).sum(1)>=3).mean())>=.95,
                 native_posture=float(pose[:,5].min())>=np.cos(np.deg2rad(20.)),
                 native_min_com=float(pose[:,2].min())>=1.20,native_nonfoot=float(active[:,12].max()/W)<.01)
    gates={key:bool(value) for key,value in gates.items()}
    return dict(passed=all(gates.values()),gates=gates,mean_support_W=float(support.sum(1).mean()),
                speed_rms=sampled.get("speed_rms"),minimum_com=float(pose[:,2].min()),
                nonfoot_peak_W=float(active[:,12].max()/W))


def roll(output, plan):
    output.mkdir()
    cfg=bench.read(plan)
    ms=int(cfg["walk_ms"])
    total = WARM+ms+STOP
    initial = bench.STARTS/"nominal.npz"
    base_sha=bench.read(bench.STARTS/"BANK.json")["nominal"]["sha256"]
    fixture=stan.Bench(output)
    bench.load_initial(fixture,initial,base_sha)
    core=fixture.body
    core.physics_order=stan.physics.rc.SENS_IDX
    np.testing.assert_array_equal(core.physics_order,bench.SENSOR_BY_LEG)
    if cfg.get("stock"):
        stock=stock_policy()
        controller=lambda x,cmd:stock(x,int(cmd[0]!=0.))
    else:
        controller=PhaseFollower(core)
    metadata = dict(scope="clocked physical target pace teacher" if not cfg.get("stock") else "frozen 0/1 control",
                    fields=["requested forward native length/s", "requested yaw in rad/s"],
                    desired_nominal_v=NOMINAL_V,initial_sha256=base_sha,plan=cfg)
    sources=[Path(__file__),HERE/"speed_teacher.py",plan,REFERENCE,initial,SOURCE/"prototype.py",
             SOURCE/"PROTOTYPES.npz",bench.STOP_SCRIPT,
             HERE.parents[1]/"20261116T004235Z"/"native_contact.so"]
    ident=stan.identity.capture(core,metadata,extra_paths=sources)
    stan.put(output/"RUN_IDENTITY.json",ident)
    record=Record(core,total)
    core.recorder=record
    m=core.measure()
    trace,decisions=[],[]
    a=np.zeros(42)
    command=np.zeros(2)

    def snapshot():
        d=core.d
        trace.append(dict(time=float(d.time),qpos=d.qpos.copy(),qvel=d.qvel.copy(),com=d.subtree_com[core.th].copy(),
                          loads=m["loads"].copy(),nonfoot=float(m["nonfoot"]),padz=m["padz"].copy(),
                          speed=m["speed"].copy(),foot=m["foot"].copy(),upright=float(d.xmat[core.th,8]),
                          target=core.actuator.target_rad.copy(),ctrl=d.ctrl.copy(),applied=a.copy(),command=command.copy()))

    snapshot()
    for tick in range(total):
        if tick%5==0:
            command=np.array([profile_value(cfg,tick-WARM),0.])
            x=bench.features(core,m)
            a=bench.legacy.bounded_action(controller(x,command))
            decisions.append((tick,x,command.copy(),a.copy()))
        m=core.tick(a)
        if np.any(core.d.xfrc_applied) or np.any(core.d.qfrc_applied):
            raise RuntimeError("Frozen body applied force changed")
        snapshot()
    n=record.count
    assert n==80*total and not record.poses[:,7].any()
    z={name:np.asarray([step[name] for step in trace]) for name in trace[0]}
    np.testing.assert_array_equal(record.endpoints,z["qpos"])
    np.testing.assert_allclose(record.poses[:,6],z["time"][0]+np.arange(n)*DT,rtol=0.,atol=5.e-8)
    np.savez_compressed(output/"TRACE.npz",**z)
    np.savez_compressed(output/"DECISIONS.npz",**{key:np.asarray(values) for key,values in zip(
        ("ms","x","command","action"),zip(*decisions,strict=True),strict=True)})
    np.savez_compressed(output/"NATIVE.npz",forces=record.forces,pose=record.poses)
    counts,_=bench.cycles(z,WARM,WARM+ms,arm_before=20)
    w=raw_window(record.forces,record.poses,z,WARM,WARM+ms)
    initial_stand=standing_window(record.forces,record.poses,z,WARM-200,WARM)
    terminal=standing_window(record.forces,record.poses,z,total-200,total)
    gates=dict(forward=w["signed_forward_native_s"]>=.05,support=w["ground_foot_support_mean_W"]>=.9,
               nonfoot=w["ground_nonfoot_peak_W"]<=.2,com=w["minimum_com"]>=1.2,slip=w["max_ground_normalized_slip"]<=.50,
               cycles=all(counts.values()),initial=initial_stand["passed"],terminal=terminal["passed"])
    windows={}
    for a1,b1 in cfg.get("observe_ms",[]):
        windows[f"{a1}-{b1}"]=raw_window(record.forces,record.poses,z,WARM+a1,WARM+b1)
    source_hashes={name:stan.sha(output/name) for name in ("TRACE.npz","DECISIONS.npz","NATIVE.npz")}
    stan.identity.verify_runtime(ident,core)
    result=dict(config=cfg,actual_ms=total,walk_ms=ms,native_steps=n,native_integrations=n,
                native_cross_checks=record.cross_checks,zero_extra_force=True,native_joint_pass=all(gates.values()),
                native_gates=gates,gait_cycles=counts,initial_stand=initial_stand,terminal_stand=terminal,
                walk=w,windows=windows,trace_sha256=source_hashes["TRACE.npz"],
                decision_sha256=source_hashes["DECISIONS.npz"],native_sha256=source_hashes["NATIVE.npz"],
                physical_command_fields=metadata["fields"],scope=metadata["scope"])
    stan.put(output/"RESULT.json",result)
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    p.add_argument("plan",type=Path)
    a=p.parse_args()
    torch.set_num_threads(1)
    roll(a.output.resolve(),a.plan.resolve())
