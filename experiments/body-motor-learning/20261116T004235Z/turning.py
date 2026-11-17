"""Continuous turn-in-place and walking-arc trials for the frozen fly body."""
import argparse
import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "20261116T000536Z"
sys.path.insert(0, str(PARENT))
from prototype import policy as straight_policy
from walking_bench import STARTS, PROJ, LEGS, W, StaticTeacher, features, load_initial, read, stand, standing

PIVOT_ROOT = PROJ / "experiments/locomotion-enhancement/20261113T161918Z"
TEACHER = PIVOT_ROOT / "pivot_fixed_shadow_v2/teacher.py"
spec = importlib.util.spec_from_file_location("frozen_turn_teacher", TEACHER)
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
sys.path.insert(0, str(PIVOT_ROOT/"pivot_teacher_design"))
from common import PivotCycles

MODES = ("pivot_left", "pivot_right", "arc_left", "arc_right")
WARM = 500
TURN = 9600
COAST = 4800
STOP = 1200


class ArcReference(fixed.old.PivotReference):
    """Private desired body arc. Feet are anchored in the world during support."""
    def __init__(self, c, teacher, condition, order, speed, omega):
        super().__init__(c, teacher, condition, order)
        self.speed = speed
        self.omega = np.copysign(omega,self.omega)
        h=self.rot0[:2,0]
        self.heading=h/np.linalg.norm(h)
        self.left=np.array([-self.heading[1],self.heading[0]])

    def center(self,t):
        theta=self.omega*fixed.old.ramp_integral(t)
        xy=(self.speed/self.omega)*(np.sin(theta)*self.heading+(1.-np.cos(theta))*self.left)
        return self.origin+np.r_[xy,0.]

    def sample(self,t):
        if self.last_t is not None and not np.isclose(t-self.last_t,.005,atol=1e-9,rtol=0):
            raise ValueError("Teacher time discontinuity")
        self.last_t=t
        slot,leg,u=fixed.old.schedule(t,self.order,self.step_s)
        if slot!=self.slot:
            if self.slot>=0:
                self.anchors[self.order[self.slot%6]]=self.swing_end
            self.slot=slot
            self.swing_start=self.anchors[leg].copy()
            lead=(slot+1)*self.step_s+2.
            self.swing_end=self.center(lead)+self.rotation(lead)@self.rest[leg]
        world=self.anchors.copy()
        s=fixed.old.blend(u)
        world[leg]=(1.-s)*self.swing_start+s*self.swing_end
        heights=np.zeros(6)
        stance=np.ones(6,dtype=bool)
        heights[leg]=self.lift*16.*u*u*(1.-u)*(1.-u)
        stance[leg]=False
        desired=self.center(t)
        desired_rot=self.rotation(t)
        self.last_snapshot=dict(t=t,active_leg=leg,desired_origin=desired.tolist(),yaw=self.omega*fixed.old.ramp_integral(t))
        return dict(target_xy=((world-desired)@desired_rot)[:,:2],target_z=heights,stance=stance,
                    source_time_s=None,phase="arc")


def make_teacher(c,mode,speed,omega):
    condition=("left" if mode.endswith("left") else "right")+"_pivot"
    probe=StaticTeacher(c)
    order=probe.reference.order_indices
    teacher=fixed.make_teacher(c,condition,order_indices=order)
    if mode.startswith("arc"):
        teacher.reference=ArcReference(c,teacher,condition,order,speed,omega)
    return teacher


def sample(c,m):
    return dict(time=float(c.d.time),qpos=c.d.qpos.copy(),qvel=c.d.qvel.copy(),com=c.d.subtree_com[c.th].copy(),
        loads=m["loads"].copy(),nonfoot=float(m["nonfoot"]),padz=m["padz"].copy(),speed=m["speed"].copy(),
        foot=m["foot"].copy(),upright=float(c.d.xmat[c.th,8]),yaw=float(np.arctan2(c.d.xmat[c.th,3],c.d.xmat[c.th,0])),
        target=c.actuator.target_rad.copy(),ctrl=c.d.ctrl.copy())


def score(ms,mode,bounds):
    begin,end,after,finish=bounds
    actual=min(end,len(ms["time"])-1)
    if actual<=begin:
        return dict(turn_gates={},actual_ms=len(ms["time"])-1,function_pass=False,low_slip_pass=False,joint_pass=False)
    yaw=np.unwrap(ms["yaw"])
    heading=np.c_[np.cos(yaw),np.sin(yaw)]
    delta=ms["com"][begin+1:actual+1,:2]-ms["com"][begin:actual,:2]
    local=(delta*heading[begin:actual]).sum(1)
    window=slice(begin+1,actual+1)
    yaw_change=float(np.rad2deg(yaw[actual]-yaw[begin]))
    signed=yaw_change*(1 if mode.endswith("left") else -1)
    drift=float(np.linalg.norm(ms["com"][window,:2]-ms["com"][begin,:2],axis=1).max()/stand.LENGTH)
    speed=float(local.sum()/((actual-begin)*.001))
    wt=ms["loads"][window]
    slip=(wt*ms["speed"][window]).sum(0)/np.maximum(wt.sum(0),1e-12)
    pivot=PivotCycles(W)
    for i in range(begin-10,actual+1):
        pivot.update(ms["time"][i],ms["loads"][i],ms["padz"][i])
    pivoting=mode.startswith("pivot")
    gates=dict(duration=actual==end, support=float(wt.sum(1).mean()/W)>=.9,
               com=float(ms["com"][window,2].min())>=1.2,nonfoot=float(ms["nonfoot"][window].max())==0.,
               turn=signed>=(45. if pivoting else 30.),six_cycles=bool(all(pivot.counts)))
    gates.update(drift=drift<=.5) if pivoting else gates.update(forward=speed>=.05)
    ratio=float((slip/speed).max()) if speed>1.e-8 and not pivoting else None
    low=bool(pivoting or ratio is not None and ratio<=.5)
    standing_before=standing(ms,0,min(WARM,len(ms["time"])-1))
    terminal=standing(ms,after,min(finish,len(ms["time"])-1))
    continued=None
    if after>end and len(ms["time"])>after:
        u=ms["com"][end+1:after+1,:2]-ms["com"][end:after,:2]
        h=heading[end:after]
        resumed_speed=float(np.sum(u*h)/((after-end)*.001))
        resume_nf=float(ms["nonfoot"][end+1:after+1].max())
        resume_com=float(ms["com"][end+1:after+1,2].min())
        continued=dict(forward_body=resumed_speed,nonfoot=resume_nf,minimum_com=resume_com,
                       passed=resumed_speed>=.05 and resume_nf==0. and resume_com>=1.2)
    complete=len(ms["time"])==finish+1
    function=bool(all(gates.values()))
    return dict(turn_gates=gates,actual_ms=len(ms["time"])-1,function_pass=function,low_slip_pass=low,
                joint_pass=bool(function and low and standing_before["passed"] and terminal["passed"]
                                and (after==end or continued is not None and continued["passed"]) and complete),
                signed_turn_deg=signed,actual_turn_deg=yaw_change,drift_body_lengths=drift,forward_body=speed,
                path_length_native=float(np.linalg.norm(delta,axis=1).sum()),
                abs_slip_native_s=slip.tolist(),arc_slip_ratio=ratio, cycles=[int(k) for k in pivot.counts],
                tail=terminal,initial_stand=standing_before,continue_walk=continued,
                turn_com_min=float(ms["com"][window,2].min()),turn_nonfoot_peak=float(ms["nonfoot"][window].max()),
                support_over_weight=float(wt.sum(1).mean()/W))


def run(output,start,mode,kind,speed,omega,entry,exit_walk):
    output.mkdir()
    bench=stand.Bench(output)
    c=bench.body
    c.physics_order=stand.physics.rc.SENS_IDX
    bank=read(STARTS/"BANK.json")
    snapshot=STARTS/f"{start}.npz"
    load_initial(bench,snapshot,bank[start]["sha256"])
    straight=straight_policy()
    controller=load_proto(mode) if kind=="model" else None
    provenance=[HERE/"PREREG.zh-CN.md",snapshot,TEACHER,PARENT/"PROTOTYPES.npz",
                PARENT/"FIT_INITIAL/step_1200.pt"]
    if kind=="model":
        provenance.extend((HERE/f"BANK_{mode}.npz",HERE/"BUILD.json"))
    ident=stand.identity.capture(c,dict(kind=kind,mode=mode,start=start,speed=speed,omega=omega,
        entry_ms=entry,exit_walk_ms=exit_walk),extra_paths=provenance)
    stand.put(output/"TURN_IDENTITY.json",ident)
    begin=WARM+entry
    end=begin+TURN
    after=end+exit_walk
    finish=after+STOP
    m=c.measure()
    rows=[sample(c,m)]
    events=[]
    teacher=None
    unsafe=0
    clock=time.monotonic()
    for tick in range(finish):
        if tick%5==0:
            x=features(c,m)
            if begin<=tick<end:
                if kind=="teacher":
                    if teacher is None:
                        teacher=make_teacher(c,mode,speed,omega)
                    a=teacher.action(c,(tick-begin)*.001)[0]
                else:
                    a=controller(x)
                command=2
            else:
                a=straight(x,int(WARM<=tick<after))
                command=int(WARM<=tick<after)
            if not np.isfinite(a).all() or np.max(np.abs(a))>1.:
                raise ValueError("Invalid bounded command")
            events.append((tick,command,x.copy(),a.copy()))
        m=c.tick(a)
        if np.any(c.d.xfrc_applied) or np.any(c.d.qfrc_applied):
            raise ValueError("External wrench")
        rows.append(sample(c,m))
        unsafe=unsafe+1 if m["com"]<.80 or m["nonfoot"]>.5*W else 0
        if unsafe>=100:
            break
    ms={k:np.asarray([v[k] for v in rows]) for k in rows[0]}
    decisions={k:np.asarray(v) for k,v in zip(("ms","command","x","action"),zip(*events),strict=True)}
    trace=output/"TRACE.npz"
    d=output/"DECISIONS.npz"
    np.savez_compressed(trace,**ms)
    np.savez_compressed(d,**decisions)
    result=score(ms,mode,(begin,end,after,finish))
    result.update(mode=mode,kind=kind,start=start,bounds=[begin,end,after,finish],nsteps=(len(rows)-1)*80,
                  settle_steps=80000,teacher_speed=speed,teacher_omega=omega,
                  trace_sha256=stand.sha(trace),decisions_sha256=stand.sha(d),wall=time.monotonic()-clock)
    stand.identity.verify_runtime(ident,c)
    stand.put(output/"RESULT.json",result)
    print(result,flush=True)
    return result


def build(paths):
    entries={}
    body=stand.physics.Core(adhesion=False)
    half=np.maximum(np.ptp(body.actuator.target_bounds_rad,axis=1)*.5,.1)
    for p in paths:
        result=read(p/"RESULT.json")
        mode=result["mode"]
        if result["kind"]!="teacher" or not result["function_pass"]:
            raise ValueError(f"Teacher function gates not met: {p}")
        f=p/"DECISIONS.npz"
        digest=stand.sha(f)
        assert digest==result["decisions_sha256"]
        with np.load(f,allow_pickle=False) as z:
            select=z["command"]==2
            x=z["x"][select]
            a=z["action"][select]
        target=HERE/f"BANK_{mode}.npz"
        with target.open("xb") as fp:
            np.savez_compressed(fp,target=x[:,126:168],angle=x[:,:42],a=a,half=half)
        entries[mode]=dict(samples=len(a),sha=stand.sha(target),source=str(p.resolve()),decisions_sha=digest,
                          result_sha=stand.sha(p/"RESULT.json"))
    stand.put(HERE/"BUILD.json",entries)


def load_proto(mode):
    p=HERE/f"BANK_{mode}.npz"
    identity=read(HERE/"BUILD.json")[mode]
    if stand.sha(p)!=identity["sha"]:
        raise ValueError("Prototype identity changed")
    with np.load(p,allow_pickle=False) as z:
        target,angle,action,half=(z[k] for k in ("target","angle","a","half"))

    def evaluate(observation):
        d=observation[126:168]-target
        position=observation[:42]-angle
        i=int((np.square(d).sum(1)+.001*np.square(position).sum(1)).argmin())
        return np.clip(action[i]-45*half*d[i],-1.,1.)

    return evaluate


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--build",nargs="*",type=Path)
    p.add_argument("--out",type=Path)
    p.add_argument("--start",default="nominal")
    p.add_argument("--mode",choices=MODES)
    p.add_argument("--kind",choices=("teacher","model"),default="teacher")
    p.add_argument("--speed",type=float,default=.13)
    p.add_argument("--omega",type=float,default=.115)
    p.add_argument("--entry",type=int)
    p.add_argument("--exit-walk",type=int,default=0)
    a=p.parse_args()
    torch.set_num_threads(1)
    if a.build is not None:
        build(a.build)
    else:
        entry=(4800 if a.mode.startswith("arc") else 0) if a.entry is None else a.entry
        run(a.out.resolve(),a.start,a.mode,a.kind,a.speed,a.omega,entry,a.exit_walk)
