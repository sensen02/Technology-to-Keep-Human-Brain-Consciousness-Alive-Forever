"""Native body bench for a continuous stand, walk and stop experiment."""
import argparse
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJ = HERE.parents[2]
SNAPSHOT = HERE.parent / "20261115T203716Z"
sys.path.insert(0, str(SNAPSHOT))
from snapshots import load as load_initial, model as stand_model, stand
import student_wave as legacy
from static_teacher import StaticTeacher
from cycle_reward import CycleConfig, CycleTracker

STARTS = SNAPSHOT / "RUN/INITIAL"
STOP_SCRIPT = PROJ / "experiments/locomotion-enhancement/20261113T161918Z/stop_shadow_control/teacher.py"
spec = importlib.util.spec_from_file_location("fixed_shadow_stop_reference", STOP_SCRIPT)
hold = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hold)
LEGS = tuple(legacy.LEGS)
W = legacy.W if hasattr(legacy, "W") else stand.physics.W
SENSOR_BY_LEG = (0, 2, 1, 3, 5, 4)
INPUTS = 213


def read(path):
    return json.loads(path.read_text())


def features(c, m):
    proprio = c.sensory.features(c.shim, c.d, c.actuator.target_rad, m["loads"][c.physics_order])
    x = np.r_[proprio, m["foot"].ravel(), m["padz"], m["speed"]]
    assert x.shape == (INPUTS,) and np.isfinite(x).all()
    return x


def feature_from_old(x238, t):
    foot = t["relativefoot"]
    return np.column_stack((x238[:, :183], foot.reshape(-1, 18), t["padz"], t["speed"]))


def standing(ms, t0, end):
    if end-t0 < 200:
        return dict(passed=False)
    s = slice(end-199,end+1)
    rms = np.sqrt(np.square(np.diff(ms["com"][end-200:end+1,:2],axis=0)/.001/stand.LENGTH).sum(1).mean())
    w = ms["loads"][s]/W
    gates = dict(speed=bool(rms<=.01), foot=bool(w.sum(1).mean()>=.9),
                 at_least_three=bool(((w>=.02).sum(1)>=3).mean()>=.95),
                 posture=bool(ms["upright"][s].min() >= np.cos(np.deg2rad(20.))),
                 com=bool(ms["com"][s,2].min()>=1.2),
                 nonfoot=bool(ms["nonfoot"][s].max()<.01*W))
    return dict(passed=all(gates.values()), speed_rms=float(rms), gates=gates,
                com_min=float(ms["com"][s,2].min()), nonfoot_peak=float(ms["nonfoot"][s].max()))


def cycles(ms, start, end, *, arm_before=0):
    cfg = CycleConfig(timeout=4.)
    tracker = CycleTracker(cfg).reset(start_time=ms["time"][start-arm_before])
    events = []
    for i in range(start-arm_before+1,end+1):
        result = tracker.update(dict(time=ms["time"][i], loads=ms["loads"][i], padz=ms["padz"][i],
                                     relativefoot=ms["foot"][i]))
        events.extend(result["events"])
    counts = {leg:sum(e["type"]=="complete" and e["leg"]==leg for e in events) for leg in LEGS}
    return counts, events


def metrics(ms, warm, walk, whole):
    end = warm+walk
    g = min(end,len(ms["time"])-1)
    sample = [(ms["time"][i],ms["com"][i],ms["loads"][i],ms["nonfoot"][i],ms["speed"][i])
              for i in range(warm+1,g+1)]
    walking = stand.physics.summarise(sample,len(sample)*80) if len(sample)>1 else None
    counts, events = cycles(ms,warm,g,arm_before=20) if g>=warm else ({leg:0 for leg in LEGS},[])
    event0 = next((e for e in events if e["type"]=="complete"),None)
    walking_pass = walking is not None and walking["all_gates"] and all(counts.values()) and g==end
    initial = standing(ms,0,min(warm,len(ms["time"])-1))
    terminal = standing(ms,end,min(whole,len(ms["time"])-1))
    return dict(initial_stand=initial, terminal_stand=terminal, walk=walking, cycles=counts, first_step=event0,
                walking_pass=walking_pass, joint_sampled_pass=bool(walking_pass and initial["passed"]
                    and terminal["passed"] and len(ms["time"])==whole+1), actual_ms=len(ms["time"])-1)


class Teacher:
    def __init__(self,c,standing_policy):
        self.walk = None
        self.stop = None
        self.standing = standing_policy
        self.walk_ticks = 0
        self.stop_ticks = 0
        self.walk_model = c

    def action(self,c,x,intent):
        if intent:
            if self.walk is None:
                self.walk = StaticTeacher(c,legacy.teacher_config(4))
            result=self.walk.action(c,self.walk_ticks*.005)[0]
            self.walk_ticks+=1
            return result
        if self.walk is None:
            with torch.no_grad():
                return self.standing(torch.from_numpy(x[:183])).numpy()
        if self.stop is None:
            self.stop=hold.make_teacher(c,"stop",order_indices=self.walk.reference.order_indices)
        result=self.stop.action(c,self.stop_ticks*.005)[0]
        self.stop_ticks+=1
        return result


def rollout(output, start, walk_ms, policy=None, *, light=False, native=False):
    output.mkdir()
    bench=stand.Bench(output)
    c=bench.body
    c.physics_order=stand.physics.rc.SENS_IDX
    if not np.array_equal(c.physics_order,SENSOR_BY_LEG):
        raise RuntimeError("Current load order differs from named plant legs")
    bank=read(STARTS/"BANK.json")
    path=STARTS/f"{start}.npz"
    load_initial(bench,path,bank[start]["sha256"])
    ident=stand.identity.capture(c,dict(start=start,start_sha256=bank[start]["sha256"],scope="stand–walk–stop body bench"),
                                 extra_paths=[HERE/"PREREG.zh-CN.md",STOP_SCRIPT,path])
    stand.put(output/"RUN_IDENTITY.json",ident)
    teacher=Teacher(c,stand_model()) if policy is None else None
    aux = None
    if native:
        aux=stand.physics.NativeRecorder(output/"native")
        c.recorder=aux
    m=c.measure()
    warm=500
    end=warm+walk_ms
    final=end+1200
    start_time=time.monotonic()
    trace, decisions = [], []
    unsafe=0

    def save(m, a):
        trace.append(dict(time=float(c.d.time), qpos=c.d.qpos.copy(), qvel=c.d.qvel.copy(),
            com=c.d.subtree_com[c.th].copy(),loads=m["loads"].copy(),nonfoot=float(m["nonfoot"]),
            padz=m["padz"].copy(),speed=m["speed"].copy(),foot=m["foot"].copy(),upright=float(c.d.xmat[c.th,8]),
            target=c.actuator.target_rad.copy(),ctrl=c.d.ctrl.copy(),applied=a.copy()))

    save(m,bench.zero)
    try:
        for tick in range(final):
            if tick%5==0:
                intent = int(warm<=tick<end)
                x=features(c,m)
                action=teacher.action(c,x,intent) if teacher is not None else policy(x,intent)
                action=legacy.bounded_action(action)
                decisions.append((x.copy(), intent, action.copy(),tick))
            imposed=np.zeros(42)
            if light and warm+2000<=tick<warm+2020:
                imposed=np.random.default_rng(3011).normal(0.,.035,42)
            applied=np.clip(action+imposed,-1,1)
            m=c.tick(applied)
            if np.any(c.d.xfrc_applied) or np.any(c.d.qfrc_applied):
                raise RuntimeError("Unexpected external wrench")
            save(m,applied)
            unsafe=unsafe+1 if m["com"]<.8 or m["nonfoot"]>.5*W else 0
            if unsafe>=100:
                break
    finally:
        if aux is not None:
            aux.close()
    z={k:np.asarray([record[k] for record in trace]) for k in trace[0]}
    np.savez_compressed(output/"TRACE.npz",**z)
    np.savez_compressed(output/"DECISIONS.npz",**{k:np.array(v) for k,v in zip(("x","intent","action","ms"),zip(*decisions))})
    result=metrics(z,warm,walk_ms,final)
    stand.identity.verify_runtime(ident,c)
    result.update(start=start,control="teacher" if teacher else "model",light=light, walk_ms=walk_ms,
        elapsed=time.monotonic()-start_time, native_integrations=(len(trace)-1)*80, unsafe_stop=unsafe>=100,
        trace_sha256=stand.sha(output/"TRACE.npz"),decision_sha256=stand.sha(output/"DECISIONS.npz"),
        raw_native_contacts=bool(aux), full_D1_D8=False)
    stand.put(output/"RESULT.json",result)
    print(start,result["actual_ms"],result["walk"],result["cycles"],result["terminal_stand"],flush=True)
    return result


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("start")
    a.add_argument("output",type=Path)
    a.add_argument("--walk-ms",type=int,default=4800)
    a.add_argument("--checkpoint",type=Path)
    a.add_argument("--light",action="store_true")
    a.add_argument("--native",action="store_true")
    args=a.parse_args()
    torch.set_num_threads(1)
    policy=None
    if args.checkpoint:
        from fit_walk import load_policy
        policy=load_policy(args.checkpoint)
    rollout(args.output.resolve(),args.start,args.walk_ms,policy,light=args.light,native=args.native)


if __name__=="__main__":
    main()
