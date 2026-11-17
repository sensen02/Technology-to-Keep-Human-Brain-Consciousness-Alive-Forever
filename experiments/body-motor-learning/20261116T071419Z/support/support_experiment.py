"""Native paired geometry/actuator experiment for an initially unloaded helper."""
import argparse
import ctypes
from pathlib import Path
import sys

import mujoco as mj
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parents[1] / "20261116T063132Z" / "teacher"
sys.path.insert(0, str(PRIOR))
from physical import ARCHIVE, NATIVE_LIB, W, bench, build_scene, read, stan, terrain_masks, Record
from prototype import policy as frozen_policy

PER_MS = 80
STEP = .0000125


def group(i):
    return slice(i*7, (i+1)*7)


class Capture(Record):
    def __init__(self, body, duration, target, helper):
        super().__init__(body, duration)
        self.roles = np.ascontiguousarray(terrain_masks(body.plant), dtype=np.int32)
        self.pad_ids = np.asarray([body.pads[bench.LEGS[i]] for i in (target, helper)])
        self.radii = body.plant.geom_size[self.pad_ids,0].copy()
        self.pad_pose = np.zeros((duration*PER_MS,2,4))
        self.target_ground = np.zeros((duration*PER_MS,4))
        self.record = np.zeros(17)
        self.target = target
        lib = ctypes.CDLL(str(NATIVE_LIB))
        self.contact = lib.gap_totals
        self.contact.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int] + [ctypes.c_void_p]*2
        self.contact.restype = None

    def before(self, c):
        super().before(c)
        points = c.d.geom_xpos[self.pad_ids]
        self.pad_pose[self.count,:,:3] = points.mean(1)
        self.pad_pose[self.count,:,3] = np.min(points[:,:,2] - self.radii,axis=1)

    def solved(self, c, before):
        self.contact(c.plant._address,c.d._address,self.foot_map.ctypes.data,self.roles.ctypes.data,
                     self.target,self.velocity.ctypes.data,self.record.ctypes.data)
        self.forces[self.count] = self.record[:13]
        self.target_ground[self.count] = self.record[13:]
        if self.count%4031 == 0:
            original = stan.physics.NativeRecorder.solved(None,c,{"qvel_before":self.velocity})
            expected = np.r_[original["loads_native"],
                             original["loads_native"] * original["contact_speed_native_per_s"],
                             original["nonfoot_force_native"]]
            np.testing.assert_allclose(self.record[:13], expected, atol=1e-9, rtol=1e-12)
            self.cross_checks += 1


class Teacher:
    """Same state-triggered braking and replant; optional helper placement arm."""
    def __init__(self, core, plan, opening, base, reference, enabled):
        self.core,self.plan,self.opening,self.base = core,plan,opening,base
        self.ids = [bench.LEGS.index(plan[k]) for k in ("target","helper")]
        self.pads = [np.asarray(core.pads[bench.LEGS[i]]) for i in self.ids]
        self.jac = np.zeros((3,core.plant.nv))
        self.expected = reference
        self.enabled = enabled
        self.first = self.helper_first = self.replanted = None
        self.stage = 0

    def pose(self,index):
        return self.core.d.geom_xpos[self.pads[index]].mean(0).copy()

    def rate(self, i, goal, gain, cap):
        c = self.core
        j = group(self.ids[i])
        current = self.pose(i)
        geom = self.pads[i][0]
        mj.mj_jac(c.plant,c.d,self.jac,None,current,int(c.plant.geom_bodyid[geom]))
        jac = self.jac[:,c.a_dof[j]]
        speed = np.clip(gain*(goal-current),-cap,cap)
        return jac.T @ np.linalg.solve(jac @ jac.T + .010*np.eye(3),speed) / stan.physics.MAX_RATE

    def __call__(self, t, x, measured, nominal, intended):
        p = self.plan
        missing, available = self.ids
        if self.first is None:
            if not intended:
                return nominal
            site = self.pose(0)
            x0,x1,y0,y1 = self.opening
            if not (x0 < site[0] < x1 and y0 < site[1] < y1
                    and self.expected[t]>=p["expected_load_W"]
                    and measured["loads"][missing]<=p["target_unloaded_W"]*W
                    and measured["padz"][missing] < -p["below_rim_native"]):
                return nominal
            self.first = t
            self.target_anchor = site.copy()
            self.target_anchor[0] = min(self.target_anchor[0]-p["target_back_native"],x0-.020)
            self.target_anchor[2] = p["target_stance_z"]
            self.helper_anchor = self.pose(1)
            self.helper_anchor[2] = p["helper_stance_z"]
            self.available_at_trigger = bool(measured["loads"][available] == 0 and measured["padz"][available] > 0.)
        age = t-self.first
        act = self.base(x,0).copy()
        if self.replanted is None or t-self.replanted <= p["postcontact_ms"]:
            lost = self.pose(0)
            if self.stage == 0 and measured["padz"][missing] > .003:
                self.stage = 1
            if self.stage == 1 and np.linalg.norm(lost[:2]-self.target_anchor[:2]) < .018:
                self.stage = 2
            if self.stage == 2 and measured["loads"][missing] > .03*W and self.replanted is None:
                self.replanted = t
            if self.stage == 0:
                aim = np.r_[lost[:2],p["target_lift_z"]]
            else:
                aim = self.target_anchor.copy()
                if self.stage == 1:
                    aim[2] = p["target_lift_z"]
            act[group(missing)] = self.rate(0,aim,p["target_gain"],.65)
        if self.enabled and self.available_at_trigger and age < p["helper_hold_ms"]:
            if self.helper_first is None:
                self.helper_first = t
            aim = self.helper_anchor
            speed = self.rate(1,aim,p["helper_gain"],.74)
            fade = min(1.,(p["helper_hold_ms"]-age)/70.)
            target_slice = group(available)
            act[target_slice] = (1.-fade)*act[target_slice]+fade*speed
        return np.clip(act,-1.,1.)


def run(mode, output, case, plan_file, reference):
    output.mkdir()
    p = read(plan_file)
    shape = read(case)
    duration = p["duration_ms"]
    torch.set_num_threads(1)
    missing, helper = [bench.LEGS.index(p[k]) for k in ("target","helper")]
    initial = bench.STARTS / "nominal.npz"
    initial_sha = read(bench.STARTS/"BANK.json")["nominal"]["sha256"]
    core, error = build_scene(shape,initial,initial_sha)
    with np.load(reference,allow_pickle=False) as z:
        expected = z["loads"][:,missing].copy()/W
    base = frozen_policy()
    law = Teacher(core,p,shape["opening"],base,expected,mode=="on") if mode in ("on","off") else None
    evidence = [Path(__file__),case,plan_file,reference,initial,NATIVE_LIB,PRIOR/"floor_contacts.cpp",
                PRIOR/"physical.py",PRIOR/"geometry.py",ARCHIVE/"RESULT.json"]
    config = dict(scope="native previously-unloaded helper placement",mode=mode,scene=shape,plan=p,
                  initial_sha256=initial_sha,initial_observation_error=error)
    identity = stan.identity.capture(core,config,extra_paths=evidence)
    stan.put(output/"RUN_IDENTITY.json",identity)
    rec = Capture(core,duration,missing,helper)
    core.recorder = rec
    trace, decisions = [], []
    zeros = np.zeros(42)
    desired = zeros.copy()
    correction = zeros.copy()
    m = core.measure()

    def snap():
        points = core.d.geom_xpos[rec.pad_ids]
        return dict(time=float(core.d.time),qpos=core.d.qpos.copy(),qvel=core.d.qvel.copy(),
                    com=core.d.subtree_com[core.th].copy(),loads=m["loads"].copy(),nonfoot=float(m["nonfoot"]),
                    padz=m["padz"].copy(),speed=m["speed"].copy(),foot=m["foot"].copy(),
                    upright=float(core.d.xmat[core.th,8]),target=core.actuator.target_rad.copy(),
                    ctrl=core.d.ctrl.copy(),applied=desired.copy(),correction=correction.copy(),
                    pair_pad_xyz=points.mean(1))
    trace.append(snap())
    unsafe = 0
    for t in range(duration):
        if t%5 == 0:
            x = bench.features(core,m)
            intent = int(t>=500)
            nominal = base(x,intent)
            desired = law(t,x,m,nominal,intent) if law else nominal.copy()
            desired = bench.legacy.bounded_action(desired)
            correction = desired-nominal
            decisions.append((t,intent,x.copy(),nominal.copy(),desired.copy(),expected[t]))
        m = core.tick(desired)
        if np.any(core.d.xfrc_applied) or np.any(core.d.qfrc_applied):
            raise RuntimeError("Unexpected applied force")
        trace.append(snap())
        unsafe = unsafe+1 if m["com"]<.8 or m["nonfoot"]>.5*W else 0
        if unsafe>=100:
            break
    results = {k:np.asarray([row[k] for row in trace]) for k in trace[0]}
    count = rec.count
    assert count==(len(trace)-1)*PER_MS
    np.testing.assert_array_equal(rec.endpoints[:len(trace)],results["qpos"])
    assert not rec.poses[:count,7].any()
    np.testing.assert_allclose(rec.poses[:count,6],results["time"][0]+np.arange(count)*STEP,rtol=0.,atol=5e-8)
    np.savez_compressed(output/"TRACE.npz",**results)
    np.savez_compressed(output/"DECISIONS.npz",**{k:np.asarray(v) for k,v in zip(
        ("ms","intent","x","base","action","expected_target_W"),zip(*decisions,strict=True),strict=True)})
    np.savez_compressed(output/"NATIVE.npz",forces=rec.forces[:count],pose=rec.poses[:count],
                        pair_pad=rec.pad_pose[:count],missing_floor=rec.target_ground[:count])
    final=len(trace)-1
    terminal=bench.standing(results,max(0,final-600),final)
    stan.identity.verify_runtime(identity,core)
    out=dict(mode=mode,scenario=shape,initial_snapshot_sha256=initial_sha,initial_observation_error=error,
             native_count=count,contact_cross_checks=rec.cross_checks,actual_ms=final,terminal_stand=terminal,
             target=p["target"],helper=p["helper"],first_trigger_ms=law.first if law else None,
             first_helper_ms=law.helper_first if law else None,replant_detection_ms=law.replanted if law else None,
             observed_helper_available_at_trigger=getattr(law,"available_at_trigger",None),
             reference=str(reference),trace_sha256=stan.sha(output/"TRACE.npz"),
             decision_sha256=stan.sha(output/"DECISIONS.npz"),native_sha256=stan.sha(output/"NATIVE.npz"))
    stan.put(output/"RESULT.json",out)
    print(out,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",choices=("control","on","off"))
    parser.add_argument("output",type=Path)
    parser.add_argument("--case",type=Path,default=HERE/"CASE_A.json")
    parser.add_argument("--plan",type=Path,default=HERE/"PLAN_RH_LF.json")
    parser.add_argument("--reference",type=Path,default=ARCHIVE/"TRACE.npz")
    a=parser.parse_args()
    run(a.mode,a.output.resolve(),a.case.resolve(),a.plan.resolve(),a.reference.resolve())
