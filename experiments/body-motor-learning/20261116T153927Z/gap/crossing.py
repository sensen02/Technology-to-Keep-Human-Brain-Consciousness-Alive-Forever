"""Paired shallow slot with current-state contact trigger and direct far-rim replant."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parents[1] / "20261116T063132Z" / "teacher"
RESTART = HERE.parents[1] / "20261116T071419Z" / "restart"
sys.path[:0] = [str(PRIOR), str(RESTART)]
import physical
from physical import RF, RM, FloorRecord, Local, bench, build_scene, read, stan, W
from gap_weight import Weight

SOURCE = PRIOR / "AS.json"
FILLED = PRIOR / "AS_FILLED.json"
PLAN = PRIOR / "PLAN_BRAKE_RF_ONLY.json"
BASIS = RESTART / "RESTART_ACCEPTANCE.json"
FIRST_WALK = 500
RELEASE = 6450
RESUMED_WALK = 9600
TERMINAL = 1200
END = RELEASE + RESUMED_WALK + TERMINAL
PER_MS = 80


class TerrainReflex(Weight):
    """Geometry-aware physical teacher. Deployment observations remain separate."""

    def __init__(self, sensor, plan, shape, mode, reach, gain, retain):
        super().__init__(sensor, plan)
        self.mode = mode
        self.shape = shape
        self.reach = reach
        self.p["gain"] = gain
        self.p["postcontact_ms"] = retain
        self.edges = []
        self.rf_enabled = mode in ("far", "near")
        self.radii = sensor.body.plant.geom_size[self.pads[RF], 0]

    def cross_ready(self):
        c = self.s.body
        pad_left = np.min(c.d.geom_xpos[self.pads[RF], 0] - self.radii)
        return pad_left > self.shape["opening"][1] + .002

    def __call__(self, t, x, intent, local, base, measure):
        if not intent:
            return base
        if self.onset is None:
            if not (local[0] >= .004 and local[1] <= .004 and measure["padz"][RF] < -.012):
                return base
            if self.mode in ("off", "flat"):
                return base
            self.edges.append((t, "support_loss"))
            self.onset = t
            self.anchor = self.position(RF).copy()
            if self.mode == "far":
                self.anchor[0] = self.shape["opening"][1] + self.reach
            else:
                self.anchor[0] -= self.p["back_native"]
            self.anchor[2] = self.p["stance_z"]
            self.rm_anchor = self.position(RM).copy()
            # Initialize Weight's physically bounded RM preload in the same pose.
            self.onset = None
            if self.mode == "near":
                return super().__call__(t, x, intent, local, base, measure)
            far_anchor = self.anchor
            saved = super().__call__(t, x, intent, local, base, measure)
            self.anchor = far_anchor
            if self.rf_enabled:
                return saved
            saved[physical.RF_JOINTS] = self.base(x, 0)[physical.RF_JOINTS]
            return saved
        if self.mode == "near":
            return super().__call__(t, x, intent, local, base, measure)
        action = self.base(x, 0).copy()
        if self.rf_enabled and (self.reload_seen is None or t - self.reload_seen <= self.p["postcontact_ms"]):
            current = self.position(RF)
            if self.stage == 0 and measure["padz"][RF] > .008:
                self.stage = 1
                self.edges.append((t, "retracted"))
            if self.stage == 1 and self.cross_ready():
                self.stage = 2
                self.edges.append((t, "far_rim_cleared"))
            if self.stage == 2 and measure["loads"][RF] > .03*W and self.reload_seen is None:
                self.reload_seen = t
                self.edges.append((t, "reloaded"))
            goal = self.anchor.copy()
            if self.stage == 0:
                goal = np.r_[current[:2], self.p["lift_z"]]
            if self.stage == 1:
                goal[2] = self.p["lift_z"]
            action[physical.RF_JOINTS] = self.joint_rate(RF, current, goal, cap=.65)
        age = t - self.onset
        if age < 460:
            from gap_weight import smooth
            c = self.s.body
            weight = smooth((age+5)/95.) * (1. - smooth((age+5-280.)/180.))
            joint = self.helper_target + self.helper_offset*weight
            rate = (joint-c.actuator.target_rad[physical.RM_JOINTS])/(.005*stan.physics.MAX_RATE)
            action[physical.RM_JOINTS] = np.clip(rate,-1.,1.)
        return np.clip(action, -1., 1.)


class Capture(FloorRecord):
    def __init__(self, c, duration):
        super().__init__(c, duration)
        self.rf_spatial = np.zeros((PER_MS * duration, 5))
        self.all_pads = np.asarray([c.pads[leg] for leg in bench.LEGS])

    def before(self, c):
        super().before(c)
        balls = c.d.geom_xpos[self.pad_ids]
        point = balls.mean(0)
        self.rf_spatial[self.count] = (*point, (balls[:,0]-self.pad_radii).min(),
                                      (balls[:,0]+self.pad_radii).max())


TRACE_KEYS = ("time","qpos","qvel","com","loads","nonfoot","padz","speed","foot","upright",
              "target","ctrl","applied","correction","rf_pad_xyz","all_pad_xyz","requested")

def run(output, mode, reach, gain, retain, duration, native, keep_progressing=False, shape_path=None,
        branch_factory=TerrainReflex):
    torch.set_num_threads(1)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir()
    clock = time.monotonic()
    shape_path = shape_path or (FILLED if mode == "flat" else SOURCE)
    shape = read(shape_path)
    initial = bench.STARTS / "nominal.npz"
    initial_sha = read(bench.STARTS / "BANK.json")["nominal"]["sha256"]
    c, delta = build_scene(shape, initial, initial_sha)
    sensor = Local(c, RESUMED_WALK)
    policy = branch_factory(sensor, read(PLAN), shape, mode, reach, gain, retain)
    base = physical.frozen_policy()
    rec = Capture(c, duration) if native else None
    c.recorder = rec
    all_pads = np.asarray([c.pads[leg] for leg in bench.LEGS])
    sources = [Path(__file__), shape_path, PLAN, BASIS, initial, PRIOR/"physical.py",
               PRIOR/"gap_stop.py", PRIOR/"gap_weight.py", PRIOR/"geometry.py",
               physical.BASE/"prototype.py", physical.BASE/"walking_bench.py",
               physical.ARCHIVE/"TRACE.npz", PRIOR/"floor_contacts.cpp", physical.NATIVE_LIB]
    see = stan.identity.capture(c, dict(scope="single RF-leg lane slot", source=str(shape_path),
                    initial_snapshot_sha256=initial_sha, initial_observation_max_abs_delta=delta,
                    mode=mode, reach=reach, gain=gain, retain=retain, bench_renewal_ms=RELEASE,
                    keep_progressing=keep_progressing), extra_paths=sources)
    stan.put(output/"RUN_IDENTITY.json", see)
    rows = []
    decisions = []
    m = c.measure()
    a = np.zeros(42)
    correction = a.copy()
    request = np.zeros(2)
    def snapshot():
        pads = c.d.geom_xpos[all_pads].mean(1)
        rows.append((float(c.d.time), c.d.qpos.copy(), c.d.qvel.copy(), c.d.subtree_com[c.th].copy(),
            m["loads"].copy(),float(m["nonfoot"]),m["padz"].copy(),m["speed"].copy(),m["foot"].copy(),
            float(c.d.xmat[c.th,8]),c.actuator.target_rad.copy(),c.d.ctrl.copy(),a.copy(),correction.copy(),
            pads[RF].copy(),pads.copy(),request.copy()))
    snapshot()
    unsafe = 0
    for t in range(duration):
        if t % 5 == 0:
            request = np.array([.071713 if FIRST_WALK <= t < RELEASE+RESUMED_WALK else 0., 0.])
            intended = int(request[0] != 0.)
            local = sensor.sense(m)
            x = bench.features(c,m)
            b = np.asarray(base(x,intended), dtype=np.float64)
            a = policy(t, x, intended, local, b, m) if t < RELEASE else b.copy()
            if keep_progressing and t < RELEASE and policy.onset is not None and policy.reload_seen is not None:
                if t > policy.reload_seen + retain:
                    a = b.copy()
            a = bench.legacy.bounded_action(a)
            correction = a-b
            decisions.append((t,x.copy(),request.copy(),intended,local.copy(),b.copy(),a.copy()))
        m = c.tick(a)
        if np.any(c.d.qfrc_applied) or np.any(c.d.xfrc_applied):
            raise RuntimeError("External force")
        snapshot()
        unsafe = unsafe+1 if m["com"]<.8 or m["nonfoot"]>.5*W else 0
        if unsafe>=100:
            break
    arrays = {k:np.asarray(v) for k,v in zip(TRACE_KEYS,zip(*rows),strict=True)}
    np.savez_compressed(output/"TRACE.npz", **arrays)
    labels = ("ms","x","command","intent","local","base","action")
    np.savez_compressed(output/"DECISIONS.npz", **dict(zip(labels,map(np.asarray,zip(*decisions)),strict=True)))
    last = len(rows)-1
    pre_renewal = bench.standing(arrays,max(0,RELEASE-260),RELEASE) if last >= RELEASE else None
    result = dict(mode=mode, actual_ms=last, native_iterations=last*PER_MS, edges=policy.edges,
                  onset_ms=policy.onset, reload_ms=policy.reload_seen, stage=policy.stage,
                  anchor=policy.anchor.tolist() if policy.anchor is not None else None,
                  com_min=float(arrays["com"][:,2].min()), max_nonfoot_W=float(arrays["nonfoot"].max()/W),
                  initial_stand=bench.standing(arrays,0,min(last,FIRST_WALK)),
                  before_renewal_stand=pre_renewal,
                  previous_acceptance_sha256=stan.sha(BASIS), scenario=shape,
                  trace_sha256=stan.sha(output/"TRACE.npz"), decisions_sha256=stan.sha(output/"DECISIONS.npz"))
    if last == END:
        post = bench.metrics(arrays,RELEASE,RESUMED_WALK,END)
        post["walk_ms"] = RESUMED_WALK
        result["post_renewal"] = post
        if rec is not None:
            before = physical.WARM
            physical.WARM = RELEASE
            try:
                physical.native_metrics(rec,output,post,arrays)
            finally:
                physical.WARM = before
            result["native_audit_sha256"] = stan.sha(output/"NATIVE_AUDIT.json")
    if rec is not None:
        n = rec.count
        assert n == last*PER_MS and not rec.poses[:n,7].any()
        np.savez_compressed(output/"GEOMETRY_NATIVE.npz", rf_spatial=rec.rf_spatial[:n])
        result["geometry_native_sha256"] = stan.sha(output/"GEOMETRY_NATIVE.npz")
        if last != END:
            np.savez_compressed(output/"NATIVE.npz", forces=rec.forces[:n], pose=rec.poses[:n],
                                rf_floor=rec.rf_contact[:n], rf_pad_bottom_z=rec.fk_z[:n])
        result["native_sha256"] = stan.sha(output/"NATIVE.npz")
        result["native_contact_cross_checks"] = rec.cross_checks
    stan.identity.verify_runtime(see,c)
    result["elapsed_wall_s"] = time.monotonic()-clock
    stan.put(output/"RESULT.json",result)
    print(json.dumps({k: result[k] for k in ("mode","actual_ms","edges","onset_ms","reload_ms",
                      "com_min","max_nonfoot_W","before_renewal_stand","elapsed_wall_s")}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("far","near","off","flat","brake_only"))
    p.add_argument("output",type=Path)
    p.add_argument("--reach",type=float,default=.022)
    p.add_argument("--gain",type=float,default=12.8)
    p.add_argument("--retain",type=int,default=400)
    p.add_argument("--duration",type=int,default=END)
    p.add_argument("--native",action="store_true")
    p.add_argument("--keep-progressing",action="store_true")
    p.add_argument("--shape",type=Path)
    a=p.parse_args()
    if not 500 < a.duration <= END:
        p.error("Duration outside the physical bench")
    run(a.output.resolve(),a.mode,a.reach,a.gain,a.retain,a.duration,a.native,a.keep_progressing,a.shape)
