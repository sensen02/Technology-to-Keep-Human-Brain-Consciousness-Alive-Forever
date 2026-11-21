"""Cache only teacher-controlled preparation; integrate every released step on the native body."""
import copy
import json
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / '20261120T081056Z-stance-warmup'
sys.path.insert(0, str(PARENT))
from warm_release import (body, BodyObservations, ConductanceNetwork, DT_MS, HANDOFF, SETTLE,
                          Interfaces, MuscleTeacher, PREVIOUS, load_pose, mj, sha, support)

BASE = PREVIOUS / 'verified/PARAMETERS.npz'


class Bench:
    def __init__(self):
        stance = json.loads((PARENT/'stance.json').read_text())
        self.model = body(stance['floor_contact_time_ms'])
        self.io = Interfaces(self.model)
        self.io.restore(BASE)
        self.obs = BodyObservations(self.model)
        self.target = np.array(stance['tonic_activation'])
        pose = np.array(stance['qpos'])
        self.handoff = load_pose(self.model, pose)
        d = self.handoff
        d.ctrl[:] = d.act[:] = self.target
        mj.mj_forward(self.model, d)
        self.height = d.subtree_com[0,2]
        self.angle = d.qpos[self.obs.tibia_qpos].copy()
        teacher = MuscleTeacher(self.model, pose, self.target)
        preload = []
        with np.load(PARENT/'results/warm_release.npz') as old:
            history = old['qpos']
        for k in range(HANDOFF):
            if k % 5 == 0:
                d.ctrl[:] = teacher.command(d)
                if k >= SETTLE:
                    preload.append(self.inputs(self.obs.measure(d)))
            mj.mj_step(self.model, d)
            mj.mj_forward(self.model, d)
        np.testing.assert_array_equal(d.qpos, history[HANDOFF])
        self.preload = np.array(preload)
        self.motor_rows, self.motor_cols = np.nonzero(self.io.exact | self.io.gaps)
        self.weights = self.io.parameters()
        self.cache = None

    def inputs(self, z):
        io = self.io
        load = z['ground_load_w'][io.feet]
        feco = np.where(io.angle,(z['tibia_angle_rad']-self.angle)[io.feco_feet],
                        z['tibia_velocity_rad_s'][io.feco_feet])
        return np.r_[load, feco]

    def current(self, features, gains, drive):
        n = len(self.io.sensors)
        drive[self.io.sensors] = np.clip(features[:n]*gains['load_gain'], 0, .12)
        drive[self.io.feco] = np.clip(features[n:]*gains['feco_gain'], 0, .12)

    def prepare(self, params):
        io = self.io
        net = ConductanceNetwork(len(io.roots), dt_ms=DT_MS)
        io.vnc_gain = params['vnc_gain_uS']
        net.set_connectivity(*io.connectivity([params['exc_fraction'], params['inh_fraction']]))
        drive = np.zeros(net.n)
        hist = []
        counts = np.zeros(net.n, dtype=np.int64)
        trace = np.zeros(len(io.motor))
        for tick in range(HANDOFF//5):
            if tick >= SETTLE//5:
                self.current(self.preload[tick-SETTLE//5], params, drive)
            s = net.step(drive)
            trace *= np.exp(-DT_MS/10)
            trace += s[io.motor]
            if tick >= SETTLE//5:
                hist.append(trace.copy())
                counts += s
        return net, trace, np.array(hist), counts

    def evaluate(self, params, cached, duration_ms=250., full=False, strict=False):
        net, trace, *_ = self.cache if cached else self.prepare(params)
        net = copy.deepcopy(net)
        trace = trace.copy()
        d = copy.copy(self.handoff)
        gains = np.zeros_like(self.io.motor_gain)
        gains[self.motor_rows, self.motor_cols] = params['motor_gain']
        drive = np.zeros(net.n)
        command = np.clip(gains@trace, 0, 1)
        first = None
        losses = []
        bad = 0
        excess = []
        for tick in range(round(duration_ms*10)):
            if tick % 5 == 0:
                d.ctrl[:] = command
                z = self.obs.measure(d)
                self.current(self.inputs(z), params, drive)
                trace *= np.exp(-DT_MS/10)
                trace += net.step(drive)[self.io.motor]
                command = np.clip(gains@trace, 0, 1)
            mj.mj_step(self.model,d)
            mj.mj_forward(self.model,d)
            z = self.obs.measure(d)
            depth = min((c.dist for c in d.contact if self.obs.floor in c.geom), default=0.)
            ok = support(z, depth, self.height)
            if not ok and first is None:
                first = (tick+1)*.1
            bad += not ok
            foot = z['ground_load_w'][:-1].sum()
            h = z['torso'][0]/self.height
            losses.append(min(2.,abs(foot-1)) + min(2.,abs(h-1))*3)
            excess.append(max(0., abs(foot-1)-.2) + max(0., .9-h)*5)
            if not full and (z['ground_load_w'][-1] > 0 or h < .88):
                break
        elapsed = (tick+1)*.1
        # Both the time to first strict failure and the time to collapse matter.
        objective = -(duration_ms if first is None else first) - .45*elapsed + .2*bad*.1 + 4*np.mean(losses)
        if strict:
            # Smooth early load surplus breaks ties among instantaneous failures.
            objective = -(duration_ms if first is None else first) - .025*elapsed + 8*np.mean(excess[:100])
        if strict == 'handoff':
            objective = -(duration_ms if first is None else first) - .01*elapsed + 30*max(excess[:10]) + 5*np.mean(excess)
        return float(objective), first, elapsed


if __name__=='__main__':
    b=Bench()
    b.cache=b.prepare(b.weights)
    hist=b.cache[2]; a=b.io
    active=(hist[-100:].mean(axis=0)>0)
    print('silent muscles and their target',[(i, round(b.target[i],3)) for i,row in enumerate((a.exact|a.gaps)[:,active]) if not row.any()])
    print('baseline',b.evaluate(b.weights,True),flush=True)
    np.savez_compressed(HERE/'WARM_ACTIVITY.npz',traces=hist,counts=b.cache[3],motor=a.motor)
