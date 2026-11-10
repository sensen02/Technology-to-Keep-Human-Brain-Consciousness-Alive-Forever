"""Can the brain -> ANN -> incremental-target route WALK?

The standing route proved static holding works. This script changes exactly one
thing: the teacher that supplies training labels is no longer the static PD-to-q0
hold, but a rate teacher that tracks the real untethered-fly gait reference
(GAIT.npz, retargeted from STANDARD_GAIT_FK, already physically verified to walk
this body on the same incremental actuator). The deployed controller is still the
exported ANN reading ONLY brain rate history + own target state; the teacher is a
labelled algorithm-assisted collection stage, never the deployed controller.

Measured after handover (ann mode): does the body keep advancing, do the feet keep
carrying it, is non-foot contact zero. Controls (brain_cut, frozen history, cold
start, hold, motor_cut) are run under identical conditions; no neural-necessity
claim is made unless they separate.

Run as: python -B walk_ann.py pilot | full
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

PROJ = '/run/media/sensen/Data2/cell_wound_prototype'
ANN = PROJ + '/experiments/s5-ann-incremental/20261107T010000Z'
sys.path[:0] = [PROJ, PROJ + '/tools', ANN]
import learn_stand as ls
from incremental_actuator import IncrementalActuator
from ann_translator import NumpyFullBrainTranslator, FullBrainTranslator

OUT = Path('/home/sensen/Desktop/engine修复验收/s5-ann-walk/20261107T024636Z')
GAIT_NPZ = Path('/home/sensen/Desktop/engine修复验收/s5-drive-chain-diagnosis/20261105T054144Z/baseline/EVAL/RUN/GAIT.npz')
WEIGHT = 24.446354755878005
FREQ = 1.0
RAMP = 0.5
RATE_CAP = 30.0
ls.O = OUT          # every run()/dump() now writes under this timestamp only
ls.RATE = RATE_CAP  # scaling used inside ls.Runtime.run


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


class WalkRuntime(ls.Runtime):
    """Same runtime; teacher tracks the real gait reference instead of q0."""

    def __init__(self, ndt=.125, freq=FREQ, ramp=RAMP, rate_cap=RATE_CAP):
        super().__init__(ndt)
        self.freq = float(freq)
        self.ramp = float(ramp)
        self.rate_cap = float(rate_cap)
        g = np.load(GAIT_NPZ)
        assert np.array_equal(g['qpos_addresses'], self.p.qa), 'gait columns must match actuator order'
        self.spline = CubicSpline(g['phase'], g['qpos_hinges'], axis=0, bc_type='periodic')
        self.lo = self.p.m.jnt_range[self.p.jids, 0]
        self.hi = self.p.m.jnt_range[self.p.jids, 1]

    def gait_target(self, t):
        if t < self.ramp:
            u = t / self.ramp
            blend = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            return (1 - blend) * self.p.q0[self.p.qa] + blend * np.clip(self.spline(0.0), self.lo, self.hi)
        phase = 2 * np.pi * self.freq * (t - self.ramp)
        return np.clip(self.spline(phase % (2 * np.pi)), self.lo, self.hi)

    def teacher(self, s):
        # Policy commands are held 5 ms (learn_stand.run updates every 5th tick),
        # so a tracking rate must be divided by the 5 ms hold, not by 1 ms.
        rate = (self.gait_target(float(s['d'].time)) - s['a'].target_rad) / 0.005
        return np.clip(rate, -self.rate_cap, self.rate_cap)

    def start(self):
        d = self.p.fresh()
        lim = self.p.limits
        a = IncrementalActuator(self.p.m, d, kp=lim / .2 * ls.SCALE,
                                kd=lim * .005 * np.sqrt(ls.SCALE),
                                max_rate_rad_s=self.rate_cap)
        n = ls.ConductanceNetwork(12000, ls.ConductanceParams(), dt_ms=self.ndt)
        n.set_connectivity(self.we, self.wi)
        return dict(d=d, a=a, n=n, enc=ls.ChannelSensoryEncoder(self.groups, 12000, ls.ChannelConfig()),
                    rates=np.zeros(12000), history=np.zeros((4, 12000), np.float32))


def walk_metrics(folder, handover_s=1.0):
    nat = np.load(folder / 'NATIVE.npz')['rows']
    with np.load(folder / 'BODY.npz') as z:
        body = {k: z[k].copy() for k in z.files}
    t = nat[:, 0]
    rel = nat[t >= handover_s]
    foot = rel[:, 1:7].sum(1) / WEIGHT
    bt = body['time']
    i0 = int(np.argmax(bt >= handover_s))
    return dict(
        release_seconds=float(rel[-1, 0] - rel[0, 0]),
        forward_displacement=float(body['qpos'][-1, 0] - body['qpos'][i0, 0]),
        lateral_displacement=float(body['qpos'][-1, 1] - body['qpos'][i0, 1]),
        whole_run_forward=float(body['qpos'][-1, 0] - body['qpos'][0, 0]),
        forward_speed=float((body['qpos'][-1, 0] - body['qpos'][i0, 0]) / (bt[-1] - bt[i0])),
        foot_support_mean=float(foot.mean()), foot_support_p10=float(np.percentile(foot, 10)),
        foot_support_min=float(foot.min()),
        nonfoot_positive_max=float(rel[:, 7].max()),
        nonfoot_contact_fraction=float((rel[:, 7] > 1e-9).mean()),
        com_z_min=float(rel[:, 10].min()), com_z_end=float(rel[-1, 10]),
        bearing_legs_median=float(np.median((rel[:, 1:7] > 1e-9).sum(1))),
        per_leg_duty=[float((rel[:, 1 + i] > 1e-9).mean()) for i in range(6)],
        liftoff_counts=[int(np.sum(np.diff((rel[:, 1 + i] > 1e-9).astype(int)) == -1)) for i in range(6)])


def eval_all(rt, model, tags, seconds=3.0):
    results = []
    for tag, mode in tags:
        r = rt.run(tag, mode, 55, model, seconds=seconds, disturb=False)
        r['walk'] = walk_metrics(Path(r['folder']), 1.0)
        results.append(r)
        print(json.dumps(dict(tag=tag, mode=mode, passed=r['validation']['passed'],
                              fwd=r['walk']['forward_displacement'],
                              foot=r['walk']['foot_support_mean'],
                              nonfoot=r['walk']['nonfoot_positive_max'],
                              com_min=r['walk']['com_z_min'])), flush=True)
    return results


def frozen_model_callable(model):
    class Frozen:
        def __init__(self):
            self.h = None

        def __call__(self, h, t):
            if self.h is None:
                self.h = h.copy()
            return model(self.h, t)
    return Frozen()


def pilot():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'learning').mkdir(exist_ok=True)
    rt = WalkRuntime()
    tests = []
    teacher = rt.run('teacher_walk_baseline', 'teacher', 77, None, seconds=3.0, disturb=False)
    teacher['walk'] = walk_metrics(Path(teacher['folder']), 0.5)
    print(json.dumps(dict(tag='teacher_walk_baseline', walk=teacher['walk'])), flush=True)
    train = rt.run('walk_train0', 'teacher', 11, None, seconds=3.0, disturb=True)
    train['walk'] = walk_metrics(Path(train['folder']), 0.5)
    # Train on the single walking trajectory (pilot only).
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(20261107)
    with np.load(Path(train['folder']) / 'DATA.npz') as z:
        ds = {k: z[k].copy() for k in ['frames', 'target_states', 'teacher_actions']}
    ids = np.array([(0, i) for i in range(40, len(ds['frames']))])
    rng = np.random.default_rng(20261107)
    model = FullBrainTranslator()
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    best = float('inf')
    best_state = None
    for epoch in range(100):
        model.train()
        rng.shuffle(ids)
        losses = []
        for s0 in range(0, len(ids), 32):
            b = ids[s0:s0 + 32]
            hs = np.array([ds['frames'][np.maximum(0, np.arange(i - 3, i + 1))] for _, i in b])
            ts = np.array([ds['target_states'][i] for _, i in b])
            ys = np.array([ds['teacher_actions'][i] for _, i in b])
            pred = model(torch.tensor(hs, dtype=torch.float32), torch.tensor(ts))
            loss = ((pred - torch.tensor(ys)) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach()))
        if float(np.mean(losses)) < best:
            best = float(np.mean(losses))
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            print(json.dumps(dict(epoch=epoch, loss=float(np.mean(losses)))), flush=True)
    model.load_state_dict(best_state)
    model_path = OUT / 'learning' / 'TRANSLATOR_WALK.npz'
    model.export(model_path)
    print(json.dumps(dict(trained=True, train_loss=best, path=str(model_path))), flush=True)
    npop = NumpyFullBrainTranslator(model_path)
    tags = [('walk_ann', 'ann'), ('walk_brain_cut', 'brain_cut'),
            ('walk_frozen', 'ann'), ('walk_hold', 'hold')]
    results = []
    for tag, mode in tags:
        m = npop if mode in ('ann', 'brain_cut') else None
        if tag == 'walk_frozen':
            m = frozen_model_callable(npop)
        r = rt.run(tag, mode, 55, m, seconds=3.0, disturb=False)
        r['walk'] = walk_metrics(Path(r['folder']), 1.0)
        results.append(r)
        print(json.dumps(dict(tag=tag, fwd=r['walk']['forward_displacement'],
                              foot=r['walk']['foot_support_mean'],
                              nonfoot=r['walk']['nonfoot_positive_max'],
                              com_min=r['walk']['com_z_min'],
                              passed=r['validation']['passed'])), flush=True)
    # cold start: ANN from t=0, no teacher ever
    class Cold(ls.Runtime):
        def teacher(self, s):
            state = (s['a'].target_rad - self.p.q0[self.p.qa]).astype(np.float32)
            return ls.RATE * npop(s['history'][None], state[None])[0]

    cold_rt = Cold(.125)
    cold = cold_rt.run('walk_cold_ann', 'ann', 66, npop, seconds=3.0, disturb=False)
    cold['walk'] = walk_metrics(Path(cold['folder']), 0.0)
    results.append(cold)
    dump(OUT / 'PILOT_SUMMARY.json', dict(importance='核心必需',
                                          model=str(model_path),
                                          teacher_walk=teacher['walk'],
                                          walk_train=train['walk'],
                                          results=[dict(tag=r['tag'], walk=r['walk'], passed=r['validation']['passed']) for r in results],
                                          note='algorithm-assisted teacher only; deployed action is exported ANN; no neural-necessity claim without control separation'))


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'pilot'
    assert mode == 'pilot', 'only pilot implemented in this file'
    pilot()
