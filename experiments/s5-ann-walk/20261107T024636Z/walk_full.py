"""Full walking test on the brain -> ANN -> incremental-target route.

Pilot (1 trajectory, 100 epochs) showed the real gait teacher walks this body on
this actuator route (fwd +0.29 native, foot support 1.0, non-foot 0), but the ANN
trained on a single trajectory fails to continue walking after handover. This
script gives the route a fair attempt: 3 independent teacher trajectories + 1 dev,
longer training, then the same handover protocol and controls as the standing run.

Deployed controller is still the exported ANN reading only brain rate history +
own target state. The gait teacher is an algorithm-assisted collection stage and
is reported separately; no neural-necessity claim is made unless controls separate.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
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
RUN = OUT / 'full'
ls.O = RUN
ls.RATE = RATE_CAP


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


class WalkRuntime(ls.Runtime):
    def __init__(self, ndt=.125, freq=FREQ, ramp=RAMP, rate_cap=RATE_CAP):
        super().__init__(ndt)
        self.freq = float(freq)
        self.ramp = float(ramp)
        self.rate_cap = float(rate_cap)
        g = np.load(GAIT_NPZ)
        assert np.array_equal(g['qpos_addresses'], self.p.qa)
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
        # Policy commands are held 5 ms, so track over the 5 ms hold.
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


def walk_metrics(folder, handover_s):
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


def batch(data, ids):
    hs, ts, ys = [], [], []
    for ep, i in ids:
        ds = data[ep]
        hs.append(ds['frames'][np.maximum(0, np.arange(i - 3, i + 1))])
        ts.append(ds['target_states'][i])
        ys.append(ds['teacher_actions'][i])
    return (torch.tensor(np.array(hs), dtype=torch.float32),
            torch.tensor(np.array(ts)), torch.tensor(np.array(ys)))


def train(epochs=140):
    torch.set_num_threads(8)
    torch.manual_seed(20261107)
    root = RUN / 'learning'
    trainsets, devsets = [], []
    for name in ['walk_train0', 'walk_train1', 'walk_train2', 'walk_dev0']:
        with np.load(root / name / 'DATA.npz') as z:
            ds = {k: z[k].copy() for k in ['frames', 'target_states', 'teacher_actions']}
        (devsets if name.startswith('walk_dev') else trainsets).append(ds)
    ids = np.array([(e, i) for e, ds in enumerate(trainsets) for i in range(40, len(ds['frames']))])
    dev = np.array([(e, i) for e, ds in enumerate(devsets) for i in range(40, len(ds['frames']))])
    rng = np.random.default_rng(20261107)
    model = FullBrainTranslator()
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    best, best_state, history = float('inf'), None, []
    for epoch in range(epochs):
        model.train()
        rng.shuffle(ids)
        losses = []
        for s0 in range(0, len(ids), 32):
            x, tt, y = batch(trainsets, ids[s0:s0 + 32])
            pred = model(x, tt)
            loss = ((pred - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        errs = []
        with torch.no_grad():
            for s0 in range(0, len(dev), 32):
                x, tt, y = batch(devsets, dev[s0:s0 + 32])
                errs.append(float(((model(x, tt) - y) ** 2).mean()))
        score = float(np.mean(errs))
        history.append(dict(epoch=epoch, train=float(np.mean(losses)), dev=score))
        if score < best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            print(json.dumps(history[-1]), flush=True)
    model.load_state_dict(best_state)
    path = root / 'TRANSLATOR_WALK_FULL.npz'
    model.export(path)
    npop = NumpyFullBrainTranslator(path)
    # parity
    x, tt, y = batch(devsets, dev[:8])
    with torch.no_grad():
        expected = model(x, tt).numpy()
    actual = npop(x.numpy(), tt.numpy())
    parity = float(np.max(np.abs(actual - expected)))
    dump(root / 'WALK_TRAINING.json', dict(importance='核心必需', history=history, best_dev_mse=best,
                                           numpy_parity_max=parity, model=str(path),
                                           train_samples=len(ids), dev_samples=len(dev),
                                           split='entire independent physical trajectories',
                                           input='all12000 brain rates plus own42 target state',
                                           label='teacher incremental correction (gait tracking), not torque',
                                           brain_specificity='not established by fitting'))
    return path


def eval_all(model_path):
    root = RUN / 'learning'
    model = NumpyFullBrainTranslator(model_path)
    rt = WalkRuntime()
    results = []
    tests = [('walk_eval_ann', 'ann', model),
             ('walk_eval_brain_cut', 'brain_cut', model),
             ('walk_eval_hold', 'hold', None),
             ('walk_eval_motor_cut', 'motor_cut', None)]
    for tag, mode, m in tests:
        r = rt.run(tag, mode, 55, m, seconds=3.0, disturb=False)
        r['walk'] = walk_metrics(Path(r['folder']), 1.0)
        results.append(r)
        print(json.dumps(dict(tag=tag, mode=mode, walk=r['walk'])), flush=True)

    class Frozen:
        def __init__(self):
            self.h = None

        def __call__(self, h, t):
            if self.h is None:
                self.h = h.copy()
            return model(self.h, t)

    # cold: ANN from t=0, no teacher at all; same gait-target integration.
    class ColdWalk(WalkRuntime):
        def teacher(self, s):
            state = (s['a'].target_rad - self.p.q0[self.p.qa]).astype(np.float32)
            return ls.RATE * model(s['history'][None], state[None])[0]

    cold = ColdWalk(.125)
    r = cold.run('walk_eval_cold_ann', 'ann', 66, model, seconds=3.0, disturb=False)
    r['walk'] = walk_metrics(Path(r['folder']), 0.0)
    results.append(r)
    print(json.dumps(dict(tag='walk_eval_cold_ann', walk=r['walk'])), flush=True)
    # frozen history: same ANN but history frozen after handover
    fz = WalkRuntime()
    r = fz.run('walk_eval_frozen', 'ann', 55, Frozen(), seconds=3.0, disturb=False)
    r['walk'] = walk_metrics(Path(r['folder']), 1.0)
    results.append(r)
    print(json.dumps(dict(tag='walk_eval_frozen', walk=r['walk'])), flush=True)
    dump(root / 'WALK_EVALUATION.json', dict(importance='核心必需',
                                             model=model_path,
                                             results=[dict(tag=x['tag'], mode=x['mode'], walk=x['walk'],
                                                           validation_passed=x['validation']['passed']) for x in results]))
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (RUN / 'learning').mkdir(parents=True, exist_ok=True)
    rt = WalkRuntime()
    # teacher baseline and data collection
    teacher = rt.run('walk_teacher_baseline', 'teacher', 77, None, seconds=3.0, disturb=False)
    teacher['walk'] = walk_metrics(Path(teacher['folder']), 0.5)
    print(json.dumps(dict(tag='walk_teacher_baseline', walk=teacher['walk'])), flush=True)
    for tag, seed in [('walk_train0', 11), ('walk_train1', 22), ('walk_train2', 33), ('walk_dev0', 44)]:
        r = rt.run(tag, 'teacher', seed, None, seconds=3.0, disturb=True)
        r['walk'] = walk_metrics(Path(r['folder']), 0.5)
        print(json.dumps(dict(tag=tag, walk=r['walk'])), flush=True)
    path = train()
    eval_all(path)


if __name__ == '__main__':
    main()
