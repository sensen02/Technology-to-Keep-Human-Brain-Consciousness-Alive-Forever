"""State-projected limit-cycle teacher, then ANN takeover.

Why this design: the previous teacher was time-indexed  rate = (ref(t) - target)/0.005.
Its label contains almost no positional correction when tracking is good, so the learned
policy reproduces "reference velocity" and has no term that pulls a drifting target back.
Measured on the TH takeover: post-handover target speed ran at 13.87 rad/s mean (reference
mean 0.58), i.e. the loop diverged instead of being corrected.

This teacher instead defines the reference by the nearest point on the gait loop:
    phase* = argmin_phi || loop(phi) - target ||
    desired = loop(phase* + omega*dt)
    rate = clip((desired - target)/0.005, +-30)
so every label is a self-correcting vector field on the target space: off-loop states are
pulled back, on-loop states advance along the loop. That is consistent (a function of the
target state alone) and therefore learnable, unlike the time-indexed variant.

Teacher is data collection only; deployment is the single exported ANN.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = '/run/media/sensen/Data2/cell_wound_prototype'
ANN = PROJ + '/experiments/s5-ann-incremental/20261107T010000Z'
HERE = Path(__file__).resolve().parent
sys.path[:0] = [PROJ, PROJ + '/tools', ANN, str(HERE)]
import walk_full as wf
import learn_stand as ls
import translator_th as T
from translator_th import TranslatorTH, NumpyTranslatorTH, THWrapper, parity_check

OUT = wf.OUT
RUN = OUT / 'loop'
wf.RUN = RUN
ls.O = RUN
K = 20
DT = 0.005
FREQ = 1.0
RAMP = 0.5
CAP = 30.0
GRID = 720


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


class LoopRuntime(wf.WalkRuntime):
    """Teacher replaced by a nearest-phase projection onto the gait loop."""

    def __init__(self, ndt=.125):
        super().__init__(ndt, freq=FREQ, ramp=RAMP, rate_cap=CAP)
        phases = np.linspace(0, 2 * np.pi, GRID, endpoint=False)
        loop = np.clip(self.spline(phases), self.lo, self.hi)
        self.phases = phases
        self.loop_abs = loop
        self.loop_rel = loop - self.p.q0[self.p.qa]      # same offset as target_state

    def nearest_phase(self, target_rel):
        d = np.linalg.norm(self.loop_rel - target_rel, axis=1)
        return float(self.phases[int(np.argmin(d))])

    def teacher(self, s):
        target = s['a'].target_rad
        rel = target - self.p.q0[self.p.qa]
        phi = self.nearest_phase(rel)
        step = self.spline(phi + 2 * np.pi * FREQ * DT)
        step = np.clip(step, self.lo, self.hi)
        t = float(s['d'].time)
        if t < RAMP:
            u = t / RAMP
            blend = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
            desired = (1 - blend) * self.p.q0[self.p.qa] + blend * step
        else:
            desired = step
        return np.clip((desired - target) / DT, -CAP, CAP)


def load_sets():
    root = RUN / 'learning'
    train, dev = [], None
    for name in ['loop_train0', 'loop_train1', 'loop_train2']:
        with np.load(root / name / 'DATA.npz') as z:
            train.append({k: z[k].copy() for k in ['frames', 'target_states', 'teacher_actions']})
    with np.load(root / 'loop_dev0' / 'DATA.npz') as z:
        dev = {k: z[k].copy() for k in ['frames', 'target_states', 'teacher_actions']}
    return train, dev


def batch(data, ids):
    hs, th, ys = [], [], []
    for ep, i in ids:
        ds = data[ep]
        hs.append(ds['frames'][np.maximum(0, np.arange(i - 3, i + 1))])
        th.append(ds['target_states'][np.maximum(0, np.arange(i - K + 1, i + 1))])
        ys.append(ds['teacher_actions'][i])
    return (torch.tensor(np.array(hs), dtype=torch.float32),
            torch.tensor(np.array(th), dtype=torch.float32),
            torch.tensor(np.array(ys), dtype=torch.float32))


def train(train_sets, dev, rounds=200):
    torch.set_num_threads(8)
    torch.manual_seed(20261107)
    ids = np.array([(e, i) for e, ds in enumerate(train_sets) for i in range(40, len(ds['frames']))])
    dv = np.array([(0, i) for i in range(40, len(dev['frames']))])
    rng = np.random.default_rng(20261107)
    model = TranslatorTH(K)
    opt = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-6)
    best, best_state, hist = float('inf'), None, []
    for epoch in range(rounds):
        model.train()
        rng.shuffle(ids)
        losses = []
        for s0 in range(0, len(ids), 32):
            x, t, y = batch(train_sets, ids[s0:s0 + 32])
            loss = ((model(x, t) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            x, t, y = batch([dev], dv)
            score = float(((model(x, t) - y) ** 2).mean())
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            hist.append(dict(epoch=epoch, train=float(np.mean(losses)), dev=score))
            print(json.dumps(hist[-1]), flush=True)
    model.load_state_dict(best_state)
    path = RUN / 'learning' / 'TRANSLATOR_LOOP.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    model.export(path)
    return path, best, hist, model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (RUN / 'learning').mkdir(parents=True, exist_ok=True)
    rt = LoopRuntime()
    # teacher feasibility check on the new teacher itself
    r = rt.run('loop_teacher', 'teacher', 77, None, seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
    print(json.dumps(dict(tag='loop_teacher', walk=r['walk'])), flush=True)
    for tag, seed in [('loop_train0', 11), ('loop_train1', 22), ('loop_train2', 33), ('loop_dev0', 44)]:
        x = rt.run(tag, 'teacher', seed, None, seconds=3.0, disturb=True)
        x['walk'] = wf.walk_metrics(Path(x['folder']), 1.0)
        print(json.dumps(dict(tag=tag, walk=x['walk'])), flush=True)
    train_sets, dev = load_sets()
    path, score, hist, model = train(train_sets, dev)
    np_model = NumpyTranslatorTH(path)
    print(json.dumps(dict(loop_dev_mse=score, parity=parity_check(model, np_model, K),
                          label_rms=float(np.sqrt((np.concatenate([d['teacher_actions'][40:] for d in train_sets]) ** 2).mean())))), flush=True)
    results = []
    w1 = THWrapper(np_model, K)
    r = LoopRuntime().run('loop_ann', 'ann', 55, w1, seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)
    cut = THWrapper(np_model, K)
    cut.zero_brain = True
    r = LoopRuntime().run('loop_brain_cut', 'ann', 55, cut, seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)

    class ColdLoop(LoopRuntime):
        def teacher(self, s):
            state = (s['a'].target_rad - self.p.q0[self.p.qa]).astype(np.float32)
            return CAP * w1(s['history'][None], state[None])[0]

    r = ColdLoop().run('loop_cold', 'ann', 66, w1, seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 0.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)

    best = results[0]['walk']
    if best['foot_support_mean'] > .97 and best['nonfoot_positive_max'] == 0 and best['com_z_min'] > .95:
        w2 = THWrapper(np_model, K)
        r = LoopRuntime().run('loop_ann_6s', 'ann', 55, w2, seconds=6.0, disturb=False)
        r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
        results.append(r)
        print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)
    dump(RUN / 'learning' / 'LOOP_RESULTS.json',
         dict(importance='核心必需', dev_mse=score, model=str(path), history=hist,
              results=[dict(tag=x['tag'], mode=x['mode'], walk=x['walk']) for x in results]))


if __name__ == '__main__':
    main()
