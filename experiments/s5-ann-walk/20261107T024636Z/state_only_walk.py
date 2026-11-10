"""Decisive ablation: can the ANN controller class walk at all, independent of the brain?

Diagnosis so far:
  * linear readout from 4-frame brain history -> teacher label: dev R^2 = -5.6 (no phase info)
  * linear readout from own target state   -> teacher label: dev R^2 = +0.76
  * the exported full ANN ignores brain history (frozen-history eval is bit-identical)
  * full ANN after handover: model output nearly constant on drifted states -> gait collapses

So the walk phase can only come from the actuator target state (a vector field on the gait
loop). This script trains the SAME architecture with brain history forced to zero, i.e. a
pure state-feedback limit-cycle policy, and runs the identical handover protocol.

Interpretation rules set before the run:
  state_only walks  -> the route CAN walk; the brain is simply not the signal that carries phase.
  state_only fails  -> this controller class cannot sustain the gait; no brain wiring fixes it.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJ = '/run/media/sensen/Data2/cell_wound_prototype'
ANN = PROJ + '/experiments/s5-ann-incremental/20261107T010000Z'
sys.path[:0] = [PROJ, PROJ + '/tools', ANN]
import walk_full as wf
import learn_stand as ls
from ann_translator import NumpyFullBrainTranslator, FullBrainTranslator

OUT = wf.OUT
RUN = OUT / 'state_only'
wf.RUN = RUN
ls.O = RUN


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


def load_sets():
    root = OUT / 'full' / 'learning'
    train, dev = [], None
    for name in ['walk_train0', 'walk_train1', 'walk_train2']:
        with np.load(root / name / 'DATA.npz') as z:
            train.append({k: z[k].copy() for k in ['frames', 'target_states', 'teacher_actions']})
    with np.load(root / 'walk_dev0' / 'DATA.npz') as z:
        dev = {k: z[k].copy() for k in ['frames', 'target_states', 'teacher_actions']}
    return train, dev


def batch(data, ids, zero_brain=True):
    hs, ts, ys = [], [], []
    for ep, i in ids:
        ds = data[ep]
        hs.append(ds['frames'][np.maximum(0, np.arange(i - 3, i + 1))])
        ts.append(ds['target_states'][i])
        ys.append(ds['teacher_actions'][i])
    h = torch.tensor(np.array(hs), dtype=torch.float32)
    if zero_brain:
        h = torch.zeros_like(h)
    return h, torch.tensor(np.array(ts)), torch.tensor(np.array(ys))


def train(train, dev, rounds=160):
    torch.set_num_threads(8)
    torch.manual_seed(20261107)
    ids = np.array([(e, i) for e, ds in enumerate(train) for i in range(40, len(ds['frames']))])
    dv = np.array([(0, i) for i in range(40, len(dev['frames']))])
    rng = np.random.default_rng(20261107)
    model = FullBrainTranslator()
    opt = torch.optim.Adam(model.parameters(), lr=.002, weight_decay=1e-6)
    best, best_state, hist = float('inf'), None, []
    for epoch in range(rounds):
        model.train()
        rng.shuffle(ids)
        losses = []
        for s0 in range(0, len(ids), 32):
            x, tt, y = batch(train, ids[s0:s0 + 32])
            loss = ((model(x, tt) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            x, tt, y = batch([dev], dv)
            score = float(((model(x, tt) - y) ** 2).mean())
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            hist.append(dict(epoch=epoch, train=float(np.mean(losses)), dev=score))
            print(json.dumps(hist[-1]), flush=True)
    model.load_state_dict(best_state)
    path = RUN / 'learning' / 'TRANSLATOR_STATE_ONLY.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    model.export(path)
    return path, best, hist


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (RUN / 'learning').mkdir(parents=True, exist_ok=True)
    train_sets, dev_set = load_sets()
    path, score, hist = train(train_sets, dev_set)
    print(json.dumps(dict(state_only_dev_mse=score)), flush=True)
    model = NumpyFullBrainTranslator(path)
    rt = wf.WalkRuntime()
    results = []
    r = rt.run('state_only_ann', 'ann', 55, model, seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)

    class Frozen:
        def __init__(self):
            self.h = None

        def __call__(self, h, t):
            if self.h is None:
                self.h = np.zeros_like(h)
            return model(self.h, t)

    r = rt.run('state_only_frozen', 'ann', 55, Frozen(), seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)

    class ColdWalk(wf.WalkRuntime):
        def teacher(self, s):
            state = (s['a'].target_rad - self.p.q0[self.p.qa]).astype(np.float32)
            return wf.RATE_CAP * model(np.zeros_like(s['history'])[None], state[None])[0]

    cold = ColdWalk(.125)
    r = cold.run('state_only_cold', 'ann', 66, model, seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 0.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)

    # teacher reference in the same protocol for direct comparison
    r = wf.WalkRuntime().run('state_only_teacher', 'teacher', 77, None, seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 0.5)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)
    dump(RUN / 'learning' / 'STATE_ONLY.json',
         dict(importance='核心必需', description='brain input zeroed: pure state-feedback limit-cycle policy',
              dev_mse=score, history=hist, model=str(path),
              results=[dict(tag=x['tag'], mode=x['mode'], walk=x['walk']) for x in results]))


if __name__ == '__main__':
    main()
