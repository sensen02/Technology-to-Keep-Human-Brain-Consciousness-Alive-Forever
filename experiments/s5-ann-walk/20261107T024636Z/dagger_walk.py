"""DAgger round for the walking route: add the ANN's own drifted states + teacher labels.

Round 1 (walk_full.py) failed after handover: the model tracks the teacher label in
distribution (dev MSE ~8e-4) but outputs an almost constant action once the body
leaves the reference trajectory, so the gait collapses. The saved eval run already
contains exactly what DAgger needs: for every visited state (including drifted ones)
the run recorded the teacher's would-be correction as teacher_actions.

This script retrains the SAME frozen architecture on teacher rollouts + drift states
from the failed evaluations, then re-runs handover/controls. The deployed controller
is still the exported ANN; no extra inputs are added.
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
RUN = OUT / 'dagger'
wf.RUN = RUN
ls.O = RUN


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


def load_sets():
    root = OUT / 'full' / 'learning'
    teacher = []
    for name in ['walk_train0', 'walk_train1', 'walk_train2']:
        with np.load(root / name / 'DATA.npz') as z:
            teacher.append({k: z[k].copy() for k in ['frames', 'target_states', 'teacher_actions']})
    with np.load(root / 'walk_dev0' / 'DATA.npz') as z:
        dev = {k: z[k].copy() for k in ['frames', 'target_states', 'teacher_actions']}
    drift = []
    for name in ['walk_eval_ann', 'walk_eval_frozen', 'walk_eval_brain_cut']:
        with np.load(root / name / 'DATA.npz') as z:
            drift.append({k: z[k].copy() for k in ['frames', 'target_states', 'teacher_actions']})
    return teacher, dev, drift


def batch(data, ids):
    hs, ts, ys = [], [], []
    for ep, i in ids:
        ds = data[ep]
        hs.append(ds['frames'][np.maximum(0, np.arange(i - 3, i + 1))])
        ts.append(ds['target_states'][i])
        ys.append(ds['teacher_actions'][i])
    return (torch.tensor(np.array(hs), dtype=torch.float32),
            torch.tensor(np.array(ts)), torch.tensor(np.array(ys)))


def train_round(trainsets, devset, rounds=140, path=None):
    torch.set_num_threads(8)
    torch.manual_seed(20261107 + (path is not None) * 1)
    ids = np.array([(e, i) for e, ds in enumerate(trainsets) for i in range(40, len(ds['frames']))])
    dev = np.array([(0, i) for i in range(40, len(devset['frames']))])
    rng = np.random.default_rng(20261107)
    model = FullBrainTranslator()
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-5)
    best, best_state = float('inf'), None
    for epoch in range(rounds):
        model.train()
        rng.shuffle(ids)
        losses = []
        for s0 in range(0, len(ids), 32):
            x, tt, y = batch(trainsets, ids[s0:s0 + 32])
            loss = ((model(x, tt) - y) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            x, tt, y = batch([devset], dev)
            score = float(((model(x, tt) - y) ** 2).mean())
        if score < best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if epoch % 20 == 0:
            print(json.dumps(dict(round=path.name if path else 'r1', epoch=epoch,
                                  train=float(np.mean(losses)), dev=score)), flush=True)
    model.load_state_dict(best_state)
    model.export(path)
    return best


def evaluate(model_path, tag_prefix='dagger'):
    root = RUN / 'learning'
    root.mkdir(parents=True, exist_ok=True)
    model = NumpyFullBrainTranslator(model_path)
    rt = wf.WalkRuntime()
    results = []
    # seed 55 reproduces the failed eval trajectory family
    r = rt.run(f'{tag_prefix}_ann', 'ann', 55, model, seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)
    for tag, mode in [(f'{tag_prefix}_brain_cut', 'brain_cut'), (f'{tag_prefix}_hold', 'hold')]:
        m = model if mode == 'brain_cut' else None
        # output folders are created by run(); keep them distinct from full run
        r = rt.run(tag, mode, 55, m, seconds=3.0, disturb=False)
        r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
        results.append(r)
        print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)

    class Frozen:
        def __init__(self):
            self.h = None

        def __call__(self, h, t):
            if self.h is None:
                self.h = h.copy()
            return model(self.h, t)

    r = rt.run(f'{tag_prefix}_frozen', 'ann', 55, Frozen(), seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)

    class ColdWalk(wf.WalkRuntime):
        def teacher(self, s):
            state = (s['a'].target_rad - self.p.q0[self.p.qa]).astype(np.float32)
            return wf.RATE_CAP * model(s['history'][None], state[None])[0]

    cold = ColdWalk(.125)
    r = cold.run(f'{tag_prefix}_cold_ann', 'ann', 66, model, seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 0.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / 'learning').mkdir(exist_ok=True)
    teacher, dev, drift = load_sets()
    round1 = RUN / 'learning' / 'TRANSLATOR_DAGGER1.npz'
    score1 = train_round(teacher + drift, dev, 140, round1)
    print(json.dumps(dict(round='dagger1', dev_mse=score1)), flush=True)
    results = evaluate(round1)
    dump(RUN / 'learning' / 'DAGGER1_EVALUATION.json',
         dict(importance='核心必需', round='dagger1', dev_mse=score1,
              model=str(round1),
              results=[dict(tag=r['tag'], mode=r['mode'], walk=r['walk']) for r in results],
              note='teacher labels only; deployed controller is the exported ANN'))


if __name__ == '__main__':
    main()
