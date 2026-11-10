"""Walking with the target-history translator (TH-20).

Motivated by the geometric result that a memoryless policy cannot resolve gait phase:
adds 20 previous actuator-target states (95 ms) to the input. Same incremental actuator,
same PD servo, same 12000-cell brain, same causal policy tick.

Pre-registered interpretation:
  th walks after handover with feet carrying the body and no non-foot contact
      -> the route walks; nothing else is needed to stand or walk.
  th fails -> report the mechanism (interface needs longer context or recurrence) as the
      blocker, not the data volume.
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
from translator_th import TranslatorTH, NumpyTranslatorTH, THWrapper, parity_check

OUT = wf.OUT
RUN = OUT / 'th'
wf.RUN = RUN
ls.O = RUN
K = 20


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


def load_sets():
    root = OUT / 'full' / 'learning'
    train = []
    for name in ['walk_train0', 'walk_train1', 'walk_train2']:
        with np.load(root / name / 'DATA.npz') as z:
            train.append({k: z[k].copy() for k in ['frames', 'target_states', 'teacher_actions']})
    with np.load(root / 'walk_dev0' / 'DATA.npz') as z:
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
    path = RUN / 'learning' / 'TRANSLATOR_TH.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    model.export(path)
    return path, best, hist, model


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (RUN / 'learning').mkdir(parents=True, exist_ok=True)
    train_sets, dev = load_sets()
    import os
    existing = RUN / 'learning' / 'TRANSLATOR_TH.npz'
    if os.environ.get('RESUME') == '1' and existing.exists():
        path, score, hist, model = existing, float('nan'), [], None
        print(json.dumps(dict(resumed=str(path))), flush=True)
    else:
        path, score, hist, model = train(train_sets, dev)
    np_model = NumpyTranslatorTH(path)
    parity = parity_check(model, np_model, K) if model is not None else None
    print(json.dumps(dict(th_dev_mse=score, numpy_parity_max=parity)), flush=True)

    results = []
    wrapper = THWrapper(np_model, K)
    rt = wf.WalkRuntime()
    r = rt.run('th_ann', 'ann', 55, wrapper, seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)

    cut = THWrapper(np_model, K)
    cut.zero_brain = True
    r = rt.run('th_brain_cut', 'ann', 55, cut, seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)

    class ColdTH(wf.WalkRuntime):
        def teacher(self, s):
            state = (s['a'].target_rad - self.p.q0[self.p.qa]).astype(np.float32)
            return wf.RATE_CAP * wrapper(s['history'][None], state[None])[0]

    cold = ColdTH(.125)
    r = cold.run('th_cold', 'ann', 66, wrapper, seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 0.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)

    # long run if the 3 s handover kept the body up
    ann_walk = results[0]['walk']
    if ann_walk['foot_support_mean'] > 0.97 and ann_walk['nonfoot_positive_max'] == 0.0 \
            and ann_walk['com_z_min'] > 0.95 and ann_walk['forward_displacement'] > 0.15:
        w2 = THWrapper(np_model, K)
        r = wf.WalkRuntime().run('th_ann_6s', 'ann', 55, w2, seconds=6.0, disturb=False)
        r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
        results.append(r)
        print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)
    dump(RUN / 'learning' / 'TH_RESULTS.json',
         dict(importance='核心必需', description='target-history translator, 20 frames (95 ms)',
              dev_mse=score, numpy_parity_max=parity, history=hist, model=str(path),
              results=[dict(tag=x['tag'], mode=x['mode'], walk=x['walk']) for x in results]))


if __name__ == '__main__':
    main()
