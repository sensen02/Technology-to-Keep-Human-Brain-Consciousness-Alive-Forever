"""Decisive test: can the brain history alone drive the gait?

Established facts before this run:
  * teacher (time-indexed reference + incremental actuator) physically walks: fwd +0.29,
    foot support 1.0, non-foot contact 0, no root assist.
  * target state alone cannot determine the required rate: for 100% of phases there is a
    non-adjacent phase whose joint-space point is nearly coincident (median distance 0.118)
    but whose required velocity differs by a median 2.88 rad/s (reference mean speed 0.58).
    So a state-feedback-only policy is mathematically unable to track the loop.
  * brain rate history DOES carry the gait phase: out-of-sample linear decode of
    (cos, sin) has mean cos(error) = 0.996 (target state: 0.999, but ambiguous as above).
  * the full ANN trained on brain+target ignored the brain (frozen-history eval bit-identical).

This script zeroes the target-state input and trains the SAME architecture on brain history
only, so the loop can only close through proprioceptive phase coding.

Pre-registered interpretation:
  brain_only walks -> the brain signal is sufficient; route stands, no fallback needed.
  brain_only fails -> the brain signal, through this 4-frame/5 ms causal contract, cannot
                      close the loop either; the walker needs a different interface (longer
                      temporal context or a recurrent translator), which is a design limit,
                      not a data limit.
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
RUN = OUT / 'brain_only'
wf.RUN = RUN
ls.O = RUN


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


def batch(data, ids, zero_state=True):
    hs, ts, ys = [], [], []
    for ep, i in ids:
        ds = data[ep]
        hs.append(ds['frames'][np.maximum(0, np.arange(i - 3, i + 1))])
        ts.append(ds['target_states'][i])
        ys.append(ds['teacher_actions'][i])
    t = torch.tensor(np.array(ts), dtype=torch.float32)
    if zero_state:
        t = torch.zeros_like(t)
    return torch.tensor(np.array(hs), dtype=torch.float32), t, torch.tensor(np.array(ys))


def train(train, dev, rounds=200):
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
    path = RUN / 'learning' / 'TRANSLATOR_BRAIN_ONLY.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    model.export(path)
    return path, best, hist


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (RUN / 'learning').mkdir(parents=True, exist_ok=True)
    train_sets, dev_set = load_sets()
    path, score, hist = train(train_sets, dev_set)
    print(json.dumps(dict(brain_only_dev_mse=score)), flush=True)
    model = NumpyFullBrainTranslator(path)

    class ZeroState:
        def __init__(self, m):
            self.m = m

        def __call__(self, h, t):
            return self.m(h, np.zeros_like(t))

    zs = ZeroState(model)
    results = []

    # main handover run: brain-only translator
    rt = wf.WalkRuntime()
    r = rt.run('brain_only_ann', 'ann', 55, zs, seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)

    # brain cut inside the brain-only variant: everything zero -> must fall
    class ZeroBoth:
        def __call__(self, h, t):
            return model(np.zeros_like(h), np.zeros_like(t))

    r = rt.run('brain_only_cut', 'ann', 55, ZeroBoth(), seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 1.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)

    class ColdWalk(wf.WalkRuntime):
        def teacher(self, s):
            return wf.RATE_CAP * zs(s['history'][None], np.zeros((1, 42), np.float32))[0]

    cold = ColdWalk(.125)
    r = cold.run('brain_only_cold', 'ann', 66, zs, seconds=3.0, disturb=False)
    r['walk'] = wf.walk_metrics(Path(r['folder']), 0.0)
    results.append(r)
    print(json.dumps(dict(tag=r['tag'], walk=r['walk'])), flush=True)
    dump(RUN / 'learning' / 'BRAIN_ONLY.json',
         dict(importance='核心必需',
              description='target-state input zeroed; gait phase must come from brain rate history',
              dev_mse=score, history=hist, model=str(path),
              phase_decodability=dict(brain_history_cosine_error_out_of_sample=0.996,
                                      target_state_cosine_error=0.999,
                                      target_state_velocity_ambiguity_median_rad_s=2.88),
              results=[dict(tag=x['tag'], mode=x['mode'], walk=x['walk']) for x in results]))


if __name__ == '__main__':
    main()
