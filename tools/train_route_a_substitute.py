"""Route A: learn only where the animal has no command at all.

The 600 dimensional run let the search touch everything: the connectome readout pools, every
per actuator gain, every per cell readout weight, the sensory channel gains, and a replacement
layer for the actuators that biology never commands. It improved J from 3.4892 to 3.4024, and
nothing in that result says WHICH part did the work.

Route A separates them by construction. Everything biological is FROZEN at the recorded best.
The only free parameters are the ones that exist because BANC has nothing to say:

  - the 226 weights of the replacement layer, which map the driven actuators of a leg onto the
    actuators of that same leg that have no motor neuron, or whose motor neurons never fire
  - the 26 per actuator gains of exactly those actuators

That is 252 numbers, against 600 before, with the same 768 evaluation cap, the same objective and
the same seed run, so the comparison against the parent is budget matched.

Because the frozen part is the recorded best, evaluation zero reproduces 3.402395132105888 exactly.
Anything after that is attributable to the substitution layer and nothing else.

The honest reading to keep in front of the result: 64 of the 90 actuators are still commanded by
their own motor neurons in this arm, and on their own they were not enough to hold the body up.
Route A can therefore raise the score without showing that the connectome can carry the body.

env: RA_OUT RA_GENS RA_LAM RA_SEED RA_SIGMA RA_WORKERS
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ['PCR_SYNTH'] = '1'
import json
import math
import multiprocessing as mp
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R)); sys.path.insert(0, str(R / 'tools'))
O = Path(os.environ.get('RA_OUT', '/home/sensen/Desktop/engine修复验收/s5-route-a-substitute/20261108T040000Z'))
BEST = Path('/home/sensen/Desktop/engine修复验收/s5-synthetic-readout-cont/20261107T120000Z')
REF_J = 3.402395132105888
LAM = int(os.environ.get('RA_LAM', '48')); MU = LAM // 2
GENS = int(os.environ.get('RA_GENS', '15'))
SEED = int(os.environ.get('RA_SEED', '20261108'))
WORKERS = int(os.environ.get('RA_WORKERS', '12'))
SIGMA0 = float(os.environ.get('RA_SIGMA', '0.15'))
CTX = mp.get_context('fork')

_S = {}


def js(p):
    return json.loads(Path(p).read_text())


def _san(v):
    if isinstance(v, float):
        return v if math.isfinite(v) else str(v)
    if isinstance(v, dict):
        return {k: _san(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_san(x) for x in v]
    return v


def dump(p, v):
    Path(p).write_text(json.dumps(_san(v), ensure_ascii=False, indent=2, allow_nan=False,
                                  default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')


def prepare():
    import train_per_cell_readout as pcr
    import train_joint_pool_readout as jp
    if not (O / 'LAYOUT.json').exists():
        shutil.copyfile(BEST / 'LAYOUT.json', O / 'LAYOUT.json')
    pcr.O = O
    pcr.SET = O / 'CONNECTIVITY_SET'
    pcr.make_set()
    plan = pcr.synthetic_plan()
    ncell = pcr.cell_dim()
    nch_new = len(pcr.CHANNELS) - 1
    syn0 = pcr.BASE_DIM + ncell + nch_new
    nsw = sum(len(p['sources']) for p in plan)
    dead = [int(p['index']) for p in plan]
    vec0 = np.asarray(js(BEST / 'EVALS' / 'eval_556' / 'SCORE.json')['vec'], float)
    assert vec0.size == syn0 + nsw, 'vector layout moved: %d vs %d + %d' % (vec0.size, syn0, nsw)
    gain_idx = [int(jp.POOL_DIM) + i for i in dead]
    learn_idx = list(range(syn0, syn0 + nsw)) + gain_idx
    _S.update(pcr=pcr, plan=plan, syn0=syn0, nsw=nsw, dead=dead, gain_idx=gain_idx,
              learn_idx=learn_idx, vec0=vec0, x0=vec0[learn_idx].copy())
    dump(O / 'ARM_DEFINITION.json', dict(
        arm='A: learn only the actuators that BANC never commands',
        frozen='the recorded best vector, every entry except the learnable indices',
        frozen_count=int(vec0.size - len(learn_idx)),
        learnable_count=int(len(learn_idx)),
        synthetic_block=dict(start=int(syn0), count=int(nsw),
                             why='one signed weight per (dead actuator, driven actuator of the same leg) pair'),
        gain_block=dict(pool_dim=int(jp.POOL_DIM), count=len(gain_idx),
                        why='an actuator with no command is still scaled by its gain, so the gain belongs to the substitution'),
        dead_actuators=[dict(index=int(p['index']), leg=p['leg'], actuator=p['actuator'],
                             sources=list(p['sources'])) for p in plan],
        dead_by_leg={leg: sum(1 for p in plan if p['leg'] == leg) for leg in ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')},
        seed_vector=str(BEST / 'EVALS' / 'eval_556' / 'SCORE.json'),
        seed_J=REF_J,
        budget='same 768 evaluation cap as the parent 600 dimensional run, so the comparison is budget matched',
        honest_limit='64 of 90 actuators remain commanded by their own motor neurons here; raising J does not show the connectome can carry the body'))
    return vec0, learn_idx, _S['x0']


def encode(x):
    v = _S['vec0'].copy()
    nsw = _S['nsw']
    v[_S['learn_idx'][:nsw]] = np.clip(np.asarray(x, float)[:nsw], -4.0, 4.0)
    v[_S['learn_idx'][nsw:]] = np.clip(np.asarray(x, float)[nsw:], -1.0, 1.0)
    return v


def evaluate(args):
    idx, x = args
    pcr = _S['pcr']
    folder = O / 'EVALS' / ('eval_%04d' % idx)
    if folder.exists():
        shutil.rmtree(folder)
    folder.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = pcr.eval_one((idx, encode(x), folder))
    except BaseException:
        dump(folder / 'SCORE.json', dict(failed=True, traceback=traceback.format_exc()[-1000:]))
        return dict(idx=idx, J=float('inf'), failed=True)
    return dict(idx=idx, J=float(r.get('J', float('inf'))), parts=r.get('parts'),
                failed=bool(r.get('failed', False)))


def run():
    vec0, learn_idx, x0 = prepare()
    rng = np.random.default_rng(SEED)
    dim = int(x0.size)
    history = []

    r0 = evaluate((-1, x0))
    seed_reproduced = bool(abs(r0['J'] - REF_J) == 0.0)
    history.append(dict(generation=-1, kind='seed', J=r0['J'], parts=r0.get('parts'),
                        reproduced_exactly=seed_reproduced))
    dump(O / 'SEED_CHECK.json', history[0])
    print('seed J=%.15f reproduced_exactly=%s' % (r0['J'], seed_reproduced), flush=True)
    if not seed_reproduced:
        dump(O / 'INTERRUPTION.json', dict(reason='the frozen seed did not reproduce the parent score',
                                           expected=REF_J, got=r0['J']))
        raise SystemExit('seed mismatch: %r vs %r' % (r0['J'], REF_J))

    mean, sigma = x0.copy(), SIGMA0
    C = np.ones(dim)
    psigma = np.zeros(dim); pc = np.zeros(dim)
    cs = (4 + MU / dim) / (dim + 4 + 2 * MU / dim)
    ds = 1 + 2 * max(0.0, math.sqrt((MU - 1) / (dim + 1)) - 1) + cs
    chin = math.sqrt(MU / (dim + 2 * MU))
    mueff = MU
    cc = 4.0 / (dim + 4)
    c1 = 2.0 / ((dim + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
    w = math.log(MU + 0.5) - np.log(np.arange(1, MU + 1)); w = w / w.sum()
    best_x, best_J, best_parts = x0.copy(), r0['J'], r0.get('parts')
    evals = 1
    pool = CTX.Pool(WORKERS)

    for g in range(GENS):
        t0 = time.time()
        z = rng.standard_normal((LAM, dim))
        X = mean + sigma * z * np.sqrt(C)
        ys = list(pool.imap_unordered(evaluate, [(g * LAM + i, X[i]) for i in range(LAM)]))
        evals += LAM
        J = np.array([y['J'] for y in ys])
        order = np.argsort(J)
        if J[order[0]] < best_J:
            best_J = float(J[order[0]]); best_x = X[order[0]].copy()
            best_parts = ys[order[0]].get('parts')
        ysel = z[order[:MU]]
        yw = np.tensordot(w, ysel, axes=1)
        mean = mean + sigma * yw * np.sqrt(C)
        invsqrt = 1.0 / np.sqrt(C)
        psigma = (1 - cs) * psigma + math.sqrt(cs * (2 - cs) * mueff) * (yw * invsqrt)
        hsig = (np.linalg.norm(psigma) / math.sqrt(1 - (1 - cs) ** (2 * (g + 1)))
                / (1.4 + 2 / (dim + 1))) < 1.3
        pc = (1 - cc) * pc + (1.0 if hsig else 0.0) * math.sqrt(cc * (2 - cc) * mueff) * yw
        C = np.clip(C * (1 - c1 - cmu)
                    + c1 * (pc ** 2 + (1 - hsig) * cc * (2 - cc) * C)
                    + cmu * (w[:, None] * ysel ** 2).sum(axis=0), 1e-8, 1e8)
        sigma = float(np.clip(sigma * math.exp((cs / ds) * (np.linalg.norm(psigma) / chin - 1)), 1e-4, 2.0))
        history.append(dict(generation=g, best=best_J, gen_best=float(J[order[0]]),
                            median=float(np.median(J)), mean=float(J.mean()), sigma=sigma,
                            evals=evals, wall_s=time.time() - t0, best_idx=int(g * LAM + order[0])))
        print('gen %2d  gen_best=%.6f  best=%.6f  median=%.6f  sigma=%.4f  %.0fs'
              % (g, J[order[0]], best_J, np.median(J), sigma, time.time() - t0), flush=True)
        dump(O / 'CMA_HISTORY.json', history)
        np.save(O / 'BEST_X.npy', best_x); np.save(O / 'CENTER.npy', mean)
        nsw = _S['nsw']
        dump(O / 'BEST_VEC.json', dict(J=best_J, parts=best_parts, vec=encode(best_x).tolist(),
                                       synthetic_weights=np.clip(best_x[:nsw], -4, 4).tolist(),
                                       dead_gains=np.clip(best_x[nsw:], -1, 1).tolist()))
    pool.close(); pool.join()
    dump(O / 'FINAL.json', dict(arm='A', seed_J=REF_J, best_J=best_J, gain=REF_J - best_J,
                                evaluations=evals, generations=GENS, lam=LAM, dim=dim,
                                seed=SEED, sigma0=SIGMA0, reproduced_seed=seed_reproduced,
                                best_parts=best_parts,
                                note='frozen part is the recorded best; only the substitution layer and the dead actuator gains moved'))
    print('FINAL best_J=%.6f gain=%.6f' % (best_J, REF_J - best_J), flush=True)


if __name__ == '__main__':
    run()
