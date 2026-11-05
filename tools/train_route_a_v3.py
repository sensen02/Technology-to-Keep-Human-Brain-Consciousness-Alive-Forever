"""Route A, third attempt: use the parent's own optimiser instead of the one I invented.

Two earlier attempts at this arm both diverged, and the cause turned out to be my optimiser rather
than the arm.

  The parent 600 dimensional run does NOT use a CMA. tools/train_per_cell_readout.py:192-202 samples
  candidates as

      sig = [0.25 repeated 115 times] + [0.10 repeated for every newer block]
      X   = mean + sig * randn(lam, dim),  with X[0] forced to be the mean itself
      mean += sum over the top mu candidates of w * (X - mean)

  That is a plain evolution strategy with a FIXED per block step and a top-mu mean. It cannot
  diverge, because sigma never moves.

  The Route A trainers I wrote replaced that with a separable CMA-ES whose cumulative step size
  adaptation was wrong in two ways at once: cs came from a non-standard formula, and the
  normalisation used sqrt(mu/(n+2mu)) = 0.28 where E||N(0,I_n)|| is sqrt(n) = 15.9. The result was a
  sigma that grew by about a factor of eleven per generation, from 0.15 to 0.76 in one attempt and
  from 0.04 to 2.00 (its clip) in the second, with the best score never improving.

So this attempt removes the invented optimiser entirely and uses the parent's scheme unchanged, with
the parent's own step sizes per block: 0.10 for the substitution weights and 0.10 for the gains,
because those are the values the parent used for exactly these two blocks.

That choice also makes the comparison stricter rather than looser: the arm now differs from the
parent only in WHICH dimensions are free (252 against 600), not in how they are searched.

Everything else is unchanged: the frozen biology, the 576 evaluation cap, the seed vector, the
objective, the timestep.

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
O = Path(os.environ.get('RA_OUT', '/home/sensen/Desktop/engine修复验收/s5-route-a-substitute-v3/20261108T120000Z'))
BEST = Path('/home/sensen/Desktop/engine修复验收/s5-synthetic-readout-cont/20261107T120000Z')
REF_J = 3.402395132105888
TIMESTEP_BAND = 0.027848515225160853
LAM = int(os.environ.get('RA_LAM', '48')); MU = LAM // 2
GENS = int(os.environ.get('RA_GENS', '12'))
SEED = int(os.environ.get('RA_SEED', '20261108'))
WORKERS = int(os.environ.get('RA_WORKERS', '8'))
SIGMA = float(os.environ.get('RA_SIGMA', '0.10'))     # the parent's value for exactly these blocks
ARM = os.environ.get('RA_ARM', 'A').strip().upper()
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
    if ARM != 'A':
        # the ablation arms from Route B, installed where the network actually receives its matrix
        import scipy.sparse as sparse
        import engine.neural_cond as nc
        arm_src = Path('/home/sensen/Desktop/engine修复验收/s5-route-b-ablation/20261108T060000Z/ARMS') / ARM
        we_arm = sparse.load_npz(arm_src / 'ACTUAL_We.npz').tocsr()
        wi_arm = sparse.load_npz(arm_src / 'ACTUAL_Wi.npz').tocsr()
        sparse.save_npz(pcr.SET / 'ACTUAL_We.npz', we_arm)
        sparse.save_npz(pcr.SET / 'ACTUAL_Wi.npz', wi_arm)
        _orig_sc = nc.ConductanceNetwork.set_connectivity

        def _substituted(self, exc_uS, inh_uS):
            del exc_uS, inh_uS
            return _orig_sc(self, we_arm, wi_arm)

        nc.ConductanceNetwork.set_connectivity = _substituted
        print('arm %s: recurrent matrices installed (%d + %d edges)' % (ARM, we_arm.nnz, wi_arm.nnz), flush=True)
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
        arm=ARM,
        arm_meaning=('A: learn only the actuators that BANC never commands, on the real graph'
                     if ARM == 'A' else
                     'A search space applied to ablation arm %s instead of the real graph' % ARM),
        frozen='the recorded best vector, every entry except the learnable indices',
        frozen_count=int(vec0.size - len(learn_idx)),
        learnable_count=int(len(learn_idx)),
        optimiser="the parent's fixed step evolution strategy, identical to tools/train_per_cell_readout.py",
        sigma_per_block=dict(substitution_weights=SIGMA, dead_actuator_gains=SIGMA,
                            why='the parent used 0.10 for exactly these two blocks'),
        lam=LAM, mu=MU, generations=GENS, evaluations=GENS * LAM,
        dead_actuators=[dict(index=int(p['index']), leg=p['leg'], actuator=p['actuator'],
                             sources=list(p['sources'])) for p in plan],
        dead_by_leg={leg: sum(1 for p in plan if p['leg'] == leg) for leg in ('LF', 'LM', 'LH', 'RF', 'RM', 'RH')},
        seed_vector=str(BEST / 'EVALS' / 'eval_556' / 'SCORE.json'), seed_J=REF_J,
        honest_limit='64 of 90 actuators remain commanded by their own motor neurons here; raising J does not show the connectome can carry the body'))
    return learn_idx, _S['x0']


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
    learn_idx, x0 = prepare()
    rng = np.random.default_rng(SEED)
    dim = int(x0.size)
    sigma = np.full(dim, SIGMA)
    w = np.ones(MU) / MU
    history = []

    r0 = evaluate((-1, x0))
    reproduced = bool(abs(r0['J'] - REF_J) == 0.0)
    history.append(dict(generation=-1, kind='seed', J=r0['J'], parts=r0.get('parts'), reproduced_exactly=reproduced))
    dump(O / 'SEED_CHECK.json', dict(history[0], arm=ARM,
        note=('reproduced the recorded score exactly' if reproduced else
              'scored %r on arm %s; that is this arm\'s seed, not the recorded score' % (r0['J'], ARM))))
    print('seed J=%.15f reproduced_exactly=%s' % (r0['J'], reproduced), flush=True)
    if ARM == 'A' and not reproduced:
        dump(O / 'INTERRUPTION.json', dict(reason='arm A did not reproduce the parent score',
                                          expected=REF_J, got=r0['J']))
        raise SystemExit('seed mismatch')

    mean = x0.copy()
    best_x, best_J, best_parts = x0.copy(), r0['J'], r0.get('parts')
    evals = 1
    pool = CTX.Pool(WORKERS)
    for g in range(GENS):
        t0 = time.time()
        z = np.zeros((LAM, dim)) if g == 0 else rng.standard_normal((LAM, dim))
        X = mean[None, :] + sigma[None, :] * z
        if g > 0:
            X[0] = mean            # member zero is always the current mean, never a fresh sample
        # imap_unordered returns results in COMPLETION order, so the list has to be put back into
        # candidate order before it is used to index X. Without this the top mu candidates' SCORES
        # were paired with other candidates' VECTORS, which silently turns the ordered mean update
        # into a random one and makes any recorded best_idx point at the wrong evaluation. The
        # parent never had this bug because it calls the ordered pool.map.
        raw = list(pool.imap_unordered(evaluate, [(g * LAM + i, X[i]) for i in range(LAM)]))
        ys = [None] * LAM
        for y in raw:
            ys[y['idx'] - g * LAM] = y
        assert all(y is not None for y in ys), 'a candidate result went missing'
        evals += LAM
        J = np.array([y['J'] for y in ys])
        order = np.argsort(J)[:MU]
        if J[order[0]] < best_J:
            best_J = float(J[order[0]]); best_x = X[order[0]].copy(); best_parts = ys[order[0]].get('parts')
        mean = mean + np.sum(w[:, None] * (X[order] - mean[None, :]), axis=0)
        history.append(dict(generation=g, best=best_J, gen_best=float(J.min()), median=float(np.median(J)),
                            mean=float(J.mean()), sigma=float(sigma[0]), evals=evals,
                            wall_s=time.time() - t0, best_idx=int(g * LAM + int(np.argmin(J)))))
        print('gen %2d  gen_best=%.6f  best=%.6f  median=%.6f  %.0fs'
              % (g, J.min(), best_J, np.median(J), time.time() - t0), flush=True)
        dump(O / 'CMA_HISTORY.json', history)
        np.save(O / 'BEST_X.npy', best_x); np.save(O / 'CENTER.npy', mean)
        nsw = _S['nsw']
        dump(O / 'BEST_VEC.json', dict(J=best_J, parts=best_parts, vec=encode(best_x).tolist(),
                                       synthetic_weights=np.clip(best_x[:nsw], -4, 4).tolist(),
                                       dead_gains=np.clip(best_x[nsw:], -1, 1).tolist()))
    pool.close(); pool.join()
    dump(O / 'FINAL.json', dict(arm=ARM, seed_J=REF_J, seed_J_on_this_arm=r0['J'],
                                best_J=best_J, gain=r0['J'] - best_J, gain_vs_recorded=REF_J - best_J,
                                gain_in_timestep_bands=(r0['J'] - best_J) / TIMESTEP_BAND,
                                evaluations=evals, generations=GENS, lam=LAM, mu=MU, dim=dim,
                                seed=SEED, sigma=SIGMA, reproduced_seed=reproduced, best_parts=best_parts,
                                optimiser="the parent's fixed step ES",
                                note='compared against the timestep band of 0.0278; a gain below that is inside the discretisation band'))
    print('FINAL best_J=%.6f gain=%.6f (%.2f timestep bands)'
          % (best_J, r0['J'] - best_J, (r0['J'] - best_J) / TIMESTEP_BAND), flush=True)


if __name__ == '__main__':
    run()
