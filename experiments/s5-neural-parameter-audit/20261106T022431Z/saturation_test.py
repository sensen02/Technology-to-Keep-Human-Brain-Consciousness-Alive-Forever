"""Follow-up: is the large command change under tiny weight error a real sensitivity,
or an artefact of the saturated readout quantising joints to +-limit?

Same perturbations, same replay; the only change is that the command is also evaluated
before clipping and at a small gain where almost nothing saturates.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, json
from pathlib import Path
import numpy as np
from scipy import sparse

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = R / 'experiments/s5-neural-translator/20261105T200021Z'
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-neural-parameter-audit/20261106T022431Z')
sys.path[:0] = [str(BASE), str(R), str(R / 'tools')]
import runtime as rt
from engine.neural_cond import ConductanceNetwork, ConductanceParams

rt.O = OUT
rt.b.O = OUT
js, sha, dump = rt.js, rt.sha, rt.dump
AUD = Path('/home/sensen/Desktop/engine修复验收/s5-sensory-neural-audit/20261106T015244Z/neural/SENSORY_CURRENTS.npz')
NDT = .0005
STEPS = 1200
SIGMAS = [0.001, 0.01, 0.05, 0.20]
SEEDS = [1, 2, 3]


def arrays(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def main():
    run = rt.Runtime()
    poolmat = sparse.csr_matrix(run.poolmat)
    matrix, limits = run.matrix, run.limits
    We = sparse.load_npz(rt.b.AB / 'A' / 'ACTUAL_We.npz').tocsr()
    Wi = sparse.load_npz(rt.b.AB / 'A' / 'ACTUAL_Wi.npz').tocsr()
    seq = np.repeat(arrays(AUD)['currents'].astype(np.float64), 2, axis=0)[:STEPS]

    def pool_track(we, wi):
        net = ConductanceNetwork(12000, ConductanceParams(), dt_ms=NDT * 1000)
        net.set_connectivity(we, wi)
        rates = np.zeros(66)
        decay = np.exp(-NDT / .005)
        out = np.zeros((STEPS // 2, 66))
        for k in range(STEPS):
            sp = net.step(current_nA=seq[k])
            rates = decay * rates + (1 - decay) * np.asarray(poolmat @ sp, float) / NDT
            if k % 2 == 1:
                out[k // 2] = rates
        return out

    ref_pool = pool_track(We, Wi)
    raw = ref_pool @ matrix.T                      # unitless command before any scaling
    means = {}

    def compare(pool):
        unitless = pool @ matrix.T
        row = {}
        # (a) as used today: gain 1.0, clipped
        t1 = np.clip(limits / 100 * unitless, -limits, limits)
        r1 = np.clip(limits / 100 * raw, -limits, limits)
        row['clipped_gain1_rms_rel'] = float(np.sqrt(np.mean((t1 - r1) ** 2)) / (np.mean(np.abs(r1)) + 1e-12))
        row['clipped_gain1_sign_flip_fraction'] = float(np.mean(np.sign(t1) != np.sign(r1)))
        # (b) pre-clip (no saturation at all)
        row['unclipped_rms_rel'] = float(np.sqrt(np.mean((unitless - raw) ** 2)) / (np.mean(np.abs(raw)) + 1e-12))
        # (c) small gain where almost nothing saturates
        g = 0.02
        t3 = np.clip(limits / 100 * g * unitless, -limits, limits)
        r3 = np.clip(limits / 100 * g * raw, -limits, limits)
        row['gain0.02_rms_rel'] = float(np.sqrt(np.mean((t3 - r3) ** 2)) / (np.mean(np.abs(r3)) + 1e-12))
        row['gain0.02_saturation'] = float(np.mean(np.abs(t3) >= (limits[None, :] - 1e-9)))
        row['pool_rate_rms_rel'] = float(np.sqrt(np.mean((pool - ref_pool) ** 2)) / (np.mean(np.abs(ref_pool)) + 1e-12))
        return row

    for sigma in SIGMAS:
        rows = []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            we = We.copy()
            wi = Wi.copy()
            we.data = we.data * np.exp(rng.normal(0, sigma, We.nnz))
            wi.data = wi.data * np.exp(rng.normal(0, sigma, Wi.nnz))
            rows.append(compare(pool_track(we, wi)))
        means[f'sigma_{sigma:g}'] = dict(
            sigma=sigma,
            clipped_gain1_rms_rel=[min(r['clipped_gain1_rms_rel'] for r in rows), max(r['clipped_gain1_rms_rel'] for r in rows)],
            clipped_gain1_sign_flip=[min(r['clipped_gain1_sign_flip_fraction'] for r in rows),
                                     max(r['clipped_gain1_sign_flip_fraction'] for r in rows)],
            unclipped_rms_rel=[min(r['unclipped_rms_rel'] for r in rows), max(r['unclipped_rms_rel'] for r in rows)],
            gain0p02_rms_rel=[min(r['gain0.02_rms_rel'] for r in rows), max(r['gain0.02_rms_rel'] for r in rows)],
            gain0p02_saturation=[min(r['gain0.02_saturation'] for r in rows), max(r['gain0.02_saturation'] for r in rows)],
            pool_rate_rms_rel=[min(r['pool_rate_rms_rel'] for r in rows), max(r['pool_rate_rms_rel'] for r in rows)])

    out = dict(stage='saturation vs dynamical amplification follow-up', importance='核心必需',
               baseline=dict(clipped_saturation=float(np.mean(np.abs(np.clip(limits / 100 * raw, -limits, limits))
                                                                 >= (limits[None, :] - 1e-9))),
                             pool_rate_mean_hz=float(ref_pool.mean())),
               by_sigma=means,
               interpretation='If the clipped command changes a lot at tiny sigma while the unclipped command and the '
                              'pool rates change little, the apparent fragility is readout quantisation at the actuator '
                              'limit, not amplification in the neural dynamics.',
               boundaries=['1200-step replay, single real sensory segment', 'strength-only perturbation, structure fixed',
                           'absolute forces and SI remain NOT_ESTABLISHED'])
    dump(OUT / 'SATURATION_VS_AMPLIFICATION.json', out)
    print(json.dumps(out, ensure_ascii=False, indent=1)[:3000])


if __name__ == '__main__':
    main()
