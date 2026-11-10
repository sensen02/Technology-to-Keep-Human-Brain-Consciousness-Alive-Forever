"""Locate the readout bottleneck: at what output gain does the saturated bang-bang
command become graded, given unchanged connectivity, neurons and sensory drive?

Only the decoder scalar is varied; the neuron model and connectome are untouched.
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


def arrays(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def main():
    run = rt.Runtime()
    poolmat = sparse.csr_matrix(run.poolmat)
    matrix, limits = run.matrix, run.limits
    We = sparse.load_npz(rt.b.AB / 'A' / 'ACTUAL_We.npz')
    Wi = sparse.load_npz(rt.b.AB / 'A' / 'ACTUAL_Wi.npz')
    seq = np.repeat(arrays(AUD)['currents'].astype(np.float64), 2, axis=0)[:STEPS]

    net = ConductanceNetwork(12000, ConductanceParams(), dt_ms=NDT * 1000)
    net.set_connectivity(We, Wi)
    rates = np.zeros(66)
    decay = np.exp(-NDT / .005)
    pool = np.zeros((STEPS // 2, 66))
    for k in range(STEPS):
        sp = net.step(current_nA=seq[k])
        rates = decay * rates + (1 - decay) * np.asarray(poolmat @ sp, float) / NDT
        if k % 2 == 1:
            pool[k // 2] = rates

    raw = np.tanh(0)  # placeholder to keep linters quiet
    command_unitless = pool @ matrix.T          # exactly the quantity inside limits/100 * (...)
    rows = []
    for gain in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
        torque = np.clip(limits / 100 * gain * command_unitless, -limits, limits)
        sat = float(np.mean(np.abs(torque) >= (limits[None, :] - 1e-9)))
        graded = float(np.mean((np.abs(torque) > 1e-9) & (np.abs(torque) < limits[None, :] * 0.99)))
        rows.append(dict(readout_gain=gain, saturation_fraction=sat, graded_fraction=graded,
                         torque_mean_abs=float(np.mean(np.abs(torque))),
                         torque_std_over_time=float(np.mean(torque.std(axis=0)))))
    # Natural reference: current behaviour uses gain 1.0
    base = rows[0]
    # What gain gives saturation near 0.10 (mostly graded control)?
    target = None
    for r in rows:
        if r['saturation_fraction'] <= 0.10:
            target = r
    out = dict(pool_rate_mean_hz=float(pool.mean()), pool_rate_max_hz=float(pool.max()),
               pool_rate_p95_hz=float(np.percentile(pool, 95)),
               current_gain=1.0, current_saturation=base['saturation_fraction'],
               scan=rows, gain_for_saturation_below_0p10=target,
               sources={str(p): sha(p) for p in [AUD, Path(rt.b.AB / 'A' / 'ACTUAL_We.npz'),
                                                Path(rt.b.AB / 'A' / 'ACTUAL_Wi.npz'),
                                                R / 'engine/neural_cond.py', BASE / 'runtime.py', Path(__file__)]},
               note='Only the decoder gain is swept. Connectivity, neuron model and sensory drive are unchanged, so this '
                    'isolates whether the saturated command is a readout-scale problem rather than a neural one.')
    dump(OUT / 'READOUT_GAIN_SCAN.json', out)
    print(json.dumps(out, indent=1)[:2500])


if __name__ == '__main__':
    main()
