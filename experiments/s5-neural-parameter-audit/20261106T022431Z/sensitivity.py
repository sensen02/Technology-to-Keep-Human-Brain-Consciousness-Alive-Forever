"""Bounded electrical-parameter sensitivity audit on the UNCHANGED connectome.

Connectivity (A: We/Wi) is never modified. Only shared scalar parameters of
ConductanceParams float inside declared bounds; per-neuron threshold jitter is a
separate, clearly labelled arm. No physics; sensory drive is the same real
archived replay used by the fidelity audit, so differences are attributable to
parameters alone.

Purpose: decide whether a bounded physical search is worth its cost, by measuring
(1) which parameters actually move the population rate and the motor command,
(2) whether any admissible setting escapes the ~99 Hz dense-firing regime and the
saturated readout, and (3) how much output variance parameters explain compared
with the input-driven temporal variance.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, json, time, hashlib
from dataclasses import replace, asdict
from pathlib import Path
import numpy as np
from scipy import sparse

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = R / 'experiments/s5-neural-translator/20261105T200021Z'
OUT = Path('/home/sensen/Desktop/engine修复验收/s5-neural-parameter-audit/20261106T022431Z')
SRC = Path(__file__).resolve().parent
sys.path[:0] = [str(BASE), str(R), str(R / 'tools')]
import runtime as rt
from engine.neural_cond import ConductanceNetwork, ConductanceParams

rt.O = OUT
rt.b.O = OUT
js, sha, dump = rt.js, rt.sha, rt.dump
AUD = Path('/home/sensen/Desktop/engine修复验收/s5-sensory-neural-audit/20261106T015244Z/neural/SENSORY_CURRENTS.npz')
NDT = .0005
STEPS = 1200           # 600 ms of real sensory replay per variant
WINDOW = 20            # 10 ms rate windows


def arrays(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    run = rt.Runtime()
    cov = OUT / 'COVERAGE.json'
    if cov.exists():
        ledger = OUT / 'CLEANUP_LEDGER.json'
        d = js(ledger) if ledger.exists() else dict(importance='重要辅助', deleted=[])
        d['deleted'].append(dict(path=str(cov), sha256=sha(cov), bytes=cov.stat().st_size,
                                 classification='可再生临时',
                                 reason='Legacy dominant-count table emitted by the inherited constructor; unused here.'))
        cov.unlink()
        dump(ledger, d)

    poolmat = sparse.csr_matrix(run.poolmat)
    matrix, limits = run.matrix, run.limits
    We = sparse.load_npz(rt.b.AB / 'A' / 'ACTUAL_We.npz')
    Wi = sparse.load_npz(rt.b.AB / 'A' / 'ACTUAL_Wi.npz')

    z = arrays(AUD)
    currents = z['currents'].astype(np.float64)
    seq = np.repeat(currents, 2, axis=0)[:STEPS]
    assert seq.shape == (STEPS, 12000)

    base = ConductanceParams()

    # Declared admissible bounds: each value must stay inside these, and derived
    # constraints (reset < threshold, delay an integer number of steps) must hold.
    BOUNDS = {
        'capacitance_nF': (0.01, 0.04),
        'leak_uS': (0.0005, 0.002),
        'rest_mV': (-70.0, -60.0),
        'threshold_mV': (-56.0, -44.0),
        'reset_mV': (-72.0, -62.0),
        'exc_reversal_mV': (-5.0, 5.0),
        'inh_reversal_mV': (-75.0, -65.0),
        'exc_tau_ms': (1.5, 6.0),
        'inh_tau_ms': (4.0, 16.0),
        'adaptation_tau_ms': (50.0, 200.0),
        'adaptation_increment_nA': (0.001, 0.008),
        'refractory_ms': (1.0, 4.0),
        'delay_ms': (1.0, 4.0),
    }

    def evaluate(params, thr_offset=None, weight_scale=1.0):
        net = ConductanceNetwork(12000, params, dt_ms=NDT * 1000, threshold_offset_mV=thr_offset)
        net.set_connectivity(We * weight_scale, Wi * weight_scale)
        cum = np.zeros(STEPS, np.int64)
        total = 0
        rates = np.zeros(66)
        decay = np.exp(-NDT / .005)
        torque = np.zeros((STEPS // 2, 42))
        pool = np.zeros((STEPS, 66))
        adapt = np.zeros(STEPS)
        spikes_by_cell = np.zeros(12000, np.int64)
        for k in range(STEPS):
            sp = net.step(current_nA=seq[k])
            spikes_by_cell += sp
            n = int(sp.sum())
            total += n
            cum[k] = total
            rates = decay * rates + (1 - decay) * np.asarray(poolmat @ sp, float) / NDT
            pool[k] = rates
            if k % 2 == 1:
                torque[k // 2] = np.clip(limits / 100 * (matrix @ rates), -limits, limits)
            adapt[k] = float(np.mean(net.adaptation))
        idx = np.arange(WINDOW - 1, STEPS, WINDOW)
        prev = np.concatenate([[0], cum[idx[:-1]]])
        window_rate = (cum[idx] - prev) / (WINDOW * NDT) / 12000
        return dict(total=int(total), rate_mean_hz=float(window_rate.mean()), rate_std_hz=float(window_rate.std()),
                    rate_min_hz=float(window_rate.min()), rate_max_hz=float(window_rate.max()),
                    active_cells=int(np.count_nonzero(spikes_by_cell)),
                    silent_cells=int(np.count_nonzero(spikes_by_cell == 0)),
                    pool_rate_mean_hz=float(pool.mean()), pool_rate_max_hz=float(pool.max()),
                    torque_mean_abs=float(np.mean(np.abs(torque))),
                    saturation_fraction=float(np.mean(np.abs(torque) >= (limits[None, :] - 1e-9))),
                    adapt_end_nA=float(adapt[-1]),
                    torque=torque, window_rate=window_rate)

    reference = evaluate(base)
    ref_torque = reference['torque']
    ref_scale = float(np.mean(np.abs(ref_torque))) + 1e-12

    sweep = {}
    for name, (lo, hi) in BOUNDS.items():
        levels = np.linspace(lo, hi, 7)
        rows = []
        for v in levels:
            if name == 'delay_ms':
                v = float(round(v / NDT) * NDT) if abs(v / NDT - round(v / NDT)) < 1e-9 else float(round(v / NDT)) * NDT
                v = float(round(v / NDT)) * NDT
            cand = replace(base, **{name: float(v)})
            try:
                cand.validate()
            except ValueError as e:
                rows.append(dict(value=float(v), status='rejected_by_model_constraint', error=str(e)))
                continue
            res = evaluate(cand)
            diff = float(np.sqrt(np.mean((res['torque'] - ref_torque) ** 2)) / ref_scale)
            rows.append(dict(value=float(v), status='ok', rate_mean_hz=res['rate_mean_hz'], rate_std_hz=res['rate_std_hz'],
                             active_cells=res['active_cells'], torque_mean_abs=res['torque_mean_abs'],
                             saturation_fraction=res['saturation_fraction'],
                             cmd_relative_rms_change=diff, adapt_end_nA=res['adapt_end_nA']))
        ok = [r for r in rows if r['status'] == 'ok']
        el = None
        if len(ok) >= 3:
            x = np.array([r['value'] for r in ok], float)
            y = np.array([r['rate_mean_hz'] for r in ok], float)
            if np.all(x > 0) and np.all(y > 0) and np.ptp(np.log(x)) > 1e-12:
                slope = np.polyfit(np.log(x), np.log(y), 1)[0]
                el = float(slope)
        sweep[name] = dict(bounds=[lo, hi], levels=rows, log_log_rate_elasticity=el,
                           rate_range_hz=[min(r['rate_mean_hz'] for r in ok), max(r['rate_mean_hz'] for r in ok)] if ok else None,
                           max_cmd_relative_rms_change=max((r['cmd_relative_rms_change'] for r in ok), default=None),
                           sat_range=[min(r['saturation_fraction'] for r in ok), max(r['saturation_fraction'] for r in ok)] if ok else None,
                           reaches_sparse_regime=bool(ok and min(r['rate_mean_hz'] for r in ok) <= 20.0),
                           reaches_zero=sorted({round(r['rate_mean_hz'], 6) for r in ok} & {0.0}) != [])

    # Arm: per-neuron threshold jitter (declared separate; connectivity untouched).
    rng = np.random.default_rng(20261106)
    jitter = {}
    for sd in [0.5, 1.0, 2.0, 3.0, 4.0]:
        off = rng.normal(0, sd, 12000)
        try:
            res = evaluate(base, thr_offset=off)
        except ValueError as e:
            jitter[f'threshold_jitter_sd_{sd:g}_mV'] = dict(status='rejected_by_model_constraint', error=str(e),
                                                           min_offset_mV=float(off.min()))
            continue
        jitter[f'threshold_jitter_sd_{sd:g}_mV'] = dict(rate_mean_hz=res['rate_mean_hz'], rate_std_hz=res['rate_std_hz'],
                                                       active_cells=res['active_cells'],
                                                       saturation_fraction=res['saturation_fraction'],
                                                       cmd_relative_rms_change=float(np.sqrt(np.mean((res['torque'] - ref_torque) ** 2)) / ref_scale))

    # Arm: global synaptic scale (weights only; connectivity untouched) - flagged as a
    # strength change, not a parameter of the neuron model.
    scale_arm = {}
    for s in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        res = evaluate(base, weight_scale=s)
        scale_arm[f'synaptic_scale_{s:g}'] = dict(rate_mean_hz=res['rate_mean_hz'], rate_std_hz=res['rate_std_hz'],
                                                  active_cells=res['active_cells'],
                                                  torque_mean_abs=res['torque_mean_abs'],
                                                  saturation_fraction=res['saturation_fraction'],
                                                  cmd_relative_rms_change=float(np.sqrt(np.mean((res['torque'] - ref_torque) ** 2)) / ref_scale))

    # Variance decomposition: parameters vs time.
    traces = {}
    for name in ['threshold_mV', 'leak_uS', 'refractory_ms', 'exc_tau_ms']:
        lo, hi = BOUNDS[name]
        traces[name] = np.stack([evaluate(replace(base, **{name: float(v)}))['torque'] for v in [lo, (lo + hi) / 2, hi]])
    across = np.mean([t.std(axis=0).mean() for t in traces.values()])
    within = float(np.mean([np.abs(np.diff(ref_torque, axis=0)).mean()]))
    jit_off = rng.normal(0, 2.0, 12000)
    jit_t = evaluate(base, thr_offset=jit_off)['torque']
    ref_detrended = ref_torque - ref_torque.mean(axis=0, keepdims=True)
    # Effective spread of torque across parameter settings, relative to within-run temporal spread.
    decomp = dict(across_setting_spread=float(across), within_run_step_change=within,
                  ratio_setting_over_time=float(across / within),
                  threshold_jitter2mV_change_abs_max=float(np.abs(jit_t - ref_torque).max()),
                  threshold_jitter2mV_change_relative=float(np.abs(jit_t - ref_torque).max() / ref_scale))

    admissibility = dict(
        rule='A setting is admissible only if it stays inside the declared bounds, satisfies the model constraints '
             '(reset < threshold, integer delay steps) and does not silence the network (rate > 0) nor exceed 60 Hz mean '
             'population rate for a sparse regime claim.',
        sparse_candidates=[n for n, v in sweep.items() if v['reaches_sparse_regime']],
        silent_arms=[n for n, v in sweep.items() if v['reaches_zero']],
        note='Sparse-regime means mean population rate <= 20 Hz inside bounds; 0 Hz arms mean the bound range can silence the network.')

    report = dict(stage='bounded electrical-parameter sensitivity audit, connectivity unchanged',
                  importance='核心必需',
                  bounds=BOUNDS, steps=STEPS, simulated_s=STEPS * NDT,
                  sensory_drive=str(AUD), sensory_drive_sha256=sha(AUD),
                  reference=dict(rate_mean_hz=reference['rate_mean_hz'], rate_std_hz=reference['rate_std_hz'],
                                 active_cells=reference['active_cells'], silent_cells=reference['silent_cells'],
                                 pool_rate_mean_hz=reference['pool_rate_mean_hz'], pool_rate_max_hz=reference['pool_rate_max_hz'],
                                 torque_mean_abs=reference['torque_mean_abs'],
                                 saturation_fraction=reference['saturation_fraction'],
                                 adapt_end_nA=reference['adapt_end_nA']),
                  per_parameter=sweep, threshold_jitter=jitter, synaptic_scale=scale_arm,
                  variance_decomposition=decomp, admissibility=admissibility,
                  provenance=dict(connectivity=f'ARMS/A actual matrices, unchanged: We nnz={We.nnz}, Wi nnz={Wi.nnz}',
                                  parameters='ConductanceParams; only the swept scalar is replaced via dataclasses.replace',
                                  classification='engineering sensitivity, NOT biological calibration; no animal electrophysiology exists in this project'),
                  sources={str(p): sha(p) for p in [AUD, Path(rt.b.AB / 'A' / 'ACTUAL_We.npz'), Path(rt.b.AB / 'A' / 'ACTUAL_Wi.npz'),
                                                    R / 'engine/neural_cond.py', BASE / 'runtime.py', Path(__file__)]},
                  wall_s=time.monotonic() - t0)
    dump(OUT / 'PARAMETER_SENSITIVITY.json', report)
    np.savez_compressed(OUT / 'TORQUE_TRACES.npz', reference=ref_torque.astype(np.float32),
                        **{f'{k}_lo_mid_hi': v.astype(np.float32) for k, v in traces.items()},
                        window_rate_reference=reference['window_rate'].astype(np.float32))
    print('REF', json.dumps(report['reference']), flush=True)
    for n, v in sweep.items():
        print(f"{n:26s} el={v['log_log_rate_elasticity']} rate={v['rate_range_hz']} sat={v['sat_range']} cmd={v['max_cmd_relative_rms_change']}", flush=True)
    print('JITTER', json.dumps(jitter), flush=True)
    print('SCALE', json.dumps(scale_arm), flush=True)
    print('DECOMP', json.dumps(decomp), flush=True)
    print('ADM', json.dumps(admissibility), flush=True)
    print('WALL_S', report['wall_s'], flush=True)


if __name__ == '__main__':
    main()
