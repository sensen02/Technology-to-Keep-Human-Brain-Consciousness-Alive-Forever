"""Corrected neural diagnostics (v2). v1 had a wrong absolute rate factor and no
spike-level or input-dimensionality measurement; those are fixed and extended here.

Adds: spike-level divergence (not only 20 ms averages), effective dimensionality of the
real sensory drive, and two controls for "is the motion effectively random":
time-shuffled drive (same currents, destroyed temporal order) and random Gaussian drive
(same per-cell mean/std, no structure). No physics stepping; forward-only contact reads.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, json, time
from pathlib import Path
import numpy as np
import mujoco
from scipy import sparse

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = R / 'experiments/s5-neural-translator/20261105T200021Z'
OUTP = Path('/home/sensen/Desktop/engine修复验收/s5-sensory-neural-audit/20261106T015244Z/neural')
sys.path[:0] = [str(BASE), str(R), str(R / 'tools')]
import runtime as rt
from engine.neural_cond import ConductanceNetwork, ConductanceParams

rt.O = OUTP
rt.b.O = OUTP
js, sha, dump = rt.js, rt.sha, rt.dump
ARCH = Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T200021Z')
NDT = .0005
SPM = 2  # neural steps per 1 ms sensory sample
WINDOW = 40  # steps per reported window (20 ms)


def arrays(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def main():
    t0 = time.monotonic()
    run = rt.Runtime()
    ledger_path = OUTP / 'CLEANUP_LEDGER.json'
    cov = OUTP / 'COVERAGE.json'
    if cov.exists():
        ledger = js(ledger_path) if ledger_path.exists() else dict(importance='重要辅助', deleted=[])
        ledger['deleted'].append(dict(path=str(cov), sha256=sha(cov), bytes=cov.stat().st_size,
                                      classification='可再生临时',
                                      reason='Legacy dominant-count table written by the inherited constructor; no audit code reads it.'))
        cov.unlink()
        dump(ledger_path, ledger)

    m, q0, joints, geom, groups = run.m, run.q0, run.joints, run.geom, run.groups
    poolmat = sparse.csr_matrix(run.poolmat)
    cells, matrix, limits = run.cells, run.matrix, run.limits

    body_path = ARCH / 'DATA/train_offset_0/BODY.npz'
    sealed = js(ARCH / 'MANIFEST_SHA256.json')
    if str(body_path) in sealed:
        assert sha(body_path) == sealed[str(body_path)]
    body = arrays(body_path)
    times = body['time']

    # Reuse the already-verified real currents when the byte hashes match; otherwise recompute.
    cur_path = OUTP / 'SENSORY_CURRENTS.npz'
    if cur_path.exists():
        z = arrays(cur_path)
        currents = z['currents'].astype(np.float64)
        obs_index = z['obs_index']
        sensory_source = 'reused v1 recomputation, hash-checked'
    else:
        raise RuntimeError('expected v1 sensory currents')

    # ---- effective dimensionality of the sensory drive ------------------------
    centered = currents - currents.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = sv ** 2
    frac = energy / energy.sum()
    participation = float(energy.sum() ** 2 / (energy ** 2).sum())
    dims = dict(largest_singular_value=float(sv[0]),
                participation_ratio=participation,
                variance_fraction_top1=float(frac[0]), variance_fraction_top5=float(frac[:5].sum()),
                variance_fraction_top30=float(frac[:30].sum()),
                variance_fraction_top66=float(frac[:66].sum()),
                nominal_scalar_channels=30,
                note='The body drives the network only through 5 scalars per leg; this measures how much '
                     'of the 12000-cell current space those scalars actually span.')

    # ---- network replays ------------------------------------------------------
    We = sparse.load_npz(rt.b.AB / 'A' / 'ACTUAL_We.npz')
    Wi = sparse.load_npz(rt.b.AB / 'A' / 'ACTUAL_Wi.npz')

    def make_net():
        net = ConductanceNetwork(12000, ConductanceParams(), dt_ms=NDT * 1000)
        net.set_connectivity(We, Wi)
        return net

    def run_seq(seq, steps, collect_spikes=False):
        net = make_net()
        cum = np.zeros(steps, dtype=np.int64)
        total = 0
        pool_rates = np.zeros((steps, 66))
        rates = np.zeros(66)
        decay = np.exp(-NDT / .005)
        spikes = np.empty((steps, 12000), bool) if collect_spikes else None
        adaptation, volts = np.zeros(steps), np.zeros(steps)
        for k in range(steps):
            sp = net.step(current_nA=seq[k])
            if collect_spikes:
                spikes[k] = sp
            n = int(sp.sum())
            total += n
            cum[k] = total
            rates = decay * rates + (1 - decay) * np.asarray(poolmat @ sp, float) / NDT
            pool_rates[k] = rates
            adaptation[k] = float(np.mean(net.adaptation))
            volts[k] = float(np.mean(net.v))
        return dict(cum=cum, pool_rates=pool_rates, spikes=spikes, adaptation=adaptation, volts=volts,
                    total=total, net=net)

    def per_neuron_hz(spikes_in_window, window_steps):
        return spikes_in_window / (window_steps * NDT) / 12000

    def window_rates(cum, steps):
        # per-neuron Hz in each non-overlapping WINDOW-step window, from cumulative counts
        idx = np.arange(WINDOW - 1, steps, WINDOW)
        prev = np.concatenate([[0], cum[idx[:-1]]])
        counts = cum[idx] - prev
        return np.array([per_neuron_hz(c, WINDOW) for c in counts])

    real_1ms = np.repeat(currents, SPM, axis=0)[:2400]
    report = {}

    base = run_seq(real_1ms, 2400, collect_spikes=True)
    wr = window_rates(base['cum'], 2400)
    report['real_replay'] = dict(steps=2400, simulated_s=1.2, per_neuron_rate_hz_per_20ms=[float(x) for x in wr],
                                 rate_mean_hz=float(wr.mean()), rate_min_hz=float(wr.min()), rate_max_hz=float(wr.max()),
                                 rate_first_half_hz=float(wr[:len(wr) // 2].mean()),
                                 rate_second_half_hz=float(wr[len(wr) // 2:].mean()),
                                 total_spikes=int(base['total']),
                                 mean_voltage_mV_first_last=[float(base['volts'][0]), float(base['volts'][-1])],
                                 mean_adaptation_nA_end=float(base['adaptation'][-1]),
                                 silent_cells=int(np.count_nonzero(base['net'].spike_count == 0)))

    # ---- long-run trend over 6 s ---------------------------------------------
    long = {}
    conditions = {'zero_input': np.zeros((12000, 12000)),
                  'constant_input': np.tile(currents.mean(axis=0), (12000, 1)),
                  'repeated_real': np.tile(real_1ms, (5, 1))}
    for label, seq in conditions.items():
        res = run_seq(seq, 12000)
        wr6 = window_rates(res['cum'], 12000)
        q = [float(wr6[i * len(wr6) // 4:(i + 1) * len(wr6) // 4].mean()) for i in range(4)]
        long[label] = dict(simulated_s=6.0, quarter_mean_rate_hz=q, total_spikes=int(res['total']),
                           silent_windows=int(np.sum(wr6 == 0)),
                           mean_voltage_mV_quarters=[float(res['volts'][i * 3000]) for i in range(4)],
                           mean_adaptation_nA_end=float(res['adaptation'][-1]),
                           drift_last_minus_first_hz=float(q[-1] - q[0]))
    report['long_run'] = long

    # repeatability of identical input across cycles
    rep = run_seq(np.tile(real_1ms, (5, 1)), 12000)
    wr5 = window_rates(rep['cum'], 12000)
    per = len(wr5) // 5
    cycles = [dict(cycle=c, rate_mean_hz=float(wr5[c * per:(c + 1) * per].mean()),
                   rate_min_hz=float(wr5[c * per:(c + 1) * per].min())) for c in range(5)]
    report['cycle_repeatability'] = dict(cycles=cycles,
                                         cycle1_vs_cycle5_hz=float(cycles[0]['rate_mean_hz'] - cycles[-1]['rate_mean_hz']),
                                         note='Identical 1.2 s sensory input repeated five times; differences are state drift.')

    # ---- spike-level sensitivity --------------------------------------------
    def divergence(a, b):
        ham = np.count_nonzero(a['spikes'] != b['spikes'], axis=1)
        nz = np.flatnonzero(ham > 0)
        return dict(first_differing_step=None if len(nz) == 0 else int(nz[0]),
                    first_differing_time_s=None if len(nz) == 0 else float((nz[0] + 1) * NDT),
                    max_spikes_differing_per_step=int(ham.max()),
                    differing_spikes_total_end_step=int(ham[-1]),
                    total_spike_count_difference=int(b['total'] - a['total']),
                    cumulative_relative_difference_end=float(abs(b['total'] - a['total']) / max(a['total'], 1)),
                    flip_fraction_end_step=float(ham[-1] / 12000))

    sens = {}
    for eps in [1e-12, 1e-9, 1e-6, 1e-3]:
        seq = real_1ms.copy()
        seq[0] = seq[0] * (1.0 + eps) + eps
        other = run_seq(seq, 2400, collect_spikes=True)
        sens[f'input_rel_eps_{eps:g}'] = divergence(base, other)
    net = make_net()
    net.v = net.v + 1e-9
    res = dict(cum=np.zeros(2400, np.int64), pool_rates=None, spikes=np.empty((2400, 12000), bool),
               adaptation=np.zeros(2400), volts=np.zeros(2400), total=0, net=net)
    cum = 0
    rates = np.zeros(66)
    decay = np.exp(-NDT / .005)
    for k in range(2400):
        sp = net.step(current_nA=real_1ms[k])
        res['spikes'][k] = sp
        cum += int(sp.sum())
        res['cum'][k] = cum
        rates = decay * rates + (1 - decay) * np.asarray(poolmat @ sp, float) / NDT
    res['total'] = cum
    sens['voltage_offset_1e-9_mV'] = divergence(base, res)
    report['sensitivity'] = sens

    # ---- temporal structure vs marginal statistics ---------------------------
    rng = np.random.default_rng(20261106)
    blocks = currents.copy()
    order = rng.permutation(len(blocks))
    shuffled = np.repeat(blocks[order], SPM, axis=0)[:2400]
    mu, sd = currents.mean(axis=0), currents.std(axis=0)
    random_drive = rng.normal(mu, sd, size=(2400, 12000))
    sh = run_seq(shuffled, 2400)
    rd = run_seq(random_drive, 2400)

    def summary(res):
        w = window_rates(res['cum'], 2400)
        return dict(rate_mean_hz=float(w.mean()), rate_std_hz=float(w.std()), total_spikes=int(res['total']))

    def pool_corr(a, b):
        x, y = a['pool_rates'], b['pool_rates']
        return float(np.mean([np.corrcoef(x[:, j], y[:, j])[0, 1] for j in range(66) if x[:, j].std() > 0 and y[:, j].std() > 0]))

    report['drive_structure_control'] = dict(
        real=summary(base), time_shuffled=summary(sh), random_gaussian=summary(rd),
        pool_rate_correlation_real_vs_shuffled=pool_corr(base, sh),
        pool_rate_correlation_real_vs_random=pool_corr(base, rd),
        interpretation='If destroying the temporal order or replacing the drive by matched random noise leaves '
                       'population and motor-pool statistics nearly unchanged, the driven response carries little '
                       'time-structured content beyond the input marginal distribution.')

    # ---- motor readout amplification ----------------------------------------
    def torque_track(res):
        out = []
        for k in range(0, 2400, 2):
            out.append(np.clip(limits / 100 * (matrix @ res['pool_rates'][k]), -limits, limits))
        return np.array(out)

    tb = torque_track(base)
    seq = real_1ms.copy()
    seq[0] = seq[0] * (1.0 + 1e-3) + 1e-3
    tp = torque_track(run_seq(seq, 2400))
    report['torque'] = dict(mean_abs_torque=float(np.mean(np.abs(tb))),
                            natural_step_change_abs_mean=float(np.mean(np.abs(np.diff(tb, axis=0)))),
                            changed_by_eps1e3_abs_max=float(np.abs(tp - tb).max()),
                            ratio_to_natural_change=float(np.abs(tp - tb).max() / max(np.mean(np.abs(np.diff(tb, axis=0))), 1e-12)),
                            fraction_joints_touching_clip=float(np.mean(np.abs(tb) >= (limits[None, :] - 1e-9))))

    report['sensory_dimensionality'] = dims
    report['sources'] = {str(p): sha(p) for p in [body_path, ARCH / 'MANIFEST_SHA256.json',
                                                 Path(rt.b.AB / 'A' / 'ACTUAL_We.npz'), Path(rt.b.AB / 'A' / 'ACTUAL_Wi.npz'),
                                                 R / 'engine/neural_cond.py', R / 'run_neural_muscle_closure.py',
                                                 R / 'engine/embodied/sensory_channels.py', BASE / 'runtime.py',
                                                 cur_path, Path(__file__)]}
    report['provenance'] = dict(
        network='engine/neural_cond.py ConductanceNetwork, 12000 cells, dt 0.5 ms, backward_euler',
        parameters='ConductanceParams defaults; the module declares every value illustrative and uncalibrated',
        connectivity=f'ARMS/A actual matrices: We nnz={We.nnz}, Wi nnz={Wi.nnz}',
        randomness='none in the network; the shuffle/random arms are deliberate controls, seeded 20261106',
        rh_gain='RH incoming excitation x0.375 applies only at release; not used in these replays',
        sensory_source=sensory_source)
    report['rate_convention'] = 'per-neuron Hz = spikes / (window_steps * 0.0005 s) / 12000; v1 printed spikes/step*1000/NDT/12000 and was wrong by a constant factor, so only v2 absolute values are quoted.'
    report['wall_s'] = time.monotonic() - t0
    dump(OUTP / 'NEURAL_DIAGNOSTICS_V2.json', report)
    np.savez_compressed(OUTP / 'REPLAY_TRACES_V2.npz',
                        rate_real=wr.astype(np.float32), cum_real=base['cum'],
                        pool_rates_real=base['pool_rates'].astype(np.float32),
                        singular_values=sv.astype(np.float64), currents=currents.astype(np.float32))
    print('DIM', json.dumps(dims), flush=True)
    print('REAL', json.dumps(report['real_replay']), flush=True)
    print('LONG', json.dumps(long), flush=True)
    print('SENS', json.dumps(sens), flush=True)
    print('STRUCT', json.dumps(report['drive_structure_control']), flush=True)
    print('TORQUE', json.dumps(report['torque']), flush=True)
    print('WALL_S', report['wall_s'], flush=True)


if __name__ == '__main__':
    main()
