"""Read-only neural fidelity diagnostics on the unchanged archived network.

Three questions, measured rather than assumed:
1. Real sensory replay: what currents does the actual body contact geometry deliver?
2. Long-run stability: does population activity degrade over simulated time?
3. Sensitivity: how fast do arbitrarily small input differences amplify?

No physics stepping after the initial forward-only sensory reconstruction; the
physics arrays are only used through mj_forward to read contacts per archived row.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MUJOCO_GL', 'egl')
import sys, json, hashlib, time
from pathlib import Path
import numpy as np
import mujoco
from scipy import sparse

R = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = R / 'experiments/s5-neural-translator/20261105T200021Z'
sys.path[:0] = [str(BASE), str(R), str(R / 'tools')]
import runtime as rt
from engine.neural_cond import ConductanceNetwork, ConductanceParams

O = Path('/home/sensen/Desktop/engine修复验收/s5-sensory-neural-audit/20261106T015244Z/neural')
created_now = not O.exists()
O.mkdir(parents=True, exist_ok=True)
C = Path(__file__).resolve().parent
rt.O = O
rt.b.O = O
js, sha, dump = rt.js, rt.sha, rt.dump
ARCH = Path('/home/sensen/Desktop/engine修复验收/s5-neural-translator/20261105T200021Z')
FAILED_ATTEMPTS = [
    dict(attempt=1, interpreter='venv_body (no torch)', exit_code=1, reason='runtime imports translator which imports torch', physics=False),
    dict(attempt=2, interpreter='translator .venv (torch + mujoco)', exit_code=1,
         reason='ChannelSensoryEncoder called with wrong positional signature (groups, cfg) instead of (groups, n_neurons, cfg)',
         physics=False, note='Shell pipeline hid the exit code in the first attempt; failures were re-checked with an explicit EXIT marker.')]
NDT = .0005
STEPS_PER_MS = 2


def cleanup_coverage():
    p = O / 'COVERAGE.json'
    if p.exists():
        ledger = dict(importance='重要辅助', deleted=[dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size,
                                                         classification='可再生临时',
                                                         reason='Legacy dominant-count table written by inherited constructor; no audit code reads it.')])
        p.unlink()
        dump(O / 'CLEANUP_LEDGER.json', ledger)


def arrays(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def main():
    t0 = time.monotonic()
    rt_runtime = rt.Runtime()
    cleanup_coverage()
    m, q0, joints, geom, groups = rt_runtime.m, rt_runtime.q0, rt_runtime.joints, rt_runtime.geom, rt_runtime.groups
    poolmat = rt_runtime.poolmat
    poolmat = sparse.csr_matrix(poolmat)
    cells = rt_runtime.cells
    matrix = rt_runtime.matrix
    limits = rt_runtime.limits

    body_path = ARCH / 'DATA/train_offset_0/BODY.npz'
    sealed = js(ARCH / 'MANIFEST_SHA256.json')
    if str(body_path) in sealed:
        assert sha(body_path) == sealed[str(body_path)]
    body = arrays(body_path)
    times = body['time']
    assert abs(times[1] - times[0] - NDT) < 1e-12

    from run_neural_muscle_closure import observe_quantities
    from engine.embodied.sensory_channels import ChannelSensoryEncoder, ChannelConfig

    enc = ChannelSensoryEncoder(groups, 12000, ChannelConfig())
    d = rt_runtime.mj.MjData(m)
    obs_index = np.arange(400, len(times), 2)  # network starts at 0.2 s, sensory sampled every 1 ms
    currents = []
    per_leg_series = {leg: {k: [] for k in ('force', 'touch', 'tib', 'tibvel', 'multi', 'multi_spread')} for leg in joints}
    contact_records = 0
    touch_with_zero_force = 0
    force_values = []
    for i in obs_index:
        d.qpos[:] = body['qpos'][i]
        d.qvel[:] = body['qvel'][i]
        d.ctrl[:] = 0.0
        rt_runtime.mj.mj_forward(m, d)
        q = observe_quantities(m, d, q0, joints, geom)
        cur, rec = enc.encode(q)
        currents.append(np.asarray(cur, float))
        for leg in joints:
            j = joints[leg]
            per_leg_series[leg]['force'].append(q[leg].contact_force_norm)
            per_leg_series[leg]['touch'].append(q[leg].foot_contact)
            per_leg_series[leg]['tib'].append(q[leg].tibia_joint_dev_rad)
            per_leg_series[leg]['tibvel'].append(q[leg].tibia_joint_vel_rad_s)
            per_leg_series[leg]['multi'].append(q[leg].multi_joint_dev_rad)
            spread = float(np.max(np.abs(d.qpos[j['q']] - q0[j['q']])))
            per_leg_series[leg]['multi_spread'].append(spread)
            if q[leg].foot_contact > 0 and q[leg].contact_force_norm == 0.0:
                touch_with_zero_force += 1
            if q[leg].contact_force_norm > 0:
                force_values.append(q[leg].contact_force_norm)
        # Count raw foot-plane contacts and their reported force magnitudes independently.
        for ci in range(d.ncon):
            c = d.contact[ci]
            for g, other in ((int(c.geom1), int(c.geom2)), (int(c.geom2), int(c.geom1))):
                if g in geom and m.geom_type[other] == mujoco.mjtGeom.mjGEOM_PLANE:
                    contact_records += 1
    currents = np.array(currents, dtype=np.float64)
    assert currents.shape == (len(obs_index), 12000)

    # Sensory geometry summary: how often each channel is active, and how the signed scalars are spread.
    legs = sorted(joints)
    geom_summary = {}
    cancel_events = 0
    for leg in legs:
        s = {k: np.asarray(v, float) for k, v in per_leg_series[leg].items()}
        cancel = (s['multi_spread'] > .05) & (np.abs(s['multi']) < .05)
        cancel_events += int(cancel.sum())
        geom_summary[leg] = dict(
            touch_fraction=float(np.mean(s['touch'])),
            force_max=float(s['force'].max()), force_median=float(np.median(s['force'])),
            force_p90=float(np.quantile(s['force'], .9)),
            tib_range=[float(s['tib'].min()), float(s['tib'].max())],
            tibvel_abs_max=float(np.abs(s['tibvel']).max()),
            multi_range=[float(s['multi'].min()), float(s['multi'].max())],
            multi_spread_median=float(np.median(s['multi_spread'])),
            mean_cancellation_samples=int(cancel.sum()),
            samples=int(len(s['touch'])))
    encoder_coverage = enc.coverage()
    nonzero_cell_fraction = float(np.mean(np.any(currents != 0, axis=0)))
    per_cell_active = np.mean(currents != 0, axis=0)
    current_abs = np.abs(currents)
    sensory_summary = dict(
        samples=int(len(currents)), sample_interval_ms=1.0, total_cells=12000,
        cells_reached_by_any_current=int(np.sum(nonzero_cell_fraction > 0) * 12000) if False else int(np.count_nonzero(np.any(currents != 0, axis=0))),
        nonzero_cell_fraction=nonzero_cell_fraction,
        active_fraction_per_cell_mean=float(per_cell_active.mean()),
        current_abs_max=float(current_abs.max()), current_abs_p99=float(np.quantile(current_abs, .99)),
        norm_l2_range=[float(np.linalg.norm(currents, axis=1).min()), float(np.linalg.norm(currents, axis=1).max())],
        force_records=int(len(force_values)), foot_plane_contact_records=contact_records,
        touch_with_zero_force_samples=touch_with_zero_force,
        mean_absolute_cancellation_samples=cancel_events,
        encoder_coverage=encoder_coverage,
        per_leg=geom_summary,
        note='multi_joint_dev_rad is a MEAN over joint displacements, so opposing joint motions cancel; '
             'mean_cancellation_samples counts samples with large per-joint spread but small mean.')

    # ---- neural replays -------------------------------------------------------
    We = sparse.load_npz(rt.b.AB / 'A' / 'ACTUAL_We.npz')
    Wi = sparse.load_npz(rt.b.AB / 'A' / 'ACTUAL_Wi.npz')

    def make_net():
        net = ConductanceNetwork(12000, ConductanceParams(), dt_ms=NDT * 1000)
        net.set_connectivity(We, Wi)
        return net

    def run(net, current_sequence, steps, record_every=40):
        rates, volts, adapt, spikes_per_window = [], [], [], []
        spike_total = 0
        for k in range(steps):
            cur = current_sequence[k] if current_sequence is not None else 0.0
            sp = net.step(current_nA=cur)
            spike_total += int(sp.sum())
            if (k + 1) % record_every == 0:
                rates.append(spike_total / record_every * 1000 / NDT / 12000)
                volts.append(float(np.mean(net.v)))
                adapt.append(float(np.mean(net.adaptation)))
                spike_total = 0
                spikes_per_window.append(int(sp.sum()))
        return np.array(rates), np.array(volts), np.array(adapt), np.array(spikes_per_window)

    def sequence(currents_1ms, reps=1):
        seq = np.repeat(currents_1ms, STEPS_PER_MS, axis=0)
        return np.tile(seq, (reps, 1)) if reps > 1 else seq

    real_seq = sequence(currents)
    report = {}

    # (1) real sensory replay, 1.2 s, matching the warmup window exactly.
    net = make_net()
    r_real, v_real, a_real, w_real = run(net, real_seq, len(real_seq))
    report['real_replay'] = dict(steps=int(len(real_seq)), simulated_s=len(real_seq) * NDT,
                                 rate_hz_per_20ms=[float(x) for x in r_real],
                                 rate_first_half_mean=float(r_real[:len(r_real) // 2].mean()),
                                 rate_second_half_mean=float(r_real[len(r_real) // 2:].mean()),
                                 silent_windows=int(np.sum(w_real == 0)), windows=int(len(w_real)),
                                 mean_voltage_first_last=[float(v_real[0]), float(v_real[-1])],
                                 mean_adaptation_first_last=[float(a_real[0]), float(a_real[-1])],
                                 total_spikes=int(net.spike_count.sum()))

    # (2) long-run stability: zero, constant and repeated real input over 6 s.
    zero_runs = {}
    for label, seq6 in [('zero_input', None), ('constant_input', np.tile(currents.mean(axis=0), (12000, 1))),
                        ('repeated_real', sequence(currents, reps=5))]:
        net = make_net()
        r, v, a, w = run(net, seq6, 12000)
        quarters = [float(r[i * len(r) // 4:(i + 1) * len(r) // 4].mean()) for i in range(4)]
        zero_runs[label] = dict(steps=12000, simulated_s=6.0, quarter_rate_hz=quarters,
                                windows=int(len(w)), silent_windows=int(np.sum(w == 0)),
                                mean_voltage_quarters=[float(v[i * len(v) // 4]) for i in range(4)],
                                mean_adaptation_end=float(a[-1]), total_spikes=int(net.spike_count.sum()))
    report['long_run'] = zero_runs

    # (3) repeatability of the same real input across cycle repeats (drift test).
    net = make_net()
    seq = sequence(currents, reps=5)
    per_cycle = []
    one_cycle = 2400
    prev_snapshot = None
    r_all, v_all, a_all, w_all = run(net, seq, len(seq))
    per_window = 5 * len(r_all) // 5
    windows_per_cycle = len(r_all) // 5
    for c in range(5):
        seg = r_all[c * windows_per_cycle:(c + 1) * windows_per_cycle]
        per_cycle.append(dict(cycle=c, rate_mean=float(seg.mean()), rate_std=float(seg.std()),
                              first_window=float(seg[0]), last_window=float(seg[-1])))
    report['cycle_repeatability'] = dict(
        cycles=per_cycle,
        cycle1_vs_cycle5_rate_mean_abs_diff=float(abs(per_cycle[0]['rate_mean'] - per_cycle[-1]['rate_mean'])),
        note='Identical 1.2 s sensory input repeated; differences across cycles measure state drift, not external change.')

    # (4) sensitivity: tiny perturbation of the sensory current at the first step only.
    base_net = make_net()
    base_rates, base_v, base_a, base_w = run(base_net, real_seq, 2400)
    sens = {}
    for eps in [1e-9, 1e-7, 1e-5, 1e-3]:
        net = make_net()
        perturbed = real_seq.copy()
        perturbed[0] = perturbed[0] * (1.0 + eps) + eps
        r, v, a, w = run(net, perturbed, 2400)
        diff = np.abs(r - base_rates)
        nz = np.flatnonzero(diff > 0)
        sens[f'rel_eps_{eps:g}'] = dict(
            first_divergent_window_index=None if len(nz) == 0 else int(nz[0]),
            first_divergent_time_s=None if len(nz) == 0 else float((nz[0] + 1) * 40 * NDT),
            max_abs_rate_diff=float(diff.max()),
            final_abs_rate_diff=float(diff[-1]),
            relative_max_diff=float(diff.max() / max(base_rates.max(), 1e-12)))
    # Voltage-only perturbation (no input change at all).
    net = make_net()
    net.v = net.v + 1e-9
    r, v, a, w = run(net, real_seq, 2400)
    diff = np.abs(r - base_rates)
    nz = np.flatnonzero(diff > 0)
    sens['voltage_eps_1e-9'] = dict(
        first_divergent_window_index=None if len(nz) == 0 else int(nz[0]),
        first_divergent_time_s=None if len(nz) == 0 else float((nz[0] + 1) * 40 * NDT),
        max_abs_rate_diff=float(diff.max()), final_abs_rate_diff=float(diff[-1]),
        relative_max_diff=float(diff.max() / max(base_rates.max(), 1e-12)))
    report['sensitivity'] = sens

    # (5) motor readout sensitivity: does the decided torque change materially?
    def torque_track(seq_in, perturb=False, eps=1e-9):
        net = make_net()
        s = seq_in.copy()
        if perturb:
            s[0] = s[0] * (1.0 + eps) + eps
        rates = np.zeros(66)
        decay = np.exp(-NDT / .005)
        out = []
        for k in range(2400):
            sp = net.step(current_nA=s[k])
            rates = decay * rates + (1 - decay) * np.asarray(poolmat @ sp, float) / NDT
            if (k + 1) % 2 == 0:
                out.append(np.clip(limits / 100 * (matrix @ rates), -limits, limits))
        return np.array(out)

    tb = torque_track(real_seq)
    tp = torque_track(real_seq, True, 1e-9)
    tq = torque_track(real_seq, True, 1e-3)
    scale = float(np.mean(np.abs(tb))) + 1e-12
    report['torque_sensitivity'] = dict(
        mean_abs_torque=float(np.mean(np.abs(tb))),
        eps1e9_max_abs_torque_diff=float(np.abs(tp - tb).max()),
        eps1e9_relative_to_mean=float(np.abs(tp - tb).max() / scale),
        eps1e3_max_abs_torque_diff=float(np.abs(tq - tb).max()),
        eps1e3_relative_to_mean=float(np.abs(tq - tb).max() / scale),
        time_variation_abs_mean=float(np.mean(np.abs(np.diff(tb, axis=0)))))

    report['provenance'] = dict(
        network='engine/neural_cond.py ConductanceNetwork, 12000 cells, dt=0.5 ms, backward_euler',
        parameters='ConductanceParams defaults, engine/neural_cond.py declares every value "illustrative, uncalibrated point reduction, not fly measurement"',
        connectivity=f'ARMS/A ACTUAL_We nnz={We.nnz}, ACTUAL_Wi nnz={Wi.nnz}',
        random_generation='none: ConductanceNetwork.step contains no RNG and this diagnostic injects no noise',
        rh_gain='incoming RH excitation scaled by 0.375 at release only; not applied in these replay tests')
    report['sources'] = {str(p): sha(p) for p in [body_path, ARCH / 'MANIFEST_SHA256.json', Path(rt.b.AB / 'A' / 'ACTUAL_We.npz'),
                                                 Path(rt.b.AB / 'A' / 'ACTUAL_Wi.npz'), R / 'engine/neural_cond.py',
                                                 R / 'run_neural_muscle_closure.py', R / 'engine/embodied/sensory_channels.py',
                                                 BASE / 'runtime.py', Path(__file__)]}
    report['failed_attempts'] = FAILED_ATTEMPTS
    report['output_dir_created_by_this_run'] = created_now
    report['wall_s'] = time.monotonic() - t0
    dump(O / 'NEURAL_DIAGNOSTICS.json', report)
    np.savez_compressed(O / 'SENSORY_CURRENTS.npz', currents=currents.astype(np.float32),
                        obs_index=obs_index, time=times[obs_index],
                        rate_real=r_real.astype(np.float32), rate_base=r_all.astype(np.float32))
    print('DIAGNOSTICS_DONE', json.dumps({k: report[k] for k in ['real_replay']}, default=str)[:400], flush=True)
    print('SENS', json.dumps(sens, default=str), flush=True)
    print('LONG', json.dumps(zero_runs, default=str), flush=True)
    print('TORQUE', json.dumps(report['torque_sensitivity'], default=str), flush=True)
    print('SENSORY', json.dumps({k: v for k, v in sensory_summary.items() if k != 'per_leg'}, default=str), flush=True)
    print('WALL_S', report['wall_s'], flush=True)


if __name__ == '__main__':
    main()
