"""Offline NumPy reconstruction of S108 archives; never imports/runs physics.

Only the translator's validated pool mapping is reused, NOT its FIR/consume code.
Matplotlib is used only to render the comparison. Computational initialization
is not physical support, standing, preheating, or evidence of walking.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / 'run_neural_initialized_startup.py'
INITIAL = Path('/run/media/sensen/Data2/engine修复验收/s7-posture-ik/20261023T021141Z/DERIVED_INITIAL.npz')
MANIFEST = Path('/run/media/sensen/Data2/engine修复验收/s4-endpoints/20261023T011559Z/MANIFEST.json')
GATE_NAMES = ('height', 'angle', 'legs', 'load', 'nonfoot', 'limit')
BINS = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80))
DT = .0005


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    require(all(np.isfinite(v).all() for v in arrays.values()), f'Nonfinite archive: {path}')
    return arrays


def equal(actual, expected, label, atol=0):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, f'{label}: shape mismatch {actual.shape} != {expected.shape}')
    require(np.allclose(actual, expected, rtol=0, atol=atol), f'{label}: values differ')


def fir_recompute(spikes, pools, actuator_pool, weights):
    """Independent one-tick recurrence with a zero initial 80-tick FIFO."""
    require(spikes.ndim == 2 and np.all((spikes == 0) | (spikes == 1)), 'Nonbinary spikes')
    history = np.zeros((80, len(pools)))
    features = np.empty((len(spikes), len(pools), 5))
    controls = np.zeros((len(spikes), len(actuator_pool)))
    mapped = actuator_pool >= 0
    for i, spike in enumerate(spikes):
        history[1:] = history[:-1].copy()
        history[0] = [np.mean(spike[list(pool)]) / DT / 100 for pool in pools]
        features[i] = np.stack([np.mean(history[lo:hi], axis=0) for lo, hi in BINS], axis=1)
        pooled = np.clip(np.sum(features[i] * weights, axis=1), 0, 1)
        controls[i, mapped] = pooled[actuator_pool[mapped]]
    return features, controls, history


def first_nonzero(values, times):
    indices = np.flatnonzero(np.any(np.asarray(values) != 0, axis=tuple(range(1, np.asarray(values).ndim))))
    return None if not len(indices) else float(times[indices[0]] * 1000)


def support_metrics(mask, time):
    """Trapezoid occupancy and conservative contiguous true-endpoint spans."""
    mask = np.asarray(mask, bool)
    dt = np.diff(time)
    cumulative = np.r_[0., np.cumsum((mask[:-1].astype(float) + mask[1:]) * .5 * dt)] * 1000
    padded = np.r_[False, mask, False].astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    longest = max((time[b] - time[a] for a, b in zip(starts, ends)), default=0.) * 1000
    return float(cumulative[-1]), float(longest), cumulative


def physical_metrics(a, weight):
    time = a['time']
    height = a['qpos'][:, 2] / a['qpos'][0, 2]
    quat = a['qpos'][:, 3:7]
    norms = np.linalg.norm(quat, axis=1)
    require(np.all(norms > 0), 'Invalid root quaternion')
    angle = 2 * np.arccos(np.clip(np.abs(quat @ quat[0]) / (norms * norms[0]), 0, 1))
    load = np.sum(a['foot_force'][:, :, 2], axis=1) / weight
    legs = np.count_nonzero(a['contacts'] > 0, axis=1)
    gates = np.column_stack((height >= .85, angle <= .35, legs >= 3, load >= .8,
                             a['nonfoot_depth'] <= 1e-12, a['limit_excess'] <= 1e-6))
    total, longest, cumulative = support_metrics(gates.all(axis=1), time)
    first = {name: (float(time[np.flatnonzero(~gates[:, j])[0]] * 1000)
                    if (~gates[:, j]).any() else None) for j, name in enumerate(GATE_NAMES)}
    metrics = dict(first_failures_ms=first, all_gates_support_ms=total,
                   longest_all_gates_true_endpoint_span_ms=longest,
                   first_nonzero_applied_physical_start_ms=first_nonzero(a['applied'], np.arange(20) * DT),
                   first_nonzero_saved_ctrl_ms=first_nonzero(a['ctrl'], time),
                   first_nonzero_activation_ms=first_nonzero(a['act'], time),
                   min_root_height_native=float(a['qpos'][:, 2].min()),
                   min_height_ratio=float(height.min()), min_root_vertical_velocity_native=float(a['qvel'][:, 2].min()),
                   max_downward_velocity_native=float(np.maximum(-a['qvel'][:, 2], 0).max()),
                   max_angle_rad=float(angle.max()), max_nonfoot_depth_native=float(a['nonfoot_depth'].max()),
                   max_limit_excess_rad=float(a['limit_excess'].max()), min_load_ratio=float(load.min()),
                   min_contact_legs=int(legs.min()))
    return metrics, (load, height, cumulative, gates.all(axis=1))


def verify(out):
    require('mujoco' not in sys.modules, 'Physics module already imported')
    require(not (out / 'FAILED.json').exists(), 'FAILED.json present; refuse successful certification')
    require(load_json(out / 'COMPLETE.json')['status'] == 'COMPLETE', 'Runner incomplete')
    prereg = load_json(out / 'PREREG.json')
    hashes = prereg['source_hashes']
    require(str(RUNNER) in hashes and str(INITIAL) in hashes and str(MANIFEST) in hashes,
            'Missing required source hashes')
    for path, digest in hashes.items():
        require(sha(path) == digest, f'Source hash mismatch: {path}')
    require(prereg['initialization_ms'] == [0, 40] and prereg['physical_duration_ms'] == 10
            and prereg['dt_ms'] == .1 and prereg['neural_dt_ms'] == .5, 'Prereg timing changed')
    weights_path = Path(prereg['same_weights'])
    require(str(weights_path) in hashes, 'Missing weights source hash')
    weights = load_npz(weights_path)['weights']
    initial_q = load_npz(INITIAL)['new_qpos']
    results = load_json(out / 'RESULTS.json')
    require([r['label'] for r in results] == ['init00ms', 'init40ms'], 'Unexpected results arms')
    ledger = load_json(out / 'CALL_LEDGER.json')
    require(ledger == dict(model_load=2, forward=202, step=200, neural_ticks=120,
                           active=None, completed=['init00ms', 'init40ms']), 'Call ledger mismatch')
    # Safe NumPy-only module: use constructor ONLY to validate/recover pool identity.
    from engine.embodied.motor_translation import FiniteMotorTranslator as MotorTranslator
    mapping = MotorTranslator(load_json(MANIFEST), weights)
    require(mapping.n == 17099 and mapping.nu == 90 and len(mapping.pools) == 66, 'Mapping dimensions')
    require(tuple(mapping.BINS) == BINS and mapping.DT == DT, 'Translator clock/kernel changed')
    checks, plotted, archive_hashes = [], [], {}
    baseline_cold = baseline_init = None
    for r, pre in zip(results, (0, 80)):
        label = r['label']; dest = out / label
        for filename, key in (('TRAJECTORY.npz', 'trajectory_sha256'), ('INITIALIZATION.npz', 'initialization_sha256')):
            require(sha(dest / filename) == r[key], f'Archive hash mismatch: {dest / filename}')
        for filename in ('TRAJECTORY.npz', 'INITIALIZATION.npz', 'NEURAL_COLD.npz', 'SCHEDULE.json',
                         'INITIAL_OBSERVATION.json', 'OBSERVATION_00.json', 'OBSERVATION_10.json'):
            archive_hashes[str(dest / filename)] = sha(dest / filename)
        a = load_npz(dest / 'TRAJECTORY.npz'); init = load_npz(dest / 'INITIALIZATION.npz')
        cold = load_npz(dest / 'NEURAL_COLD.npz')
        require(bool(cold), 'Empty cold neural snapshot')
        if baseline_cold is None:
            baseline_cold, baseline_init = cold, init
        require(set(cold) == set(baseline_cold), 'Cold network keys differ')
        for key in cold:
            equal(cold[key], baseline_cold[key], f'{label}: cold {key}')
            require(key in init, f'Missing release neural field {key}')
            if pre == 0:
                equal(init[key], cold[key], f'{label}: cold release {key}')
        for key, shape in dict(time=(101,), qpos=(101,49), qvel=(101,48), act=(101,90), ctrl=(101,90),
                               contacts=(101,6), foot_force=(101,6,3), nonfoot_depth=(101,),
                               limit_excess=(101,), spikes=(20,17099), features=(20,66,5),
                               applied=(20,90), current=(20,17099), weights=(66,5)).items():
            require(a[key].shape == shape, f'{label}: {key} shape')
        equal(a['time'], np.arange(101) * .0001, f'{label}: body clock', 1e-12)
        for key in ('qfrc_applied', 'xfrc_applied', 'warnings'):
            require(not a[key].any(), f'{label}: nonzero {key}')
        for key in ('act', 'ctrl', 'applied'):
            require(np.all((a[key] >= 0) & (a[key] <= 1)), f'{label}: {key} bounds')
        equal(init['body_qpos'], initial_q, f'{label}: initial qpos')
        equal(a['qpos'][0], initial_q, f'{label}: rollout initial qpos')
        for key in ('qvel', 'act', 'ctrl'):
            require(not init['body_' + key].any() and not a[key][0].any(), f'{label}: nonzero initial {key}')
            equal(init['body_' + key], a[key][0], f'{label}: initial {key}')
        equal(init['weights'], weights, f'{label}: initialization weights')
        equal(a['weights'], weights, f'{label}: rollout weights')
        equal(init['current'], baseline_init['current'], f'{label}: held initial current')
        equal(a['current'][:10], np.repeat(init['current'][None], 10, axis=0), f'{label}: initial current hold')
        equal(a['current'][10:], np.repeat(a['current'][10:11], 10, axis=0), f'{label}: second current hold')
        require(init['spikes'].shape == (pre,17099), f'{label}: initialization spike shape')
        wf, wu, history = fir_recompute(init['spikes'], mapping.pools, mapping.actuator_pool, weights)
        equal(init['features'], wf, f'{label}: initialization FIR', 1e-14)
        equal(init['decoded'], wu, f'{label}: initialization commands', 1e-14)
        equal(init['history'], history, f'{label}: release FIFO', 1e-14)
        release_features = np.stack([history[lo:hi].mean(0) for lo,hi in BINS], axis=1)
        equal(init['release_features'], release_features, f'{label}: release features', 1e-14)
        release_ctrl = np.zeros(90); mapped = mapping.actuator_pool >= 0
        release_ctrl[mapped] = np.clip((release_features * weights).sum(1),0,1)[mapping.actuator_pool[mapped]]
        equal(init['release_decoded'], release_ctrl, f'{label}: release decoded', 1e-14)
        features, controls, _ = fir_recompute(np.concatenate((init['spikes'], a['spikes'])),
                                              mapping.pools, mapping.actuator_pool, weights)
        equal(a['features'], features[pre:], f'{label}: continuous rollout FIR', 1e-14)
        equal(a['applied'], controls[pre:], f'{label}: continuous rollout commands', 1e-14)
        equal(a['ctrl'][1:], np.repeat(a['applied'], 5, axis=0), f'{label}: each command held five body steps')
        schedule = load_json(dest / 'SCHEDULE.json')
        require(len(schedule) == 20, f'{label}: schedule length')
        for k, row in enumerate(schedule):
            require(row['tick'] == k and row['consume_id'] == pre+k, f'{label}: consumption IDs')
            for key, value in dict(source_start_s=(pre+k)*DT, source_end_s=(pre+k+1)*DT,
                                   physical_start_s=k*DT, physical_end_s=(k+1)*DT, source_offset_s=pre*DT).items():
                require(abs(row[key]-value) <= 1e-12, f'{label}: schedule {key}')
        require(r['initialization_ticks'] == pre and abs(r['source_offset_s']-pre*DT) <= 1e-12, 'Result offset')
        for key, expected in dict(neural_final_s=(pre+20)*DT, FIR_final_s=(pre+20)*DT, physical_final_s=.01).items():
            require(abs(r[key]-expected) <= 1e-12, f'{label}: final {key}')
        require(np.isfinite(r['weight_native']) and r['weight_native'] > 0, 'Invalid saved native weight')
        if checks:
            require(r['weight_native'] == results[0]['weight_native'], 'Native weight differs')
        metrics, series = physical_metrics(a, r['weight_native'])
        metrics.update(label=label, initialization_ms=pre*.5, passed=True,
                       first_nonzero_initialization_decoded_source_end_ms=first_nonzero(wu, (np.arange(pre)+1)*DT),
                       first_nonzero_rollout_command_source_end_ms=first_nonzero(a['applied'], (pre+np.arange(20)+1)*DT),
                       release_decoded_max=float(release_ctrl.max()), saved_cold_neural_fields=sorted(cold))
        checks.append(metrics); plotted.append((label, a, series))
    require(load_json(out/'init00ms/INITIAL_OBSERVATION.json') == load_json(out/'init40ms/INITIAL_OBSERVATION.json'),
            'Held initial observations differ')
    require('mujoco' not in sys.modules, 'Verifier imported physics')
    return dict(passed=True, verifier=str(Path(__file__).resolve()), verifier_sha256=sha(__file__),
                new_physics_calls=0, new_neural_ticks=0, ledger=ledger, source_hashes=hashes,
                observed_archive_hashes=archive_hashes, mapping_digest=mapping.mapping_digest,
                physical_units='NOT_ESTABLISHED', behavior_claim='NONE', checks=checks,
                gate_thresholds=dict(height_ratio_min=.85, angle_rad_max=.35, penetrating_contact_legs_min=3,
                                     foot_load_native_weight_min=.8, nonfoot_depth_max=1e-12, limit_excess_max=1e-6),
                support_convention='Physical 0..10ms ONLY. Cumulative trapezoid binary occupancy; longest is span between first/last true samples in contiguous run (isolated true sample spans zero). No substep crossing interpolation.',
                caveats=['40ms initialization is neural/FIR computation with fixed observation, NOT physical support or standing.',
                         'Independent spike-to-FIR/command arithmetic; network dynamics and physical dynamics are not rerun or certified.',
                         'Saved cold ndarray fields compared; connectivity CSR and full neural scalar state were not archived here.',
                         'Native weight, aggregated foot force/contact/depth/limit arrays are trusted saved inputs, not independently reconstructed from raw contacts/model.',
                         'NEURAL_COLD, schedule and observation hashes are observed now, not preregistered runner commitments.',
                         'Ledger and source identity do not independently prove absence of all unrecorded runtime calls/callbacks; XML mesh dependency bytes not separately committed.',
                         'Source tick is computed before its physical interval by declared operator split; source END equals physical START + initialization offset + 0.5ms, not a future physical observation. First saved activation has 0.1ms endpoint resolution.',
                         'Ten physical milliseconds and earlier nonzero command do not establish sustained support, autonomous walking, or biological calibration.']), plotted


def plot(out, plotted):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4,1,figsize=(12,12),sharex=True)
    for label, a, (load, height, cumulative, support) in plotted:
        t = a['time']*1000
        axes[0].step(np.arange(20)*.5, a['applied'].max(1), where='post', label=label+' max command')
        axes[0].plot(t,a['act'].max(1),ls='--',label=label+' max activation')
        axes[1].plot(t,load,label=label)
        axes[2].plot(t,height,label=label)
        axes[3].plot(t,cumulative,label=label+' cumulative six-gate support')
    axes[1].axhline(.8,color='black',ls=':'); axes[2].axhline(.85,color='black',ls=':')
    for ax, ylabel in zip(axes, ('Normalized command / activation','Foot Fz / saved native weight',
                                'Root height / initial height','Six-gate occupancy (ms)')):
        ax.set_ylabel(ylabel); ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[-1].set_xlabel('Physical time after release (ms); initialization NOT physical support')
    fig.suptitle('Matched 0ms vs 40ms neural/FIR computational initialization\nIndependent saved-array checks; NOT standing/walking certification')
    fig.tight_layout(); fig.savefig(out/'NEURAL_INITIALIZED_STARTUP_COMPARISON.png',dpi=150); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    out=parser.parse_args().output.resolve()
    report, plotted=verify(out)
    plot(out, plotted)
    report['comparison_png']=str(out/'NEURAL_INITIALIZED_STARTUP_COMPARISON.png')
    (out/'INDEPENDENT_CHECK.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
