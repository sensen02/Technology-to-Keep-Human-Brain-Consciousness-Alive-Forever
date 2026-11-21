"""Recalculate the P1 airborne gates from recorded native and control steps."""
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from flight_manifest import CASES, COLUMNS, ROOT, digest

HERE = Path(__file__).resolve().parent
BASELINE = ROOT / '20261117T051249Z-bidirectional'


def json_file(path):
    return json.loads(path.read_text())


def verify(path, sha):
    if digest(path) != sha:
        raise ValueError(f'Changed artifact: {path}')
    return np.load(path, allow_pickle=False)


def read_run(path):
    facts = json_file(path / 'RESULT.json')
    return facts, {
        name: verify(path / filename, facts[f'{name}_sha256'])
        for name, filename in [('trace', 'TRACE.npy'), ('pose', 'QPOS.npy'),
                               ('native', 'NATIVE.npy'), ('initial', 'INITIAL.npy')]}


def cycle_range(values):
    peaks, _ = find_peaks(values, distance=16, prominence=1.)
    assert len(peaks) >= 10
    amplitudes = [np.ptp(values[lo:hi + 1]) for lo, hi in zip(peaks[:-1], peaks[1:])]
    return float(np.median(amplitudes))


def measure(trace, baseline):
    sampled = (trace[:, 0] >= .110) & (trace[:, 0] <= .180)
    target = [cycle_range(trace[sampled, COLUMNS.index(axis)])
              for axis in ('target_left_rad', 'target_right_rad')]
    stroke = [cycle_range(trace[sampled, COLUMNS.index(axis)])
              for axis in ('wing_left_rad', 'wing_right_rad')]
    return {
        'stroke_rad': stroke,
        'target_stroke_rad': target,
        'yaw_delta_deg': float(np.rad2deg(trace[-1, 4] - baseline[-1, 4])),
        'lateral_delta_mm': float(10 * (trace[-1, 2] - baseline[-1, 2])),
        'min_height_mm': float(10 * trace[:, 3].min()),
        'height_span_mm': float(10 * np.ptp(trace[:, 3])),
    }


def audit_neck():
    source = HERE / 'NECK'
    identity = json_file(source / 'IDENTITY.json')
    for name, file in {
        'fixed_neck_source_sha256': ROOT / '20261117T164207Z-fixed-neck/fixed_neck.py',
        'parameters_sha256': ROOT / '20261117T164207Z-fixed-neck/PARAMETERS.json',
        'catalog_sha256': BASELINE / 'CATALOG.json',
        'eon_mapping_sha256': BASELINE / 'EON/RESULT.json',
    }.items():
        assert digest(file) == identity[name]
    ports = list(identity['ports'])
    with verify(source / 'POST_NECK.npz', identity['data_sha256']) as data:
        assert list(data['labels']) == ports
        cases = tuple(data['conditions'])
        assert np.isfinite(data['levels']).all()
        onset = {}
        for name in cases:
            index = cases.index(name)
            emitted, arrivals = data['emitted'][:, index], data['arrived'][:, index]
            assert emitted.sum(axis=0).tolist() == identity['cases'][name]['emitted']
            assert arrivals.sum(axis=0).tolist() == identity['cases'][name]['arrived']
            for column, label in enumerate(ports):
                source_events = np.flatnonzero(emitted[:, column])
                body_events = np.flatnonzero(arrivals[:, column])
                delay = identity['ports'][label]['delay_ticks']
                blocked = name.endswith('_cut')
                assert np.array_equal(body_events, [] if blocked else source_events + delay)
                if len(body_events):
                    for field in ('synaptic_pa', 'membrane_mv', 'levels'):
                        voltage = data[field][:, index, column]
                        assert not np.any(voltage[:body_events[0]])
                        assert voltage.max() > 0.
            if arrivals.any():
                onset[name] = np.flatnonzero(arrivals.any(axis=1))[0] * identity['dt_ms'] / 1000
            if not arrivals.any():
                assert not np.any(data['levels'][:, index])
                assert not np.any(data['synaptic_pa'][:, index])
        for intact, cut in [('amp_high', 'amp_cut'), ('left', 'left_cut'), ('right', 'right_cut'),
                            ('takeoff_probe', 'takeoff_cut'), ('landing_probe', 'landing_cut')]:
            assert np.array_equal(data['emitted'][:, cases.index(intact)],
                                  data['emitted'][:, cases.index(cut)])
    return identity, onset


def audit_air(directory, identity, onset, legged=False):
    provenance = json_file(directory / 'IDENTITY.json')
    assert provenance['post_neck_sha256'] == identity['data_sha256']
    assert provenance['training_updates'] == 0
    if legged:
        assert provenance['frozen_policy_sha256'] == digest(BASELINE / 'flight_reference/flight/saved_model.pb')
        assert len(provenance['adapter']['stock_action_names']) == 12
        assert len(provenance['adapter']['legged_action_names']) == 66
    else:
        assert provenance['fluid']
        assert digest(BASELINE / 'flight_reference/wing_pattern_fmech.npy') == provenance['wing_pattern_sha256']
        for name, sha in provenance['policy_files'].items():
            assert digest(BASELINE / 'flight_reference/flight' / name) == sha
    facts, blank = read_run(directory / 'sham')
    baseline = blank['trace']
    stats = {}
    steps = 0
    cases = [case for case in CASES if not legged or case not in ('amp_low', 'combined')]
    for case in cases:
        facts, arrays = read_run(directory / case)
        trace, native = arrays['trace'], arrays['native']
        assert tuple(facts['columns']) == COLUMNS
        assert trace.shape == (1400, len(COLUMNS))
        assert not facts['terminal'] and facts['ground_collision_mask'] == [1, 1]
        assert all(value > 0. for value in facts['density_viscosity'])
        assert native.shape == (5600, 2) and not np.any(native)
        assert not facts['ground_contacts'] and facts['max_external_applied'] == 0.
        assert np.isclose(trace[-1, 0], .28) and trace[:, 3].min() >= .8
        assert np.array_equal(arrays['initial'], blank['initial'])
        if case.endswith('_cut'):
            assert all(np.array_equal(value, blank[name]) for name, value in arrays.items())
        elif case != 'sham':
            before = trace[:, 0] <= onset[case] + 1e-12
            assert np.array_equal(arrays['pose'][before], blank['pose'][before])
            assert np.array_equal(trace[before], baseline[before])
        stats[case] = measure(trace, baseline)
        steps += len(native)
    sham, high = (np.asarray(stats[name]['stroke_rad']) for name in ('sham', 'amp_high'))
    assert (high / sham >= 1.01).all()
    stats['amp_high']['stroke_gain_pct'] = (100 * (high / sham - 1)).tolist()
    for case, sign in [('left', 1), ('right', -1)]:
        assert sign * stats[case]['yaw_delta_deg'] >= 15 and sign * stats[case]['lateral_delta_mm'] >= 5
    if not legged:
        low = np.asarray(stats['amp_low']['stroke_rad'])
        assert (high > low).all()
        stats['amp_low']['stroke_gain_pct'] = (100 * (low / sham - 1)).tolist()
        combined = stats['combined']
        combined['stroke_gain_pct'] = (100 * (np.asarray(combined['stroke_rad']) / sham - 1)).tolist()
        assert min(combined['stroke_gain_pct']) >= 1.
        assert combined['yaw_delta_deg'] >= 15 and combined['lateral_delta_mm'] >= 5
    return stats, steps, blank


def audit_transitions(identity):
    def pair(directory, mode):
        assert identity['data_sha256'] == json_file(directory / 'IDENTITY.json')['post_neck_sha256']
        files = {}
        for case in (f'{mode}_probe', f'{mode}_cut'):
            folder = directory / case
            facts = json_file(folder / 'RESULT.json')
            arrays = {name: verify(folder / name, sha) for name, sha in facts['files'].items()}
            trace = arrays['TRACE.npy']
            assert np.isfinite(trace).all() and not np.any(trace[:, 8])
            assert len(trace) == facts['native_steps']
            assert np.count_nonzero(trace[:, 5]) == facts['nonfoot_contact_samples']
            assert facts['reset_only']
            if case.endswith('_cut'):
                assert not np.any(trace[:, 9:])
            if mode == 'landing':
                assert np.all(trace[-400:, 4] == 6) and not np.any(trace[-400:, 5])
                assert trace[-400:, 6].max() < .005  # 0.05 mm/s
            files[case] = facts, arrays
        probe, cut = (files[f'{mode}_{suffix}'] for suffix in ('probe', 'cut'))
        assert np.array_equal(probe[1]['INITIAL.npy'], cut[1]['INITIAL.npy'])
        return probe, cut

    (launch, jumped), (stand, stayed) = pair(HERE / 'TRANSITION_A', 'takeoff')
    flight = jumped['TRACE.npy']
    contact = stayed['TRACE.npy']
    assert np.array_equal(jumped['QPOS.npy'][:400], stayed['QPOS.npy'][:400])
    clear = np.flatnonzero((flight[:, 4] + flight[:, 5] == 0) & (flight[:, 0] > .08))
    assert len(clear) >= 40 and np.all(np.diff(clear) == 1) and (flight[:1600, 4] > 0).all()
    assert not np.any(contact[:, 4] + contact[:, 5] == 0)
    assert not launch['nonfoot_contact_samples'] and not stand['nonfoot_contact_samples']
    landing = {}
    for folder, expectation in [('TRANSITION_A', 'both_clear'),
                                ('TRANSITION_B', 'probe_partial'),
                                ('TRANSITION_C', 'both_impacted')]:
        (probe, arrays), (blocked, _) = pair(HERE / folder, 'landing')
        assert np.any(arrays['TRACE.npy'][:, 9:] > 0.)
        assert 0 < probe['first_any_floor_contact_ms'] < 100
        probe_nonfoot, cut_nonfoot = (item['nonfoot_contact_samples'] for item in (probe, blocked))
        if expectation == 'both_clear':
            assert probe_nonfoot == cut_nonfoot == 0
        elif expectation == 'probe_partial':
            assert 0 < probe_nonfoot / cut_nonfoot < .20
        else:
            assert probe_nonfoot > 0 and cut_nonfoot > 0
        landing[folder] = {'probe': probe, 'cut': blocked}
    return {'launch': launch, 'launch_cut': stand, 'clear_s': float(.00005 * len(clear)),
            'clear_window_ms': (1000 * flight[clear[[0, -1]], 0]).tolist(), 'landing_approaches': landing}


def audit_replay(identity):
    replay = json_file(HERE / 'NECK_REPLAY/IDENTITY.json')
    assert replay == identity
    assert digest(HERE / 'NECK_REPLAY/POST_NECK.npz') == identity['data_sha256']
    assert json_file(HERE / 'REPLAY_PHASE29/IDENTITY.json') == json_file(HERE / 'VALID_29/IDENTITY.json')
    cases = ('sham', 'left', 'right', 'amp_high')
    for case in cases:
        replay, reference = (read_run(HERE / path / case)[1] for path in ('REPLAY_PHASE29', 'VALID_29'))
        assert all(np.array_equal(value, reference[name]) for name, value in replay.items())
    return {'neck_all_channels_same_hash': True, 'airborne_array_replays': len(cases) * 4}


def main():
    identity, onset = audit_neck()
    seed17, n17, blank17 = audit_air(HERE / 'VALID_17', identity, onset)
    seed29, n29, blank29 = audit_air(HERE / 'VALID_29', identity, onset)
    legged_air, n_legged, _ = audit_air(HERE / 'LEGGED_AIR', identity, onset, legged=True)
    assert not np.array_equal(blank17['initial'], blank29['initial'])
    old = BASELINE / 'FREE_FLIGHT_QUAL/intact/QPOS.npy'
    assert np.array_equal(blank17['pose'], np.load(old, allow_pickle=False))
    dry, no_fluid = read_run(HERE / 'VALID_NO_FLUID/sham')
    assert np.array_equal(no_fluid['initial'], blank17['initial'])
    assert dry['density_viscosity'] == [0., 0.]
    assert dry['terminal'] and dry['simulated_s'] < .10 and dry['min_height_mm'] < 2.
    legged = json_file(HERE / 'LEGGED_COLLIDABLE/RESULT.json')
    assert legged['legged_joint_count'] == 66
    assert legged['stock_action_shape'] == [12] and legged['legged_action_shape'] == [66]
    assert 'legged_teacher_error' in legged
    stance = legged['stance']
    footing = verify(HERE / 'LEGGED_COLLIDABLE/TRACE.npy', stance['trace_sha256'])
    assert np.all(footing[-400:, 4] == 6) and not np.any(footing[-400:, 5])
    assert not np.any(footing[:, 6]) and np.max(np.ptp(10 * footing[-400:, 1:4], axis=0)) < .02
    transitions = audit_transitions(identity)
    replay = audit_replay(identity)
    result = {
        'status': 'AIRBORNE_COMMANDS_PASS',
        'scope': 'P1 isolated mapped DN commands with frozen FlyBody low-level policy',
        'body_native_steps_air_comparisons': n17 + n29,
        'body_native_steps_fluid_ablation': dry['native_steps'],
        'legged_contact_fixture_steps': stance['native_steps'],
        'legged_air_comparisons_steps': n_legged,
        'baseline_seed17_equals_archived_poses': True,
        'first_arrival_ms': {key: float(value * 1000) for key, value in onset.items()},
        'by_initial_wing_phase': {'seed17': seed17, 'seed29': seed29},
        'no_fluid': dry,
        'legged_fixture': legged,
        'legged_air': legged_air,
        'transition_stages': transitions,
        'independent_replay': replay,
        'takeoff': 'BRIEF_DETACHMENT_PASS_SUSTAINED_POWERED_TAKEOFF_UNVERIFIED',
        'landing': 'NEAR_FLOOR_SUPPORT_PASS_SOME_APPROACHES_HIT_NONFEET',
        'autonomous_flight': 'UNVERIFIED',
        'frozen_policy_updates': 0,
        'inputs': {name: digest(HERE / name)
                   for name in ['NECK/IDENTITY.json', 'VALID_17/IDENTITY.json', 'VALID_29/IDENTITY.json',
                                'VALID_NO_FLUID/IDENTITY.json', 'LEGGED_COLLIDABLE/RESULT.json',
                                'LEGGED_AIR/IDENTITY.json', 'TRANSITION_A/IDENTITY.json',
                                'TRANSITION_B/IDENTITY.json', 'TRANSITION_C/IDENTITY.json',
                                'REPLAY_PHASE29/IDENTITY.json', 'NECK_REPLAY/IDENTITY.json']},
    }
    (HERE / 'FINAL_AIRBORNE.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
