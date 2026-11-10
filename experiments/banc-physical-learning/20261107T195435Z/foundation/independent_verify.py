"""Independent Round10 foundation verifier.

This checker is intentionally separate from build_foundation.py.  It reads only
frozen source/artifact arrays, performs no physics or optimization, and writes
only INDEPENDENT_VERIFICATION.json beside this file.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
RUNTIME_PATH = HERE / 'runtime.py'
CORE = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T192931Z/dynamics/module.py')
ENCODER_PATH = Path('/run/media/sensen/Data2/cell_wound_prototype/experiments/banc-local-learning/20261107T185910Z/encoder/encoder.py')
TRACE = PROJECT / 'experiments/banc-local-learning/20261107T192931Z/dynamics/SOURCE_TRACE.npz'
OLD_TRAIN = Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T185910Z/readout/TRAIN_TRACE.npz')
PRIOR = Path('/home/sensen/Desktop/engine修复验收/banc-local-learning/20261107T192931Z/closed_loop')
INTERFACE = PROJECT / 'experiments/flybody-teacher/20261107T180844Z/mapping/STUDENT_INTERFACE.json'
FOUNDATION = HERE
POLICY = PRIOR / 'POLICY.npz'
HANDOVER = PRIOR / 'HANDOVER.npz'
START = FOUNDATION / 'NEURAL_START.npz'
NEXT_INPUT = FOUNDATION / 'CONTINUATION_INPUT.npz'
EXPECTED = FOUNDATION / 'CONTINUATION_EXPECTED.npz'
BASE = FOUNDATION / 'BASE.npz'
FAILED_BUILD = FOUNDATION / 'FAILED_BUILD.json'
LEG_ORDER = ('LF', 'LH', 'LM', 'RF', 'RH', 'RM')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def npz(path: Path, keys=None):
    with np.load(path, allow_pickle=False) as z:
        names = z.files if keys is None else keys
        return {k: z[k].copy() for k in names}


def equal(a, b):
    return set(a) == set(b) and all(np.array_equal(a[k], b[k]) for k in a)


def max_abs(a, b):
    vals = []
    for k in a:
        if a[k].shape == b[k].shape and a[k].dtype.kind in 'fiu' and b[k].dtype.kind in 'fiu':
            vals.append(float(np.max(np.abs(a[k].astype(float) - b[k].astype(float)))))
    return max(vals, default=0.0)


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def manual_restore(c, state, runtime):
    """Restore mutable neural state without calling runtime.restore_circuit."""
    for k in runtime.IDENTITY_FIELDS:
        if not np.array_equal(getattr(c, k), state[k]):
            raise AssertionError('manual identity mismatch: ' + k)
    for k in runtime.ARRAY_FIELDS:
        setattr(c.net, k, state[k].copy())
    c.net._ring_i = int(state['_ring_i'])
    c.net.steps = int(state['steps'])
    c.net.t_ms = float(state['t_ms'])
    c.filtered_rates_hz = state['filtered_rates_hz'].copy()
    if not equal(runtime.snapshot(c), state):
        raise AssertionError('manual restore did not reproduce complete snapshot')


def main():
    runtime = import_module(RUNTIME_PATH, 'independent_round10_runtime')
    encoder_module = import_module(ENCODER_PATH, 'independent_round10_encoder')
    encoder = encoder_module.SensoryEncoder()
    policy = npz(POLICY, ('time_s', 'mn_rates_hz', 'mn_voltages_mV', 'sensory_current_nA', 'teacher_warmup'))
    hand = npz(HANDOVER, ('mn_rates_hz', 'neural_v', 'neural_ge', 'neural_gi', 'neural_g_adapt_uS', 'neural_delay_ring', 'neural_ring_i'))
    start = npz(START)
    next_input = npz(NEXT_INPUT)
    expected = npz(EXPECTED)
    base = npz(BASE)
    trace = npz(TRACE, ('time_s', 'mn_rates_hz', 'root_ids', 'mn_local', 'mn_leg'))
    old = npz(OLD_TRAIN, ('time_s', 'train_mask', 'mn_rates_hz', 'sensory_local_indices'))

    report = {
        'status': 'PASS_INDEPENDENT_VERIFICATION',
        'checker': str(Path(__file__).resolve()),
        'contract': {
            'runtime_path': str(RUNTIME_PATH),
            'core_path': str(CORE),
            'core_sha256': sha256(CORE),
            'expected_core_sha256': str(start['neural_module_sha256']),
            'physical_steps': 0,
            'optimizer_updates': 0,
            'labels_loaded': [],
        },
        'checks': {},
        'actual_network_steps': {},
        'artifacts': {},
    }

    # Core identity and exact saved source/current contracts.
    assert sha256(CORE) == str(start['neural_module_sha256'])
    assert encoder.qualified_count == 556
    assert np.array_equal(encoder.local_indices, old['sensory_local_indices'])
    assert policy['sensory_current_nA'].shape == (600, 556)
    assert np.isfinite(policy['sensory_current_nA'][:201]).all()
    assert np.all(policy['sensory_current_nA'][:201] >= 0)
    assert policy['teacher_warmup'][:200].all() and not policy['teacher_warmup'][200]
    assert abs(float(policy['time_s'][200]) - 1.0) < 1e-12
    assert np.array_equal(next_input['sensory_current_nA'][encoder.local_indices], policy['sensory_current_nA'][200])
    assert np.count_nonzero(next_input['sensory_current_nA'][np.setdiff1d(np.arange(20000), encoder.local_indices)]) == 0
    assert np.array_equal(next_input['descending_current_nA'], np.zeros(20000))
    assert int(next_input['source_policy_row']) == 200
    report['checks']['saved_policy_and_input_contract'] = True

    # Replay exactly 200 saved 5 ms sensory-current rows on a fresh unchanged core.
    replay = runtime.new_circuit()
    zero = np.zeros(20000, dtype=float)
    replay_steps = 0
    replay_policy_errors = []
    for row in range(200):
        replay_policy_errors.append({
            'row': row,
            'mn_rate_error': float(np.max(np.abs(replay.filtered_rates_hz[replay.mn_local] - policy['mn_rates_hz'][row]))),
            'mn_voltage_error': float(np.max(np.abs(replay.net.v[replay.mn_local] - policy['mn_voltages_mV'][row]))),
        })
        before = int(replay.net.steps)
        sensory = np.zeros(20000, dtype=float)
        sensory[encoder.local_indices] = policy['sensory_current_nA'][row]
        replay.advance(sensory, zero, 5.0)
        replay_steps += int(replay.net.steps) - before
    replay_state = runtime.snapshot(replay)
    assert replay_steps == 2000 and int(replay.net.steps) == 2000 and float(replay.net.t_ms) == 1000.0
    assert max(x['mn_rate_error'] for x in replay_policy_errors) == 0.0
    assert max(x['mn_voltage_error'] for x in replay_policy_errors) == 0.0
    assert equal(replay_state, start)
    report['actual_network_steps']['saved_policy_replay_200x5ms'] = replay_steps
    report['checks']['replay_matches_all_neural_start_arrays'] = True
    report['checks']['replay_policy_pre_step_max_error'] = 0.0

    # Every available HANDOVER neural field is compared exactly.
    handover_map = {
        'v': 'neural_v',
        'ge': 'neural_ge',
        'gi': 'neural_gi',
        'g_adapt': 'neural_g_adapt_uS',
        '_ring': 'neural_delay_ring',
        '_ring_i': 'neural_ring_i',
    }
    handover_matches = {k: bool(np.array_equal(getattr(replay.net, k), hand[v])) for k, v in handover_map.items()}
    handover_matches['MN_rates'] = bool(np.array_equal(replay.filtered_rates_hz[replay.mn_local], hand['mn_rates_hz']))
    handover_matches['POLICY200_MN_rates'] = bool(np.array_equal(replay.filtered_rates_hz[replay.mn_local], policy['mn_rates_hz'][200]))
    handover_matches['POLICY200_MN_voltage'] = bool(np.array_equal(replay.net.v[replay.mn_local], policy['mn_voltages_mV'][200]))
    assert all(handover_matches.values())
    report['checks']['all_available_handover_neural_fields_exact'] = handover_matches
    report['checks']['handover_non_neural_fields_skipped_without_physics'] = ['physics_state', 'actuator_target', 'predictor_history_rates_hz', 'predictor_observations_seen', 'predictor_lags_bins', 'predictor_bin_ms']

    # Continuous recorded next 5 ms, then two independent restorations of the same start.
    before = int(replay.net.steps)
    replay.advance(next_input['sensory_current_nA'], next_input['descending_current_nA'], 5.0)
    continuous_steps = int(replay.net.steps) - before
    continuous = runtime.snapshot(replay)
    assert continuous_steps == 10 and equal(continuous, expected)
    report['actual_network_steps']['continuous_recorded_next5ms'] = continuous_steps
    report['checks']['continuous_next5ms_matches_expected'] = True

    manual = runtime.new_circuit()
    manual_restore(manual, start, runtime)
    before = int(manual.net.steps)
    manual.advance(next_input['sensory_current_nA'], next_input['descending_current_nA'], 5.0)
    manual_steps = int(manual.net.steps) - before
    manual_state = runtime.snapshot(manual)
    assert manual_steps == 10 and equal(manual_state, expected)
    report['actual_network_steps']['manual_array_restore_next5ms'] = manual_steps
    report['checks']['manual_restore_without_runtime_restore_matches_expected'] = True

    # Verify default-relative load_circuit while the process cwd is workspace root.
    root_cwd = '/run/media/sensen/Data2'
    os.chdir(root_cwd)
    loaded = runtime.load_circuit()
    assert equal(runtime.snapshot(loaded), start)
    before = int(loaded.net.steps)
    loaded.advance(next_input['sensory_current_nA'], next_input['descending_current_nA'], 5.0)
    loaded_steps = int(loaded.net.steps) - before
    assert loaded_steps == 10 and equal(runtime.snapshot(loaded), expected)
    report['actual_network_steps']['cwd_root_load_circuit_next5ms'] = loaded_steps
    report['checks']['load_circuit_default_relative_from_workspace_root'] = True

    # Rebuild source basis independently, using the saved old mask exactly.
    assert np.array_equal(trace['time_s'], old['time_s'])
    train = old['train_mask'].copy()
    recomputed_mask = (trace['time_s'] >= 0.2) & (trace['time_s'] < 6.0)
    assert train.dtype == np.dtype(bool) and int(train.sum()) == 1160
    assert bool(train[40]) and float(trace['time_s'][40]) == 0.19999999999998003
    assert not bool(recomputed_mask[40])
    assert not np.array_equal(train, recomputed_mask)
    assert np.array_equal(trace['root_ids'], replay.root_ids)
    assert np.array_equal(trace['mn_local'], replay.mn_local)
    assert np.array_equal(trace['mn_leg'], replay.mn_leg)
    assert np.array_equal(old['sensory_local_indices'], encoder.local_indices)
    interface = json.loads(INTERFACE.read_text())
    joint_names = np.array(interface['output']['ordered_joint_names'])
    joint_leg = np.array([name.split('_')[1][:2] for name in joint_names])
    assert np.array_equal(joint_names, encoder.joint_names)
    assert np.array_equal(joint_leg, np.repeat(np.array(LEG_ORDER), 7))
    assert tuple(base['leg_order']) == LEG_ORDER
    assert np.array_equal(base['joint_names'], joint_names)
    assert np.array_equal(base['joint_leg'], joint_leg)
    assert np.array_equal(base['mn_local'], trace['mn_local'])
    assert np.array_equal(base['mn_root_ids'], trace['root_ids'][trace['mn_local']])
    assert np.array_equal(base['mn_leg'], trace['mn_leg'])

    independent_raw = np.zeros((323, 24), dtype=float)
    independent_basis = np.zeros((323, 24), dtype=float)
    independent_rms = np.empty(24, dtype=float)
    independent_scale = np.empty(24, dtype=float)
    independent_singular = np.empty((6, 4), dtype=float)
    basis_stats = {}
    for i, leg in enumerate(LEG_ORDER):
        ni = np.flatnonzero(trace['mn_leg'] == leg.lower())
        x = trace['mn_rates_hz'][train][:, ni] / 100.0
        _, singular, vh = np.linalg.svd(x, full_matrices=False)
        vectors = vh[:4].T.copy()
        for col in range(4):
            pivot = int(np.argmax(np.abs(vectors[:, col])))
            if vectors[pivot, col] < 0:
                vectors[:, col] *= -1
        projected = x @ vectors
        rms = np.sqrt(np.mean(projected ** 2, axis=0))
        scale = np.maximum(rms, 0.05)
        independent_raw[np.ix_(ni, np.arange(4 * i, 4 * i + 4))] = vectors
        independent_basis[np.ix_(ni, np.arange(4 * i, 4 * i + 4))] = vectors / scale
        independent_rms[4 * i:4 * i + 4] = rms
        independent_scale[4 * i:4 * i + 4] = scale
        independent_singular[i] = singular[:4]
        assert np.max(np.abs(vectors.T @ vectors - np.eye(4))) < 1e-12
        assert np.all(np.isfinite(vectors)) and np.all(np.isfinite(scale))
        basis_stats[leg] = {
            'mn_count': int(len(ni)),
            'orthogonality_max_error': float(np.max(np.abs(vectors.T @ vectors - np.eye(4)))),
            'max_cross_leg_loading': float(np.max(np.abs(independent_basis[np.ix_(np.flatnonzero(trace['mn_leg'] != leg.lower()), np.arange(4 * i, 4 * i + 4))]))),
        }
    assert np.array_equal(independent_raw, base['unwhitened_basis'])
    assert np.array_equal(independent_basis, base['basis'])
    assert np.array_equal(independent_rms, base['training_latent_rms'])
    assert np.array_equal(independent_scale, base['normalization_denominator'])
    assert np.array_equal(1.0 / independent_scale, base['normalization_gain'])
    assert np.array_equal(independent_singular, base['singular_values_top4'])
    assert np.all(base['normalization_gain'] <= 20.0)
    for i, leg in enumerate(LEG_ORDER):
        mask = base['mn_leg'] == leg.lower()
        assert np.all(base['basis'][~mask, 4 * i:4 * i + 4] == 0)
    report['checks']['preserved_old_train_mask_exact'] = {
        'training_rows': int(train.sum()),
        'boundary_index_40_time_s': float(trace['time_s'][40]),
        'boundary_index_40_train': bool(train[40]),
        'recomputed_inequality_index_40': bool(recomputed_mask[40]),
        'mask_difference_count': int(np.count_nonzero(train != recomputed_mask)),
    }
    report['checks']['independent_uncentered_per_leg_svd_exact'] = True
    report['checks']['basis_identity_csv_mn_order_joint_order_leg_order'] = True
    report['checks']['basis_stats'] = basis_stats

    # Actor math and bounds, including explicit zero/silent and seeded theta cases.
    joint_cols = [np.flatnonzero(joint_leg == leg) for leg in LEG_ORDER]
    mn_rows = [np.flatnonzero(base['mn_leg'] == leg.lower()) for leg in LEG_ORDER]
    rng = np.random.default_rng(195435)
    actor_max_error = 0.0
    actions_checked = 0
    clipped_counts = []
    theta_cases = [np.zeros((6, 4, 7)), np.full((6, 4, 7), 0.25), np.full((6, 4, 7), -0.25)]
    theta_cases += [rng.uniform(-0.25, 0.25, (6, 4, 7)) for _ in range(16)]
    for theta in theta_cases:
        actor = runtime.Actor(theta)
        manual_w = np.zeros((323, 42), dtype=float)
        raw_w = np.zeros((323, 42), dtype=float)
        for i in range(6):
            raw_w[np.ix_(mn_rows[i], joint_cols[i])] = base['basis'][np.ix_(mn_rows[i], np.arange(4 * i, 4 * i + 4))] @ theta[i]
            manual_w[np.ix_(mn_rows[i], joint_cols[i])] = np.clip(raw_w[np.ix_(mn_rows[i], joint_cols[i])], -1.0, 1.0)
        actor_max_error = max(actor_max_error, float(np.max(np.abs(actor.weights - manual_w))))
        assert np.all(actor.weights >= -1.0) and np.all(actor.weights <= 1.0)
        assert np.all(actor.weights[np.char.lower(base['mn_leg'])[:, None] != np.char.lower(joint_leg)[None, :]] == 0)
        features = rng.uniform(0, 500, 323)
        action = actor.action(features)
        expected_action = 30.0 * np.tanh((features / 100.0) @ manual_w)
        actor_max_error = max(actor_max_error, float(np.max(np.abs(action - expected_action))))
        assert np.all(np.isfinite(action)) and np.all(np.abs(action) <= 30.0)
        assert np.array_equal(actor.action(np.zeros(323)), np.zeros(42))
        clipped_counts.append(int(np.count_nonzero(np.abs(actor.weights) == 1.0)))
        actions_checked += 1
    assert actor_max_error < 1e-12
    assert np.array_equal(runtime.Actor(np.full((6, 4, 7), 0.25)).action(np.zeros(323)), np.zeros(42))
    report['checks']['independent_actor_math_zero_silent_same_leg_clip_bounds'] = True
    report['checks']['actor_cases'] = actions_checked
    report['checks']['actor_manual_max_abs_error'] = actor_max_error
    report['checks']['actor_clipped_weight_counts'] = clipped_counts

    # Invalid actor/MN/current inputs and zero-step guarantee.
    guard_rejections = 0
    for invalid in (np.zeros(167), np.full(168, 0.251), np.full(168, np.nan), np.full(168, np.inf), np.zeros((6, 4, 6))):
        try:
            runtime.Actor(invalid)
        except ValueError:
            guard_rejections += 1
        else:
            raise AssertionError('invalid theta accepted')
    actor0 = runtime.Actor()
    for invalid in (np.zeros(322), np.full(323, -1.0), np.full(323, np.nan), np.full(323, np.inf)):
        try:
            actor0.action(invalid)
        except ValueError:
            guard_rejections += 1
        else:
            raise AssertionError('invalid MN action input accepted')
    invalid_currents = {
        'negative_sensory': lambda: np.where(np.arange(20000) == int(encoder.local_indices[0]), -0.01, 0.0),
        'nan_sensory': lambda: np.where(np.arange(20000) == int(encoder.local_indices[0]), np.nan, 0.0),
        'outside_sensory_identity': lambda: np.where(np.arange(20000) == int(replay.mn_local[0]), 0.01, 0.0),
        'wrong_sensory_shape': lambda: np.zeros(19999),
        'negative_descending': lambda: np.where(np.arange(20000) == int(replay.descending_local[0]), -0.01, 0.0),
    }
    guard_state = runtime.new_circuit()
    manual_restore(guard_state, start, runtime)
    before_guard = runtime.snapshot(guard_state)
    guard_step_count_before = int(guard_state.net.steps)
    for label, make in invalid_currents.items():
        bad = make()
        try:
            guard_state.advance(bad, zero, 5.0) if label != 'negative_descending' else guard_state.advance(zero, bad, 5.0)
        except ValueError:
            guard_rejections += 1
        else:
            raise AssertionError('invalid current accepted: ' + label)
        assert equal(runtime.snapshot(guard_state), before_guard)
    for duration in (0.0, 0.25):
        try:
            guard_state.advance(zero, zero, duration)
        except ValueError:
            guard_rejections += 1
        else:
            raise AssertionError('invalid duration accepted')
        assert equal(runtime.snapshot(guard_state), before_guard)
    assert int(guard_state.net.steps) == guard_step_count_before == 2000
    report['checks']['invalid_actor_current_duration_guards_transactional_zero_step'] = True
    report['checks']['guard_rejections'] = guard_rejections
    report['actual_network_steps']['invalid_input_tests'] = 0

    # Transactional restore checks use in-memory monkeypatched runtime.load only.
    transactional = runtime.new_circuit()
    manual_restore(transactional, start, runtime)
    before_transaction = runtime.snapshot(transactional)
    original_load = runtime.load
    rejected = []
    cases = [
        ('shape_v', 'v', lambda x: x[:-1]),
        ('nan_v', 'v', lambda x: np.full_like(x, np.nan)),
        ('negative_ge', 'ge', lambda x: -np.ones_like(x)),
        ('negative_gi', 'gi', lambda x: -np.ones_like(x)),
        ('negative_g_adapt', 'g_adapt', lambda x: -np.ones_like(x)),
        ('negative_rate', 'filtered_rates_hz', lambda x: -np.ones_like(x)),
        ('negative_spike_count', 'spike_count', lambda x: -np.ones_like(x)),
        ('bad_ring_shape', '_ring', lambda x: x[:-1]),
        ('bad_ring_index', '_ring_i', lambda x: x + 1),
        ('bad_clock', 't_ms', lambda x: x + 1),
        ('bad_root_identity', 'root_ids', lambda x: x[::-1]),
        ('bad_mn_identity', 'mn_local', lambda x: x[::-1]),
        ('bad_contract', 'parameter_contract_json', lambda x: np.array('invalid')),
    ]
    try:
        for label, key, change in cases:
            bad = {k: v.copy() for k, v in start.items()}
            bad[key] = change(bad[key])
            runtime.load = lambda _path, bad=bad: {k: v.copy() for k, v in bad.items()}
            try:
                runtime.restore_circuit(transactional, HERE / 'in_memory_corruption_not_written.npz')
            except ValueError:
                rejected.append(label)
            else:
                raise AssertionError('corrupt snapshot accepted: ' + label)
            assert equal(runtime.snapshot(transactional), before_transaction)
    finally:
        runtime.load = original_load
    assert len(rejected) == len(cases)
    report['checks']['transactional_restore_rejections'] = rejected
    report['actual_network_steps']['transactional_restore_tests'] = 0

    # Preserve any pre-existing builder failure artifact without modifying it.
    if FAILED_BUILD.exists():
        failed = json.loads(FAILED_BUILD.read_text())
        report['preserved_builder_failure_artifact'] = {
            'path': str(FAILED_BUILD),
            'sha256': sha256(FAILED_BUILD),
            'error': failed.get('error'),
            'traceback_tail': str(failed.get('traceback', '')).splitlines()[-1] if failed.get('traceback') else None,
            'preserved': True,
        }

    total_steps = sum(int(v) for v in report['actual_network_steps'].values())
    assert total_steps == 2030
    report['actual_network_steps']['total'] = total_steps
    report['checks']['no_physics_or_optimizer_steps'] = True
    report['checks']['expected_primary_total_2030'] = True
    for path in (START, NEXT_INPUT, EXPECTED, BASE, FOUNDATION / 'CANONICAL.json', FOUNDATION / 'BASIS.json', FOUNDATION / 'FOUNDATION_RESULTS.json', RUNTIME_PATH):
        report['artifacts'][path.name] = {'path': str(path), 'sha256': sha256(path)}
    report['artifacts']['independent_verify.py'] = {'path': str(Path(__file__).resolve()), 'sha256': sha256(Path(__file__).resolve())}
    report['source_identity'] = {
        'leg_order': list(LEG_ORDER),
        'mn_count': 323,
        'qualified_sensory_count': 556,
        'training_rows': int(train.sum()),
        'basis_shape': list(base['basis'].shape),
        'joint_count': int(len(joint_names)),
    }
    (HERE / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        failure = {
            'status': 'FAIL_INDEPENDENT_VERIFICATION',
            'error': repr(exc),
            'cwd': os.getcwd(),
            'checker': str(Path(__file__).resolve()),
            'physical_steps': 0,
            'optimizer_updates': 0,
        }
        (HERE / 'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(failure, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps(failure, indent=2, ensure_ascii=False))
        raise
