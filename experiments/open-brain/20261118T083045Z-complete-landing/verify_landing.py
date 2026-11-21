"""Audit physical landing gates from recorded native steps."""
import json
from pathlib import Path

import numpy as np

from land import BASELINE, COLUMNS, TAKEOFF, commands_from, digest

HERE = Path(__file__).resolve().parent
SEEDS = (17, 29, 53)
CASES = ('intact', 'cut', 'sham')


def read_json(path):
    return json.loads(path.read_text())


def read_case(folder):
    facts = read_json(folder / 'RESULT.json')
    assert tuple(facts['columns']) == COLUMNS
    arrays = {}
    for name, sha in facts['files'].items():
        assert digest(folder / name) == sha
        arrays[name] = np.load(folder / name, allow_pickle=False)
        assert np.isfinite(arrays[name]).all()
    return arrays, facts


def measure(traces, facts, arrival_s):
    trace = dict(zip(COLUMNS, traces.T, strict=True))
    t = trace['time_s']
    feet, body = trace['feet'], trace['nonfoot_contacts']
    floor = feet + body > 0
    cruise = (t >= .380) & (t <= .420)
    after = t > arrival_s
    last = t > t[-1] - .100
    v = traces[:, 4:7] * 10
    angles = np.rad2deg(traces[:, 7:10])
    contact = np.flatnonzero(after & floor)
    recent_floor = np.flatnonzero((t < arrival_s) & floor)
    preceding_air_ms = (arrival_s - t[recent_floor[-1]]) * 1000
    landed = float(t[contact[0]] * 1000) if len(contact) else None
    if len(contact):
        before = v[contact[0] - 1]
        all_feet = np.flatnonzero((t >= landed / 1000) & (feet == 6))
        interruptions = np.flatnonzero((t >= landed / 1000) & ~floor)
        six_steady = np.flatnonzero((feet != 6) | (body != 0))
        first_feet = int(feet[contact[0]])
        vertical = float(before[2])
        horizontal = float(np.linalg.norm(before[:2]))
        all_feet_ms = float(t[all_feet[0]] * 1000) if len(all_feet) else None
        stable_ms = float(t[six_steady[-1] + 1] * 1000) if six_steady[-1] + 1 < len(t) else None
    else:
        first_feet, vertical, horizontal, all_feet_ms, stable_ms = (None,) * 5
        interruptions = np.array([], int)
    return {
        'initial_oscillator_phase': facts['initial_oscillator_phase'],
        'continuous_air_before_arrival_ms': float(preceding_air_ms),
        'cruise_mean_forward_mm_s': float(v[cruise, 0].mean()),
        'cruise_min_height_mm': float(10 * trace['z_cm'][cruise].min()),
        'landing_trigger_ms': None if facts['landing_trigger_s'] is None else 1000 * facts['landing_trigger_s'],
        'first_contact_ms': landed, 'first_contact_feet': first_feet,
        'horizontal_speed_before_contact_mm_s': horizontal,
        'vertical_speed_before_contact_mm_s': vertical,
        'first_six_feet_ms': all_feet_ms, 'continuous_six_feet_from_ms': stable_ms,
        'air_steps_after_touchdown': len(interruptions),
        'nonfoot_contact_native_steps': int(np.count_nonzero(body)),
        'late_ground_contact_native_steps': int(np.count_nonzero(after & floor)),
        'last_100ms_min_feet': int(feet[last].min()),
        'last_100ms_height_mm': (10 * np.array([trace['z_cm'][last].min(), trace['z_cm'][last].max()])).tolist(),
        'last_100ms_max_speed_mm_s': float(np.linalg.norm(v[last], axis=1).max()),
        'last_100ms_max_roll_pitch_deg': np.abs(angles[last, 1:]).max(axis=0).tolist(),
        'last_100ms_wings_folded': bool(np.all(trace['wing_mix'][last] == 0)
                                       and np.all(trace['teacher_mix'][last] == 0)
                                       and np.max(np.ptp(traces[last, 14:16], axis=0)) < .02),
    }


def gates(metrics):
    speed = metrics['horizontal_speed_before_contact_mm_s']
    vertical = metrics['vertical_speed_before_contact_mm_s']
    lo, hi = metrics['last_100ms_height_mm']
    return {
        'proven_cruise': (metrics['continuous_air_before_arrival_ms'] >= 200
                          and metrics['cruise_mean_forward_mm_s'] >= 150
                          and metrics['cruise_min_height_mm'] >= 8),
        'feet_first_and_soft': (metrics['first_contact_feet'] is not None
                                and metrics['first_contact_feet'] >= 1
                                and speed <= 80 and -75 <= vertical <= 75),
        'no_nonfoot_contacts': metrics['nonfoot_contact_native_steps'] == 0,
        'six_feet_steady': (metrics['last_100ms_min_feet'] == 6
                            and .75 <= lo <= hi <= 1.10
                            and metrics['last_100ms_max_speed_mm_s'] <= 1.
                            and max(metrics['last_100ms_max_roll_pitch_deg']) <= 12.
                            and metrics['last_100ms_wings_folded']),
    }


def audit_neck(folder):
    info, _ = commands_from(folder)
    with np.load(folder / 'COMMANDS.npz', allow_pickle=False) as npz:
        emitted = npz['emitted']
        arrived = npz['arrived']
        assert np.array_equal(emitted[:, 0], emitted[:, 1])
        assert not np.any(emitted[:, 2, 2:])
        assert emitted[:, 0].sum(axis=0).tolist() == [28, 28, 30, 30]
        assert not np.any(arrived[:, 1:, 2:])
        assert not np.any(npz['synaptic_pa'][:, 1:, 2:])
        assert not np.any(npz['membrane_mv'][:, 1:, 2:])
        assert not np.any(npz['levels'][:, 1:, 2:])
        for idx, label in enumerate(info['ports']):
            delay = info['ports'][label]['delay_ticks']
            assert np.array_equal(np.flatnonzero(emitted[:, 0, idx]) + delay,
                                  np.flatnonzero(arrived[:, 0, idx]))
        first = float(np.flatnonzero(arrived[:, 0, 2:].any(axis=1))[0] * info['dt_ms'] / 1000)
    assert np.isclose(first, .4214)
    return info, first


def main():
    info, arrival = audit_neck(HERE / 'NECK')
    outcomes = {}
    total = 0
    for seed in SEEDS:
        folder = HERE / f'FORMAL{seed}'
        idn = read_json(folder / 'IDENTITY.json')
        assert idn['neck_data_sha256'] == info['sha256']
        assert idn['implementation_sha256'] == digest(HERE / 'land.py')
        assert idn['takeoff_config_sha256'] == digest(TAKEOFF / 'CALIBRATED.json')
        assert idn['flight_policy_sha256'] == digest(BASELINE / 'flight_reference/flight/saved_model.pb')
        assert idn['training_updates'] == 0
        runs = {case: read_case(folder / case) for case in CASES}
        active, meta = runs['intact']
        for case, (arrays, facts) in runs.items():
            t = arrays['TRACE.npy'][:, 0]
            total += len(t)
            assert len(t) == 21000 and not facts['terminal']
            assert facts['approach'] == read_json(HERE / 'CALIBRATED.json')
            assert np.allclose(t, np.arange(1, len(t) + 1) * .00005, rtol=0, atol=1e-9)
            assert not np.any(arrays['TRACE.npy'][:, 12])
            adhesion = [col for col, name in enumerate(facts['actuator_names']) if 'adhere' in name]
            assert len(adhesion) == 6 and not np.any(arrays['ACTION.npy'][:, adhesion])
            old = TAKEOFF / f'FORMAL{seed}/takeoff_probe'
            for name, count in [('INITIAL.npy', None), ('ACTION.npy', 2000), ('QPOS.npy', 2000)]:
                expected = np.load(old / name, allow_pickle=False)
                actual = arrays[name][:count]
                assert np.array_equal(actual, expected)
            expected = np.load(old / 'TRACE.npy', allow_pickle=False)
            # The longer neck run carries the tiny DNp11 tail past 280 ms.
            # All physical channels and both controllers remain exactly equal.
            columns = [i for i in range(22) if i not in (16, 17)]
            assert np.array_equal(arrays['TRACE.npy'][:8000, columns], expected[:, columns])
            before = t <= arrival + 1e-9
            assert np.array_equal(arrays['TRACE.npy'][before], active['TRACE.npy'][before])
            assert np.array_equal(arrays['QPOS.npy'][before[3::4]], active['QPOS.npy'][before[3::4]])
        cut, cut_meta = runs['cut']
        assert all(np.array_equal(cut[name], runs['sham'][0][name]) for name in cut)
        blank = measure(cut['TRACE.npy'], cut_meta, arrival)
        assert blank['late_ground_contact_native_steps'] == 0 and blank['landing_trigger_ms'] is None
        metrics = measure(active['TRACE.npy'], meta, arrival)
        assert metrics['landing_trigger_ms'] > 1000 * arrival
        results = gates(metrics)
        outcomes[str(seed)] = dict(metrics, gates=results, pass_all=all(results.values()))
    for case in ('intact', 'cut'):
        reference, _ = read_case(HERE / 'FORMAL29' / case)
        replay, _ = read_case(HERE / 'REPLAY29' / case)
        assert all(np.array_equal(reference[name], replay[name]) for name in reference)
    success = sum(item['pass_all'] for item in outcomes.values())
    result = {
        'status': 'CONTINUOUS_LANDING_PASS' if success == len(SEEDS) else 'LANDING_PHASE_DEPENDENT',
        'pass_count': success, 'total_count': len(SEEDS), 'cases_by_seed': outcomes,
        'source_hash': digest(HERE / 'land.py'), 'configuration_hash': digest(HERE / 'CALIBRATED.json'),
        'post_neck_hash': info['sha256'], 'post_neck_first_arrival_ms': 1000 * arrival,
        'physical_takeoff_prefix_exact': True, 'all_cut_equals_sham_exact': True,
        'independent_arrays_exact': 8, 'native_steps_formal': total, 'native_steps_replay': 42000,
        'controller_training_updates': 0,
    }
    (HERE / 'FINAL_LANDING.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(result['status'], success, '/', len(SEEDS), flush=True)


if __name__ == '__main__':
    main()
