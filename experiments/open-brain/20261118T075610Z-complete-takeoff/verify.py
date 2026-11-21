"""Recompute the continuous takeoff acceptance from native steps and paired controls."""
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

from takeoff import BASELINE, COLUMNS, PARENT, digest, load_neck

HERE = Path(__file__).resolve().parent
SEEDS = (17, 29, 53)


def read_json(path):
    return json.loads(path.read_text())


def read_rollout(path):
    facts = read_json(path / 'RESULT.json')
    assert tuple(facts['columns']) == COLUMNS
    arrays = {}
    for name, sha in facts['files'].items():
        assert digest(path / name) == sha
        values = np.load(path / name, allow_pickle=False)
        assert np.isfinite(values).all()
        arrays[name] = values
    trace = arrays['TRACE.npy']
    assert trace.shape == (8000, len(COLUMNS))
    assert arrays['QPOS.npy'].shape[0] == 2000 and arrays['ACTION.npy'].shape == (2000, 66)
    assert facts['control_steps'] == 2000 and facts['native_steps'] == 8000 and not facts['terminal']
    data = dict(zip(COLUMNS, trace.T, strict=True))
    assert np.allclose(data['time_s'], np.arange(1, 8001) * facts['dt_s'], rtol=0, atol=1e-9)
    assert not np.any(data['external_applied'])
    adhesion_cols = [idx for idx, name in enumerate(facts['actuator_names']) if 'adhere' in name]
    assert len(adhesion_cols) == 6 and not np.any(arrays['ACTION.npy'][:, adhesion_cols])
    return facts, data, arrays


def wing_metrics(data, last, dt):
    output = {}
    for side in ('left', 'right'):
        angle = data[f'{side}_stroke'][last]
        peaks, _ = find_peaks(angle, distance=round(.003 / dt), prominence=1.)
        assert len(peaks) >= 17
        period = np.median(np.diff(peaks)) * dt
        amplitude = np.median([np.ptp(angle[lo:hi + 1]) for lo, hi in zip(peaks[:-1], peaks[1:])])
        output[side] = {'hz': float(1 / period), 'stroke_deg': float(np.rad2deg(amplitude))}
    return output


def flight_metrics(facts, data):
    time = data['time_s']
    ground = (data['feet'] + data['nonfoot_contacts']) > 0
    pre = (time >= .020) & (time <= .075)
    initial_support = bool(np.all(data['feet'][pre] == 6) and not np.any(data['nonfoot_contacts'][pre]))
    first = np.flatnonzero(~ground & (time > .080))
    last = np.flatnonzero(ground)
    last_sample = time >= .300
    last_contact = float(time[last[-1]]) if len(last) > 0 else 0.
    first_air = float(time[first[0]]) if len(first) > 0 else None
    amplitude = (wing_metrics(data, last_sample, facts['dt_s'])
                 if facts['engaged_flight'] and facts['fluid'] and np.any(data['wing_mix'][last_sample])
                 else {})
    return {
        'initial_six_feet': initial_support,
        'first_all_clear_ms': first_air * 1000 if first_air else None,
        'last_floor_contact_ms': 1000 * last_contact,
        'first_to_last_floor_contact_ms': 1000 * (last_contact - first_air) if first_air else None,
        'continuous_final_air_ms': float(1000 * (time[-1] - last_contact)),
        'nonfoot_contact_native_steps': int(np.count_nonzero(data['nonfoot_contacts'])),
        'end_height_mm': float(10 * data['z_cm'][-1]),
        'last_100ms_min_height_mm': float(10 * data['z_cm'][last_sample].min()),
        'last_100ms_forward_mm': float(10 * (data['x_cm'][last_sample][-1] - data['x_cm'][last_sample][0])),
        'last_100ms_abs_max_roll_deg': float(np.rad2deg(abs(data['roll_rad'][last_sample]).max())),
        'last_100ms_abs_max_pitch_deg': float(np.rad2deg(abs(data['pitch_rad'][last_sample]).max())),
        'last_100ms_wings': amplitude,
        'trigger_ms': None if facts['trigger_s'] is None else float(1000 * facts['trigger_s']),
        'initial_oscillator_phase': facts['initial_wing_phase'],
    }


def accept(metrics):
    return (metrics['initial_six_feet'] and metrics['first_all_clear_ms'] is not None
            and metrics['first_to_last_floor_contact_ms'] <= 25.
            and metrics['continuous_final_air_ms'] >= 200.
            and metrics['nonfoot_contact_native_steps'] == 0
            and metrics['last_100ms_min_height_mm'] >= 5.
            and metrics['last_100ms_forward_mm'] >= 10.
            and metrics['last_100ms_abs_max_roll_deg'] <= 30.
            and metrics['last_100ms_abs_max_pitch_deg'] <= 75.
            and len(metrics['last_100ms_wings']) == 2
            and all(170 <= wing['hz'] <= 240 and wing['stroke_deg'] >= 90.
                    for wing in metrics['last_100ms_wings'].values()))


def main():
    receipt, _ = load_neck(HERE / 'NECK_FRESH')
    delays = [receipt['ports'][f'takeoff/{side}']['delay_ticks'] for side in ('left', 'right')]
    arrival = (receipt['pulse_window_ms'][0] + min(delays) * receipt['dt_ms']) / 1000
    results = {}
    common_initial = None
    for seed in SEEDS:
        folder = HERE / f'FORMAL{seed}'
        identity = read_json(folder / 'IDENTITY.json')
        assert identity['post_neck_sha256'] == receipt['data_sha256']
        assert Path(identity['post_neck_source']) == HERE / 'NECK_FRESH'
        assert identity['training_updates'] == 0 and identity['source_sha256'] == digest(HERE / 'takeoff.py')
        for name, sha in identity['policy'].items():
            assert digest(BASELINE / 'flight_reference/flight' / name) == sha
        assert digest(BASELINE / 'flight_reference/wing_pattern_fmech.npy') == identity['wing_pattern_sha256']
        active_facts, active, active_arr = read_rollout(folder / 'takeoff_probe')
        cut_facts, cut, cut_arr = read_rollout(folder / 'takeoff_cut')
        blank_facts, blank, blank_arr = read_rollout(folder / 'sham')
        assert active_facts['settings'] == cut_facts['settings'] == blank_facts['settings']
        assert all(active_facts['settings'][key] == value
                   for key, value in read_json(HERE / 'CALIBRATED.json').items())
        assert active_facts['settings']['seed'] == seed
        assert active_facts['fluid'] and not cut_facts['engaged_flight'] and not blank_facts['engaged_flight']
        assert cut_facts['trigger_s'] is None and blank_facts['trigger_s'] is None
        assert all(np.array_equal(cut_arr[name], blank_arr[name]) for name in cut_arr)
        if common_initial is None:
            common_initial = active_arr['INITIAL.npy']
        for trial in (active_arr, cut_arr, blank_arr):
            assert np.array_equal(common_initial, trial['INITIAL.npy'])
        before_arrival = active['time_s'] <= arrival + 1e-10
        assert np.array_equal(active_arr['TRACE.npy'][before_arrival], cut_arr['TRACE.npy'][before_arrival])
        pose_before_arrival = active['time_s'][3::4] <= arrival + 1e-10
        assert np.array_equal(active_arr['QPOS.npy'][pose_before_arrival], cut_arr['QPOS.npy'][pose_before_arrival])
        assert not any(np.any(cut[field]) for field in ('vnc_left', 'vnc_right', 'wing_mix', 'teacher_mix', 'phase'))
        assert np.all(cut['feet'][(cut['time_s'] > .020)] == 6) and not np.any(cut['nonfoot_contacts'])
        metrics = flight_metrics(active_facts, active)
        assert active['vnc_left'].max() > .22 and active['vnc_right'].max() > .22
        assert active['phase'].max() == 2. and accept(metrics), metrics
        results[str(seed)] = metrics
    assert len({row['initial_oscillator_phase'] for row in results.values()}) == len(SEEDS)
    dry_facts, dry, dry_arr = read_rollout(HERE / 'FORMAL17/fluid_off')
    leg_facts, legs, leg_arr = read_rollout(HERE / 'FORMAL17/legs_only')
    assert not dry_facts['fluid'] and leg_facts['fluid'] and not np.any(legs['wing_mix'])
    assert np.array_equal(common_initial, dry_arr['INITIAL.npy'])
    assert np.array_equal(common_initial, leg_arr['INITIAL.npy'])
    _, normal, _ = read_rollout(HERE / 'FORMAL17/takeoff_probe')
    for data in (dry, legs):
        for field in ('vnc_left', 'vnc_right'):
            assert np.array_equal(normal[field], data[field])
    ablation = {'fluid_off': flight_metrics(dry_facts, dry), 'legs_only': flight_metrics(leg_facts, legs)}
    assert all(not accept(row) and row['continuous_final_air_ms'] < 200. for row in ablation.values())
    for case in ('takeoff_probe', 'takeoff_cut'):
        old_facts, _, old = read_rollout(HERE / 'FORMAL29' / case)
        replay_facts, _, replay = read_rollout(HERE / 'REPLAY29' / case)
        assert old_facts == replay_facts and all(np.array_equal(replay[name], old[name]) for name in replay)
    result = {
        'status': 'CONTINUOUS_POWERED_TAKEOFF_PASS', 'experiment': 'P1 FlyBody with 66 leg DoFs',
        'stimulus_source': 'scheduled bilateral DNp11 anterior pulses with confirmed BANC crossing',
        'first_neck_arrival_ms': arrival * 1000,
        'neck_hash': receipt['data_sha256'],
        'previous_neck_trace_byte_identical': digest(PARENT / 'NECK/POST_NECK.npz') == receipt['data_sha256'],
        'by_seed': results, 'ablation_seed17': ablation,
        'matching_cut_and_sham': 'all arrays bit-exact in each seed',
        'all_initial_qpos_and_qvel_exact': True,
        'body_native_steps_formal': 11 * 8000, 'body_native_steps_independent_replay': 2 * 8000,
        'independent_replay_arrays': 8, 'controller_training_updates': 0,
        'config_sha256': digest(HERE / 'CALIBRATED.json'),
        'body_source_sha256': digest(HERE / 'takeoff.py'),
    }
    (HERE / 'FINAL_TAKEOFF.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
