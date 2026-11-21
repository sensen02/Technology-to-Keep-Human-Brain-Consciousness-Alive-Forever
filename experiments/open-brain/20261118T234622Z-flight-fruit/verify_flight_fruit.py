"""Check the paired visual trigger, post-neck delivery, landing, and ingestion."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

MIN_PERCH_S = .30
MIN_INTAKE = .07


def read(folder):
    facts = json.loads((folder / 'RESULT.json').read_text())
    if 'files' in facts:
        for name, expected in facts['files'].items():
            assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    return facts


def array(folder, name):
    return np.load(folder / f'{name}.npy', allow_pickle=False)


def series(path):
    facts = read(path)
    return facts, dict(zip(facts['columns'], array(path, 'TRACE').T, strict=True))


def longest(mask, dt):
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return float(np.max(edges[1::2] - edges[::2], initial=0) * dt)


def check_prefix(guide_path, body_path, neck_path):
    guide, g = series(guide_path)
    body, b = series(body_path)
    receipt = json.loads((neck_path / 'IDENTITY.json').read_text())
    emitted_ms = np.array(receipt['command_windows_ms']['DNp10'])
    trigger = emitted_ms[0] / 1000.
    r = np.load(neck_path / 'COMMANDS.npz', allow_pickle=False)
    first_arrival = float(np.flatnonzero(np.any(r['arrived'][:, 0, 2:], axis=1))[0] * receipt['dt_ms'] / 1000.)
    assert guide['seed'] == body['seed'] and guide['geometry'] == body['geometry']
    assert guide['visual_trigger_s'] == body['visual_trigger_s'] == trigger
    assert body['landing_s'] > first_arrival > trigger
    step_prefix = round(trigger / .0002)
    assert np.array_equal(array(guide_path, 'INITIAL'), array(body_path, 'INITIAL'))
    assert np.array_equal(array(guide_path, 'EYE'), array(body_path, 'EYE'))
    for name in ('QPOS', 'ACTION'):
        assert np.array_equal(array(guide_path, name)[:step_prefix], array(body_path, name)[:step_prefix]), name
    assert np.array_equal(np.column_stack(tuple(g.values()))[:step_prefix * 4],
                          np.column_stack(tuple(b.values()))[:step_prefix * 4])
    return first_arrival, step_prefix


def assess(root, primary, second):
    results = {}
    for key, label in [('NEAR', primary), ('FAR', second)]:
        case = root / label
        data, trace = series(case)
        food = read(root / ('MEAL_' + key))
        times = trace['time_s']
        gates = {
            'visual_trigger_matches_causal_prefix': False,
            'flight_before_fruit': False,
            'six_feet_continuously_perched': False,
            'no_off_fruit_support_after_touch': False,
            'contact_gates_actual_intake': False,
            'no_external_load': data['external_force_max'] == 0,
        }
        first_arrival = prefix = None
        if data['visual_trigger_s'] is not None and data['landing_s'] is not None:
            first_arrival, prefix = check_prefix(root / ('GUIDE_' + key), case, root / ('NECK_' + key))
            gates['visual_trigger_matches_causal_prefix'] = True
        quiet = (trace['fruit_feet'] == 6) & (trace['floor_feet'] == 0) & (trace['nonfoot_contacts'] == 0)
        confirmed = (trace['perch_ready'] > 0) & quiet
        perch = longest(confirmed, data['native_dt_s'])
        gates['six_feet_continuously_perched'] = perch >= MIN_PERCH_S and bool(confirmed[-1])
        touch = data['first_fruit_contact_s']
        if touch is not None and data['landing_s'] is not None:
            air = (times >= .16) & (times < touch - data['native_dt_s'])
            after = times >= touch
            gates['flight_before_fruit'] = bool(air.any() and touch > data['landing_s']
                and np.all(trace['feet'][air] == 0) and np.all(trace['nonfoot_contacts'][air] == 0))
            gates['no_off_fruit_support_after_touch'] = bool(
                np.all(trace['floor_feet'][after] == 0) and np.all(trace['nonfoot_contacts'][after] == 0))
        if food['first_intake_s'] is not None and data['first_taste_s'] is not None:
            meal = array(root / ('MEAL_' + key), 'MEAL')
            advances = np.diff(meal[:, 5], prepend=0.) > 0
            assert food['trace_sha256'] == data['files']['TRACE.npy']
            gates['contact_gates_actual_intake'] = bool(
                food['intake'] >= MIN_INTAKE and food['first_intake_s'] >= data['first_taste_s']
                and np.all(meal[advances, 1] > 0) and np.all(meal[advances, 4] > 0)
                and data['first_taste_s'] > data['first_fruit_contact_s'] and abs(food['nutrient_balance_error']) < 1e-9)
        results[key.lower()] = dict(gates=gates, first_command_to_vnc_s=first_arrival, prefix_steps=prefix,
                    seed=data['seed'], fruit=data['geometry'], takeoff_s=data['takeoff_s'],
                    eye_trigger_s=data['visual_trigger_s'], landing_command_s=data['landing_s'],
                    first_fruit_contact_s=touch, first_all_six_s=data['first_all_six_s'],
                    confirmed_perch_s=perch, final_six=bool(confirmed[-1]),
                    end_xy_mm=[float(trace['x_cm'][-1] * 10), float(trace['y_cm'][-1] * 10)],
                    first_intake_s=food['first_intake_s'], intake=food['intake'],
                    absorbed_nutrient=food['absorbed_nutrient'], mn9_spikes=food['mn9_spikes'])
    actual_cut, cut = series(root / 'CUT')
    blind, blindness = series(root / 'BLIND')
    masked_meal = read(root / 'MEAL_BLOCKED')
    cut_meal = read(root / 'MEAL_CUT')
    selected, _ = series(root / primary)
    actual_replay = root / 'REPLAY'
    loop_replay = all(np.array_equal(array(root / primary, name), array(actual_replay, name))
                      for name in ('INITIAL', 'TRACE', 'QPOS', 'ACTION', 'EYE'))
    cut_gates = dict(
        vision_blocked=blind['visual_trigger_s'] is None and blind['landing_s'] is None
            and not np.any(blindness['perch_ready']),
        neck_cut=actual_cut['visual_trigger_s'] == selected['visual_trigger_s']
            and actual_cut['landing_s'] is None and not np.any(cut['perch_ready']) and cut_meal['intake'] == 0.,
        food_blocked=masked_meal['trace_sha256'] == selected['files']['TRACE.npy']
            and masked_meal['intake'] == 0. and masked_meal['mn9_spikes'] == 0,
        replay_identical=loop_replay,
    )
    status = ('PASS_STRAIGHT_FLIGHT_PERCH_FEED'
              if all(cut_gates.values()) and all(all(r['gates'].values()) for r in results.values())
              else 'PARTIAL_STRAIGHT_FLIGHT_PERCH_FEED')
    verdict = dict(status=status, limits={'continuous_perch_s': MIN_PERCH_S, 'min_intake': MIN_INTAKE},
                   results=results, controls=cut_gates)
    (root / 'FLIGHT_FRUIT_CURRENT.json').write_text(json.dumps(verdict, indent=2, allow_nan=False) + '\n')
    print(json.dumps(verdict, indent=2), flush=True)
    return status.startswith('PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--near', default='NEAR')
    parser.add_argument('--far', default='FAR')
    args = parser.parse_args()
    raise SystemExit(0 if assess(args.root, args.near, args.far) else 1)
