"""Compare old and corrected physical runs, with terminal contacts and food transfer."""
import json
from pathlib import Path

import numpy as np

from inspect_pose import FLIGHT, audit, build_scene

HERE = Path(__file__).resolve().parent
WINGS = tuple(f'wing_{a}_{s}' for s in ('left', 'right') for a in ('yaw', 'roll', 'pitch'))
PAST_WING_WINDOW = .20


def read(folder):
    return json.loads((folder / 'RESULT.json').read_text())


def load(folder, name):
    return np.load(folder / (name + '.npy'), allow_pickle=False)


def leg_stand(traces, dt):
    active = (traces['perch_ready'] == 1) & (traces['fruit_feet'] == 6) & (traces['floor_feet'] == 0)
    suffix = len(active) - 1 - np.flatnonzero(~active)[-1]
    return float(dt * suffix)


def contact_scan(phy, positions):
    model, data = phy.model, phy.data
    names = tuple(model.id2name(i, 'geom') or '' for i in range(model.ngeom))
    wing = np.array([n.startswith('walker/wing_') for n in names])
    abdomen = np.array([n.startswith('walker/abdomen') for n in names])
    contacts = np.zeros(len(positions), dtype=int)
    pairs = set()
    for i, qpos in enumerate(positions):
        data.qpos[:] = qpos
        phy.forward()
        for con in data.contact[:data.ncon]:
            a, b = con.geom
            if (wing[a] and abdomen[b]) or (wing[b] and abdomen[a]):
                contacts[i] += 1
                pairs.add(tuple(sorted((names[a], names[b]))))
    return contacts, sorted(pairs)


def compare(name):
    original, fixed = FLIGHT / name, HERE / name
    base, result = read(original), read(fixed)
    body = load(fixed, 'TRACE')
    traces = dict(zip(result['columns'], body.T, strict=True))
    original_qpos, fixed_qpos = load(original, 'QPOS'), load(fixed, 'QPOS')
    assert base['visual_trigger_s'] == result['visual_trigger_s']
    assert base['landing_s'] == result['landing_s']
    assert base['fold_s'] == result['fold_s']
    assert base['first_fruit_contact_s'] == result['first_fruit_contact_s']
    assert np.array_equal(load(original, 'INITIAL'), load(fixed, 'INITIAL'))
    assert np.array_equal(load(original, 'EYE'), load(fixed, 'EYE'))
    cut = round(base['fold_s'] / .0002)
    assert np.array_equal(original_qpos[:cut], fixed_qpos[:cut])
    assert np.array_equal(load(original, 'ACTION')[:cut], load(fixed, 'ACTION')[:cut])
    assert np.array_equal(load(original, 'TRACE')[:cut*4], body[:cut*4])
    geometry = result['geometry']
    env = build_scene(geometry['x_cm'], geometry['y_cm'])
    env.reset()
    phy = env.physics
    model = phy.model
    indexes = [model.name2id('walker/' + j, 'joint') for j in WINGS]
    addr = model.jnt_qposadr[indexes]
    stow = model.qpos_spring[addr]
    control_steps = round(PAST_WING_WINDOW / .0002)
    err_old = np.rad2deg(original_qpos[:, addr] - stow)
    err_new = np.rad2deg(fixed_qpos[:, addr] - stow)
    old_contacts, old_pairs = contact_scan(phy, original_qpos[cut:])
    new_contacts, new_pairs = contact_scan(phy, fixed_qpos[cut:])
    orig_audit, fix_audit = audit(original), audit(fixed)
    assert orig_audit['initial_matches_xml'] and fix_audit['initial_matches_xml']
    after = traces['time_s'] >= result['first_fruit_contact_s']
    meal = read(HERE / ('MEAL_' + name))
    assert meal['trace_sha256'] == result['files']['TRACE.npy']
    support_time = leg_stand(traces, result['native_dt_s'])
    old_error = float(np.max(np.abs(err_old[-control_steps:])))
    new_error = float(np.max(np.abs(err_new[-control_steps:])))
    assert old_error > 10. and new_error <= 2.
    assert np.all(old_contacts[-control_steps:] > 0) and not np.any(new_contacts[-control_steps:])
    assert result['external_force_max'] == 0.0 and np.all(traces['nonfoot_contacts'][after] == 0)
    assert np.all(traces['floor_feet'][after] == 0) and support_time >= .30
    assert meal['intake'] >= .07 and meal['first_intake_s'] >= result['first_taste_s']
    assert abs(meal['nutrient_balance_error']) < 1e-9
    np.savez(HERE / f'{name}_POSES.npz', time_s=np.arange(1, len(err_old) + 1) * .0002,
             old_error_deg=err_old, corrected_error_deg=err_new, start_index=cut,
             old_contact_count=old_contacts, corrected_contact_count=new_contacts)
    return dict(case=name, trigger_s=result['visual_trigger_s'], touchdown_s=result['first_fruit_contact_s'],
                terminal_wing_error_deg={'original': old_error, 'corrected': new_error},
                terminal_wing_abdomen_samples={'original': int(np.count_nonzero(old_contacts[-control_steps:])),
                                               'corrected': int(np.count_nonzero(new_contacts[-control_steps:])),
                                               'total': control_steps},
                wing_abdomen_contact_pairs={'original': old_pairs, 'corrected': new_pairs},
                abdomen_tip_relative_to_thorax_mm={'original': orig_audit['samples']['1.42']['abdomen_tip_thorax_mm'][2],
                                                  'corrected': fix_audit['samples']['1.42']['abdomen_tip_thorax_mm'][2]},
                continuous_six_foot_perch_s=support_time, first_intake_s=meal['first_intake_s'],
                simulated_food_intake=meal['intake'], flight_prefix_control_steps=cut)


if __name__ == '__main__':
    facts = [compare(name) for name in ('NEAR', 'FAR')]
    replay = all(np.array_equal(load(HERE / 'NEAR', name), load(HERE / 'REPLAY', name))
                 for name in ('TRACE', 'QPOS', 'ACTION', 'INITIAL', 'EYE'))
    assert replay
    outcome = dict(status='PASS_PERCH_WING_STOW', unchanged_flight_before_fold=True,
                   independent_replay_identical=replay, results=facts)
    (HERE / 'PERCH_WING_CURRENT.json').write_text(json.dumps(outcome, indent=2) + '\n')
    print(json.dumps(outcome, indent=2))
