"""Verify the archived neural and paired mechanical traces."""
import math
import numpy as np

from build_catalog import HERE, hash_file, read, record


def neural(catalog):
    folder = HERE / 'NEURAL_EXPANDED'
    proof = read(folder / 'RESULT.json')
    assert proof['catalog_sha256'] == hash_file(HERE / 'CATALOG.json')
    assert proof['total'] == proof['verified'] == len(catalog['ports']) == 52
    for path, sha in catalog['files'].items():
        from pathlib import Path
        assert hash_file(Path(path)) == sha
    cases = proof['cases'][1:]
    assert cases == list(catalog['ports'])
    shifted = np.arange(1200) + 20
    journal = {}
    for mode, details in proof['conditions'].items():
        with np.load(folder / f'{mode}.npz') as trace:
            emitted, arrived, voltage = trace['emitted'], trace['arrived'], trace['voltage']
        cut = np.array([root in details['cut_roots'] for root in cases])
        allowed = (~cut[None, :]) | (shifted[:, None] < 400) | (np.arange(1200)[:, None] >= 680)
        assert not arrived[:20].any() and not arrived[:, 0].any()
        assert np.array_equal(arrived[20:], (emitted & allowed[:, None, :])[:-20])
        assert np.isfinite(voltage).all() and int(arrived.sum()) == details['delivered']
        journal[mode] = int(arrived.sum())
        if mode == 'intact':
            for lane, root in enumerate(cases):
                info = proof['ports'][root]
                port = catalog['ports'][root]
                assert int(arrived[:, lane+1, lane].sum()) == info['delivered']
                assert set(info['monitored_roots']) <= {r for r, _ in port['outgoing']}
                assert (((voltage[:, lane]+52)*port['sign']).max() > .0001)

    cross = read(HERE / 'EON/RESULT.json')
    source = read(HERE / 'CATALOG_PRE_ABDOMEN.json')
    for root, port in source['ports'].items():
        assert catalog['ports'][root] == port
    assert len(set(cross['admitted'].values())) == len(cross['admitted']) == 25
    assert set(cross['admitted']) <= {r for r, p in catalog['ports'].items() if p['direction'] == 'DN'}
    assert all(count > 0 for group in cross['direct_probe_counts'].values() for count in group.values())
    with np.load(HERE / 'EON/EVENTS.npz') as traces:
        emitted, intact, cut = traces['emitted'], traces['delivered'], traces['cut']
        assert np.array_equal(emitted[:-20], intact[20:]) and not intact[:20].any()
        assert np.array_equal(intact[:320], cut[:320]) and not cut[320:510].any()
        assert np.array_equal(intact[510:], cut[510:])
    return {'admitted_axons': len(cases), 'path_groups': len(catalog['groups']), 'subgraph_nodes': proof['nodes'],
            'cross_edges': proof['cross_edges'], 'local_receptor_edges': proof['local_edges'],
            'delivered_by_condition': journal, 'cross_donor_flywire_descending_ports': len(cross['admitted'])}


def load_run(folder, case):
    path = HERE / folder / case
    result = read(path / 'RESULT.json')
    assert hash_file(path / 'TRACE.npy') == result['trace_sha256']
    assert hash_file(path / 'QPOS.npy') == result['pose_sha256']
    assert (HERE / folder / f'{case}.exit').read_text().strip() == '0'
    trace = np.load(path / 'TRACE.npy')
    assert np.isfinite(trace).all()
    return result, trace


def at(trace, time):
    return trace[np.argmin(np.abs(trace[:, 0]-time))]


def physical(catalog):
    groups = ('ground_forward','ground_speed','ground_halt','ground_back','ground_left','ground_right',
              'ground_cut','ground_taste','wing_power','wing_turn','wing_takeoff','wing_landing','wing_cut','wing_no_air')
    traces, stats, senses = {}, {}, {}
    for case in groups:
        info, values = load_run('PHYSICAL_SENSORY', case)
        assert info['catalog_sha256'] == hash_file(HERE / 'CATALOG_PRE_ABDOMEN.json')
        assert info['implementation_sha256'] == hash_file(HERE / 'physical_sensor_archive.py')
        traces[case] = values
        senses[case] = {group: sum(info['arrival_counts'].get(root, 0) for root in roots)
                        for group, roots in catalog['groups'].items()}
        if case.startswith('ground'):
            initial = at(values, .23)
            stats[case] = {'dx_mm': float(values[-1, 1] - initial[1]),
                           'yaw_deg': math.degrees(values[-1, 4] - initial[4]),
                           'late_dx_mm': float(values[-1, 1] - at(values, .83)[1]),
                           'stable_window_displacement_mm': float(np.linalg.norm(at(values, 1.34)[1:3]-at(values, 1.15)[1:3])),
                           'active_forward_speed_mm_s': float((at(values, 1.20)[1]-at(values, .46)[1])/.74),
                           'end_height_mm': float(values[-1,3])}
        else:
            stats[case] = {'wing_drive_lr': values[300:580, 4:6].mean(0).tolist(),
                           'wing_actuator_effort': float(np.abs(values[:, 12:18]).sum()),
                           'end_height_cm': float(values[-1,3]), 'max_midleg_drive': float(values[:,18].max())}
    base = stats['ground_forward']
    assert base['dx_mm'] > 0 and stats['ground_speed']['active_forward_speed_mm_s'] > base['active_forward_speed_mm_s']
    assert stats['ground_back']['dx_mm'] < 0
    assert stats['ground_left']['yaw_deg'] > 0 > stats['ground_right']['yaw_deg']
    assert np.array_equal(traces['ground_forward'][:229], traces['ground_cut'][:229])
    assert np.array_equal(traces['ground_forward'][:829], traces['ground_halt'][:829])
    assert abs(stats['ground_cut']['dx_mm']) < .02
    assert abs(stats['ground_halt']['late_dx_mm']) < .2 * base['late_dx_mm']
    assert senses['ground_cut']['front_touch'] > 0  # Ascending continues during the DN cut.
    assert all(senses['ground_forward'][group] > 0 for group in ('front_touch','front_load','leg_vibration',
                 'joint_position','joint_position_alt','joint_movement'))
    assert senses['ground_taste']['front_taste'] > 0 and not senses['ground_forward']['front_taste']
    assert stats['wing_power']['wing_actuator_effort'] > 0 == stats['wing_cut']['wing_actuator_effort']
    assert stats['wing_power']['end_height_cm'] > stats['wing_no_air']['end_height_cm']
    assert stats['wing_turn']['wing_drive_lr'][1] > stats['wing_turn']['wing_drive_lr'][0]
    assert stats['wing_landing']['wing_actuator_effort'] < stats['wing_power']['wing_actuator_effort']
    assert stats['wing_takeoff']['max_midleg_drive'] > 0 == stats['wing_power']['max_midleg_drive']
    assert senses['wing_power']['wing_strain'] and senses['wing_power']['haltere_strain']
    stats['pivots'] = {}
    for case in ('ground_pivot_left','ground_pivot_right'):
        _, values = load_run('PHYSICAL_PIVOT_GAIN', case)
        delta = values[-1] - at(values, .49)
        stats['pivots'][case] = {'yaw_deg': math.degrees(delta[4]), 'translation_mm': float(np.linalg.norm(delta[1:3]))}
    return stats, senses


def flight():
    proof = {}
    traces = {}
    for case in ('intact','cut','cut_restore_long','cut_restore_short'):
        info, trace = load_run('FREE_FLIGHT_QUAL', case)
        assert info['world_collision_geoms'] == ['groundplane']
        assert info['ground_contact_steps'] == int(np.count_nonzero(trace[:,6]))
        assert info['disabled_wing_force_max'] == 0
        traces[case] = trace
        proof[case] = info
    assert proof['intact']['simulated_time'] > .279 and proof['intact']['ground_contact_steps'] == 0
    assert proof['intact']['displacement'][0] > 5 and proof['intact']['min_height_cm'] > .9
    assert proof['cut']['simulated_time'] < .279 and proof['cut']['normal_wing_force_max'] == 0
    for case in ('cut_restore_long','cut_restore_short'):
        trace = traces[case]
        window = np.flatnonzero(trace[:,4] == 0)
        assert len(window)
        assert np.array_equal(traces['intact'][:window[0]], trace[:window[0]])
        reenabled = bool(window[-1] + 1 < len(trace))
        if reenabled:
            assert trace[window[-1]+1, 5] > 0
        proof[case]['wing_reenabled_before_termination'] = reenabled
        proof[case]['observed_gain_off_ms'] = float(window.size * .2)
        proof[case]['recovered_flight'] = bool(proof[case]['simulated_time'] > .279
                                             and proof[case]['ground_contact_steps'] == 0)
    assert proof['cut_restore_short']['wing_reenabled_before_termination']
    assert proof['cut_restore_short']['recovered_flight']
    return proof


def ascending_pair(catalog):
    before, a = load_run('PHYSICAL_AN_PAIR', 'ground_forward')
    after, b = load_run('PHYSICAL_AN_PAIR', 'ground_ascending_cut')
    assert before['front_touch_brain_targets'] == after['front_touch_brain_targets']
    assert np.array_equal(a[:, :-1], b[:, :-1])
    assert before['pose_sha256'] == after['pose_sha256']
    touch = catalog['groups']['front_touch']
    received = sum(before['arrival_counts'][r] for r in touch)
    assert received > 0 and not any(after['arrival_counts'][r] for r in touch)
    delta = float(np.max(np.abs(a[:, -1] - b[:, -1])))
    assert delta > 1.0
    return {'matched_pose_and_physics_bitexact': True, 'monitored_brain_cells': len(before['front_touch_brain_targets']),
            'intact_arrived': received, 'cut_arrived': 0, 'max_brain_voltage_effect_mv': delta}


if __name__ == '__main__':
    catalog = read(HERE / 'CATALOG.json')
    network = neural(catalog)
    body, senses = physical(catalog)
    flight_proof = flight()
    record(HERE / 'FINAL_AUDIT.json', {'result': 'SCOPED_PASS', 'network': network, 'ground_and_wing': body,
            'physical_sensory_arrivals': senses, 'ascending_cut_pair': ascending_pair(catalog),
            'free_flight': flight_proof, 'epochs': 0})
    print(network, flush=True)
