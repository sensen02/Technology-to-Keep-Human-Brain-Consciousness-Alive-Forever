"""Stimulate mapped anterior DN roots and export only post-neck body commands."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

from flight_manifest import GROUPS, LABELS, ROOT, digest

FIXED = ROOT / '20261117T164207Z-fixed-neck'
sys.path.insert(0, str(FIXED))
from fixed_neck import DT_MS, REST_MV, FixedCircuit, full_catalog

CONDITIONS = {
    'sham': (),
    'amp_low': (0, 1),
    'amp_high': (0, 1),
    'amp_cut': (0, 1),
    'left': (2,),
    'right': (3,),
    'left_cut': (2,),
    'right_cut': (3,),
    'combined': (0, 1, 2),
    'takeoff_probe': (4, 5),
    'takeoff_cut': (4, 5),
    'landing_probe': (6, 7),
    'landing_cut': (6, 7),
}
TICKS = 2800
START, STOP = 800, 1900
REFERENCE_PERIOD = 40
LOW_PERIOD = 80
FILTER_MS = 8.


def make_probes(circuit, roots):
    owners = defaultdict(set)
    for root in circuit.brain_roots:
        for target, _ in circuit.ports[root]['outgoing']:
            owners[target].add(root)
    result = []
    for root in roots:
        exclusive = [(target, count) for target, count in circuit.ports[root]['outgoing']
                     if owners[target] == {root} and target not in circuit.allowed_drive_roots]
        ranked = sorted(exclusive, key=lambda row: (-row[1], row[0]))
        if len(ranked) < 4:
            raise ValueError(f'Insufficient exclusive VNC targets for {root}')
        result.append(ranked[:4])
    return result


def main(output):
    catalog = full_catalog()
    binding = {f"{port['group']}/{port['side']}": root
               for root, port in catalog['ports'].items() if port['group'] in GROUPS}
    roots = [binding[label] for label in LABELS]
    mapping_path = ROOT / '20261117T051249Z-bidirectional/EON/RESULT.json'
    eon = json.loads(mapping_path.read_text())['admitted']
    assert len({eon[root] for root in roots}) == len(roots)
    calibration = {f'ref_{index}': (index,) for index in range(len(roots))}
    cases = dict(calibration, **CONDITIONS)
    names = tuple(cases)
    # Build with the full registered catalog for target ownership. Only the eight
    # flight source columns can emit scheduled external events.
    circuit = FixedCircuit(catalog, batch=len(cases))
    probes = make_probes(circuit, roots)
    probe_cols = np.array([[circuit.index[target] for target, _ in group] for group in probes])
    port_cols = np.array([circuit.port_index[root] for root in roots])
    cut_cases = [name.endswith('_cut') for name in names]
    # The fixed kernel's cut mask is common to its batch. Run cut and intact as
    # parallel groups with fresh state rather than changing kernel semantics.
    del circuit
    shape = (TICKS, len(names), len(roots))
    emitted = np.zeros(shape, dtype=bool)
    arrived = np.zeros_like(emitted)
    membrane = np.zeros(shape)
    synaptic = np.zeros(shape)
    filtered = np.zeros(shape)
    delays = None
    for blocking in (False, True):
        lanes = [index for index, cut in enumerate(cut_cases) if cut == blocking]
        circuit = FixedCircuit(catalog, batch=len(lanes))
        delays = circuit.neck_delay[port_cols].copy()
        drive = np.zeros((len(lanes), len(circuit.roots)), np.float32)
        external = np.zeros((len(lanes), len(circuit.ports)), dtype=bool)
        cut = np.zeros(len(circuit.ports), dtype=bool)
        if blocking:
            cut[port_cols] = True
        schedules = np.zeros((TICKS, len(lanes), len(roots)), dtype=bool)
        for lane, index in enumerate(lanes):
            case = names[index]
            period = LOW_PERIOD if case == 'amp_low' else REFERENCE_PERIOD
            for selected in cases[case]:
                schedules[START:STOP:period, lane, selected] = True
        level = np.zeros((len(lanes), len(roots)))
        alpha = -np.expm1(-DT_MS / FILTER_MS)
        for tick in range(TICKS):
            external[:, port_cols] = schedules[tick]
            source, arrival, _, _ = circuit.step(drive, cut, eon_output=external)
            emitted[tick, lanes] = source[:, port_cols]
            arrived[tick, lanes] = arrival[:, port_cols]
            voltage = (circuit.pre_reset_voltage[:, probe_cols] - REST_MV).mean(axis=-1)
            current = circuit.last_current.sum(axis=0)[:, probe_cols].mean(axis=-1)
            membrane[tick, lanes] = voltage
            synaptic[tick, lanes] = current
            level += alpha * (voltage - level)
            filtered[tick, lanes] = level
        for local, global_index in enumerate(lanes):
            assert np.array_equal(emitted[:, global_index], schedules[:, local])
            for channel, delay in enumerate(delays):
                expected = np.flatnonzero(emitted[:, global_index, channel]) + delay
                actual = np.flatnonzero(arrived[:, global_index, channel])
                assert np.array_equal(actual, [] if blocking else expected)
    for index, name in enumerate(names):
        if name.endswith('_cut') or name == 'sham':
            assert not np.any(filtered[:, index])
    reference = np.array([filtered[1500:1850, channel, channel].mean()
                          for channel in range(len(roots))])
    assert np.all(reference > 0.1)
    levels = np.clip(filtered / reference, 0., 1.25)
    output.mkdir()
    data = output / 'POST_NECK.npz'
    np.savez_compressed(data, conditions=np.array(names), labels=np.array(LABELS),
                        emitted=emitted, arrived=arrived, membrane_mv=membrane,
                        synaptic_pa=synaptic, filtered_mv=filtered, levels=levels)
    definition = {
        'data_sha256': digest(data), 'dt_ms': DT_MS,
        'fixed_neck_source_sha256': digest(FIXED / 'fixed_neck.py'),
        'parameters_sha256': circuit.params_sha256,
        'catalog_sha256': digest(ROOT / '20261117T051249Z-bidirectional/CATALOG.json'),
        'eon_mapping_sha256': digest(mapping_path),
        'root_source': 'scheduled anterior pulses on reviewed Eon/BANC pairs',
        'filter_ms': FILTER_MS, 'pulse_window_ms': [START * DT_MS, STOP * DT_MS],
        'reference_pulse_hz': 1000 / (DT_MS * REFERENCE_PERIOD),
        'low_pulse_hz': 1000 / (DT_MS * LOW_PERIOD),
        'ports': {label: {'banc': root, 'eon': eon[root], 'delay_ticks': int(delays[index]),
                          'targets_and_contact_counts': probes[index],
                          'reference_mv': float(reference[index])}
                  for index, (label, root) in enumerate(zip(LABELS, roots, strict=True))},
        'cases': {name: {'emitted': emitted[:, index].sum(axis=0).tolist(),
                         'arrived': arrived[:, index].sum(axis=0).tolist()}
                  for index, name in enumerate(names)},
    }
    (output / 'IDENTITY.json').write_text(json.dumps(definition, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    main(parser.parse_args().out)
