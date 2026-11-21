"""Re-integrate paired landing commands on the original BANC fixed circuit."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

FLIGHT = Path('/home/sensen/.codex/worktrees/4528/cell_wound_prototype/experiments/open-brain/20261118T072349Z-flight-descending')
sys.path.insert(0, str(FLIGHT))
from probe_neck import DT_MS, FIXED, REST_MV, FixedCircuit, digest, full_catalog, make_probes

LABELS = ('takeoff/left', 'takeoff/right', 'landing/left', 'landing/right')
CASES = ('intact', 'cut', 'sham')
TAKEOFF = (800, 1900)
LANDING = (4200, 5400)
TICKS = 11000
PERIOD = 40


def run(output):
    parent = FLIGHT / 'NECK'
    receipt = json.loads((parent / 'IDENTITY.json').read_text())
    assert digest(parent / 'POST_NECK.npz') == receipt['data_sha256']
    roots = [receipt['ports'][label]['banc'] for label in LABELS]
    scale = np.array([receipt['ports'][label]['reference_mv'] for label in LABELS])
    values = {name: np.zeros((TICKS, len(CASES), len(LABELS)), dtype=dtype)
              for name, dtype in [('emitted', bool), ('arrived', bool), ('membrane_mv', float),
                                  ('synaptic_pa', float), ('levels', float)]}
    for lanes, block in [((0, 2), False), ((1,), True)]:
        circuit = FixedCircuit(full_catalog(), batch=len(lanes))
        probes = make_probes(circuit, roots)
        assert [[list(row) for row in group] for group in probes] == [
            receipt['ports'][label]['targets_and_contact_counts'] for label in LABELS]
        cols = np.array([circuit.port_index[root] for root in roots])
        target_cols = np.array([[circuit.index[target] for target, _ in group] for group in probes])
        zeros = np.zeros((len(lanes), len(circuit.roots)), np.float32)
        mask = np.zeros(len(circuit.ports), dtype=bool)
        mask[cols[2:]] = block
        events = np.zeros((len(lanes), len(circuit.ports)), dtype=bool)
        local = np.zeros((len(lanes), len(LABELS)))
        alpha = -np.expm1(-DT_MS / receipt['filter_ms'])
        for tick in range(TICKS):
            launch = TAKEOFF[0] <= tick < TAKEOFF[1] and (tick - TAKEOFF[0]) % PERIOD == 0
            landing = LANDING[0] <= tick < LANDING[1] and (tick - LANDING[0]) % PERIOD == 0
            events[:, cols[:2]] = launch
            events[0, cols[2:]] = landing
            emitted, arrived, _, _ = circuit.step(zeros, mask, eon_output=events)
            voltage = (circuit.pre_reset_voltage[:, target_cols] - REST_MV).mean(axis=-1)
            current = circuit.last_current.sum(axis=0)[:, target_cols].mean(axis=-1)
            local += alpha * (voltage - local)
            values['emitted'][tick, lanes, :] = emitted[:, cols]
            values['arrived'][tick, lanes, :] = arrived[:, cols]
            values['membrane_mv'][tick, lanes, :] = voltage
            values['synaptic_pa'][tick, lanes, :] = current
            values['levels'][tick, lanes, :] = np.clip(local / scale, 0., 1.25)
    for channel, label in enumerate(LABELS):
        delay = receipt['ports'][label]['delay_ticks']
        for lane in range(len(CASES)):
            source = np.flatnonzero(values['emitted'][:, lane, channel])
            if lane == 1 and channel in (2, 3):
                source = np.array([], int)
            assert np.array_equal(np.flatnonzero(values['arrived'][:, lane, channel]), source + delay)
    assert np.array_equal(values['emitted'][:, 0], values['emitted'][:, 1])
    assert not np.any(values['levels'][:, 1:, 2:])
    with np.load(parent / 'POST_NECK.npz', allow_pickle=False) as historical:
        index = list(historical['conditions']).index('takeoff_probe')
        reference = historical['levels'][:, index, 4:6]
        assert np.array_equal(reference, values['levels'][:len(reference), 0, :2])
    output.mkdir()
    data = output / 'COMMANDS.npz'
    np.savez_compressed(data, labels=np.array(LABELS), cases=np.array(CASES), **values)
    (output / 'IDENTITY.json').write_text(json.dumps({
        'sha256': digest(data), 'dt_ms': DT_MS, 'ticks': TICKS,
        'command_windows_ms': {'DNp11': [value * DT_MS for value in TAKEOFF],
                               'DNp10': [value * DT_MS for value in LANDING]},
        'ports': {label: receipt['ports'][label] for label in LABELS},
        'fixed_neck_sha256': digest(FIXED / 'fixed_neck.py'),
        'parameters_sha256': circuit.params_sha256,
        'previous_takeoff_byte_exact_in_prefix': True,
        'cases': {'source_counts': values['emitted'].sum(axis=0).tolist(),
                  'arrival_counts': values['arrived'].sum(axis=0).tolist()},
    }, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)
