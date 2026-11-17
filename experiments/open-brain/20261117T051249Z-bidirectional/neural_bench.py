"""Probe each admitted DN, AN relay and directly ascending sensory cell."""
import argparse
from pathlib import Path

import numpy as np
import torch

from build_catalog import hash_file, read, record
from circuit import Circuit, HERE, TRANSMISSION_TICKS

TICKS = 1200
START, END = 150, 970
CUT_START, CUT_END = 400, 680
PERIOD = 30
AMPLITUDE_MV = 68.75


def make_stimuli(circuit):
    conditions = [None, *circuit.port_roots]
    amplitude = np.zeros((len(conditions), len(circuit.roots)), dtype=np.float32)
    for lane, root in enumerate(conditions[1:], 1):
        spec = circuit.ports[root]
        roots = [pre for pre, _ in spec['incoming_receptors']] if spec['direction'] == 'AN' else [root]
        side = 'brain' if spec['direction'] == 'DN' else 'body'
        amplitude[lane, circuit.validate_inputs(roots, side)] = AMPLITUDE_MV
    return conditions, amplitude


def replay(model, catalog, conditions, stimulus, mode, period):
    mode_targets = {'intact': [], 'cut_all': list(model.port_roots),
                    'cut_selective': [catalog['groups']['walking_steer'][0], catalog['groups']['front_touch'][0]]}
    cut = np.array([root in mode_targets[mode] for root in model.port_roots], dtype=bool)
    open_mask = np.zeros_like(cut)
    zero = np.zeros_like(stimulus)
    emitted = np.zeros((TICKS, model.batch, len(model.port_roots)), dtype=bool)
    arrived = np.zeros_like(emitted)
    voltage = np.zeros((TICKS, model.batch-1, 3), dtype=np.float32)
    targets = []
    for root in conditions[1:]:
        known = sorted(model.ports[root]['outgoing'], key=lambda pair: (-pair[1], pair[0]))
        # Monitor the three strongest directly verified target cells, excluding driven cells.
        selected = [model.index[dst] for dst, _ in known if dst not in model.allowed_drive_roots][:3]
        assert len(selected) == 3
        targets.append(selected)
    lane_index = np.arange(1, model.batch)[:, None]
    for tick in range(TICKS):
        on = START <= tick < END and (tick - START) % period == 0
        mask = cut if CUT_START <= tick < CUT_END else open_mask
        emitted[tick], arrived[tick], _, state = model.step(stimulus if on else zero, mask)
        voltage[tick] = state[lane_index, np.asarray(targets)]
    return emitted, arrived, voltage, cut, targets


def expected_transit(emitted, cut):
    expected = np.zeros_like(emitted)
    for tick, lane, channel in np.argwhere(emitted):
        due = tick + TRANSMISSION_TICKS
        if due < TICKS and (not cut[channel] or tick >= CUT_END or due < CUT_START):
            expected[due, lane, channel] = True
    return expected


def main(out, period):
    catalog = read(HERE / 'CATALOG.json')
    out.mkdir()
    brain = Circuit(catalog, len(catalog['ports']) + 1)
    cases, stimulus = make_stimuli(brain)
    baseline = None
    results = {}
    native = {}
    for mode in ('intact', 'cut_all', 'cut_selective'):
        model = brain if mode == 'intact' else Circuit(catalog, len(cases))
        emitted, arrived, monitored, cut, targets = replay(model, catalog, cases, stimulus, mode, period)
        assert np.array_equal(arrived, expected_transit(emitted, cut)), mode
        assert not emitted[:, 0].any() and not arrived[:, 0].any()
        if baseline is None:
            baseline = arrived
        else:
            assert np.array_equal(baseline[:, :, ~cut], arrived[:, :, ~cut]), mode
            assert not np.any(arrived[CUT_START:CUT_END + TRANSMISSION_TICKS, :, cut]), mode
        np.savez_compressed(out / f'{mode}.npz', emitted=emitted, arrived=arrived, voltage=monitored)
        native[mode] = {'delivered': int(arrived.sum()), 'cut_roots': [r for r, c in zip(model.port_roots, cut) if c]}
        if mode == 'intact':
            for channel, root in enumerate(model.port_roots):
                emitted_count = int(emitted[:, channel + 1, channel].sum())
                delivered_count = int(arrived[:, channel + 1, channel].sum())
                sign = model.ports[root]['sign']
                response = float(((monitored[:, channel] + 52) * sign).max())
                results[root] = {'group': model.ports[root]['group'], 'emitted': emitted_count,
                                 'delivered': delivered_count, 'peak_signed_response_mv': response,
                                 'verified': bool(emitted_count and delivered_count and response > 0.0001),
                                 'monitored_roots': [model.roots[i] for i in targets[channel]]}
        if mode == 'cut_all':
            for channel, root in enumerate(model.port_roots):
                results[root]['blocked_in_window'] = int(baseline[CUT_START:CUT_END, channel+1, channel].sum())
                results[root]['restored_after_window'] = int(arrived[CUT_END+TRANSMISSION_TICKS:, channel+1, channel].sum())
                results[root]['verified'] &= bool(results[root]['blocked_in_window'] and results[root]['restored_after_window'])
        print(mode, native[mode]['delivered'], flush=True)
    record(out / 'RESULT.json', {'cases': cases, 'ports': results, 'conditions': native,
             'catalog_sha256': hash_file(HERE / 'CATALOG.json'), 'dt_ms': 0.1, 'stimulus_period_ticks': period,
             'nodes': len(brain.roots), 'brain_inputs': len(brain.brain_roots),
             'body_receptors': len(brain.receptor_roots), 'local_edges': int(brain.local.nnz),
             'cross_edges': int(brain.neck.nnz), 'verified': sum(row['verified'] for row in results.values()),
             'total': len(results), 'epochs': 0, 'physics_steps': 0})
    print('passes', sum(row['verified'] for row in results.values()), '/', len(results), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--period', type=int, choices=(23, 30), default=PERIOD)
    torch.set_num_threads(1)
    arguments = parser.parse_args()
    main(arguments.out, arguments.period)
