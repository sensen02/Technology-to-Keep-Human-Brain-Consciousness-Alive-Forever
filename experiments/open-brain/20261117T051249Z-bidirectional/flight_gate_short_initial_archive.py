"""Generate audited VNC-only wing enable traces for the upstream flight controller."""
import argparse
from pathlib import Path

import numpy as np
import torch

from build_catalog import HERE, hash_file, read, record
from circuit import Circuit
from physical import Receiver

PREWARM = 850
TICKS = 2800
ENABLE_THRESHOLD_MV = .50


def main(out, short_trial):
    catalog = read(HERE / 'CATALOG.json')
    out.mkdir()
    conditions = ('intact', 'cut', 'cut_restore')
    cut_start, cut_end = (680, 830) if short_trial else (680, 1020)
    threshold = .70 if short_trial else ENABLE_THRESHOLD_MV
    per_case = {}
    for condition in conditions:
        circuit = Circuit(catalog)
        receiver = Receiver(circuit)
        columns = circuit.validate_inputs(catalog['groups']['wing_amplitude'], 'brain')
        drive = np.zeros((1, len(circuit.roots)), dtype=np.float32)
        cut_mask = np.array([root in catalog['groups']['wing_amplitude'] for root in circuit.port_roots])
        open_mask = np.zeros_like(cut_mask)
        records = []
        for tick in range(-PREWARM, TICKS):
            drive.fill(0)
            if tick % 35 == 0:
                drive[0, columns] = 68.75
            disconnected = condition == 'cut' or (condition == 'cut_restore' and cut_start <= tick < cut_end)
            emitted, arrived, _, voltage = circuit.step(drive, cut_mask if disconnected else open_mask)
            receiver.update(voltage, .0001)
            if tick >= 0:
                # The pretrained body needs its published gain to hold its trim.
                # A membrane response is required to enable that local controller.
                records.append([tick, receiver.pair('wing_amplitude'), receiver.pair('wing_amplitude') >= threshold,
                                int(emitted.sum()), int(arrived.sum())])
        data = np.asarray(records)
        np.save(out / f'{condition}.npy', data, allow_pickle=False)
        per_case[condition] = {'sha256': hash_file(out / f'{condition}.npy'),
                               'arrived': int(data[:, 4].sum()), 'enabled_ticks': int(data[:, 2].sum())}
    assert per_case['intact']['enabled_ticks'] == TICKS and per_case['cut']['arrived'] == 0
    assert per_case['cut']['enabled_ticks'] == 0 and 0 < per_case['cut_restore']['enabled_ticks'] < TICKS
    record(out / 'IDENTITY.json', {'catalog_sha256': hash_file(HERE / 'CATALOG.json'),
          'source': 'BANC wing_amplitude VNC membrane readout', 'threshold_mv': threshold,
          'cut_window_ticks': [cut_start, cut_end],
          'neural_prewarm_ticks': PREWARM, 'steps': TICKS, 'dt_s': .0001,
          'columns': ['tick', 'vnc_filtered_mv', 'gain_enable', 'emitted', 'arrived'], 'conditions': per_case})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path, nargs='?', default=HERE/'FLIGHT_GATE')
    parser.add_argument('--short', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    main(args.out, args.short)
