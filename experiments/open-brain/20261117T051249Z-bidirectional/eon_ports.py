"""Exercise reviewed FlyWire descending matches through the same BANC runtime."""
from collections import Counter
from pathlib import Path
import sys

import numpy as np
import torch

from circuit import Circuit, FOUNDATION, HERE, TRANSMISSION_TICKS, eon
from build_catalog import hash_file, read, record

sys.path.append(str(FOUNDATION))
from run_probe import FrozenCPUBrain, load_connectome
from sources import flywire_annotations, verify_upstream


def mapping(catalog, index):
    classification = flywire_annotations('classification')
    types = flywire_annotations('consolidated_cell_types')
    ports = {root: row for root, row in catalog['ports'].items() if row['direction'] == 'DN'}
    multiplicity = Counter(row['fafb_match'] for row in ports.values())
    admitted, excluded = {}, {}
    for root, row in ports.items():
        wire = row['fafb_match']
        identity = classification.get(wire, {})
        name = types.get(wire, {}).get('primary_type')
        label = row['cell_type']
        same_type = name == label or (label.startswith(('DNg74_', 'DNg02_')) and name == label.split('_')[0])
        if label == 'MDN' and name == 'DNp50':
            same_type = True
        if (wire and multiplicity[wire] == 1 and int(wire) in index and same_type
                and identity.get('side') == row['side'] and identity.get('super_class') == 'descending'):
            admitted[root] = wire
        else:
            excluded[root] = {'group': row['group'], 'match': wire, 'flywire_type': name,
                              'flywire_side': identity.get('side'), 'multiplicity': multiplicity[wire]}
    return admitted, excluded


@torch.inference_mode()
def main():
    verify_upstream()
    catalog = read(HERE / 'CATALOG.json')
    index, _ = eon.get_hash_tables(eon.path_comp)
    admitted, excluded = mapping(catalog, index)
    weights = load_connectome(index)
    groups = {key: [root for root in roots if root in admitted] for key, roots in catalog['groups'].items()}
    groups = {k: v for k, v in groups.items() if v}
    frozen = FrozenCPUBrain(len(groups), weights)
    state = frozen.state_init()
    stream = Circuit(catalog, len(groups))
    alternate = Circuit(catalog, len(groups))
    rates = torch.zeros((len(groups), len(index)))
    for lane, roots in enumerate(groups.values()):
        rates[lane, [index[int(admitted[root])] for root in roots]] = 160
    zero_rates = torch.zeros_like(rates)
    zero_drive = np.zeros((len(groups), len(stream.roots)), dtype=np.float32)
    open_mask = np.zeros(len(stream.ports), dtype=bool)
    cut_dn = np.array([root in stream.brain_roots for root in stream.port_roots], dtype=bool)
    source_columns = [index[int(wire)] for wire in admitted.values()]
    target_columns = [stream.port_index[root] for root in admitted]
    incoming = np.zeros((len(groups), len(stream.ports)), dtype=bool)
    rng = torch.Generator().manual_seed(986)
    emitted, delivered, cut_trace = [], [], []
    for tick in range(800):
        state, _ = frozen(rates if 100 <= tick < 650 else zero_rates, *state, generator=rng)
        incoming[:, target_columns] = state[2][:, source_columns].numpy().astype(bool)
        emitted.append(incoming.copy())
        _, arrivals, _, _ = stream.step(zero_drive, open_mask, eon_output=incoming)
        _, interrupted, _, _ = alternate.step(zero_drive, cut_dn if 320 <= tick < 490 else open_mask,
                                               eon_output=incoming)
        delivered.append(arrivals)
        cut_trace.append(interrupted)
        if (tick+1) % 200 == 0:
            print(tick+1, flush=True)
    emitted, delivered, cut_trace = map(np.asarray, (emitted, delivered, cut_trace))
    assert np.array_equal(emitted[:-TRANSMISSION_TICKS], delivered[TRANSMISSION_TICKS:])
    assert np.array_equal(delivered[:320], cut_trace[:320])
    assert not np.any(cut_trace[320:490+TRANSMISSION_TICKS])
    assert np.array_equal(delivered[490+TRANSMISSION_TICKS:], cut_trace[490+TRANSMISSION_TICKS:])
    result = {}
    for lane, (key, roots) in enumerate(groups.items()):
        result[key] = {root: int(emitted[:, lane, stream.port_index[root]].sum()) for root in roots}
    out = HERE / 'EON'
    out.mkdir()
    np.savez_compressed(out / 'EVENTS.npz', emitted=emitted, delivered=delivered, cut=cut_trace)
    record(out / 'RESULT.json', {'admitted': admitted, 'excluded': excluded, 'direct_probe_counts': result,
          'runtime_port_order': stream.port_roots, 'catalog_sha256': hash_file(HERE / 'CATALOG.json'),
          'neurons': len(index), 'whole_brain_connections': weights.nnz, 'full_delay_and_cut_pass': True,
          'total_emitted': int(emitted.sum()), 'total_blocked': int(delivered.sum() - cut_trace.sum()),
          'epochs': 0})


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
