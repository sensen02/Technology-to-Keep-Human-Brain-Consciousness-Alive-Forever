"""P1 direction recording replay. Only decode.py receives the electrode observations."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from decode import decode

PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = PROJECT / 'experiments/open-brain'
ASSAY = BASE / '20261118T051014Z-neck-fault-assay/FORMAL'
SNAPSHOT = BASE / '20261119T202650Z-neck-odor-integration/omnisim_source'
sys.path.insert(0, str(SNAPSHOT))
from omnisim.modules.banc_neck import load_fixed_neck
from omnisim.modules.electrode_observation import ElectrodeObservation


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contact_layout(catalog):
    planes = []
    for key in ('front_nm', 'rear_nm'):
        xyz = np.array([v['geometry'][key] for v in catalog['ports'].values()]) * 1e-9
        x = np.linspace(xyz[:, 0].min(), xyz[:, 0].max(), 8)
        planes.extend([[a, xyz[0, 1], xyz[:, 2].max() + 3e-6] for a in x])
    return np.array(planes)


def lead_vectors(paths, roots, contacts):
    field = ElectrodeObservation(contacts, ())
    result = []
    for root in roots:
        path = paths[root].points_m
        arc = np.r_[0., np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
        samples = field._at(path, arc, [0., 2e-6, arc[-1], arc[-1]-2e-6])
        leads = field._lead(samples, contacts)
        # Endpoint current dipoles at the two source-grounded crossing planes.
        result.append((leads[[0, 2]] - leads[[1, 3]]) * 1e-9 * 1e6)
    return np.asarray(result)


def voltages(depart, arrive, leads, noise):
    field = depart @ leads[:, 0] + arrive @ leads[:, 1]
    field[1:] -= .35 * field[:-1].copy()
    return (field + noise).astype('float32')


def reference_pairs(depart, arrive, roots, parameters, directions):
    truth = []
    for i, root in enumerate(roots):
        delay = parameters['ports'][root]['delay_ticks']
        dest = np.flatnonzero(arrive[:, i])
        src = dest - delay
        if np.any(src < 0) or not np.all(depart[src, i]):
            raise ValueError(f'arrival without recorded source {root}')
        head, thorax = (src, dest) if directions[i] == 'DN' else (dest, src)
        truth.extend(zip(head, thorax))
    return np.array(truth, dtype=int).reshape(-1, 2)


def score(pred, truth):
    # Both plane timestamps must agree exactly; each true event can match only once.
    actual = Counter(map(tuple, truth))
    hits = Counter(map(tuple, pred)) & actual
    result = {}
    for label, head_first in [('down', True), ('up', False)]:
        count = sum(n for (h, t), n in hits.items() if (h < t) == head_first)
        called = int(np.sum((pred[:, 0] < pred[:, 1]) == head_first))
        total = int(np.sum((truth[:, 0] < truth[:, 1]) == head_first))
        result[label] = dict(correct=count, called=called, traversals=total)
    return result


def run(output):
    paths, catalog, params, identity = load_fixed_neck(PROJECT)
    paths = {p.root: p for p in paths}
    contacts = contact_layout(catalog)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(geometry=identity, catalog_counts=dict(Counter(p.direction for p in paths.values())),
                  sample_period_ms=0.1, contacts_m=contacts.tolist(), trials=[])
    for seed in (10901, 11311, 11701):
        folder = ASSAY / f'seed{seed}'
        meta = read(folder / 'RESULT.json')
        source = folder / 'NEURAL.npz'
        binding, = (Path(k) for k in meta['source_sha256'] if k.endswith('/BINDING.json'))
        if sha(source) != meta['neural_sha256'] or sha(binding) != meta['source_sha256'][str(binding)]:
            raise ValueError('changed archive')
        roots = meta['dn_roots'] + [r['banc_root'] for r in read(binding)['records']]
        directions = [paths[r].direction for r in roots]
        if directions != ['DN']*17 + ['AN']*2:
            raise ValueError('unexpected event columns')
        leads = lead_vectors(paths, roots, contacts)
        with np.load(source) as data:
            for lane, arm in enumerate(meta['metrics']):
                # DN output is latched for one neural tick before entering the neck.
                dn = data['dn_emitted'][:, lane]
                entry = np.column_stack((np.vstack((np.zeros_like(dn[:1]), dn[:-1])), data['an_emitted'][:, lane]))
                arrival = np.column_stack((data['dn_arrived'][:, lane], data['an_arrived'][:, lane]))
                truth = reference_pairs(entry, arrival, roots, params, directions)
                noise = np.random.default_rng(seed).normal(0., .08, (len(entry), len(contacts)))
                uv = voltages(entry, arrival, leads, noise)
                pair, counts = decode(uv[:, :8], uv[:, 8:])
                swapped, _ = decode(uv[:, 8:], uv[:, :8])
                if sorted(map(tuple, pair[:, ::-1])) != sorted(map(tuple, swapped)):
                    raise ValueError('electrode bank swap check failed')
                key = f"{seed}_{arm['mode']}"
                np.savez_compressed(output / (key+'_observation.npz'), head_uv=uv[:, :8], thorax_uv=uv[:, 8:],
                                    head_thorax_ticks=pair)
                np.savez_compressed(output / (key+'_reference.npz'), roots=np.array(roots), direction=np.array(directions),
                                    entry=entry, arrival=arrival)
                result['trials'].append(dict(seed=seed, arm=arm['mode'], archive_sha256=meta['neural_sha256'],
                                             source=str(source), peaks=counts, score=score(pair, truth)))
    (output/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    run(parser.parse_args().output)
