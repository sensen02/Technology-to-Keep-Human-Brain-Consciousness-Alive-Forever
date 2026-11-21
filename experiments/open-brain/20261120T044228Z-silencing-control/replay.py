"""Paired offline source-silencing controls using the unchanged P1 electrode decoder."""
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

PROJECT = Path('/run/media/sensen/Data2/cell_wound_prototype')
BASE = PROJECT / 'experiments/open-brain/20261120T043500Z-neck-recording'
sys.path.insert(0, str(BASE))
from run import contact_layout, decode, lead_vectors, load_fixed_neck, read, reference_pairs, score, voltages


def replay():
    neck, catalog, params, _ = load_fixed_neck(PROJECT)
    paths = {p.root: p for p in neck}
    sums = defaultdict(lambda: defaultdict(Counter))
    trials = []
    for seed in (10901, 11311, 11701):
        prefix = BASE/'recordings'/f'{seed}_intact'
        with np.load(str(prefix)+'_reference.npz') as ref:
            roots, directions = ref['roots'], ref['direction']
            enter, arrive = ref['entry'], ref['arrival']
        lead = lead_vectors(paths, roots, contact_layout(catalog))
        noise = np.random.default_rng(seed).normal(0, .08, (len(enter), 16))
        u = np.random.default_rng([seed, 502]).random(enter.shape)
        dn = directions == 'DN'
        for arm in ('baseline', 'half_dn', 'no_dn', 'no_an'):
            keep = np.ones_like(enter)
            if arm == 'half_dn':
                keep[:, dn] = u[:, dn] < .5
            elif arm == 'no_dn':
                keep[:, dn] = False
            elif arm == 'no_an':
                keep[:, ~dn] = False
            dest = arrive.copy()
            for i, r in enumerate(roots):
                delay = params['ports'][r]['delay_ticks']
                dest[delay:, i] &= keep[:-delay, i]
            src = enter & keep
            uv = voltages(src, dest, lead, noise)
            pairs, _ = decode(uv[:, :8], uv[:, 8:])
            if arm == 'baseline':
                with np.load(str(prefix)+'_observation.npz') as obs:
                    assert np.array_equal(uv, np.column_stack([obs['head_uv'], obs['thorax_uv']]))
                    assert np.array_equal(pairs, obs['head_thorax_ticks'])
            result = score(pairs, reference_pairs(src, dest, roots, params, directions))
            for direction, row in result.items():
                sums[arm][direction].update(row)
            trials.append(dict(seed=seed, arm=arm, score=result))
    return dict(totals=sums, trials=trials)


if __name__ == '__main__':
    print(json.dumps(replay(), indent=2))
